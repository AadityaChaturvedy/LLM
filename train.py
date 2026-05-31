import os
import sys
import multiprocessing as mp

import random
import time
import math

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.distributed as dist

from src.dataset import FineWebDataset, BilingualHindiDataset
from src.tokenizer_utils import Tokenizer
from src.custom_tokenizer import CustomTokenizer
from src.encoder import Encoder
from src.config import (
    eval_every, eval_steps, save_every, max_grad_norm,
    train_loop,
    vocab_size, embedding_dim, TOTAL_ROWS,
    context_length, batch_size_encoder, 
    num_heads, d_model,
    hidden_dim_ffn,
    TOKENIZED_DATA_PATH, TOKENIZER_MERGES_PATH, TOKENIZER_VOCAB_PATH,
    num_layers,
    accumulation_steps,
    LANGUAGE
)

from src.encoder import get_batch, load_tokenized_data
from src.model import GPT

def main():
    # 1. Initialize DDP if environment variables are set
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        
        backend = os.environ.get("DDP_BACKEND", "nccl")
        dist.init_process_group(backend=backend, device_id=torch.device(device))
        is_master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        is_master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("log", exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join("log", f"train_{run_id}.log")

    def log_line(message):
        if is_master_process:
            print(message)
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(message + "\n")

    seed = 1337
    random.seed(seed + ddp_rank)
    torch.manual_seed(seed + ddp_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + ddp_rank)

    log_line(f"Using device: {device} (Rank: {ddp_rank}, World Size: {ddp_world_size})")

    # Determine amp dtype
    if "cuda" in str(device) and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        use_scaler = False
        log_line("Using bfloat16 mixed precision (no scaling required).")
    elif "cuda" in str(device):
        amp_dtype = torch.float16
        use_scaler = True
        log_line("Using float16 mixed precision with GradScaler.")
    else:
        amp_dtype = torch.float32
        use_scaler = False
        log_line("Running on CPU; using standard float32 precision.")

    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    # Only Master process handles tokenizer setup & dataset tokenization
    if is_master_process:
        if not os.path.exists(TOKENIZER_MERGES_PATH) and not os.path.exists(TOKENIZER_VOCAB_PATH):
            log_line(f"Tokenized data not found for language '{LANGUAGE}'. Starting dataset preparation on rank 0...")
            if LANGUAGE == "hindi":
                tokenizer = CustomTokenizer()
                tokenizer.train("hindi", max_docs_hindi=TOTAL_ROWS)
                dataset = BilingualHindiDataset(hindi_ratio=1.0)
            elif LANGUAGE == "hinglish":
                tokenizer = CustomTokenizer()
                tokenizer.train("hinglish", max_docs_hindi=TOTAL_ROWS, max_docs_english=TOTAL_ROWS)
                dataset = BilingualHindiDataset()
            else:
                tokenizer = Tokenizer()
                dataset = FineWebDataset()
                tokenizer.train(dataset.dataset)
        
        if not os.path.exists(TOKENIZED_DATA_PATH):
            if LANGUAGE == "hindi":
                dataset = BilingualHindiDataset(hindi_ratio=1.0)
            elif LANGUAGE == "hinglish":
                dataset = BilingualHindiDataset()
            else:
                dataset = FineWebDataset()
                
            encoder = Encoder()
            encoder.EncodeTokens(dataset.dataset)
        else:
            log_line(f"Tokenized data found at {TOKENIZED_DATA_PATH}; skipping tokenizer training and encoding.")

    # All processes synchronize here before loading tokenized data
    if ddp:
        dist.barrier()

    # Instantiate GPT Model
    model = GPT(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        d_model=d_model,
        hidden_dim_ffn=hidden_dim_ffn
    )
    model.to(device)

    # Checkpoint loading (done before DDP wrapping or compiling)
    start_step = 0
    def get_latest_checkpoint(checkpoint_dir):
        if not os.path.isdir(checkpoint_dir):
            return None
        latest_step = -1
        latest_path = None
        for name in os.listdir(checkpoint_dir):
            if not (name.startswith("ckpt_step_") and name.endswith(".pt")):
                continue
            step_str = name[len("ckpt_step_") : -len(".pt")]
            try:
                step = int(step_str)
            except ValueError:
                continue
            if step > latest_step:
                latest_step = step
                latest_path = os.path.join(checkpoint_dir, name)
        return latest_path

    resume_path = get_latest_checkpoint("checkpoints")
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        
        state_dict = checkpoint["model"]
        clean_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("_orig_mod.", "").replace("module.", "")
            clean_state_dict[name] = v
        model.load_state_dict(clean_state_dict)
        
        start_step = checkpoint.get("step", 0)
        log_line(f"Resumed from {resume_path} at step {start_step}")

    if start_step >= train_loop:
        log_line(f"Checkpoint step {start_step} is >= train_loop {train_loop}; nothing to train.")
        if ddp:
            dist.destroy_process_group()
        raise SystemExit(0)

    # Wrap model with DDP
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[ddp_local_rank])

    # Compile model for faster training
    if "cuda" in str(device):
        log_line("Compiling the model with torch.compile...")
        model = torch.compile(model)

    tokens_np = load_tokenized_data()
    tokens = torch.tensor(tokens_np, dtype=torch.long, device="cpu")

    log_line(f"Total tokens in dataset: {len(tokens):,}")
    split_idx = int(0.9 * len(tokens))
    tokens_train = tokens[:split_idx]
    tokens_val = tokens[split_idx:]
    if "cuda" in str(device):
        tokens_train = tokens_train.pin_memory()
        tokens_val = tokens_val.pin_memory()
    log_line(f"Train tokens: {len(tokens_train):,} | Val tokens: {len(tokens_val):,}")

    all_parameters = list(model.parameters())
    param_count = sum(p.numel() for p in all_parameters)
    log_line(f"Total parameters (tied lm_head): {param_count:,}")

    # Use 8-bit AdamW on CUDA device to reduce VRAM overhead and run purely on GPU
    if "cuda" in str(device):
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(all_parameters, lr=3e-4)
    else:
        optimizer = optim.AdamW(all_parameters, lr=3e-4)
    
    if resume_path:
        optimizer.load_state_dict(checkpoint["optimizer"])

    warmup_steps = min(2000, max(100, train_loop // 100))
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        progress = (current_step - warmup_steps) / max(1, train_loop - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if resume_path:
        scheduler.load_state_dict(checkpoint["scheduler"])

    def estimate_loss(data):
        model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(eval_steps):
                xb, yb = get_batch(data, batch_size_encoder, context_length)
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", dtype=amp_dtype):
                    logits = model(xb)
                    B, T, C = logits.shape
                    loss = F.cross_entropy(logits.view(B * T, C), yb.view(B * T))
                losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    model.train()

    effective_batch_size = batch_size_encoder * accumulation_steps * ddp_world_size
    tokens_per_step = effective_batch_size * context_length
    steps_per_epoch = len(tokens_train) / tokens_per_step

    try:
        for step in range(start_step, train_loop):
            if "cuda" in str(device):
                torch.cuda.synchronize()
            step_start = time.time()
            optimizer.zero_grad()

            accumulated_loss = 0.0
            for micro_step in range(accumulation_steps):
                # Only synchronize gradients on the final accumulation step
                is_last_micro_step = (micro_step == accumulation_steps - 1)
                
                # Context manager to avoid gradient sync until the last micro step
                if ddp and not is_last_micro_step:
                    ctx = model.no_sync()
                else:
                    from contextlib import nullcontext
                    ctx = nullcontext()

                with ctx:
                    xb, yb = get_batch(tokens_train, batch_size_encoder, context_length)
                    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

                    with torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", dtype=amp_dtype):
                        logits = model(xb)
                        B, T, C = logits.shape
                        logits_flat = logits.view(B * T, C)
                        targets_flat = yb.view(B * T)
                        loss = F.cross_entropy(logits_flat, targets_flat) / accumulation_steps

                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    accumulated_loss += loss.item()

            # Reduce loss across all GPUs for display/logging
            if ddp:
                loss_tensor = torch.tensor(accumulated_loss, device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                accumulated_loss = loss_tensor.item()

            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_parameters, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(all_parameters, max_grad_norm)
                optimizer.step()

            scheduler.step()

            if "cuda" in str(device):
                torch.cuda.synchronize()
            step_time = time.time() - step_start
            tokens_per_sec = tokens_per_step / max(step_time, 1e-8)

            epoch_val = (step + 1) / steps_per_epoch
            epoch_num = int(step / steps_per_epoch) + 1
            percent = (step % steps_per_epoch) / steps_per_epoch
            
            bar_length = 20
            filled_length = int(round(bar_length * percent))
            bar = '█' * filled_length + '░' * (bar_length - filled_length)

            if is_master_process:
                print(
                    f"\rEpoch {epoch_num} [{bar}] {percent*100:.1f}% | Loss : {accumulated_loss:.4f} | Tokens/sec: {tokens_per_sec:,.0f}", 
                    end="", 
                    flush=True
                )

                with open(log_path, "a", encoding="utf-8") as log_file:
                    log_file.write(
                        f"Step {step + 1} | Epoch {epoch_val:.2f} | Loss: {accumulated_loss:.4f} | Tokens/sec: {tokens_per_sec:,.0f}\n"
                    )

            if (step + 1) % eval_every == 0:
                val_loss = estimate_loss(tokens_val)
                # Average val loss across GPUs
                if ddp:
                    val_loss_tensor = torch.tensor(val_loss, device=device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                    val_loss = val_loss_tensor.item()
                log_line(f"\nEval @ step {step + 1} | Val loss: {val_loss:.4f}")

            if is_master_process and (step + 1) % save_every == 0:
                raw_model = model.module if hasattr(model, "module") else model
                if hasattr(raw_model, "_orig_mod"):
                    raw_model = raw_model._orig_mod
                checkpoint = {
                    "step": step + 1,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "model": raw_model.state_dict(),
                }
                torch.save(checkpoint, f"checkpoints/ckpt_step_{step + 1}.pt")
                
    except KeyboardInterrupt:
        if is_master_process:
            log_line(f"\nTraining interrupted by user. Saving emergency checkpoint at step {step + 1}...")
            raw_model = model.module if hasattr(model, "module") else model
            if hasattr(raw_model, "_orig_mod"):
                raw_model = raw_model._orig_mod
            checkpoint = {
                "step": step + 1,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "model": raw_model.state_dict(),
            }
            torch.save(checkpoint, f"checkpoints/ckpt_step_{step + 1}.pt")
        if ddp:
            dist.destroy_process_group()
        raise SystemExit(0)

    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()