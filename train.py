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
    # Check for GPU availability and set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("log", exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join("log", f"train_{run_id}.log")

    def log_line(message):
        print(message)
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(message + "\n")

    seed = 1337
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    log_line(f"Using device: {device}")

    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        use_scaler = False
        log_line("Using bfloat16 mixed precision (no scaling required).")
    elif device.type == "cuda":
        amp_dtype = torch.float16
        use_scaler = True
        log_line("Using float16 mixed precision with GradScaler.")
    else:
        amp_dtype = torch.float32
        use_scaler = False
        log_line("Running on CPU; using standard float32 precision.")

    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    if not os.path.exists(TOKENIZER_MERGES_PATH) and not os.path.exists(TOKENIZER_VOCAB_PATH):
        log_line(f"Tokenized data not found for language '{LANGUAGE}'. Starting dataset preparation on rank 0...")
        '''Tokenize the text'''
        if LANGUAGE == "hindi":
            tokenizer = CustomTokenizer()
            tokenizer.train("hindi", max_docs_hindi=TOTAL_ROWS)
            dataset = BilingualHindiDataset()
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
            dataset = BilingualHindiDataset()
        elif LANGUAGE == "hinglish":
            dataset = BilingualHindiDataset()
        else:
            dataset = FineWebDataset()
            
        '''Encode the tokens'''
        encoder = Encoder()
        encoder.EncodeTokens(dataset.dataset)

    else:
        log_line(f"Tokenized data found at {TOKENIZED_DATA_PATH}; skipping tokenizer training and encoding.")

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

    if device.type == "cuda":
        model = torch.compile(model)

    tokens_np = load_tokenized_data()
    tokens = torch.tensor(tokens_np, dtype=torch.long, device="cpu")

    log_line(f"Total tokens in dataset: {len(tokens):,}")
    split_idx = int(0.9 * len(tokens))
    tokens_train = tokens[:split_idx]
    tokens_val = tokens[split_idx:]
    if device.type == "cuda":
        tokens_train = tokens_train.pin_memory()
        tokens_val = tokens_val.pin_memory()
    log_line(f"Train tokens: {len(tokens_train):,} | Val tokens: {len(tokens_val):,}")

    all_parameters = list(model.parameters())

    last_param_count = None
    param_count = sum(p.numel() for p in all_parameters)
    if param_count != last_param_count:
        log_line(f"Total parameters (tied lm_head): {param_count:,}")
        last_param_count = param_count

    # Use fused AdamW on CUDA device to speed up updates and reduce VRAM overhead
    use_fused = (device.type == "cuda")
    optimizer = optim.AdamW(all_parameters, lr=3e-4, fused=use_fused)
    warmup_steps = min(2000, max(100, train_loop // 100))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        progress = (current_step - warmup_steps) / max(1, train_loop - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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

    def estimate_loss(data):
        model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(eval_steps):
                xb, yb = get_batch(data, batch_size_encoder, context_length)
                xb, yb = xb.to(device), yb.to(device)
                # Autocast validation forward pass
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                    logits = model(xb)
                    B, T, C = logits.shape
                    loss = F.cross_entropy(logits.view(B * T, C), yb.view(B * T))
                losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    model.train()

    start_step = 0
    resume_path = get_latest_checkpoint("checkpoints")
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = checkpoint.get("step", 0)
        log_line(f"Resumed from {resume_path} at step {start_step}")

    if start_step >= train_loop:
        log_line(
            f"Checkpoint step {start_step} is >= train_loop {train_loop}; "
            "nothing to train."
        )
        raise SystemExit(0)

    effective_batch_size = batch_size_encoder * accumulation_steps
    tokens_per_step = effective_batch_size * context_length
    steps_per_epoch = len(tokens_train) / tokens_per_step

    try:
        for step in range(start_step, train_loop):
            torch.cuda.synchronize()
            step_start = time.time()
            optimizer.zero_grad()

            accumulated_loss = 0.0
            for _ in range(accumulation_steps):
                xb, yb = get_batch(tokens_train, batch_size_encoder, context_length)
                xb, yb = xb.to(device), yb.to(device)

                # 1. Forward pass under autocast
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                    logits = model(xb)
                    B, T, C = logits.shape
                    logits_flat = logits.view(B * T, C)
                    targets_flat = yb.view(B * T)
                    # Scale the loss by accumulation steps
                    loss = F.cross_entropy(logits_flat, targets_flat) / accumulation_steps

                # 2. Backward step
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accumulated_loss += loss.item()

            # 3. Optimization step (after accumulating gradients)
            if scaler is not None:
                scaler.unscale_(optimizer) # Unscale gradients before clipping
                torch.nn.utils.clip_grad_norm_(all_parameters, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(all_parameters, max_grad_norm)
                optimizer.step()

            scheduler.step()

            torch.cuda.synchronize()
            step_time = time.time() - step_start
            tokens_per_sec = (batch_size_encoder * accumulation_steps * context_length) / max(step_time, 1e-8)

            epoch_val = (step + 1) / steps_per_epoch
            epoch_num = int(step / steps_per_epoch) + 1
            percent = (step % steps_per_epoch) / steps_per_epoch
            
            bar_length = 20
            filled_length = int(round(bar_length * percent))
            bar = '█' * filled_length + '░' * (bar_length - filled_length)

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
                log_line(f"\nEval @ step {step + 1} | Val loss: {val_loss:.4f}")

            if (step + 1) % save_every == 0:
                checkpoint = {
                    "step": step + 1,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "model": model.state_dict(),
                }
                torch.save(checkpoint, f"checkpoints/ckpt_step_{step + 1}.pt")
    except KeyboardInterrupt:
        log_line(f"\nTraining interrupted by user. Saving emergency checkpoint at step {step + 1}...")
        checkpoint = {
            "step": step + 1,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model": model.state_dict(),
        }
        torch.save(checkpoint, f"checkpoints/ckpt_step_{step + 1}.pt")
        raise SystemExit(0)

if __name__ == "__main__":
    main()