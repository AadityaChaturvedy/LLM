import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import multiprocessing as mp

import random
import time
import math
import datetime
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext

from src.dataset import FineWebDataset, BilingualHindiDataset
from src.tokenizer_utils import Tokenizer
from src.custom_tokenizer import CustomTokenizer
from src.encoder import Encoder
from src.config import (
    TRAIN_TOKENIZER, TRAIN_LLM,
    eval_every, eval_steps, save_every, max_grad_norm,
    train_loop, patience, min_delta,
    vocab_size, embedding_dim, TOKENIZER_ROWS,
    context_length, batch_size_encoder, 
    num_heads, d_model,
    hidden_dim_ffn,
    TOKENIZED_DATA_PATH, TOKENIZER_MERGES_PATH, TOKENIZER_VOCAB_PATH,
    num_layers,
    accumulation_steps,
    LANGUAGE,
    use_gqa, num_kv_heads,
    MoE, moe_aux_loss_weight,
    moe_aux_loss_warmup_steps, moe_log_every
)

from src.encoder import get_batch, load_tokenized_data, TokenPrefetcher
from src.model import GPT
import src.config as config

def main():
    # 1. Initialize process group (DDP or single-process)
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        
        backend = os.environ.get("DDP_BACKEND", "nccl")
        dist.init_process_group(
            backend=backend,
            device_id=torch.device(device),
            timeout=datetime.timedelta(hours=2)
        )
        is_master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        is_master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize single-process group to support Muon or other distributed utilities
        backend = "nccl" if "cuda" in str(device) else "gloo"
        dist.init_process_group(
            backend=backend,
            init_method="tcp://127.0.0.1:0",
            rank=0,
            world_size=1,
            timeout=datetime.timedelta(hours=2)
        )

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("log", exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join("log", f"train_{run_id}.log")

    def log_line(message):
        if is_master_process:
            print(message)
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(message + "\n")

    def unwrap_model(wrapped_model):
        raw_model = wrapped_model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        if hasattr(raw_model, "module"):
            raw_model = raw_model.module
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        return raw_model

    def collect_moe_stats(wrapped_model, device):
        raw_model = unwrap_model(wrapped_model)
        stats = []
        for block in getattr(raw_model, "blocks", []):
            if not getattr(block, "use_moe", False):
                continue
            block_stats = getattr(block.ffn, "last_stats", {})
            if block_stats:
                kept = block_stats["kept_per_expert"].float()
                stats.append(torch.stack([
                    block_stats["drop_rate"].float(),
                    block_stats["router_entropy"].float(),
                    kept.min(),
                    kept.max(),
                    block_stats["capacity"].float(),
                ]))
        if not stats:
            return None
        stats_tensor = torch.stack(stats).mean(dim=0).to(device)
        if ddp:
            dist.all_reduce(stats_tensor, op=dist.ReduceOp.AVG)
        return stats_tensor

    seed = 1337
    random.seed(seed + ddp_rank)
    np.random.seed(seed + ddp_rank)
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
    if is_master_process and TRAIN_TOKENIZER:
        if not os.path.exists(TOKENIZER_MERGES_PATH) and not os.path.exists(TOKENIZER_VOCAB_PATH):
            log_line(f"Tokenized data not found for language '{LANGUAGE}'. Starting dataset preparation on rank 0...")
            if LANGUAGE == "hindi":
                tokenizer = CustomTokenizer()
                tokenizer.train("hindi", max_docs_hindi=TOKENIZER_ROWS)
                dataset = BilingualHindiDataset(hindi_ratio=1.0)
            elif LANGUAGE == "hinglish":
                tokenizer = CustomTokenizer()
                tokenizer.train("hinglish", max_docs_hindi=TOKENIZER_ROWS, max_docs_english=TOKENIZER_ROWS)
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

    if not TRAIN_LLM:
        log_line("TRAIN_LLM is set to False in config. Exiting.")
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    # Instantiate GPT Model
    import json
    if os.path.exists(TOKENIZER_VOCAB_PATH):
        with open(TOKENIZER_VOCAB_PATH, "r", encoding="utf-8") as f:
            model_vocab_size = len(json.load(f))
    else:
        model_vocab_size = vocab_size

    model_config = {
        "vocab_size": model_vocab_size,
        "embedding_dim": embedding_dim,
        "context_length": context_length,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "d_model": d_model,
        "hidden_dim_ffn": hidden_dim_ffn,
        "use_gqa": use_gqa,
        "MoE": MoE,
        "moe_num_experts": config.moe_num_experts,
        "moe_top_k": config.moe_top_k,
        "moe_capacity_factor": config.moe_capacity_factor,
        "moe_eval_capacity_factor": config.moe_eval_capacity_factor,
        "moe_min_capacity": config.moe_min_capacity,
        "moe_aux_loss_weight": moe_aux_loss_weight,
        "moe_aux_loss_warmup_steps": moe_aux_loss_warmup_steps,
        "moe_router_z_loss_weight": config.moe_router_z_loss_weight,
        "moe_router_noise_std": config.moe_router_noise_std,
        "moe_router_temperature": config.moe_router_temperature,
        "moe_num_shared_experts": config.moe_num_shared_experts,
        "moe_shared_expert_weight": config.moe_shared_expert_weight,
        "moe_renormalize_after_drop": config.moe_renormalize_after_drop,
        "language": LANGUAGE,
    }

    model = GPT(
        vocab_size=model_vocab_size,
        embedding_dim=embedding_dim,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        d_model=d_model,
        hidden_dim_ffn=hidden_dim_ffn,
        use_gqa=use_gqa
    )
    if "cuda" in str(device) and amp_dtype == torch.bfloat16:
        model.to(device, dtype=torch.bfloat16)
    else:
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
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        
        state_dict = checkpoint["model"]
        clean_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("_orig_mod.", "").replace("module.", "")
            clean_state_dict[name] = v
        missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=not MoE)
        if MoE and (missing_keys or unexpected_keys):
            log_line(
                f"Loaded checkpoint with MoE strict=False "
                f"({len(missing_keys)} missing, {len(unexpected_keys)} unexpected keys)."
            )
        
        start_step = checkpoint.get("step", 0)
        log_line(f"Resumed from {resume_path} at step {start_step}")

    if start_step >= train_loop:
        log_line(f"Checkpoint step {start_step} is >= train_loop {train_loop}; nothing to train.")
        if dist.is_initialized():
            dist.destroy_process_group()
        raise SystemExit(0)

    # Wrap model with DDP
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # Pass find_unused_parameters=True if MoE is enabled, as some experts may not be routed to in a given step.
        model = DDP(model, device_ids=[ddp_local_rank], gradient_as_bucket_view=True, find_unused_parameters=MoE)

    # Compile model for faster training
    if "cuda" in str(device):
        log_line("Compiling the model with torch.compile...")
        model = torch.compile(model)

    tokens = load_tokenized_data()

    log_line(f"Total tokens in dataset: {len(tokens):,}")
    split_idx = int(0.9 * len(tokens))
    tokens_train = tokens[:split_idx]
    tokens_val = tokens[split_idx:]
    log_line(f"Train tokens: {len(tokens_train):,} | Val tokens: {len(tokens_val):,}")

    all_parameters = list(model.parameters())
    param_count = sum(p.numel() for p in all_parameters)
    log_line(f"Total parameters (tied lm_head): {param_count:,}")

    if "cuda" in str(device):
        try:
            from muon import MuonWithAuxAdam
            muon_params = []
            adamw_params = []
            for name, param in model.named_parameters():
                if param.ndim >= 2 and 'embed' not in name:
                    muon_params.append(param)
                else:
                    adamw_params.append(param)
                    
            param_groups = [
                dict(params=muon_params, use_muon=True, lr=0.02, momentum=0.95, weight_decay=0.01),
                dict(params=adamw_params, use_muon=False, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
            ]
            optimizer = MuonWithAuxAdam(param_groups)
            
            # Muon is incompatible with GradScaler; force bf16 path
            if scaler is not None:
                scaler = None
                log_line("WARNING: GradScaler disabled — Muon requires bfloat16, not float16.")
        except ImportError:
            log_line("Muon not installed; falling back to AdamW.")
            optimizer = optim.AdamW(all_parameters, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    else:
        optimizer = optim.AdamW(all_parameters, lr=3e-4)
    
    if resume_path:
        optimizer.load_state_dict(checkpoint["optimizer"])

    warmup_steps = min(2000, max(100, train_loop // 100))
    stable_end = int(0.8 * train_loop)  # WSD: stable phase ends at 80%
    def lr_lambda(current_step):
        # Warmup-Stable-Decay (WSD) Schedule
        # Phase 1: Linear warmup
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        # Phase 2: Stable — hold at max LR
        elif current_step < stable_end:
            return 1.0
        # Phase 3: Cosine decay in the final 20%
        else:
            progress = (current_step - stable_end) / max(1, train_loop - stable_end)
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if resume_path:
        scheduler.load_state_dict(checkpoint["scheduler"])

    def estimate_loss():
        model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(eval_steps):
                xb, yb = next(eval_prefetcher)
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", dtype=amp_dtype):
                    logits, _ = model(xb)
                    B, T, C = logits.shape
                    loss = F.cross_entropy(logits.view(B * T, C), yb.view(B * T))
                losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    best_val_loss = float("inf")
    patience_counter = 0

    model.train()

    effective_batch_size = batch_size_encoder * accumulation_steps * ddp_world_size
    tokens_per_step = effective_batch_size * context_length
    steps_per_epoch = max(1, len(tokens_train) // tokens_per_step)

    train_prefetcher = TokenPrefetcher(tokens_train, batch_size_encoder, context_length)
    eval_prefetcher = TokenPrefetcher(tokens_val, batch_size_encoder, context_length)

    try:
        for step in range(start_step, train_loop):
            step_start = time.time()
            optimizer.zero_grad(set_to_none=True)

            accumulated_loss = 0.0
            for micro_step in range(accumulation_steps):
                # Only synchronize gradients on the final accumulation step
                is_last_micro_step = (micro_step == accumulation_steps - 1)
                
                # Context manager to avoid gradient sync until the last micro step
                if ddp and not is_last_micro_step:
                    ctx = model.no_sync()
                else:
                    ctx = nullcontext()

                with ctx:
                    xb, yb = next(train_prefetcher)
                    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

                    with torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", dtype=amp_dtype):
                        if MoE:
                            logits, aux_loss = model(xb, return_aux_loss=True)
                        else:
                            logits = model(xb)
                            aux_loss = logits.new_zeros(())
                        B, T, C = logits.shape
                        logits_flat = logits.view(B * T, C)
                        targets_flat = yb.view(B * T)
                        lm_loss = F.cross_entropy(logits_flat, targets_flat)
                        aux_warmup = min(1.0, (step + 1) / max(1, moe_aux_loss_warmup_steps))
                        aux_weight = moe_aux_loss_weight * aux_warmup
                        loss = (lm_loss + aux_weight * aux_loss) / accumulation_steps

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

            if "cuda" in str(device) and (step + 1) % eval_every == 0:
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

            if MoE and (step + 1) % moe_log_every == 0:
                moe_stats = collect_moe_stats(model, device)
                if is_master_process and moe_stats is not None:
                    drop_rate, router_entropy, expert_min, expert_max, capacity = moe_stats.tolist()
                    log_line(
                        f"\nMoE @ step {step + 1} | "
                        f"aux_weight: {aux_weight:.5f} | "
                        f"drop_rate: {drop_rate:.4f} | "
                        f"router_entropy: {router_entropy:.4f} | "
                        f"expert_kept_min/max: {expert_min:.1f}/{expert_max:.1f} | "
                        f"capacity: {capacity:.1f}"
                    )

            if (step + 1) % eval_every == 0:
                val_loss = estimate_loss()
                # Average val loss across GPUs
                if ddp:
                    val_loss_tensor = torch.tensor(val_loss, device=device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                    val_loss = val_loss_tensor.item()
                log_line(f"\nEval @ step {step + 1} | Val loss: {val_loss:.4f}")

                should_stop = 0
                if is_master_process:
                    if val_loss < best_val_loss - min_delta:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model separately
                        raw_model = model.module if hasattr(model, "module") else model
                        if hasattr(raw_model, "_orig_mod"):
                            raw_model = raw_model._orig_mod
                        torch.save({
                            "step": step + 1,
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "model": raw_model.state_dict(),
                            "val_loss": best_val_loss,
                            "model_config": model_config,
                        }, "checkpoints/best_model.pt")
                        log_line(f"New best val loss: {best_val_loss:.4f} — saved best_model.pt")
                    else:
                        patience_counter += 1
                        log_line(f"No improvement. Patience: {patience_counter}/{patience} (Continuing training)")


            if is_master_process and (step + 1) % save_every == 0:
                raw_model = model.module if hasattr(model, "module") else model
                if hasattr(raw_model, "_orig_mod"):
                    raw_model = raw_model._orig_mod
                checkpoint = {
                    "step": step + 1,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "model": raw_model.state_dict(),
                    "model_config": model_config,
                }
                torch.save(checkpoint, f"checkpoints/ckpt_step_{step + 1}.pt")

        # Save final model checkpoint upon successful completion of training loop
        if is_master_process:
            log_line(f"\nTraining completed successfully. Saving final model at checkpoints/final_model.pt...")
            raw_model = model.module if hasattr(model, "module") else model
            if hasattr(raw_model, "_orig_mod"):
                raw_model = raw_model._orig_mod
            checkpoint = {
                "step": train_loop,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "model": raw_model.state_dict(),
                "model_config": model_config,
            }
            torch.save(checkpoint, "checkpoints/final_model.pt")
            if train_loop % save_every != 0:
                torch.save(checkpoint, f"checkpoints/ckpt_step_{train_loop}.pt")
            
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
                "model_config": model_config,
            }
            torch.save(checkpoint, f"checkpoints/ckpt_step_{step + 1}.pt")
        if dist.is_initialized():
            dist.destroy_process_group()
        train_prefetcher.stop()
        eval_prefetcher.stop()
        raise SystemExit(0)

    train_prefetcher.stop()
    eval_prefetcher.stop()
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
