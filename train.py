import os
import random
import time
import math

import torch
from torch import nn, optim
import torch.nn.functional as F

from src.dataset import FineWebDataset
from src.tokenizer_utils import Tokenizer
from src.encoder import Encoder
from src.config import (
    eval_every, eval_steps, save_every, max_grad_norm,
    train_loop,
    vocab_size, embedding_dim, 
    context_length, batch_size_encoder, 
    num_heads, d_model,
    hidden_dim_ffn,
    TOKENIZED_DATA_PATH,
    num_layers
)
from src.encoder import get_batch, load_tokenized_data
from src.model import GPT

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

# Check for GPU availability and set device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
log_line(f"Using device: {device}")

# Determine mixed precision settings based on device capabilities
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

'''Tokenize the text'''
tokenizer = Tokenizer()
if not os.path.exists(TOKENIZED_DATA_PATH):
    '''Get Dataset'''
    dataset = FineWebDataset()
    tokenizer.train(dataset.dataset)
else:
    log_line(f"Tokenized data found at {TOKENIZED_DATA_PATH}; skipping tokenizer training.")

'''Encode the tokens'''
encoder = Encoder()
if not os.path.exists(TOKENIZED_DATA_PATH):
    encoder.EncodeTokens(dataset.dataset)
else:
    log_line(f"Tokenized data found at {TOKENIZED_DATA_PATH}; skipping encoding.")

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

tokens_np = load_tokenized_data()
tokens = torch.tensor(tokens_np, dtype=torch.long, device=device)

log_line(f"Total tokens in dataset: {len(tokens):,}")
split_idx = int(0.9 * len(tokens))
tokens_train = tokens[:split_idx]
tokens_val = tokens[split_idx:]
log_line(f"Train tokens: {len(tokens_train):,} | Val tokens: {len(tokens_val):,}")

all_parameters = list(model.parameters())

last_param_count = None
param_count = sum(p.numel() for p in all_parameters)
if param_count != last_param_count:
    log_line(f"Total parameters (tied lm_head): {param_count:,}")
    last_param_count = param_count

optimizer = optim.AdamW(all_parameters, lr=3e-4)
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

for step in range(start_step, train_loop):
    torch.cuda.synchronize()
    step_start = time.time()
    optimizer.zero_grad()
    xb, yb = get_batch(tokens_train, batch_size_encoder, context_length)

    # 1. Forward pass under autocast
    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
        logits = model(xb)
        B, T, C = logits.shape
        logits_flat = logits.view(B * T, C)
        targets_flat = yb.view(B * T)
        loss = F.cross_entropy(logits_flat, targets_flat)

    # 2. Backward & Optimization step
    if scaler is not None:
        # float16 requires scaling to prevent underflow
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer) # Unscale gradients before clipping
        torch.nn.utils.clip_grad_norm_(all_parameters, max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        # bfloat16/float32 doesn't need scaling
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_parameters, max_grad_norm)
        optimizer.step()

    scheduler.step()

    torch.cuda.synchronize()
    step_time = time.time() - step_start
    tokens_per_sec = (B * T) / max(step_time, 1e-8)
    log_line(
        f"Step {step + 1} | Loss: {loss.item():.4f} | "
        f"Tokens/sec: {tokens_per_sec:,.0f}"
    )

    if (step + 1) % eval_every == 0:
        val_loss = estimate_loss(tokens_val)
        log_line(f"Eval @ step {step + 1} | Val loss: {val_loss:.4f}")

    if (step + 1) % save_every == 0:
        checkpoint = {
            "step": step + 1,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model": model.state_dict(),
        }
        torch.save(checkpoint, f"checkpoints/ckpt_step_{step + 1}.pt")
