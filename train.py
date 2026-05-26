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
from src.embedding import Embedding
from src.config import (
    eval_every, eval_steps, save_every, max_grad_norm,
    train_loop,
    vocab_size, embedding_dim, 
    context_length, batch_size_encoder, 
    num_heads, d_model,
    hidden_dim_ffn,
    TOKENIZED_DATA_PATH
)
from src.encoder import get_batch, load_tokenized_data
from src.rmsNorm import RMSNorm
from src.multiHeadAttention import MultiHeadAttention
from src.feedForwardNetwork import FeedForwardNetwork

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)
run_id = time.strftime("%Y%m%d-%H%M%S")
log_path = os.path.join("logs", f"train_{run_id}.log")

def log_line(message):
    print(message)
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")

seed = 1337
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
log_line(f"Using device: {device}")

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

'''Embedding'''

embedding_layer = Embedding(vocab_size, embedding_dim, context_length)
norm_1 = RMSNorm(dim=embedding_dim)
mha = MultiHeadAttention(num_heads, d_model)
norm_2 = RMSNorm(dim=embedding_dim)
ffn = FeedForwardNetwork(dim=d_model, hidden_dim=hidden_dim_ffn)
final_norm = RMSNorm(dim=embedding_dim)
lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

nn.init.normal_(lm_head.weight, mean=0.0, std=0.02)
nn.init.normal_(embedding_layer.token_embedding.weight, mean=0.0, std=0.02)
lm_head.weight = embedding_layer.token_embedding.weight

modules = [embedding_layer, norm_1, mha, norm_2, ffn, final_norm, lm_head]
for module in modules:
    module.to(device)

tokens = load_tokenized_data()
log_line(f"Total tokens in dataset: {len(tokens):,}")
split_idx = int(0.9 * len(tokens))
tokens_train = tokens[:split_idx]
tokens_val = tokens[split_idx:]
log_line(f"Train tokens: {len(tokens_train):,} | Val tokens: {len(tokens_val):,}")

all_parameters = (
    list(embedding_layer.parameters()) +
    list(norm_1.parameters()) +
    list(mha.parameters()) +
    list(norm_2.parameters()) +
    list(ffn.parameters()) +
    list(final_norm.parameters())
)

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

def set_mode(is_train):
    for module in modules:
        module.train(is_train)

def forward(xb):
    x = embedding_layer(xb)
    x_norm = norm_1(x)

    attention_output = mha(x_norm, xb.size(0), context_length)
    x = x + attention_output

    x_norm = norm_2(x)
    ffn_output = ffn(x_norm)
    x = x + ffn_output

    x_final = final_norm(x)
    return lm_head(x_final)

def estimate_loss(data):
    set_mode(False)
    losses = []
    with torch.no_grad():
        for _ in range(eval_steps):
            xb, yb = get_batch(data, batch_size_encoder, context_length)
            xb = xb.to(device)
            yb = yb.to(device)
            logits = forward(xb)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), yb.view(B * T))
            losses.append(loss.item())
    set_mode(True)
    return sum(losses) / len(losses)

set_mode(True)

start_step = 0
resume_path = get_latest_checkpoint("checkpoints")
if resume_path:
    checkpoint = torch.load(resume_path, map_location=device)
    embedding_layer.load_state_dict(checkpoint["embedding_layer"])
    norm_1.load_state_dict(checkpoint["norm_1"])
    mha.load_state_dict(checkpoint["mha"])
    norm_2.load_state_dict(checkpoint["norm_2"])
    ffn.load_state_dict(checkpoint["ffn"])
    final_norm.load_state_dict(checkpoint["final_norm"])
    lm_head.load_state_dict(checkpoint["lm_head"])
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

    step_start = time.time()
    optimizer.zero_grad()
    xb, yb = get_batch(tokens_train, batch_size_encoder, context_length)
    xb = xb.to(device)
    yb = yb.to(device)
    logits = forward(xb)
    B, T, C = logits.shape

    logits_flat = logits.view(B * T, C)
    targets_flat = yb.view(B * T)

    loss = F.cross_entropy(logits_flat, targets_flat)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(all_parameters, max_grad_norm)
    optimizer.step()
    scheduler.step()

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
            "embedding_layer": embedding_layer.state_dict(),
            "norm_1": norm_1.state_dict(),
            "mha": mha.state_dict(),
            "norm_2": norm_2.state_dict(),
            "ffn": ffn.state_dict(),
            "final_norm": final_norm.state_dict(),
            "lm_head": lm_head.state_dict(),
        }
        torch.save(checkpoint, f"checkpoints/ckpt_step_{step + 1}.pt")