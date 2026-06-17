# Architecture & Training

## Architecture

The model (`src/model.py`, `src/groupedQueryAttention.py`) combines several modern decoder-only design choices:

- **Grouped-Query Attention (GQA):** default config uses `num_heads=16`, `num_kv_heads=4` (`src/config.py`).
- **Interleaved local/global attention:** every `GLOBAL_ATTENTION_INTERVAL = 4`th layer attends globally (full causal attention); all other layers use a causal sliding window of `DEFAULT_WINDOW_SIZE = 512` tokens. This keeps most layers cheap while periodically letting the model see the full context.
- **RoPE:** rotary position embeddings applied to Q/K, with explicit position-offset handling for cached autoregressive decoding.
- **QK-Norm + RMSNorm:** a parameterless RMSNorm is applied to Q and K after RoPE (with a learnable per-head scale on Q), and standard RMSNorm is used pre-attention and pre-FFN (pre-norm residual blocks).
- **SwiGLU FFN:** `down_proj(silu(gate_proj(x)) * up_proj(x))`.
- **Weight tying:** the token embedding matrix and the LM head share weights.
- **Attention backend:** computed via PyTorch's native `scaled_dot_product_attention` with `enable_gqa=True` — this dispatches to the FlashAttention-2 kernel on supported GPUs without depending on the separate `flash-attn` package.

## Training

`train.py`:

- **Optimizer:** `MuonWithAuxAdam` — Muon (lr `0.02`, momentum `0.95`) for ≥2D non-embedding weight matrices, AdamW (lr `3e-4`, betas `0.9/0.95`) for embeddings and 1D params. Falls back to plain AdamW if `muon` isn't installed.
- **LR schedule:** Warmup → Stable (80% of `train_loop`) → Decay (cosine, final 20%).
- **Distributed:** PyTorch DDP, multi-GPU.
- **Memory/throughput:** activation checkpointing (`torch.utils.checkpoint`), `torch.compile`, mixed precision (bf16 preferred, fp16 + `GradScaler` fallback on hardware without bf16 support — note Muon forces the bf16 path and disables `GradScaler`).

## Fine-Tuning

`finetune/`:

- **`finetune_lora.py`** — LoRA fine-tuning (target modules: `wq, wk, wv, out_linear, w_gate, w_up, w_down`) on a mixture of `ai4bharat/indic-align` configs (an Airavata-style instruction mixture), optimized with `bitsandbytes` `AdamW8bit`. → `sft_checkpoints_lora/ckpt_lora_fused.pt`
- **`finetune_instruct.py`** — full fine-tuning on `ai4bharat/indic-align` (`OpenAssistant_T` split), loss masked to only the response tokens (`ignore_index=-100`). → `sft_checkpoints_instruct/ckpt_instruct_epoch_2.pt`
- **`finetune_qa.py`** — full fine-tuning for extractive QA on a local `data/hindi_squad_large.jsonl`, generated from `l3cube-pune/indic-squad` via `finetune/extract_large_squad.py`; loss similarly masked to the answer span. → `sft_checkpoints/ckpt_sft_epoch_3.pt`

Base model checkpoint used for evaluation below: `checkpoints/ckpt_step_120000.pt`.
