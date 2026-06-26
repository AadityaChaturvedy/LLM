# Architecture & Training

## Architecture

The model (`src/model.py`, `src/groupedQueryAttention.py`) combines several modern decoder-only design choices:

- **Grouped-Query Attention (GQA):** default config uses `num_heads=16`, `num_kv_heads=4` (`src/config.py`).
- **Interleaved local/global attention:** every `GLOBAL_ATTENTION_INTERVAL = 4`th layer attends globally (full causal attention); all other layers use a causal sliding window of `DEFAULT_WINDOW_SIZE = 512` tokens. This keeps most layers cheap while periodically letting the model see the full context.
- **RoPE:** rotary position embeddings applied to Q/K, with explicit position-offset handling for cached autoregressive decoding.
- **QK-Norm + RMSNorm:** a parameterless RMSNorm is applied to Q and K after RoPE (with a learnable per-head scale on Q), and standard RMSNorm is used pre-attention and pre-FFN (pre-norm residual blocks).
- **SwiGLU FFN:** `down_proj(silu(gate_proj(x)) * up_proj(x))`.
- **Optional Mixture-of-Experts FFN:** set `MoE = True` in `src/config.py` to replace the dense SwiGLU FFN with a capacity-limited top-k MoE layer (`src/mixtureOfExperts.py`). With `MoE = False`, blocks use the standard dense FFN.
- **Weight tying:** the token embedding matrix and the LM head share weights.
- **Attention backend:** computed via PyTorch's native `scaled_dot_product_attention` with `enable_gqa=True` — this dispatches to the FlashAttention-2 kernel on supported GPUs without depending on the separate `flash-attn` package.

## Mixture-of-Experts

The MoE implementation keeps the same Transformer block interface while adding sparse expert capacity:

- **Token-choice routing:** each token is routed to `moe_top_k` experts according to a learned router over `moe_num_experts`.
- **Router temperature and noise:** `moe_router_temperature` controls routing sharpness, and `moe_router_noise_std` adds training-time jitter to reduce early expert collapse.
- **Expert capacity:** each expert accepts up to `ceil(capacity_factor * num_tokens * top_k / num_experts)` assignments, with `moe_min_capacity` as a floor. `moe_capacity_factor` is used in training and `moe_eval_capacity_factor` in eval.
- **Post-drop renormalization:** if capacity overflow drops some assignments, surviving expert weights can be renormalized with `moe_renormalize_after_drop = True`, preserving routed FFN scale.
- **Shared experts:** `moe_num_shared_experts` adds always-on dense expert paths on top of routed experts, scaled by `moe_shared_expert_weight`.
- **Auxiliary router losses:** the layer returns a Switch/GShard-style load-balancing loss plus router z-loss. `GPT.forward(..., return_aux_loss=True)` averages this auxiliary loss over MoE layers before training code applies `moe_aux_loss_weight`.
- **Diagnostics:** each MoE block records lightweight stats (`drop_rate`, `router_entropy`, `kept_per_expert`, `capacity`) for training logs.

Important config knobs in `src/config.py`:

```python
MoE = False
moe_num_experts = 8
moe_top_k = 2
moe_capacity_factor = 1.25
moe_eval_capacity_factor = 2.0
moe_min_capacity = 4
moe_aux_loss_weight = 0.01
moe_aux_loss_warmup_steps = 1000
moe_router_z_loss_weight = 0.001
moe_router_noise_std = 0.1
moe_router_temperature = 1.0
moe_num_shared_experts = 1
moe_shared_expert_weight = 1.0
moe_renormalize_after_drop = True
moe_log_every = 100
```

## Training

`train.py`:

- **Optimizer:** `MuonWithAuxAdam` — Muon (lr `0.02`, momentum `0.95`) for ≥2D non-embedding weight matrices, AdamW (lr `3e-4`, betas `0.9/0.95`) for embeddings and 1D params. Falls back to plain AdamW if `muon` isn't installed.
- **LR schedule:** Warmup → Stable (80% of `train_loop`) → Decay (cosine, final 20%).
- **Distributed:** PyTorch DDP, multi-GPU.
- **Memory/throughput:** activation checkpointing (`torch.utils.checkpoint`), `torch.compile`, mixed precision (bf16 preferred, fp16 + `GradScaler` fallback on hardware without bf16 support — note Muon forces the bf16 path and disables `GradScaler`).
- **MoE training path:** when `MoE = True`, training calls `model(..., return_aux_loss=True)` and optimizes `cross_entropy + aux_weight * aux_loss`. `aux_weight` linearly warms up to `moe_aux_loss_weight` over `moe_aux_loss_warmup_steps`.
- **MoE logging:** every `moe_log_every` steps, `train.py` logs average MoE drop rate, router entropy, per-expert kept-assignment min/max, expert capacity, and current auxiliary-loss weight. DDP ranks all participate in the stats reduction; only the master rank prints.

## Fine-Tuning

`finetune/`:

- **`finetune_lora.py`** — LoRA fine-tuning (target modules: `wq, wk, wv, out_linear, w_gate, w_up, w_down`) on a mixture of `ai4bharat/indic-align` configs (an Airavata-style instruction mixture), optimized with `bitsandbytes` `AdamW8bit`. → `sft_checkpoints_lora/ckpt_lora_fused.pt`
- **`finetune_instruct.py`** — full fine-tuning on `ai4bharat/indic-align` (`OpenAssistant_T` split), loss masked to only the response tokens (`ignore_index=-100`). If `MoE = True`, it includes the same warmed MoE auxiliary loss as pretraining. → `sft_checkpoints_instruct/ckpt_instruct_epoch_2.pt`
- **`finetune_qa.py`** — full fine-tuning for extractive QA on a local `data/hindi_squad_large.jsonl`, generated from `l3cube-pune/indic-squad` via `finetune/extract_large_squad.py`; loss similarly masked to the answer span. If `MoE = True`, it includes the warmed MoE auxiliary loss. → `sft_checkpoints/ckpt_sft_epoch_3.pt`
