import math
import torch
from torch import nn
from src.rotary import RotaryEmbedding, apply_rotary_pos_emb


class QKNorm(nn.Module):
    """Parameterless RMSNorm applied along the head dimension for QK normalization."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps)

class GroupedQueryAttention(nn.Module):
    def __init__(self, num_heads, num_kv_heads, d_model, window_size=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        self.d_k = d_model // num_heads
        assert self.d_k % 2 == 0, "d_k must be even for Rotary Embeddings"
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.window_size = window_size  # None = global attention, int = local sliding window

        self.wq = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.wk = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.wv = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        
        self.out_linear = nn.Linear(d_model, d_model, bias=False)
        self.out_linear.is_residual_projection = True
        
        self.rotary_emb = RotaryEmbedding(self.d_k)
        
        # QK Normalization layers
        self.q_norm = QKNorm(self.d_k)
        self.k_norm = QKNorm(self.d_k)
        
        # Learnable per-head scale parameter for QK norm
        self.q_scale = nn.Parameter(torch.ones(1, num_heads, 1, 1))

    def _make_sliding_window_mask(self, T, device, dtype):
        """Create a causal sliding window attention mask.
        
        Returns a [T, T] mask where True = masked (ignored), matching SDPA convention
        when passed as attn_mask with additive masking.
        """
        # Row i can attend to columns max(0, i - window_size + 1) ... i
        row_idx = torch.arange(T, device=device).unsqueeze(1)  # [T, 1]
        col_idx = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
        
        # Causal: can only attend to past and present
        causal_mask = col_idx > row_idx
        # Window: can only attend within window_size positions back
        window_mask = col_idx < (row_idx - self.window_size + 1)
        
        # Combined: mask out anything that's either future OR outside the window
        mask = causal_mask | window_mask
        
        # Convert to float mask for SDPA: 0 = attend, -inf = ignore
        float_mask = torch.zeros(T, T, device=device, dtype=dtype)
        float_mask.masked_fill_(mask, float('-inf'))
        return float_mask

    def forward(self, x_norm, attn_mask=None, kv_cache=None, position_offset=0):
        B, T, C = x_norm.shape

        # Linear projections
        Q = self.wq(x_norm).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.wk(x_norm).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.wv(x_norm).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)

        # Apply RoPE — use position_offset for correct positions during cached inference
        total_len = position_offset + T
        cos, sin = self.rotary_emb(Q, seq_len=total_len)
        # Slice to only the positions for current tokens
        cos = cos[position_offset:total_len]
        sin = sin[position_offset:total_len]
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # QK Normalization (Applied after RoPE to match modern LLM architectures)
        Q = self.q_norm(Q)
        K = self.k_norm(K)

        # Apply learnable per-head scaling
        Q = Q * self.q_scale

        # KV Cache: concatenate past keys/values for autoregressive inference
        new_kv_cache = None
        if kv_cache is not None:
            past_k, past_v = kv_cache
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)
            new_kv_cache = (K, V)
            # During cached inference (T=1), use full causal attention over accumulated KV
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=None,
                is_causal=False,  # Not needed: Q has 1 token, all past K/V are valid
                enable_gqa=True
            )
        else:
            # Store cache for first pass if we're starting a cached inference session
            new_kv_cache = (K, V)
            
            # Training or prefill: handle sliding window vs global attention
            if self.window_size is not None and self.training:
                # Sliding window attention during training
                sw_mask = self._make_sliding_window_mask(T, x_norm.device, x_norm.dtype)
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    Q, K, V,
                    attn_mask=sw_mask,
                    is_causal=False,  # Mask already encodes causality
                    enable_gqa=True
                )
            else:
                # Global causal attention (full attention layers, or inference prefill)
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    Q, K, V,
                    attn_mask=attn_mask,
                    is_causal=True if attn_mask is None else False,
                    enable_gqa=True
                )

        # Concatenate heads and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            B, T, self.num_heads * self.d_k
        )

        output = self.out_linear(attn_output)
        return output, new_kv_cache
