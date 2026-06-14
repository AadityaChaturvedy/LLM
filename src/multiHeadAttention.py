import math
import torch
from torch import nn

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [B, H, T, d_k]
    # cos, sin: [T, d_k]
    cos = cos.unsqueeze(0).unsqueeze(1) # [1, 1, T, d_k]
    sin = sin.unsqueeze(0).unsqueeze(1) # [1, 1, T, d_k]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build initial cache
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len):
        if seq_len > self.max_seq_len_cached:
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().to(x.dtype)
            sin = emb.sin().to(x.dtype)
            return cos, sin
        
        return (
            self.cos_cached[:seq_len].to(device=x.device, dtype=x.dtype),
            self.sin_cached[:seq_len].to(device=x.device, dtype=x.dtype),
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, window_size=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        assert self.d_k % 2 == 0, "d_k must be even for Rotary Embeddings"
        self.num_heads = num_heads
        self.window_size = window_size  # None = global attention, int = local sliding window

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.out_linear = nn.Linear(d_model, d_model, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.d_k)

    def _make_sliding_window_mask(self, T, device, dtype):
        """Create a causal sliding window attention mask."""
        row_idx = torch.arange(T, device=device).unsqueeze(1)
        col_idx = torch.arange(T, device=device).unsqueeze(0)
        
        causal_mask = col_idx > row_idx
        window_mask = col_idx < (row_idx - self.window_size + 1)
        mask = causal_mask | window_mask
        
        float_mask = torch.zeros(T, T, device=device, dtype=dtype)
        float_mask.masked_fill_(mask, float('-inf'))
        return float_mask

    def forward(self, x_norm, attn_mask=None, kv_cache=None, position_offset=0):
        B, T, C = x_norm.shape

        # Linear projections
        Q = self.wq(x_norm).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.wk(x_norm).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.wv(x_norm).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Apply RoPE with position offset for cached inference
        total_len = position_offset + T
        cos, sin = self.rotary_emb(Q, seq_len=total_len)
        cos = cos[position_offset:total_len]
        sin = sin[position_offset:total_len]
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # KV Cache for autoregressive inference
        new_kv_cache = None
        if kv_cache is not None:
            past_k, past_v = kv_cache
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)
            new_kv_cache = (K, V)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=None,
                is_causal=False
            )
        else:
            new_kv_cache = (K, V)
            
            if self.window_size is not None and self.training:
                sw_mask = self._make_sliding_window_mask(T, x_norm.device, x_norm.dtype)
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    Q, K, V,
                    attn_mask=sw_mask,
                    is_causal=False
                )
            else:
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    Q, K, V,
                    attn_mask=attn_mask,
                    is_causal=True if attn_mask is None else False
                )

        # Concatenate heads and pass through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            B, T, self.num_heads * self.d_k
        )

        output = self.out_linear(attn_output)
        return output, new_kv_cache
