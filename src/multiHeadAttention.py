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

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            inv_freq = self.inv_freq.to(x.device)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().to(x.dtype)
            sin = emb.sin().to(x.dtype)
            return cos, sin
        
        return (
            self.cos_cached[:seq_len].to(device=x.device, dtype=x.dtype),
            self.sin_cached[:seq_len].to(device=x.device, dtype=x.dtype),
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.out_linear = nn.Linear(d_model, d_model, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.d_k)

    def forward(self, x_norm, batch_size=None, context_length=None):
        B, T, C = x_norm.shape

        # Linear projections
        Q = self.wq(x_norm).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.wk(x_norm).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.wv(x_norm).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(Q, seq_len=T)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # Efficient scaled dot-product attention (utilizes FlashAttention-2 / Memory-Efficient attention)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V,
            is_causal=True
        )

        # Concatenate heads and pass through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            B, T, self.num_heads * self.d_k
        )

        output = self.out_linear(attn_output)
        return output
