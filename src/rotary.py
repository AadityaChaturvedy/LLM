import torch
from torch import nn

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q: [B, num_heads, T, d_k]
    # k: [B, num_kv_heads, T, d_k]
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
