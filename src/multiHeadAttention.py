import math

import torch
from torch import nn


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

    def forward(self, x_norm, batch_size, context_length):

        # Linear projections
        Q = self.wq(x_norm).view(batch_size, context_length, self.num_heads, self.d_k).transpose(1, 2)
        K = self.wk(x_norm).view(batch_size, context_length, self.num_heads, self.d_k).transpose(1, 2)
        V = self.wv(x_norm).view(batch_size, context_length, self.num_heads, self.d_k).transpose(1, 2)

        # Efficient scaled dot-product attention (utilizes FlashAttention-2 / Memory-Efficient attention)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V,
            is_causal=True
        )

        # Concatenate heads and pass through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, context_length, self.num_heads * self.d_k
        )

        output = self.out_linear(attn_output)
        return output
