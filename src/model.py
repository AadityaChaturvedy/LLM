import torch
import torch.nn as nn

from src.embedding import Embedding
from src.rmsNorm import RMSNorm
from src.multiHeadAttention import MultiHeadAttention
from src.feedForwardNetwork import FeedForwardNetwork

from torch.utils.checkpoint import checkpoint

class Block(nn.Module):
    def __init__(self, num_heads, d_model, hidden_dim_ffn):
        super().__init__()
        self.norm_1 = RMSNorm(dim=d_model)
        self.mha = MultiHeadAttention(num_heads, d_model)
        self.norm_2 = RMSNorm(dim=d_model)
        self.ffn = FeedForwardNetwork(dim=d_model, hidden_dim=hidden_dim_ffn)

    def forward(self, x, batch_size, context_length):
        # Pre-LN Residual Connections (out-of-place)
        h1 = self.mha(self.norm_1(x), batch_size, context_length)
        x1 = x + h1
        h2 = self.ffn(self.norm_2(x1))
        x2 = x1 + h2
        return x2

class GPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length, num_layers, num_heads, d_model, hidden_dim_ffn):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim, context_length)
        self.blocks = nn.ModuleList([
            Block(num_heads, d_model, hidden_dim_ffn) for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(dim=embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Tie weights between embedding and final linear projection head
        self.lm_head.weight = self.embedding.token_embedding.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, xb):
        batch_size, seq_len = xb.shape
        x = self.embedding(xb)
        
        for block in self.blocks:
            if self.training:
                # Activation checkpointing to reduce VRAM usage
                x = checkpoint(block, x, batch_size, seq_len, use_reentrant=False)
            else:
                x = block(x, batch_size, seq_len)
                
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits
