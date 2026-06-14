import math
import torch
import torch.nn as nn

from src.embedding import Embedding
from src.rmsNorm import RMSNorm
from src.multiHeadAttention import MultiHeadAttention
from src.groupedQueryAttention import GroupedQueryAttention
from src.feedForwardNetwork import FeedForwardNetwork
import src.config as config

from torch.utils.checkpoint import checkpoint

# Sliding window config: every GLOBAL_ATTENTION_INTERVAL-th layer uses full
# global attention; all other layers use a local sliding window.
GLOBAL_ATTENTION_INTERVAL = 4
DEFAULT_WINDOW_SIZE = 512

class Block(nn.Module):
    def __init__(self, num_heads, d_model, hidden_dim_ffn, num_kv_heads=None, use_gqa=None, window_size=None):
        super().__init__()
        self.norm_1 = RMSNorm(dim=d_model)
        
        if use_gqa is None:
            use_gqa = getattr(config, 'use_gqa', False)
        if num_kv_heads is None:
            num_kv_heads = getattr(config, 'num_kv_heads', None)
            
        if use_gqa and num_kv_heads is not None:
            self.mha = GroupedQueryAttention(num_heads, num_kv_heads, d_model, window_size=window_size)
        else:
            self.mha = MultiHeadAttention(num_heads, d_model, window_size=window_size)
        self.norm_2 = RMSNorm(dim=d_model)
        self.ffn = FeedForwardNetwork(dim=d_model, hidden_dim=hidden_dim_ffn)

    def forward(self, x, attn_mask=None, kv_cache=None, position_offset=0):
        # Pre-LN Residual Connections (out-of-place)
        h1, new_kv_cache = self.mha(self.norm_1(x), attn_mask=attn_mask, kv_cache=kv_cache, position_offset=position_offset)
        x1 = x + h1
        h2 = self.ffn(self.norm_2(x1))
        x2 = x1 + h2
        return x2, new_kv_cache

class GPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length, num_layers, num_heads, d_model, hidden_dim_ffn, num_kv_heads=None, use_gqa=None):
        super().__init__()
        self.num_layers = num_layers  # Store layer count for initialization scaling
        self.embedding = Embedding(vocab_size, embedding_dim, context_length)
        
        # Build blocks with interleaved sliding window / global attention
        # Every GLOBAL_ATTENTION_INTERVAL-th layer (4th, 8th, 12th...) gets global attention.
        # All other layers get local sliding window attention.
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            # Layer indices are 0-based; every 4th layer (index 3, 7, 11...) is global
            is_global = ((i + 1) % GLOBAL_ATTENTION_INTERVAL == 0)
            ws = None if is_global else DEFAULT_WINDOW_SIZE
            self.blocks.append(
                Block(num_heads, d_model, hidden_dim_ffn, num_kv_heads, use_gqa, window_size=ws)
            )
        
        self.final_norm = RMSNorm(dim=embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Tie weights between embedding and final linear projection head
        self.lm_head.weight = self.embedding.token_embedding.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # Scaled initialization for residual projections
            if getattr(module, "is_residual_projection", False):
                std = std / math.sqrt(2 * self.num_layers)
            
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, xb, attn_mask=None, kv_cache=None, position_offset=0):
        """
        Forward pass with optional KV cache for fast autoregressive inference.
        
        Args:
            xb: Input token IDs [B, T]
            attn_mask: Optional attention mask
            kv_cache: List of (K, V) tuples per layer, or None for training/prefill
            position_offset: Position offset for RoPE during cached inference
            
        Returns:
            logits: [B, T, vocab_size]
            new_kv_cache: List of (K, V) tuples per layer (only when kv_cache is not None
                          or when use_kv_cache is implicitly enabled)
        """
        batch_size, seq_len = xb.shape
        x = self.embedding(xb)
        
        use_cache = kv_cache is not None
        new_kv_cache = []
        
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if use_cache else None
            
            if self.training:
                # Activation checkpointing during training — no KV cache needed
                # checkpoint doesn't support extra return values well, so we wrap it
                x = checkpoint(
                    self._block_forward_no_cache, block, x, attn_mask,
                    use_reentrant=False
                )
            else:
                x, layer_kv = block(x, attn_mask=attn_mask, kv_cache=layer_cache, position_offset=position_offset)
                new_kv_cache.append(layer_kv)
                
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        if self.training:
            return logits
        return logits, new_kv_cache

    @staticmethod
    def _block_forward_no_cache(block, x, attn_mask):
        """Wrapper for activation checkpointing — discards KV cache output."""
        x, _ = block(x, attn_mask=attn_mask)
        return x
