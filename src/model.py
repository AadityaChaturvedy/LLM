import math
import torch
import torch.nn as nn

from src.rmsNorm import RMSNorm
from src.multiHeadAttention import MultiHeadAttention
from src.groupedQueryAttention import GroupedQueryAttention
from src.feedForwardNetwork import FeedForwardNetwork
from src.mixtureOfExperts import MixtureOfExperts
import src.config as config
from src.config import ACTIVATION_CHECKPOINTING, WINDOW_SIZE, GLOBAL_ATTENTION_INTERVAL

from torch.utils.checkpoint import checkpoint


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
        self.use_moe = getattr(config, "MoE", False)
        if self.use_moe:
            self.ffn = MixtureOfExperts(
                dim=d_model,
                hidden_dim=hidden_dim_ffn,
                num_experts=getattr(config, "moe_num_experts", 8),
                top_k=getattr(config, "moe_top_k", 2),
                capacity_factor=getattr(config, "moe_capacity_factor", 1.25),
                eval_capacity_factor=getattr(config, "moe_eval_capacity_factor", 2.0),
                min_capacity=getattr(config, "moe_min_capacity", 4),
                router_z_loss_weight=getattr(config, "moe_router_z_loss_weight", 0.001),
                router_noise_std=getattr(config, "moe_router_noise_std", 0.0),
                router_temperature=getattr(config, "moe_router_temperature", 1.0),
                num_shared_experts=getattr(config, "moe_num_shared_experts", 0),
                shared_expert_weight=getattr(config, "moe_shared_expert_weight", 1.0),
                renormalize_after_drop=getattr(config, "moe_renormalize_after_drop", True),
            )
        else:
            self.ffn = FeedForwardNetwork(dim=d_model, hidden_dim=hidden_dim_ffn)

    def forward(self, x, attn_mask=None, kv_cache=None, position_offset=0):
        # Pre-LN Residual Connections (out-of-place)
        h1, new_kv_cache = self.mha(self.norm_1(x), attn_mask=attn_mask, kv_cache=kv_cache, position_offset=position_offset)
        x1 = x + h1
        if self.use_moe:
            h2, aux_loss = self.ffn(self.norm_2(x1))
        else:
            h2 = self.ffn(self.norm_2(x1))
            aux_loss = x1.new_zeros(())
        x2 = x1 + h2
        return x2, new_kv_cache, aux_loss


class GPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length, num_layers, num_heads, d_model, hidden_dim_ffn, num_kv_heads=None, use_gqa=None):
        super().__init__()
        self.num_layers = num_layers  # Store layer count for initialization scaling
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Build blocks with interleaved sliding window / global attention
        # Every GLOBAL_ATTENTION_INTERVAL-th layer (4th, 8th, 12th...) gets global attention.
        # All other layers get local sliding window attention.
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            # Layer indices are 0-based; every 4th layer (index 3, 7, 11...) is global
            is_global = ((i + 1) % GLOBAL_ATTENTION_INTERVAL == 0)
            ws = None if is_global else WINDOW_SIZE
            self.blocks.append(
                Block(num_heads, d_model, hidden_dim_ffn, num_kv_heads, use_gqa, window_size=ws)
            )
        
        self.final_norm = RMSNorm(dim=embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Tie weights between embedding and final linear projection head
        self.lm_head.weight = self.embedding.weight

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

    def load_state_dict(self, state_dict, strict=True):
        if "embedding.token_embedding.weight" in state_dict:
            state_dict = dict(state_dict)
            state_dict["embedding.weight"] = state_dict.pop("embedding.token_embedding.weight")
        return super().load_state_dict(state_dict, strict=strict)


    def forward(self, xb, attn_mask=None, kv_cache=None, position_offset=0, return_aux_loss=False):
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
        aux_loss = x.new_zeros(())
        aux_loss_count = 0
        
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if use_cache else None
            
            if self.training:
                if ACTIVATION_CHECKPOINTING:
                    # Activation checkpointing during training — no KV cache needed
                    # checkpoint doesn't support extra return values well, so we wrap it
                    x, block_aux_loss = checkpoint(
                       self._block_forward_no_cache, block, x, attn_mask,
                        use_reentrant=False
                    )
                else:
                    # Forward pass directly without activation checkpointing to avoid redundant compute
                    x, _, block_aux_loss = block(x, attn_mask=attn_mask)
                aux_loss = aux_loss + block_aux_loss
                if getattr(block, "use_moe", False):
                    aux_loss_count += 1
            else:
                x, layer_kv, _ = block(x, attn_mask=attn_mask, kv_cache=layer_cache, position_offset=position_offset)
                new_kv_cache.append(layer_kv)
                
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        if self.training:
            if return_aux_loss:
                if aux_loss_count > 0:
                    aux_loss = aux_loss / aux_loss_count
                return logits, aux_loss
            return logits
        return logits, new_kv_cache

    @staticmethod
    def _block_forward_no_cache(block, x, attn_mask):
        """Wrapper for activation checkpointing — discards KV cache output."""
        x, _, aux_loss = block(x, attn_mask=attn_mask)
        return x, aux_loss
