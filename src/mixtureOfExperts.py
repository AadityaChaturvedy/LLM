import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.feedForwardNetwork import FeedForwardNetwork


class MixtureOfExperts(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        num_experts,
        top_k=2,
        capacity_factor=1.25,
        eval_capacity_factor=2.0,
        min_capacity=4,
        router_z_loss_weight=0.001,
        router_noise_std=0.0,
        router_temperature=1.0,
        num_shared_experts=0,
        shared_expert_weight=1.0,
        renormalize_after_drop=True,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        if router_temperature <= 0:
            raise ValueError("router_temperature must be > 0")
        if num_shared_experts < 0:
            raise ValueError("num_shared_experts must be >= 0")

        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.router_z_loss_weight = router_z_loss_weight
        self.router_noise_std = router_noise_std
        self.router_temperature = router_temperature
        self.num_shared_experts = num_shared_experts
        self.shared_expert_weight = shared_expert_weight
        self.renormalize_after_drop = renormalize_after_drop
        self.last_stats = {}

        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            FeedForwardNetwork(dim=dim, hidden_dim=hidden_dim)
            for _ in range(num_experts)
        )
        self.shared_experts = nn.ModuleList(
            FeedForwardNetwork(dim=dim, hidden_dim=hidden_dim)
            for _ in range(num_shared_experts)
        )

    def _capacity(self, num_tokens):
        factor = self.capacity_factor if self.training else self.eval_capacity_factor
        capacity = math.ceil(factor * num_tokens * self.top_k / self.num_experts)
        return max(self.min_capacity, capacity)

    def _router_loss(self, router_probs, top_indices, router_logits):
        # Switch Transformer-style load balancing: match probability mass to actual
        # routed token fraction so all experts get both gradients and traffic.
        tokens_per_expert = F.one_hot(top_indices, self.num_experts).float()
        tokens_per_expert = tokens_per_expert.mean(dim=(0, 1))
        prob_per_expert = router_probs.mean(dim=0)
        load_balance_loss = (
            self.num_experts
            * torch.sum(tokens_per_expert * prob_per_expert)
        )

        z_loss = torch.logsumexp(router_logits.float(), dim=-1).pow(2).mean()
        return load_balance_loss + self.router_z_loss_weight * z_loss

    def forward(self, x):
        original_shape = x.shape
        tokens = x.reshape(-1, original_shape[-1])
        num_tokens, dim = tokens.shape

        router_logits = self.router(tokens)
        if self.training and self.router_noise_std > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_noise_std

        router_probs = F.softmax(
            router_logits.float() / self.router_temperature,
            dim=-1,
        ).to(tokens.dtype)
        top_probs, top_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        aux_loss = self._router_loss(router_probs, top_indices, router_logits)
        output = torch.zeros_like(tokens)
        normalizer = torch.zeros(num_tokens, device=tokens.device, dtype=tokens.dtype)
        capacity = self._capacity(num_tokens)

        flat_expert_ids = top_indices.reshape(-1)
        flat_token_ids = (
            torch.arange(num_tokens, device=tokens.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
            .reshape(-1)
        )
        flat_weights = top_probs.reshape(-1)
        total_assignments = flat_expert_ids.numel()
        kept_assignments = 0
        kept_per_expert = torch.zeros(self.num_experts, device=tokens.device, dtype=torch.float32)

        for expert_id, expert in enumerate(self.experts):
            selected = flat_expert_ids == expert_id
            if not selected.any():
                continue

            token_ids = flat_token_ids[selected]
            weights = flat_weights[selected]

            if token_ids.numel() > capacity:
                # Keep the strongest assignments for this expert when capacity
                # overflows. Dropped assignments simply contribute zero.
                keep = torch.topk(weights, capacity, sorted=False).indices
                token_ids = token_ids[keep]
                weights = weights[keep]

            expert_out = expert(tokens.index_select(0, token_ids))
            output.index_add_(0, token_ids, expert_out * weights.unsqueeze(-1))
            normalizer.index_add_(0, token_ids, weights)
            kept_assignments += token_ids.numel()
            kept_per_expert[expert_id] = token_ids.numel()

        if self.renormalize_after_drop:
            output = output / normalizer.clamp_min(1e-9).unsqueeze(-1)

        if self.num_shared_experts > 0 and self.shared_expert_weight != 0:
            shared_output = torch.zeros_like(tokens)
            for shared_expert in self.shared_experts:
                shared_output = shared_output + shared_expert(tokens)
            shared_output = shared_output / self.num_shared_experts
            output = output + self.shared_expert_weight * shared_output

        with torch.no_grad():
            entropy = -(router_probs.float() * router_probs.float().clamp_min(1e-9).log()).sum(dim=-1).mean()
            drop_rate = 1.0 - (kept_assignments / max(1, total_assignments))
            self.last_stats = {
                "drop_rate": torch.tensor(drop_rate, device=tokens.device),
                "router_entropy": entropy.detach(),
                "kept_per_expert": kept_per_expert.detach(),
                "capacity": torch.tensor(capacity, device=tokens.device),
            }

        return output.reshape(original_shape), aux_loss
