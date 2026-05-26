import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNetwork(nn.Module):
  def __init__(self, dim, hidden_dim):
    super().__init__()
    self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
    self.w_up = nn.Linear(dim, hidden_dim, bias=False)

    self.w_down = nn.Linear(hidden_dim, dim, bias=False)

  def forward(self, x):
    gate = F.silu(self.w_gate(x))
    up = self.w_up(x)

    activated = gate * up

    return self.w_down(activated)