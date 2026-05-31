import torch
import torch.nn as nn

class Embedding(nn.Module):
  def __init__(self, vocab_size, embedding_dim, context_length):
    super().__init__()
    self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

  def forward(self, xb):
    return self.token_embedding(xb)