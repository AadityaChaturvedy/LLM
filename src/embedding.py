import torch
import torch.nn as nn

class Embedding(nn.Module):
  def __init__(self, vocab_size, embedding_dim, context_length):
    super().__init__()
    self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
    self.position_embedding = nn.Embedding(context_length, embedding_dim)

    self.register_buffer("position", torch.arange(0, context_length))

  def forward(self, xb):
    token_emb = self.token_embedding(xb)
    position_emb = self.position_embedding(self.position)

    x = token_emb + position_emb
    return x