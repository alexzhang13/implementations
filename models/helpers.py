import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.dim = dim

    def forward(self, x):
        return self.embedding(x) * torch.sqrt(self.dim)


class Residual(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x, *args, **kwargs):
        return x + self.func(x, *args, **kwargs)
