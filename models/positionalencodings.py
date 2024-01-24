import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np


class RoPE(nn.Module):
    def __init__(self):
        pass

    def forward():
        pass


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self):
        pass


class PositionalEncoding(nn.Module):
    """
    This implementation has an issue with numerical underflow! Need to fix later.
    """

    def __init__(self, dim, context_window=2048, dropout_prob=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        posencoding = torch.zeros(context_window, dim)

        pattern = torch.arange(0, dim, 2) * -(math.log(10000.0) / dim)
        pattern = pattern.unsqueeze(0)  # 1 x dim // 2
        pattern = torch.broadcast_to(pattern, (context_window, dim // 2))
        pattern = torch.exp(pattern)

        encoding = torch.arange(0, context_window, 1)
        encoding = encoding.unsqueeze(1)
        encoding = torch.broadcast_to(encoding, (context_window, dim // 2))

        posencoding[:, 0::2] = torch.sin(encoding * pattern)
        posencoding[:, 1::2] = torch.cos(encoding * pattern)
        posencoding = posencoding.unsqueeze(0)
        self.register_buffer("posencoding", posencoding)

    def forward(self, x):
        x = x + (
            torch.autograd.Variable(
                self.posencoding[:, : x.shape[1]], requires_grad=False
            )
        )
        return self.dropout(x)


if __name__ == "__main__":
    # Test cases from https://nlp.seas.harvard.edu/2018/04/03/attention.html#attention
    # It still has floating point errors... TODO: fix later
    plt.figure(figsize=(15, 5))
    pe = PositionalEncoding(60, 500)
    y = pe.forward(torch.autograd.Variable(torch.zeros(1, 100, 60)))
    plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.show()
