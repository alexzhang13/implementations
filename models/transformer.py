# Implementation of transformers only looking at https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
# (also using Vim so I familiarize myself with bindings)

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from positionalencodings import PositionalEncoding
from helpers import TokenEmbedding


class MultiHeadAttention(nn.Module):
    def __init__(self, src_dim, target_dim, out_dim, d_k, d_v=None, num_heads=8):
        if d_v is None:
            d_v = d_k
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.norm_factor = 1.0 / torch.sqrt(self.d_k * self.num_heads)

        self.Wk = nn.Linear(src_dim, d_k * num_heads)
        self.Wq = nn.Linear(target_dim, d_k * num_heads)
        self.Wv = nn.Linear(src_dim, d_v * num_heads)
        self.Wo = nn.Linear(d_k, out_dim)

    def attention(self, key, query, value, mask=None):
        """
        attention function. assume key, query, value is B x tokens x model_dim
        """
        # key = B x h x L x d
        attn_matrix = torch.einsum("bhid, bhjd -> bhij", query, key) * self.norm_factor

        # masking procedure for decoder
        if mask is not None:
            assert mask.shape == attn_matrix.shape[2:]
            attn_matrix = attn_matrix.masked_fill(mask, -torch.inf)

        attn_matrix = F.softmax(attn_matrix, dim=-1)
        out = torch.einsum("bhij, bhjd -> bhid", attn_matrix, value)

        # undo heads
        out = einops.rearrange("bhid -> bi(hd)", out)
        return out

    def forward(self, src, target, src_mask=None, target_mask=None):
        B, L, src_dim = src.shape
        _, _, target_dim = target.shape

        K = self.Wk(src)  # B x L x (h x d_k)
        V = self.Wv(src)  # B x L x (h x d_v)
        Q = self.Wq(target)  # B x L x (h x d_q)

        # Rearrange dims
        K, V, Q = map(
            lambda x: einops.rearrange(x, "b l (d h) -> b h l d", h=self.num_heads),
            (K, V, Q),
        )

        out = self.attention(key=K, query=Q, value=V, mask=target_mask)
        return self.Wo(out)


class LayerNorm(nn.Module):
    def __init__(self, input_shape):
        self.eps = 1e-5
        self.gamma = nn.Parameter(input_shape)
        self.beta = nn.Parameter(input_shape)

    def forward(self, x):
        pass


class PointwiseFFN(nn.Module):
    """
    Pointwise FFN layer described in the paper. Two linear (or convolutions) layers with ReLU in between.
    """

    def __init__(self, input_dim, hidden_dim, dropout_prob=0.1):
        """
        input and output dimensions are the same
        """
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(p=dropout_prob)

        self.ops = nn.Sequential(self.W1, nn.ReLU(), self.dropout, self.W2)

    def forward(self, x):
        return self.ops(x)


class EncoderBlock(nn.Module):
    """
    We employ a residual connection [10] around each of the two sub-layers, followed by layer normalization [1].
    """

    def __init__(self, dim_model=512, dim_ff=2048, dropout_prob=0.1):
        self.attn_norm = LayerNorm()
        self.ffn_norm = LayerNorm()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear = PointwiseFFN(dim_model, dim_ff)
        self.attn = MultiHeadAttention(
            src_dim=dim_model,
            target_dim=dim_model,
            out_dim=dim_model,
            d_k=dim_model,
            num_heads=8,
        )

    def forward(self, x):
        # MHA -> Res + Norm -> FFN -> Res + Norm
        x = self.attn_norm(x + self.dropout(self.attn(x)))
        ffn_out = self.ffn_norm(x + self.dropout(self.linear(x)))
        return ffn_out


class DecoderBlock(nn.Module):
    def __init__(self, dim_model):
        pass

    def forward(self, src, target, src_mask, target_mask):
        pass


class Transformer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        context_window: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
    ):
        super().__init__()
        self.embed = TokenEmbedding(model_dim, context_window)
        self.pe = PositionalEncoding(model_dim, context_window)
        self.encoder = nn.Sequential(
            **[EncoderBlock(dim_model=model_dim) for i in range(n_encoder_layers)]
        )
        self.decoder = nn.Sequential(
            **[DecoderBlock(dim_model=model_dim) for i in range(n_decoder_layers)]
        )


if __name__ == "__main__":
    pass
