# Implementation of transformers only looking at https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
# (also using Vim so I familiarize myself with bindings)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Callable, List, Optional, Type, Union
import einops
from positionalencodings import PositionalEncoding
from helpers import TokenEmbedding
import math


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        src_dim: int,
        target_dim: int,
        out_dim: int,
        d_k: int,
        d_v: Optional[int] = None,
        num_heads: int = 8,
    ):
        super().__init__()
        if d_v is None:
            d_v = d_k
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.norm_factor = 1.0 / math.sqrt(self.d_k * self.num_heads)

        self.Wk = nn.Linear(src_dim, d_k * num_heads)
        self.Wq = nn.Linear(target_dim, d_k * num_heads)
        self.Wv = nn.Linear(src_dim, d_v * num_heads)
        self.Wo = nn.Linear(d_k, out_dim)

    def attention(
        self, key: Tensor, query: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ):
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

    def forward(
        self,
        src: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
    ):
        B, L, src_dim = src.shape
        _, _, target_dim = target.shape

        # TODO: Implement KV Caching

        K = self.Wk(src)  # B x L x (h x d_k)
        V = self.Wv(src)  # B x L x (h x d_v)
        Q = self.Wq(target)  # B x L x (h x d_q)

        # Rearrange dims
        K, V, Q = map(
            lambda x: einops.rearrange(x, "b l (d h) -> b h l d", h=self.num_heads),
            (K, V, Q),
        )

        out = self.attention(key=K, query=Q, value=V, mask=mask)
        return self.Wo(out)


class LayerNorm(nn.Module):
    """
    Basic Layer Normalization block.
    """

    def __init__(self, input_shape: tuple):
        super().__init__()
        self.eps = 1e-5
        self.gamma = nn.Parameter(torch.ones(input_shape))
        self.beta = nn.Parameter(torch.zeros(input_shape))
        self.reduce_dims = tuple(reversed([(-1 - i) for i in range(len(input_shape))]))

    def forward(self, x: Tensor):
        """
        Compute mean and variance statistics over last len(input_shape) dimensions.
        y = (x - E[x]) / (\sqrt{Var[x] + eps}) * gamma + beta
        """
        mean = torch.mean(x, dim=self.reduce_dims, keep_dim=True)
        var = torch.var(x, dim=self.reduce_dims, keep_dim=True)
        return self.gamma * (x - mean) / (torch.sqrt(var) + self.eps) + self.beta


class PointwiseFFN(nn.Module):
    """
    Pointwise FFN layer described in the paper. Two linear (or convolutions) layers with ReLU in between.
    """

    def __init__(self, input_dim, hidden_dim, dropout_prob=0.1):
        """
        input and output dimensions are the same
        """
        super().__init__()
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

    def __init__(
        self, dim_model: int = 512, dim_ff: int = 2048, dropout_prob: int = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear = PointwiseFFN(dim_model, dim_ff)
        self.attn = MultiHeadAttention(
            src_dim=dim_model,
            target_dim=dim_model,
            out_dim=dim_model,
            d_k=dim_model,
            num_heads=8,
        )
        self.attn_norm = LayerNorm(input_shape=(dim_model,))
        self.ffn_norm = LayerNorm(input_shape=(dim_model,))

    def forward(self, x: Tensor):
        # MHA -> Res + Norm -> FFN -> Res + Norm
        x = self.attn_norm(x + self.dropout(self.attn(x)))
        ffn_out = self.ffn_norm(x + self.dropout(self.linear(x)))
        return ffn_out


class DecoderBlock(nn.Module):
    """
    Similar to the encoder...inserts a third sub layer which MHA over output of encoder stack. Also masking
    in self-attention layer.
    """

    def __init__(
        self, dim_model: int = 512, dim_ff: int = 2048, dropout_prob: int = 0.1
    ):
        super().__init__()
        self.self_attn_norm = LayerNorm(input_shape=(dim_model,))
        self.cross_attn_norm = LayerNorm(input_shape=(dim_model,))
        self.ffn_norm = LayerNorm(input_shape=(dim_model,))
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear = PointwiseFFN(dim_model, dim_ff)
        self.self_attn = MultiHeadAttention(
            src_dim=dim_model,
            target_dim=dim_model,
            out_dim=dim_model,
            d_k=dim_model,
            num_heads=8,
        )
        self.cross_attn = MultiHeadAttention(
            src_dim=dim_model,
            target_dim=dim_model,
            out_dim=dim_model,
            d_k=dim_model,
            num_heads=8,
        )

    def forward(
        self,
        src: Tensor,
        target: Tensor,
        src_mask: Optional[Tensor] = None,
        target_mask: Optional[Tensor] = None,
    ):
        # MMHA -> Res + Norm -> CMHA -> Res + Norm -> FFN -> Res + Norm
        mha_block = self.self_attn_norm(
            src + self.dropout(self.self_attn(src, src_mask))
        )
        cmha = self.self_attn_norm(
            mha_block
            + self.dropout(
                self.cross_attn(src=mha_block, target=target, mask=target_mask)
            )
        )
        out = self.ffn_norm(cmha + self.dropout(self.linear(cmha)))
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        context_window: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        src_vocab_size: int,
        target_vocab_size: int,
        use_mask: bool = True,
        model_dim: int = 512,
        ff_dim: int = 2048,
    ):
        super().__init__()
        self.embed = TokenEmbedding(model_dim, src_vocab_size)
        self.target_embed = TokenEmbedding(model_dim, target_vocab_size)
        self.pe = PositionalEncoding(model_dim, context_window)
        self.encoder = nn.Sequential(
            *[
                EncoderBlock(dim_model=model_dim, dim_ff=ff_dim)
                for i in range(n_encoder_layers)
            ]
        )
        self.decoder = nn.Sequential(
            *[
                DecoderBlock(dim_model=model_dim, dim_ff=ff_dim)
                for i in range(n_decoder_layers)
            ]
        )

        self.decoder_mask = self.compute_offset_mask(model_dim) if use_mask else None
        self.final_projection = nn.Linear(model_dim, target_vocab_size)

        self.initialize()

    def compute_offset_mask(self, model_dim: int):
        "Mask out previous tokens in decoder self attention"
        mask = torch.tril(torch.ones((1, model_dim, model_dim)), diagonal=0)
        return mask

    def initialize(
        self,
    ):
        """Initialize all layers with Xavier"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: Tensor, target: Tensor):
        src = self.pe(self.embed(src))
        src = self.encode(src)

        target = self.pe(self.target_embed(target))
        out = self.decode(src, target, self.decoder_mask)
        return out

    def encode(self, src):
        return self.encoder(src)

    def decode(self, src, target, mask):
        return self.decoder(src=src, target=target, src_mask=mask)

    def output_logits(self, sequence: Tensor):
        """
        Return the output logits for the sequence.
        Args:
            sequence (Tensor): decoder output
        """
        return self.final_projection(sequence)

    def output_probabilities(self, sequence: Tensor):
        return F.log_softmax(self.output_logits(sequence))


if __name__ == "__main__":
    transformer = Transformer(
        context_window=8,
        n_encoder_layers=2,
        n_decoder_layers=2,
        src_vocab_size=10,
        target_vocab_size=10,
        model_dim=32,
        ff_dim=64,
    )
