"""Transformer-XL with Relative Multi-Head Attention and Segment-Level Recurrence.

Implements the relative positional encoding and segment-level memory recurrence
introduced in Dai et al. (2019), "Transformer-XL: Attentive Language Models
Beyond a Fixed-Length Context".
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import nn

from ssl_for_code.models import AutoregressiveModel


class _RelativeSinusoidalEmbedding(nn.Module):
    """Generates sinusoidal relative positional encodings R_{i-j}."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq: torch.Tensor) -> torch.Tensor:
        sinusoid_inp = torch.outer(pos_seq, self.inv_freq)
        return torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)


class RelativeMultiHeadAttention(nn.Module):
    """Multi-head self-attention with relative positional encodings (u/v biases)."""

    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.r_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.r_w_bias = nn.Parameter(torch.zeros(n_heads, self.d_head))
        self.r_r_bias = nn.Parameter(torch.zeros(n_heads, self.d_head))

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        """Skew relative positional attention scores from [q, k, b, h] to aligned offsets."""
        zero_pad = torch.zeros(
            (x.size(0), 1, x.size(2), x.size(3)), device=x.device, dtype=x.dtype
        )
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), x.size(2), x.size(3))
        return x_padded[1:].view_as(x)

    def forward(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        mem: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: [q_len, batch, d_model], mem: [m_len, batch, d_model], r: [k_len, d_model]
        q_len, batch, d_model = x.shape
        cat = torch.cat([mem, x], dim=0) if mem is not None and mem.numel() > 0 else x
        k_len = cat.size(0)

        q = self.q_proj(x).view(q_len, batch, self.n_heads, self.d_head)
        kv = self.kv_proj(cat)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(k_len, batch, self.n_heads, self.d_head)
        v = v.view(k_len, batch, self.n_heads, self.d_head)
        r_head = self.r_proj(r).view(k_len, self.n_heads, self.d_head)

        rw_q = q + self.r_w_bias.unsqueeze(0).unsqueeze(0)
        rr_q = q + self.r_r_bias.unsqueeze(0).unsqueeze(0)

        ac = torch.einsum("ibnd,jbnd->ijbn", rw_q, k)
        bd = torch.einsum("ibnd,jnd->ijbn", rr_q, r_head)
        bd = self._rel_shift(bd)

        scores = (ac + bd) * self.scale
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(-1).unsqueeze(-1), float("-inf"))

        attn = self.attn_dropout(torch.softmax(scores, dim=1))
        context = torch.einsum("ijbn,jbnd->ibnd", attn, v).reshape(q_len, batch, d_model)
        return self.out_dropout(self.out_proj(context))


class TransformerXLLayer(nn.Module):
    """Single pre-norm Transformer-XL block with relative attention + FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = RelativeMultiHeadAttention(n_heads, d_model, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        mem: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        norm_x = self.norm1(x)
        norm_mem = self.norm1(mem) if mem is not None and mem.numel() > 0 else None
        x = x + self.self_attn(norm_x, r, mem=norm_mem, attn_mask=attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerXLModel(AutoregressiveModel):
    """Transformer-XL autoregressive code language model with segment-level memory."""

    def __init__(
        self,
        n_vocab: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_vocab = n_vocab
        self.d_model = d_model
        self.n_layers = n_layers

        self.src_embed = nn.Embedding(n_vocab, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        self.pos_emb = _RelativeSinusoidalEmbedding(d_model)
        self.layers = nn.ModuleList(
            [
                TransformerXLLayer(d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.generator = nn.Linear(d_model, n_vocab)

    def forward(
        self,
        x: torch.Tensor,
        mem: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # x: [seq_len, batch_size]
        q_len = x.size(0)
        m_len = mem[0].size(0) if mem and len(mem) > 0 and mem[0].numel() > 0 else 0
        k_len = q_len + m_len

        # True entries in attn_mask are masked out (future tokens).
        causal_q = torch.triu(
            torch.ones((q_len, q_len), dtype=torch.bool, device=x.device), diagonal=1
        )
        if m_len > 0:
            mem_allow = torch.zeros((q_len, m_len), dtype=torch.bool, device=x.device)
            attn_mask = torch.cat([mem_allow, causal_q], dim=1)
        else:
            attn_mask = causal_q

        pos_seq = torch.arange(k_len - 1, -1, -1.0, device=x.device)
        r = self.pos_emb(pos_seq)

        hidden = self.embed_dropout(self.src_embed(x) * math.sqrt(self.d_model))
        new_mem: List[torch.Tensor] = []

        for idx, layer in enumerate(self.layers):
            new_mem.append(hidden.detach())
            layer_mem = mem[idx] if mem and idx < len(mem) else None
            hidden = layer(hidden, r, mem=layer_mem, attn_mask=attn_mask)

        logits = self.generator(self.final_norm(hidden))
        return logits, new_mem
