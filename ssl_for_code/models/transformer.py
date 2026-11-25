"""Causal Transformer language model for Python source code."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch
from torch import nn

from ssl_for_code.models import AutoregressiveModel


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for ``[seq_len, batch_size, d_model]``."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 8192) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term[: pe[:, 0, 1::2].shape[1]])
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerModel(AutoregressiveModel):
    """Decoder-only causal Transformer language model."""

    def __init__(
        self,
        n_tokens: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model

        self.src_embed = nn.Embedding(n_tokens, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, n_tokens)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
        )

    def forward(
        self,
        src: torch.Tensor,
        state: Any = None,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        del state
        seq_len = src.size(0)
        mask = self._causal_mask(seq_len, src.device)
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        output = self.norm(self.encoder(x, mask=mask))
        logits = self.fc(output)
        return logits, None
