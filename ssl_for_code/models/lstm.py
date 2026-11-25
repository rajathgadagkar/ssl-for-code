"""Multi-layer LSTM autoregressive code language model."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from ssl_for_code.models import AutoregressiveModel


class LstmModel(AutoregressiveModel):
    """Autoregressive LSTM baseline with persistent hidden and cell states."""

    def __init__(
        self,
        *,
        n_tokens: int,
        embedding_size: int = 512,
        hidden_size: int = 512,
        n_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.embedding = nn.Embedding(n_tokens, embedding_size)
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, n_tokens)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: [seq_len, batch_size]
        embedded = self.dropout(self.embedding(x))
        out, (hn, cn) = self.lstm(embedded, state)
        logits = self.fc(self.dropout(out))
        return logits, (hn.detach(), cn.detach())
