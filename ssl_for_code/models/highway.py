"""Recurrent Highway Network (RHN) autoregressive code language model."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn

from ssl_for_code.models import AutoregressiveModel


class _RHNCell(nn.Module):
    """Single RHN layer with coupled carry/transform gates of recurrence depth ``L``."""

    def __init__(self, input_size: int, hidden_size: int, depth: int = 2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.in_h = nn.Linear(input_size, hidden_size)
        self.in_t = nn.Linear(input_size, hidden_size)

        self.rec_h = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(depth)]
        )
        self.rec_t = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(depth)]
        )

    def forward(self, x_t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        for l_idx in range(self.depth):
            if l_idx == 0:
                h = torch.tanh(self.in_h(x_t) + self.rec_h[l_idx](s))
                t = torch.sigmoid(self.in_t(x_t) + self.rec_t[l_idx](s))
            else:
                h = torch.tanh(self.rec_h[l_idx](s))
                t = torch.sigmoid(self.rec_t[l_idx](s))
            c = 1.0 - t
            s = h * t + s * c
        return s


class RhnModel(AutoregressiveModel):
    """Multi-layer Recurrent Highway Network (Zilly et al., 2017 / Srivastava et al., 2015)."""

    def __init__(
        self,
        *,
        n_tokens: int,
        embedding_size: int = 512,
        hidden_size: int = 512,
        n_layers: int = 2,
        depth: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.depth = depth

        self.embedding = nn.Embedding(n_tokens, embedding_size)
        self.layers = nn.ModuleList(
            [
                _RHNCell(embedding_size if i == 0 else hidden_size, hidden_size, depth=depth)
                for i in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, n_tokens)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [seq_len, batch_size]
        seq_len, batch_size = x.shape
        embedded = self.dropout(self.embedding(x))

        if state is None:
            layer_states: List[torch.Tensor] = [
                embedded.new_zeros(batch_size, self.hidden_size)
                for _ in range(self.n_layers)
            ]
        else:
            layer_states = [state[i] for i in range(self.n_layers)]

        outputs: List[torch.Tensor] = []
        for t in range(seq_len):
            inp = embedded[t]
            for layer_idx, cell in enumerate(self.layers):
                s = cell(inp, layer_states[layer_idx])
                layer_states[layer_idx] = s
                inp = self.dropout(s)
            outputs.append(inp)

        out = torch.stack(outputs, dim=0)
        logits = self.fc(out)
        next_state = torch.stack([s.detach() for s in layer_states], dim=0)
        return logits, next_state
