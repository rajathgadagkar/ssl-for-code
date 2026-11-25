"""Autoregressive neural language model architectures for Python source code."""

from __future__ import annotations

from typing import Any, Tuple

import torch
from torch import nn


class AutoregressiveModel(nn.Module):
    """Base interface for sequence-to-sequence autoregressive code language models.

    Expected input shape: ``[seq_len, batch_size]`` (token IDs).
    Expected return: ``(logits, new_state)`` where ``logits`` has shape
    ``[seq_len, batch_size, vocab_size]``.
    """

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def forward(self, src: torch.Tensor, state: Any = None) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError


def build_model(
    architecture: str,
    *,
    n_tokens: int,
    d_model: int = 512,
    n_layers: int = 6,
    n_heads: int = 8,
    d_ff: int = 2048,
    dropout: float = 0.2,
    rnn_size: int = 512,
    rhn_depth: int = 2,
) -> AutoregressiveModel:
    """Instantiate a model by architecture name.

    Supported architectures: ``transformer_xl``, ``transformer``, ``lstm``, ``rhn``.
    """
    from ssl_for_code.models.highway import RhnModel
    from ssl_for_code.models.lstm import LstmModel
    from ssl_for_code.models.transformer import TransformerModel
    from ssl_for_code.models.xl import TransformerXLModel

    arch = architecture.lower()
    if arch in ("transformer_xl", "transformer_xl_model", "xl"):
        return TransformerXLModel(
            n_vocab=n_tokens,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
    if arch in ("transformer", "transformer_model"):
        return TransformerModel(
            n_tokens=n_tokens,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
    if arch in ("lstm", "lstm_model"):
        return LstmModel(
            n_tokens=n_tokens,
            embedding_size=d_model,
            hidden_size=rnn_size,
            n_layers=n_layers,
            dropout=dropout,
        )
    if arch in ("rhn", "rhn_model", "highway"):
        return RhnModel(
            n_tokens=n_tokens,
            embedding_size=d_model,
            hidden_size=rnn_size,
            n_layers=n_layers,
            depth=rhn_depth,
            dropout=dropout,
        )
    raise ValueError(
        f"Unknown architecture: {architecture!r}. "
        "Expected one of: 'transformer_xl', 'transformer', 'lstm', 'rhn'."
    )


__all__ = ["AutoregressiveModel", "build_model"]
