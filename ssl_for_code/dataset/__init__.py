"""Tokenization and dataset utilities for Python source code."""

from __future__ import annotations

import string
from typing import Dict, List, Tuple

ID_CHARS = frozenset(string.ascii_letters + string.digits + "_")


class Tokenizer:
    """Abstract base interface for code tokenizers."""

    n_tokens: int
    itos: List[str]
    stoi: Dict[str, int]
    is_trained: bool

    def encode(self, data: str, *, is_silent: bool = True) -> List[int]:
        raise NotImplementedError

    def decode(self, token_ids: List[int]) -> str:
        return "".join(self.itos[idx] for idx in token_ids if 0 <= idx < len(self.itos))

    def train(self, data: str) -> None:
        pass

    def rstrip(self, data: str) -> Tuple[str, List[int]]:
        """Split off the trailing partial token for prefix-constrained beam search."""
        return data, self.encode(data)


__all__ = ["ID_CHARS", "Tokenizer"]
