"""Dataset loading, character tokenization, and PyTorch DataLoader builders."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from ssl_for_code.dataset import Tokenizer
from ssl_for_code.dataset.bpe import load_bpe_tokenizer
from ssl_for_code.utils.io import get_data_path, load_cache, save_cache


class CharacterTokenizer(Tokenizer):
    """Character-level tokenizer with JSON vocabulary caching."""

    def __init__(
        self,
        retrain: Optional[bool] = None,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.n_tokens = load_cache("n_tokens", cache_dir=cache_dir) or 0
        self.itos = load_cache("itos", cache_dir=cache_dir) or []
        self.stoi = load_cache("stoi", cache_dir=cache_dir) or {}

        has_cache = bool(self.n_tokens and self.itos and self.stoi)
        if retrain is None:
            self.is_trained = has_cache
        else:
            self.is_trained = (not retrain) and has_cache

    def encode(self, data: str, *, is_silent: bool = True) -> List[int]:
        del is_silent
        return [self.stoi[c] for c in data if c in self.stoi]

    def train(self, data: str) -> None:
        self.itos = sorted(set(data))
        self.n_tokens = len(self.itos)
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.is_trained = True

        save_cache("n_tokens", self.n_tokens, cache_dir=self.cache_dir)
        save_cache("itos", self.itos, cache_dir=self.cache_dir)
        save_cache("stoi", self.stoi, cache_dir=self.cache_dir)


class SourceCodeDataset:
    """Container holding raw train/validation Python code strings and a tokenizer."""

    def __init__(self, tokenizer: Tokenizer, train: str, valid: str) -> None:
        self.tokenizer = tokenizer
        self.train = train
        self.valid = valid

    @staticmethod
    def load(path: str | Path) -> str:
        return Path(path).read_text(encoding="utf-8")

    @classmethod
    def from_directory(
        cls,
        tokenizer: Tokenizer,
        data_dir: Optional[str | Path] = None,
        truncate_data: int = 0,
    ) -> "SourceCodeDataset":
        root = get_data_path(data_dir)
        train = cls.load(root / "train.py")
        valid = cls.load(root / "valid.py")
        if truncate_data > 0:
            train = train[:truncate_data]
            valid = valid[:truncate_data]
        if not tokenizer.is_trained:
            tokenizer.train(train + valid)
        return cls(tokenizer=tokenizer, train=train, valid=valid)

    def __repr__(self) -> str:
        return (
            f"SourceCodeDataset(train={len(self.train) / 1_000_000:,.2f}M chars, "
            f"valid={len(self.valid) / 1_000_000:,.2f}M chars, "
            f"vocab={self.tokenizer.n_tokens})"
        )


def get_tokenizer(
    kind: str = "bpe",
    *,
    retrain: Optional[bool] = None,
    cache_dir: Optional[str | Path] = None,
) -> Tokenizer:
    """Instantiate either the ``bpe`` or ``char`` tokenizer."""
    if kind == "bpe":
        return load_bpe_tokenizer(cache_dir=cache_dir)
    if kind == "char":
        return CharacterTokenizer(retrain=retrain, cache_dir=cache_dir)
    raise ValueError(f"Unsupported tokenizer kind: {kind!r}. Choose 'bpe' or 'char'.")


def transpose_batch(batch):
    """Collate a list of ``(src, tgt)`` pairs into ``[seq_len, batch_size]`` tensors."""
    import torch

    srcs, tgts = zip(*batch)
    return torch.stack(srcs, dim=1), torch.stack(tgts, dim=1)


def create_token_dataset(data, *, batch_size: int, seq_len: int, drop_last: bool = False):
    """Build a contiguous-stream autoregressive ``TokenDataset`` (PyTorch)."""
    import torch
    from torch.utils.data import Dataset

    class TokenDataset(Dataset):
        def __init__(self, tokens: torch.Tensor) -> None:
            self.data = tokens
            self.batch_size = batch_size
            self.seq_len = seq_len
            self.n_samples = max((self.data.shape[0] - 1) // self.seq_len, 0)
            if drop_last:
                self.n_batches = self.n_samples // batch_size
            else:
                self.n_batches = (self.n_samples + batch_size - 1) // batch_size

        def __len__(self) -> int:
            return self.n_batches * self.batch_size if drop_last else self.n_samples

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            batch = idx // self.batch_size
            batch_idx = idx % self.batch_size
            stream_idx = batch_idx * self.n_batches + batch
            start = stream_idx * self.seq_len
            end = start + self.seq_len
            return self.data[start:end], self.data[start + 1 : end + 1]

    return TokenDataset(data)


def build_dataloaders(
    dataset: SourceCodeDataset,
    *,
    batch_size: int = 12,
    seq_len: int = 256,
    shuffle: bool = False,
):
    """Encode train/validation splits and return ``(train_loader, valid_loader)``."""
    import torch
    from torch.utils.data import DataLoader

    train_ids = torch.tensor(dataset.tokenizer.encode(dataset.train), dtype=torch.long)
    valid_ids = torch.tensor(dataset.tokenizer.encode(dataset.valid), dtype=torch.long)

    train_ds = create_token_dataset(
        train_ids, batch_size=batch_size, seq_len=seq_len, drop_last=True
    )
    valid_ds = create_token_dataset(
        valid_ids, batch_size=batch_size, seq_len=seq_len, drop_last=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        collate_fn=transpose_batch,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        collate_fn=transpose_batch,
    )
    return train_loader, valid_loader
