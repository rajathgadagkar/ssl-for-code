"""Factory utilities for loading trained checkpoints into a Predictor."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from ssl_for_code.dataset.dataset import get_tokenizer
from ssl_for_code.evaluate import Predictor
from ssl_for_code.models import build_model
from ssl_for_code.models.state import MemoryUpdater, StateUpdater
from ssl_for_code.utils.io import get_device


def get_predictor(
    checkpoint_path: Optional[str | Path] = None,
    *,
    architecture: str = "transformer_xl",
    tokenizer_kind: str = "bpe",
    device: str = "auto",
    beam_strategy: str = "simple",
    mem_len: int = 256,
) -> Predictor:
    """Load a trained PyTorch checkpoint or initialize a fresh model for inference."""
    target_device = get_device(device)

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        payload = torch.load(Path(checkpoint_path), map_location=target_device)
        cfg = payload.get("config", {})
        arch = cfg.get("model", architecture)
        tok_kind = cfg.get("tokenizer", tokenizer_kind)
        tokenizer = get_tokenizer(tok_kind)
        model = build_model(
            arch,
            n_tokens=cfg.get("n_tokens", tokenizer.n_tokens),
            d_model=cfg.get("d_model", 512),
            n_layers=cfg.get("n_layers", 6),
            n_heads=cfg.get("n_heads", 8),
            d_ff=cfg.get("d_ff", 2048),
            dropout=cfg.get("dropout", 0.2),
            rnn_size=cfg.get("rnn_size", 512),
            rhn_depth=cfg.get("rhn_depth", 2),
        ).to(target_device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        eff_mem_len = cfg.get("mem_len", mem_len)
        updater = MemoryUpdater(eff_mem_len) if "xl" in arch.lower() else StateUpdater()
        return Predictor(
            model=model,
            tokenizer=tokenizer,
            state_updater=updater,
            is_token_by_token=cfg.get("is_token_by_token", True),
            beam_strategy=beam_strategy,
        )

    tokenizer = get_tokenizer(tokenizer_kind)
    model = build_model(
        architecture,
        n_tokens=tokenizer.n_tokens,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        dropout=0.0,
    ).to(target_device)
    model.eval()
    updater = MemoryUpdater(mem_len) if "xl" in architecture.lower() else StateUpdater()
    return Predictor(
        model=model,
        tokenizer=tokenizer,
        state_updater=updater,
        is_token_by_token=True,
        beam_strategy=beam_strategy,
    )
