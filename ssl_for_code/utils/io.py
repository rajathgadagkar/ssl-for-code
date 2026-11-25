"""Filesystem, JSON cache, and hardware device utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional


def get_project_root() -> Path:
    """Return the root directory of the repository."""
    return Path(__file__).resolve().parent.parent.parent


def get_data_path(custom_dir: Optional[str | Path] = None) -> Path:
    """Return the path to the data directory, creating it if requested."""
    if custom_dir is not None:
        return Path(custom_dir)
    env_dir = os.environ.get("SSL_FOR_CODE_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    return get_project_root() / "data"


def get_cache_path(custom_dir: Optional[str | Path] = None) -> Path:
    """Return the path to the tokenizer cache directory."""
    return get_data_path(custom_dir) / "cache"


def load_cache(name: str, cache_dir: Optional[str | Path] = None) -> Optional[Any]:
    """Load a JSON artifact from ``data/cache/<name>.json`` if it exists."""
    path = Path(cache_dir) / f"{name}.json" if cache_dir else get_cache_path() / f"{name}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_cache(name: str, payload: Any, cache_dir: Optional[str | Path] = None) -> Path:
    """Persist a JSON artifact to ``data/cache/<name>.json``."""
    target_dir = Path(cache_dir) if cache_dir else get_cache_path()
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{name}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


def get_device(preferred: str = "auto"):
    """Select the best available PyTorch device (CUDA, MPS, or CPU)."""
    import torch

    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
