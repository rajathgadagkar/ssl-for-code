"""Autoregressive Python code generation from a prompt."""

from __future__ import annotations

import argparse
from pathlib import Path

from ssl_for_code.evaluate import Predictor
from ssl_for_code.evaluate.factory import get_predictor


def generate(predictor: Predictor, prompt: str, max_tokens: int = 100) -> str:
    """Autoregressively extend ``prompt`` by ``max_tokens`` BPE/character tokens."""
    text = prompt
    state = None
    for _ in range(max_tokens):
        next_piece, state = predictor.get_next_token(text, state)
        text += next_piece
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Python code from a prompt.")
    parser.add_argument("--prompt", type=str, default="import torch\nimport torch.nn as nn\n\n")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    predictor = get_predictor(args.checkpoint)
    print(generate(predictor, args.prompt, max_tokens=args.max_tokens))


if __name__ == "__main__":
    main()
