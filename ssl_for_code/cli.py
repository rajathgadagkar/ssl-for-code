"""Unified CLI entry point for SSL-For-Code."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ssl_for_code import __version__
from ssl_for_code.dataset.bpe import load_bpe_tokenizer


def _cmd_tokenize(args: argparse.Namespace) -> int:
    tokenizer = load_bpe_tokenizer()
    code = args.file.read_text(encoding="utf-8") if args.file else args.code
    ids = tokenizer.encode(code)
    pieces = [tokenizer.itos[idx] for idx in ids]
    payload = {
        "n_chars": len(code),
        "n_tokens": len(ids),
        "compression_ratio": round(len(code) / max(len(ids), 1), 3),
        "tokens": pieces,
        "token_ids": ids,
    }
    print(json.dumps(payload, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ssl-for-code",
        description="Self-Supervised Learning for Python Code Auto-Completion CLI.",
    )
    parser.add_argument("--version", action="version", version=f"ssl-for-code {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tok_parser = subparsers.add_parser("tokenize", help="Tokenize Python code with cached BPE.")
    tok_group = tok_parser.add_mutually_exclusive_group(required=True)
    tok_group.add_argument("--code", type=str, help="Inline Python snippet to tokenize.")
    tok_group.add_argument("--file", type=Path, help="Path to a Python file to tokenize.")
    tok_parser.set_defaults(func=_cmd_tokenize)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
