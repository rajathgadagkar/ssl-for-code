"""Thread-safe Flask REST API server for real-time Python code autocompletion."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

from flask import Flask, jsonify, request

from ssl_for_code.dataset.bpe import load_bpe_tokenizer

# Lightweight contextual completions used only when running with --demo (no GPU checkpoint).
_DEMO_PATTERNS: List[Tuple[str, List[Tuple[str, float]]]] = [
    ("import tor", [("ch\n", 0.72), ("ch.nn as nn\n", 0.21), ("ch.nn.functional as F\n", 0.07)]),
    ("nn.Mod", [("ule):\n    def __init__(self", 0.81), ("uleList([", 0.12), ("uleDict({", 0.07)]),
    ("super().", [("__init__()", 0.91), ("forward(", 0.06), ("eval()", 0.03)]),
    ("self.rel", [("u(x)", 0.68), ("u = nn.ReLU()", 0.22), ("ative_attn(", 0.10)]),
    ("def forw", [("ard(self, x: torch.Tensor) -> torch.Tensor:", 0.79), ("ard(self, x):", 0.21)]),
    ("return ", [("self.relu(x)", 0.44), ("logits, new_state", 0.31), ("None", 0.15), ("x", 0.10)]),
    ("for ", [("i, batch in enumerate(loader):", 0.48), ("epoch in range(epochs):", 0.37), ("x in ", 0.15)]),
]


def _demo_complete(prefix: str, beam_size: int = 5) -> Tuple[List[str], List[float]]:
    for trigger, candidates in _DEMO_PATTERNS:
        if prefix.endswith(trigger):
            top = candidates[:beam_size]
            return [c[0] for c in top], [c[1] for c in top]
    return ["pass", "return None", "self."][:beam_size], [0.55, 0.30, 0.15][:beam_size]


def create_app(
    *,
    checkpoint_path: Optional[str | Path] = None,
    architecture: str = "transformer_xl",
    beam_strategy: str = "simple",
    demo_mode: bool = False,
    predictor: Optional[Any] = None,
) -> Flask:
    """Create and configure the Flask inference application."""
    app = Flask("ssl_for_code")
    lock = threading.Lock()
    bpe_tokenizer = load_bpe_tokenizer()

    active_predictor = predictor
    if active_predictor is None and not demo_mode:
        from ssl_for_code.evaluate.factory import get_predictor

        active_predictor = get_predictor(
            checkpoint_path=checkpoint_path,
            architecture=architecture,
            beam_strategy=beam_strategy,
        )

    @app.route("/", methods=["GET"])
    def home():
        return jsonify(
            {
                "service": "ssl-for-code",
                "status": "ok",
                "mode": "demo" if demo_mode else "model",
                "endpoints": ["/health", "/autocomplete", "/tokenize"],
            }
        )

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify(
            {
                "status": "healthy",
                "mode": "demo" if demo_mode else "model",
                "architecture": architecture if not demo_mode else "demo-heuristic",
                "vocab_size": bpe_tokenizer.n_tokens,
            }
        )

    @app.route("/tokenize", methods=["POST"])
    def tokenize():
        payload = request.get_json(silent=True) or {}
        code = payload.get("code", "")
        if not isinstance(code, str):
            return jsonify({"success": False, "error": "'code' must be a string."}), 400
        ids = bpe_tokenizer.encode(code)
        pieces = [bpe_tokenizer.itos[i] for i in ids]
        return jsonify(
            {
                "success": True,
                "token_ids": ids,
                "tokens": pieces,
                "n_tokens": len(ids),
                "compression_ratio": round(len(code) / max(len(ids), 1), 3),
            }
        )

    @app.route("/autocomplete", methods=["POST"])
    def autocomplete():
        payload = request.get_json(silent=True) or {}
        prefix = payload.get("prompt", "")
        if not isinstance(prefix, str) or not prefix:
            return jsonify({"success": False, "error": "Non-empty 'prompt' is required."}), 400

        beam_size = int(payload.get("beam_size", 5))
        min_length = int(payload.get("min_length", 15))
        t0 = time.perf_counter()

        acquired = lock.acquire(blocking=False)
        if not acquired:
            return jsonify({"success": False, "error": "Inference engine busy."}), 503

        try:
            if demo_mode or active_predictor is None:
                results, probs = _demo_complete(prefix, beam_size=beam_size)
            else:
                predictions = active_predictor.autocomplete(
                    prefix, beam_size=beam_size, min_length=min_length
                )
                results = [pred.text for pred in predictions]
                probs = [pred.prob for pred in predictions]
        finally:
            lock.release()

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return jsonify(
            {
                "success": True,
                "prediction": results,
                "probs": probs,
                "latency_ms": latency_ms,
            }
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SSL-For-Code Flask inference server.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port number (default: 5000)")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to trained .pt file")
    parser.add_argument(
        "--model",
        default="transformer_xl",
        choices=["transformer_xl", "transformer", "lstm", "rhn"],
    )
    parser.add_argument("--beam-strategy", default="simple", choices=["simple", "lengthy"])
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run lightweight demo mode without loading a PyTorch checkpoint.",
    )
    args = parser.parse_args()

    app = create_app(
        checkpoint_path=args.checkpoint,
        architecture=args.model,
        beam_strategy=args.beam_strategy,
        demo_mode=args.demo,
    )
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
