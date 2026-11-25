"""Tests for beam search decoding, evaluation utilities, and Flask inference API."""

from __future__ import annotations

import unittest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import flask  # noqa: F401

    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed in this environment")
class TestBeamSearchAndPredictor(unittest.TestCase):
    def test_prediction_complete_boundary_rules(self) -> None:
        from ssl_for_code.evaluate.beam_search import (
            NextWordNewLinePredictionComplete,
            NextWordPredictionComplete,
        )

        rule = NextWordPredictionComplete(rest="re", min_length=3)
        self.assertFalse(rule("ret", "urn"))  # Too short.
        self.assertFalse(rule("return", "s"))  # Both identifier chars.
        self.assertTrue(rule("return", " "))  # Identifier -> whitespace boundary.

        nl_rule = NextWordNewLinePredictionComplete(rest="", min_length=2)
        self.assertTrue(nl_rule("pass", "\n"))

    def test_predictor_simple_and_lengthy_beam_search(self) -> None:
        from ssl_for_code.dataset.bpe import load_bpe_tokenizer
        from ssl_for_code.evaluate import Predictor
        from ssl_for_code.evaluate.anomalies import detect_anomalies
        from ssl_for_code.evaluate.generate import generate
        from ssl_for_code.models import build_model
        from ssl_for_code.models.state import MemoryUpdater

        tokenizer = load_bpe_tokenizer()
        model = build_model(
            "transformer_xl",
            n_tokens=tokenizer.n_tokens,
            d_model=32,
            n_layers=2,
            n_heads=4,
            d_ff=64,
            dropout=0.0,
        )
        model.eval()

        for strategy in ("simple", "lengthy"):
            predictor = Predictor(
                model=model,
                tokenizer=tokenizer,
                state_updater=MemoryUpdater(mem_len=32),
                is_token_by_token=True,
                beam_strategy=strategy,
            )
            preds = predictor.autocomplete("def foo(x):\n    ret", beam_size=3, min_length=2)
            self.assertIsInstance(preds, list)

        out = generate(predictor, "import torch\n", max_tokens=5)
        self.assertTrue(out.startswith("import torch\n"))
        self.assertGreater(len(out), len("import torch\n"))

        anomalies = detect_anomalies(
            tokenizer, "x = 1 + 2\n", model, MemoryUpdater(mem_len=32)
        )
        self.assertGreater(len(anomalies), 0)


@unittest.skipUnless(HAS_FLASK, "Flask is not installed in this environment")
class TestFlaskServer(unittest.TestCase):
    def test_health_tokenize_and_demo_autocomplete(self) -> None:
        from ssl_for_code.server import create_app

        app = create_app(demo_mode=True)
        client = app.test_client()

        health = client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.get_json()["vocab_size"], 1097)

        tok = client.post("/tokenize", json={"code": "return self.relu(x)"})
        self.assertEqual(tok.status_code, 200)
        self.assertTrue(tok.get_json()["success"])

        comp = client.post("/autocomplete", json={"prompt": "return self.rel"})
        self.assertEqual(comp.status_code, 200)
        payload = comp.get_json()
        self.assertTrue(payload["success"])
        self.assertGreater(len(payload["prediction"]), 0)

        bad = client.post("/autocomplete", json={"prompt": ""})
        self.assertEqual(bad.status_code, 400)


if __name__ == "__main__":
    unittest.main()
