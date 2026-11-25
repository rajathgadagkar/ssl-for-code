"""PyTorch model forward-pass, recurrence, and training smoke tests."""

from __future__ import annotations

import unittest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed in this environment")
class TestModels(unittest.TestCase):
    def setUp(self) -> None:
        self.vocab_size = 64
        self.seq_len = 12
        self.batch_size = 2
        self.x = torch.randint(0, self.vocab_size, (self.seq_len, self.batch_size))

    def test_all_architectures_forward_and_backward(self) -> None:
        from ssl_for_code.models import build_model
        from ssl_for_code.models.state import MemoryUpdater, StateUpdater

        for arch in ("transformer_xl", "transformer", "lstm", "rhn"):
            with self.subTest(arch=arch):
                model = build_model(
                    arch,
                    n_tokens=self.vocab_size,
                    d_model=32,
                    n_layers=2,
                    n_heads=4,
                    d_ff=64,
                    rnn_size=32,
                    rhn_depth=2,
                    dropout=0.1,
                )
                updater = MemoryUpdater(mem_len=16) if arch == "transformer_xl" else StateUpdater()
                logits, state = model(self.x, None)
                self.assertEqual(logits.shape, (self.seq_len, self.batch_size, self.vocab_size))

                # Second segment verifies state/memory persistence.
                state = updater(None, state)
                logits2, state2 = model(self.x, state)
                self.assertEqual(logits2.shape, (self.seq_len, self.batch_size, self.vocab_size))

                loss = logits2.mean()
                loss.backward()
                state_after = updater(state, state2)
                if arch == "transformer_xl":
                    self.assertEqual(len(state_after), 2)
                    self.assertLessEqual(state_after[0].shape[0], 16)


if __name__ == "__main__":
    unittest.main()
