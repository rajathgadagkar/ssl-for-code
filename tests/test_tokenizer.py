"""Unit tests for SourceCodeTokenizer, BPE learning/encoding, CharacterTokenizer, and dataset splits."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ssl_for_code.dataset.bpe import BPE, BPEEnDe, BPELearner, load_bpe_tokenizer
from ssl_for_code.dataset.break_words import NoTokenizer, SourceCodeTokenizer
from ssl_for_code.dataset.create_dataset import (
    build_splits,
    get_python_files,
    get_repos_from_readme,
    read_clean_file,
)
from ssl_for_code.dataset.dataset import CharacterTokenizer


class TestSourceCodeTokenizer(unittest.TestCase):
    def test_splits_identifiers_and_symbols(self) -> None:
        tok = SourceCodeTokenizer()
        code = "def forward(self, x):\n    return self.relu(x)\n"
        pieces = tok.tokenize(code)
        self.assertEqual("".join(pieces), code)
        self.assertIn("def", pieces)
        self.assertIn("forward", pieces)
        self.assertIn("return", pieces)
        self.assertIn("relu", pieces)

    def test_collect_words_ranks_by_frequency(self) -> None:
        tok = SourceCodeTokenizer()
        tok.collect_words("self.x = self.y + self.z + self")
        words, freqs = tok.get_words()
        self.assertEqual(words[0], "self")
        self.assertEqual(freqs[0], 4)

    def test_no_tokenizer_identity(self) -> None:
        tok = NoTokenizer()
        self.assertEqual(tok.tokenize("abc = 1"), ["abc = 1"])
        self.assertEqual(tok.tokenize(""), [])


class TestBPE(unittest.TestCase):
    def test_cached_bpe_round_trip(self) -> None:
        bpe = load_bpe_tokenizer()
        self.assertEqual(bpe.n_tokens, 1097)
        snippet = (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class SimpleNet(nn.Module):\n"
            "    def forward(self, x):\n"
            "        return self.relu(x)\n"
        )
        ids = bpe.encode(snippet)
        self.assertGreater(len(ids), 0)
        self.assertLess(len(ids), len(snippet))  # Achieves genuine subword compression.
        self.assertEqual(bpe.decode(ids), snippet)

    def test_bpe_rstrip_splits_trailing_partial_word(self) -> None:
        bpe = load_bpe_tokenizer()
        stripped, ids = bpe.rstrip("return self.rel")
        self.assertEqual(stripped, "return self.")
        self.assertEqual(bpe.decode(ids), "return self.")

    def test_bpe_learner_trains_and_compresses(self) -> None:
        corpus = (
            "def train_model(model, optimizer, loader):\n"
            "    for batch in loader:\n"
            "        optimizer.zero_grad()\n"
            "        loss = model(batch)\n"
            "        loss.backward()\n"
            "        optimizer.step()\n"
        ) * 15
        wt = SourceCodeTokenizer()
        wt.collect_words(corpus)
        learner = BPELearner(*wt.get_words())
        merges = learner.learn(40)
        self.assertGreater(merges, 10)
        self.assertLess(learner.get_length(), len(corpus))

        codec = BPEEnDe()
        codec.load(learner.char_itos, learner.char_stoi, learner.bpe)
        bpe = BPE(codec, SourceCodeTokenizer())
        sample = "def train_model(model, optimizer):"
        self.assertEqual(bpe.decode(bpe.encode(sample)), sample)


class TestCharacterTokenizerAndDataset(unittest.TestCase):
    def test_character_tokenizer_train_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tok = CharacterTokenizer(retrain=True, cache_dir=tmp)
            tok.train("def foo(x):\n    return x + 1\n")
            self.assertTrue(tok.is_trained)
            encoded = tok.encode("def foo(x):")
            self.assertEqual(tok.decode(encoded), "def foo(x):")

            reloaded = CharacterTokenizer(retrain=False, cache_dir=tmp)
            self.assertTrue(reloaded.is_trained)
            self.assertEqual(reloaded.n_tokens, tok.n_tokens)

    def test_build_splits_from_local_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "repos"
            (src / "proj_a").mkdir(parents=True)
            (src / "proj_b").mkdir(parents=True)
            (src / "proj_a" / "a.py").write_text("x = 1\n", encoding="utf-8")
            (src / "proj_b" / "b.py").write_text("y = 2\n", encoding="utf-8")
            (src / "proj_b" / "notes.txt").write_text("ignore me\n", encoding="utf-8")

            py_files = get_python_files(src)
            self.assertEqual(len(py_files), 2)
            self.assertEqual(read_clean_file(py_files[0].path).strip(), "x = 1")

            out_dir = root / "out"
            n_train, n_valid = build_splits(src, out_dir, train_ratio=0.5, seed=7)
            self.assertEqual(n_train, 1)
            self.assertEqual(n_valid, 1)
            self.assertTrue((out_dir / "train.py").exists())
            self.assertTrue((out_dir / "valid.py").exists())

    def test_extract_repos_from_readme(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp) / "awesome.md"
            md.write_text(
                "- [PyTorch](https://github.com/pytorch/pytorch)\n"
                "- [Duplicate](https://github.com/pytorch/pytorch)\n"
                "- [Docs](https://pytorch.org/docs)\n",
                encoding="utf-8",
            )
            repos = get_repos_from_readme(md)
            self.assertEqual(repos, [("pytorch", "pytorch")])


if __name__ == "__main__":
    unittest.main()
