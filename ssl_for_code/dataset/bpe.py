"""Heap-accelerated Byte Pair Encoding (BPE) learner and encoder for Python code."""

from __future__ import annotations

import argparse
from functools import lru_cache
from heapq import heappop, heappush
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from ssl_for_code.dataset import Tokenizer
from ssl_for_code.dataset.break_words import SourceCodeTokenizer, WordTokenizer
from ssl_for_code.utils.io import get_data_path, load_cache, save_cache

MergeEntry = Union[int, List[int], Tuple[int, int]]


class _BPEEncoder:
    """Applies learned BPE merge rules to a character-ID sequence in O(n log n)."""

    def __init__(self, pairs: Dict[Tuple[int, int], int]) -> None:
        self.pairs = pairs
        self.codes: List[int] = []
        self.next_idx: List[int] = []
        self.prev_idx: List[int] = []
        self.heap: List[Tuple[int, int, Tuple[int, int]]] = []

    def encode(self, codes: List[int]) -> List[int]:
        if not codes:
            return []
        self.codes = list(codes)
        self.next_idx = BPELearner.default_next_pointers(len(self.codes))
        self.prev_idx = BPELearner.default_prev_pointers(len(self.codes))
        self.heap = []

        for i in range(len(self.codes) - 1):
            self._add_pair((self.codes[i], self.codes[i + 1]), i)

        while self.heap:
            _, idx, pair = heappop(self.heap)
            self._merge(idx, pair)

        return [code for code in self.codes if code != -1]

    def _merge(self, p2: int, pair: Tuple[int, int]) -> None:
        p3 = self.next_idx[p2]
        if p3 == -1 or pair[0] != self.codes[p2] or pair[1] != self.codes[p3]:
            return

        self.codes[p2] = self.pairs[pair]
        self.codes[p3] = -1
        p1 = self.prev_idx[p2]
        p4 = self.next_idx[p3]

        if p1 != -1:
            self._add_pair((self.codes[p1], self.codes[p2]), p1)
        self.next_idx[p2] = p4
        if p4 != -1:
            self.prev_idx[p4] = p2
            self._add_pair((self.codes[p2], self.codes[p4]), p2)

    def _add_pair(self, pair: Tuple[int, int], idx: int) -> None:
        rank = self.pairs.get(pair)
        if rank is not None:
            heappush(self.heap, (rank, idx, pair))


class BPEEnDe:
    """Stores a BPE vocabulary and encodes individual pre-tokenized words."""

    def __init__(self) -> None:
        self.char_itos: List[str] = []
        self.char_stoi: Dict[str, int] = {}
        self.bpe: List[MergeEntry] = []
        self.popular_words: Dict[str, List[int]] = {}

        self.bpe_itos: List[str] = []
        self.bpe_stoi: Dict[str, int] = {}
        self.pairs: Dict[Tuple[int, int], int] = {}
        self.encoder: Optional[_BPEEncoder] = None

    def load(
        self,
        char_itos: List[str],
        char_stoi: Dict[str, int],
        bpe: List[MergeEntry],
    ) -> None:
        self.char_itos = list(char_itos)
        self.char_stoi = dict(char_stoi)
        self.bpe = list(bpe)
        self.calc()

    def set_popular_words(self, popular_words: Dict[str, List[int]]) -> None:
        self.popular_words = popular_words

    def calc(self) -> None:
        self.bpe_itos = self.calc_bpe_itos()
        self.bpe_stoi = {token: idx for idx, token in enumerate(self.bpe_itos)}
        self.pairs = {
            (pair[0], pair[1]): code
            for code, pair in enumerate(self.bpe)
            if not isinstance(pair, int)
        }
        self.encoder = _BPEEncoder(self.pairs)
        self.encode.cache_clear()

    def to_char_stoi(self, word: str) -> List[int]:
        return [self.char_stoi[c] for c in word]

    def calc_bpe_itos(self) -> List[str]:
        itos = list(self.char_itos)
        for pair in self.bpe[len(self.char_itos) :]:
            assert not isinstance(pair, int)
            itos.append(itos[pair[0]] + itos[pair[1]])
        return itos

    @lru_cache(maxsize=4096)
    def encode(self, word: str) -> List[int]:
        if word in self.popular_words:
            return list(self.popular_words[word])
        if self.encoder is None:
            raise RuntimeError("BPEEnDe must be loaded before calling encode().")
        char_ids = [self.char_stoi[c] for c in word if c in self.char_stoi]
        return self.encoder.encode(char_ids)


class BPE(Tokenizer):
    """Subword BPE tokenizer combining SourceCodeTokenizer with learned merges."""

    def __init__(self, bpe_en_de: BPEEnDe, word_tokenizer: Optional[WordTokenizer] = None) -> None:
        self.bpe = bpe_en_de
        self.word_tokenizer = word_tokenizer or SourceCodeTokenizer()
        self.is_trained = True

    @property
    def n_tokens(self) -> int:
        return len(self.bpe.bpe)

    @property
    def itos(self) -> List[str]:
        return self.bpe.bpe_itos

    @property
    def stoi(self) -> Dict[str, int]:
        return self.bpe.bpe_stoi

    def encode(self, data: str, *, is_silent: bool = True) -> List[int]:
        words = self.word_tokenizer.tokenize(data, is_silent=is_silent)
        encoded: List[int] = []
        for word in words:
            encoded.extend(self.bpe.encode(word))
        return encoded

    def tokenize_pieces(self, data: str) -> List[str]:
        """Return the string representation of each BPE subword token."""
        return [self.itos[idx] for idx in self.encode(data)]

    def __call__(self, data: str) -> List[str]:
        return self.tokenize_pieces(data)

    def rstrip(self, data: str) -> Tuple[str, List[int]]:
        """Drop the trailing incomplete word span so beam search can complete it."""
        words = self.word_tokenizer.tokenize(data, is_silent=True)
        if not words:
            return "", []
        prefix_words = words[:-1]
        encoded: List[int] = []
        for word in prefix_words:
            encoded.extend(self.bpe.encode(word))
        return "".join(prefix_words), encoded


class BPELearner:
    """Trains BPE merge rules from a frequency-weighted word corpus using a max-heap."""

    def __init__(self, words_list: Sequence[str], word_freq: Sequence[int]) -> None:
        self.words_list = list(words_list)
        self.word_freq = list(word_freq)

        self.heap: List[Tuple[int, Tuple[int, int]]] = []
        self.heap_modified: set[Tuple[int, int]] = set()
        self.char_itos: List[str] = []
        self.char_stoi: Dict[str, int] = {}
        self.bpe: List[MergeEntry] = []
        self.word_codes: List[List[int]] = []
        self.word_code_prev: List[List[int]] = []
        self.word_code_next: List[List[int]] = []

        self.counts: Dict[Tuple[int, int], int] = {}
        self.locations: Dict[Tuple[int, int], Dict[int, set[int]]] = {}

        self.build_vocab()
        self.build_word_arrays()
        self.collect_pairs()

    def learn(self, merges: int) -> int:
        """Perform up to ``merges`` BPE merge operations; return actual merges performed."""
        performed = 0
        for _ in range(merges):
            merged_pair = None
            while self.heap:
                merged_pair = self.merge_pair()
                if merged_pair is not None:
                    performed += 1
                    break
            if merged_pair is None:
                break
        return performed

    def build_vocab(self) -> None:
        vocab = {char for word in self.words_list for char in word}
        self.char_itos = sorted(vocab)
        self.char_stoi = {char: idx for idx, char in enumerate(self.char_itos)}
        self.bpe = list(range(len(self.char_stoi)))

    def to_char_stoi(self, word: str) -> List[int]:
        return [self.char_stoi[c] for c in word]

    @staticmethod
    def default_next_pointers(length: int) -> List[int]:
        if length == 0:
            return []
        return [i + 1 for i in range(length - 1)] + [-1]

    @staticmethod
    def default_prev_pointers(length: int) -> List[int]:
        return [i - 1 for i in range(length)]

    def build_word_arrays(self) -> None:
        self.word_codes = [self.to_char_stoi(word) for word in self.words_list]
        self.word_code_next = [self.default_next_pointers(len(w)) for w in self.word_codes]
        self.word_code_prev = [self.default_prev_pointers(len(w)) for w in self.word_codes]

    def heap_add_all(self) -> None:
        for pair in self.heap_modified:
            count = self.counts.get(pair, 0)
            if count > 0:
                heappush(self.heap, (-count, pair))
        self.heap_modified.clear()

    def add_pair(self, w: int, i: int, nxt: int) -> None:
        pair = (self.word_codes[w][i], self.word_codes[w][nxt])
        assert pair[0] != -1 and pair[1] != -1

        if pair not in self.counts:
            self.counts[pair] = 0
            self.locations[pair] = {}

        if w not in self.locations[pair]:
            self.locations[pair][w] = set()

        self.counts[pair] += self.word_freq[w]
        self.locations[pair][w].add(i)
        self.heap_modified.add(pair)

    def collect_pairs(self) -> None:
        for w, codes in enumerate(self.word_codes):
            for i in range(len(codes) - 1):
                self.add_pair(w, i, i + 1)
        self.heap_add_all()

    def remove_pair(self, w: int, i: int, nxt: int) -> None:
        pair = (self.word_codes[w][i], self.word_codes[w][nxt])
        assert pair[0] != -1 and pair[1] != -1
        if pair not in self.counts:
            return
        self.locations[pair][w].discard(i)
        self.counts[pair] -= self.word_freq[w]
        self.heap_modified.add(pair)

    def merge_pair(self) -> Optional[Tuple[int, int]]:
        if not self.heap:
            return None
        neg_cnt, pair = heappop(self.heap)
        if pair not in self.counts or self.counts[pair] != -neg_cnt or self.counts[pair] <= 0:
            return None

        new_code = len(self.bpe)
        self.bpe.append([pair[0], pair[1]])
        del self.counts[pair]

        for w, locs in self.locations[pair].items():
            sorted_locs = sorted(locs, reverse=True)
            prev = None
            merged: List[int] = []
            for p2 in sorted_locs:
                p1 = self.word_code_prev[w][p2]
                p3 = self.word_code_next[w][p2]
                if p3 == -1 or p3 == prev:
                    continue
                p4 = self.word_code_next[w][p3]

                if p1 != -1:
                    self.remove_pair(w, p1, p2)
                if p4 != -1 and p4 != prev:
                    self.remove_pair(w, p3, p4)

                prev = p2
                merged.append(p2)

            for p2 in merged:
                p3 = self.word_code_next[w][p2]
                p4 = self.word_code_next[w][p3]
                self.word_codes[w][p2] = new_code
                self.word_codes[w][p3] = -1
                self.word_code_next[w][p3] = -1
                self.word_code_prev[w][p3] = -1
                if p4 != -1:
                    self.word_code_next[w][p2] = p4
                    self.word_code_prev[w][p4] = p2
                else:
                    self.word_code_next[w][p2] = -1

            for p2 in merged:
                p1 = self.word_code_prev[w][p2]
                p3 = self.word_code_next[w][p2]
                if p1 != -1:
                    self.add_pair(w, p1, p2)
                if p3 != -1:
                    self.add_pair(w, p2, p3)

        self.heap_add_all()
        return pair

    def bpe_itos(self) -> List[str]:
        itos = list(self.char_itos)
        for pair in self.bpe[len(self.char_itos) :]:
            assert not isinstance(pair, int)
            itos.append(itos[pair[0]] + itos[pair[1]])
        return itos

    def get_length(self) -> int:
        total = 0
        for w, codes in enumerate(self.word_codes):
            active_tokens = sum(1 for idx in codes if idx != -1)
            total += active_tokens * self.word_freq[w]
        return total

    def to_cache_dict(self) -> Dict[str, object]:
        return {
            "char_itos": self.char_itos,
            "char_stoi": self.char_stoi,
            "bpe": self.bpe,
        }


def load_bpe_tokenizer(cache_dir: Optional[str | Path] = None) -> BPE:
    """Load the cached BPE tokenizer from ``data/cache/bpe.json``."""
    bpe_cache = load_cache("bpe", cache_dir=cache_dir)
    if not bpe_cache:
        raise FileNotFoundError(
            "BPE cache not found. Run `python -m ssl_for_code.dataset.bpe` first."
        )
    codec = BPEEnDe()
    codec.load(
        char_itos=bpe_cache["char_itos"],
        char_stoi=bpe_cache["char_stoi"],
        bpe=bpe_cache["bpe"],
    )
    return BPE(codec, SourceCodeTokenizer())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and cache a BPE tokenizer on Python code.")
    parser.add_argument(
        "--input",
        type=Path,
        default=get_data_path() / "train.py",
        help="Path to the training corpus file (default: data/train.py)",
    )
    parser.add_argument(
        "--merges",
        type=int,
        default=1000,
        help="Number of BPE merge operations to learn (default: 1000)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory to save bpe.json (default: data/cache)",
    )
    args = parser.parse_args()

    data = args.input.read_text(encoding="utf-8")
    word_tokenizer = SourceCodeTokenizer()
    word_tokenizer.collect_words(data)

    learner = BPELearner(*word_tokenizer.get_words())
    merges = learner.learn(args.merges)

    out_path = save_cache("bpe", learner.to_cache_dict(), cache_dir=args.cache_dir)
    compression = len(data) / max(learner.get_length(), 1)
    print(
        f"Trained BPE with {merges} merges ({len(learner.bpe)} total tokens, "
        f"{compression:.2f}x compression) -> {out_path}"
    )


if __name__ == "__main__":
    main()
