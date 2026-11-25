"""Source-code aware word boundary pre-tokenizers."""

from __future__ import annotations

from typing import Dict, List, Tuple

from ssl_for_code.dataset import ID_CHARS


class WordTokenizer:
    """Abstract interface for pre-tokenizing source code into word spans."""

    def collect_words(self, data: str) -> None:
        raise NotImplementedError

    def get_words(self) -> Tuple[List[str], List[int]]:
        raise NotImplementedError

    def tokenize(self, data: str, *, is_silent: bool = False) -> List[str]:
        raise NotImplementedError


class SourceCodeTokenizer(WordTokenizer):
    """Splits Python source code at identifier vs. non-identifier boundaries.

    Consecutive identifier characters (``[A-Za-z0-9_]``) form a single span,
    while consecutive non-identifier characters (operators, punctuation,
    indentation, newlines) form alternating spans.
    """

    def __init__(self) -> None:
        self.words: Dict[str, int] = {}

    def add_word(self, word: str) -> None:
        if not word:
            return
        self.words[word] = self.words.get(word, 0) + 1

    def tokenize(self, data: str, *, is_silent: bool = False) -> List[str]:
        del is_silent  # Kept for API compatibility.
        if not data:
            return []

        last_idx = 0
        is_id = False
        tokens: List[str] = []

        for i, char in enumerate(data):
            if char in ID_CHARS:
                if not is_id:
                    if last_idx < i:
                        tokens.append(data[last_idx:i])
                    last_idx = i
                    is_id = True
            else:
                if is_id:
                    if last_idx < i:
                        tokens.append(data[last_idx:i])
                    last_idx = i
                    is_id = False

        if last_idx < len(data):
            tokens.append(data[last_idx:])

        return tokens

    def collect_words(self, data: str) -> None:
        for word in self.tokenize(data, is_silent=True):
            self.add_word(word)

    def get_words(self) -> Tuple[List[str], List[int]]:
        ranked = sorted(self.words.items(), key=lambda item: (-item[1], item[0]))
        return [word for word, _ in ranked], [freq for _, freq in ranked]


class NoTokenizer(WordTokenizer):
    """Identity pre-tokenizer that treats the entire input as a single span."""

    def __init__(self) -> None:
        self.data = ""

    def collect_words(self, data: str) -> None:
        self.data += data

    def get_words(self) -> Tuple[List[str], List[int]]:
        return [self.data], [1]

    def tokenize(self, data: str, *, is_silent: bool = False) -> List[str]:
        del is_silent
        return [data] if data else []
