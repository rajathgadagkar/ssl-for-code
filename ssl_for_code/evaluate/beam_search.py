"""Prefix-constrained beam search decoders for Python code completion."""

from __future__ import annotations

from heapq import heappop, heappush
from typing import Any, List, Optional, Tuple

import torch

from ssl_for_code.dataset import ID_CHARS
from ssl_for_code.models.state import StateUpdater

EPS_PROB = 1e-6
MIN_BEAM_PROB = 1e-4


class PredictionComplete:
    """Stopping criterion determining when a beam candidate forms a complete suggestion."""

    def __call__(self, text: str, token_str: str) -> bool:
        raise NotImplementedError


class NextWordPredictionComplete(PredictionComplete):
    """Stops once the generated suffix crosses an identifier/non-identifier boundary."""

    def __init__(self, rest: str, min_length: int) -> None:
        self.rest = rest
        self.min_length = min_length

    def __call__(self, text: str, token_str: str) -> bool:
        if not text or not token_str:
            return False
        if len(text) - len(self.rest) < self.min_length:
            return False

        prev_is_id = text[-1] in ID_CHARS
        last_is_id = token_str[-1] in ID_CHARS
        return prev_is_id != last_is_id


class NextWordNewLinePredictionComplete(PredictionComplete):
    """Stops on a newline or identifier/non-identifier boundary after ``min_length``."""

    def __init__(self, rest: str, min_length: int) -> None:
        self.rest = rest
        self.min_length = min_length

    def __call__(self, text: str, token_str: str) -> bool:
        if not text or not token_str:
            return False
        if len(text) - len(self.rest) < self.min_length:
            return False
        if "\n" in token_str:
            return True

        prev_is_id = text[-1] in ID_CHARS
        last_is_id = token_str[-1] in ID_CHARS
        return prev_is_id != last_is_id


class BeamSearch:
    """Abstract base class for autoregressive beam search decoders."""

    def next_batch(
        self, prompt: torch.Tensor, state: Any
    ) -> Tuple[Optional[torch.Tensor], Any]:
        raise NotImplementedError

    def update(self, next_token: torch.Tensor, state: Any, old_state: Any) -> None:
        raise NotImplementedError


class BeamSearchSimple(BeamSearch):
    """Probability-ranked prefix-constrained beam search decoder."""

    def __init__(
        self,
        *,
        beam_size: int,
        prediction_complete: PredictionComplete,
        max_beam_size: int,
        rest: str,
        state_updater: StateUpdater,
        probs: Optional[List[float]],
        is_token_by_token: bool,
        itos: List[str],
    ) -> None:
        super().__init__()
        self.itos = itos
        self.is_token_by_token = is_token_by_token
        self.state_updater = state_updater
        self.prediction_complete = prediction_complete
        self.max_beam_size = max_beam_size
        self.rest = rest

        if probs is None:
            probs = [1.0 / beam_size] * beam_size
        assert len(probs) == beam_size
        self.probs = list(probs)

        self.result_heap: List[Tuple[float, Tuple[str, Any]]] = []
        self.text: List[str] = [""] * beam_size
        self.beam_heap: List[Tuple[float, Tuple[int, int]]] = []

    @staticmethod
    def is_substr(original: Optional[str], token_str: str) -> bool:
        if not original:
            return True
        n = min(len(original), len(token_str))
        return original[:n] == token_str[:n]

    def add_prediction(self, prob: float, beam_idx: int, token_str: str, state: Any) -> bool:
        if len(self.result_heap) == self.max_beam_size:
            if self.result_heap[0][0] > prob - EPS_PROB:
                return False
            heappop(self.result_heap)

        beam_state = self.state_updater.get_from_batch(state, beam_idx)
        text = self.text[beam_idx] + token_str
        heappush(self.result_heap, (prob, (text, beam_state)))
        return True

    def add_prediction_before_token(self, prob: float, beam_idx: int, state: Any) -> bool:
        if len(self.result_heap) == self.max_beam_size:
            if self.result_heap[0][0] > prob - EPS_PROB:
                return False
            heappop(self.result_heap)

        beam_state = self.state_updater.get_from_batch(state, beam_idx)
        text = self.text[beam_idx]
        heappush(self.result_heap, (prob, (text, beam_state)))
        return True

    def add_beam(self, prob: float, beam_idx: int, token: int) -> bool:
        if self.result_heap and self.result_heap[0][0] > prob - EPS_PROB:
            return False
        if prob < MIN_BEAM_PROB:
            return False

        if len(self.beam_heap) == self.max_beam_size:
            if self.beam_heap[0][0] > prob - EPS_PROB:
                return False
            heappop(self.beam_heap)
        heappush(self.beam_heap, (prob, (beam_idx, token)))
        return True

    def next_batch(
        self, prompt: torch.Tensor, state: Any
    ) -> Tuple[Optional[torch.Tensor], Any]:
        if not self.beam_heap:
            return None, None

        new_prompt = []
        new_state = []
        prev_texts = self.text
        self.text = []
        self.probs = []

        for prob, (b, token) in self.beam_heap:
            token_tensor = prompt.new_tensor([token])
            if self.is_token_by_token or prompt.shape[0] == 0:
                new_prompt.append(token_tensor)
            else:
                new_prompt.append(torch.cat((prompt[1:, b], token_tensor)))
            new_state.append(self.state_updater.get_from_batch(state, b))
            self.probs.append(prob)
            self.text.append(prev_texts[b] + self.itos[token])

        batched_prompt = torch.stack(new_prompt, dim=1)
        batched_state = self.state_updater.make_batch(new_state)
        self.beam_heap = []
        return batched_prompt, batched_state

    def update(self, next_token: torch.Tensor, state: Any, old_state: Any) -> None:
        del state
        self.beam_heap = []

        for b, text in enumerate(self.text):
            check_rest = None if len(text) >= len(self.rest) else self.rest[len(text) :]
            tokens = next_token[b]
            sort_idx = torch.argsort(tokens)

            for i in reversed(range(len(tokens))):
                token_id = int(sort_idx[i].item())
                token_str = self.itos[token_id]
                if not self.is_substr(check_rest, token_str):
                    continue

                if self.prediction_complete(text, token_str):
                    self.add_prediction_before_token(self.probs[b], b, old_state)
                    break

                if not self.add_beam(self.probs[b] * float(tokens[token_id].item()), b, token_id):
                    break
