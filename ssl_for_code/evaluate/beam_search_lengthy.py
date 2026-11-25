"""Length-normalized beam search decoder favoring longer coherent completions."""

from __future__ import annotations

from heapq import heappop, heappush
from typing import Any, List, Optional, Tuple

import torch

from ssl_for_code.evaluate.beam_search import BeamSearch, PredictionComplete
from ssl_for_code.models.state import StateUpdater

EPS_SCORE = 1e-6
MIN_BEAM_PROB = 1e-4


class BeamSearchLengthy(BeamSearch):
    """Beam search using length-weighted scoring: ``score(t, p) = len(t)^alpha * p``."""

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
        length_alpha: float = 0.2,
    ) -> None:
        super().__init__()
        self.itos = itos
        self.is_token_by_token = is_token_by_token
        self.state_updater = state_updater
        self.prediction_complete = prediction_complete
        self.max_beam_size = max_beam_size
        self.rest = rest
        self.length_alpha = length_alpha

        if probs is None:
            probs = [1.0 / beam_size] * beam_size
        assert len(probs) == beam_size
        self.probs = list(probs)

        self.result_heap: List[Tuple[float, Tuple[str, Any]]] = []
        self.text: List[str] = [""] * beam_size
        self.beam_heap: List[Tuple[float, Tuple[int, float, int]]] = []

    @staticmethod
    def is_prefix(original: Optional[str], token_str: str) -> bool:
        if not original:
            return True
        n = min(len(original), len(token_str))
        return original[:n] == token_str[:n]

    def _get_score(self, text: str, prob: float) -> float:
        return (max(len(text), 1) ** self.length_alpha) * prob

    def _add_prediction_before_token(self, prob: float, beam_idx: int, state: Any) -> None:
        text = self.text[beam_idx]
        score = self._get_score(text, prob)

        if len(self.result_heap) == self.max_beam_size:
            if self.result_heap[0][0] > score - EPS_SCORE:
                return
            heappop(self.result_heap)

        beam_state = self.state_updater.get_from_batch(state, beam_idx)
        heappush(self.result_heap, (score, (text, beam_state)))

    def _add_beam(self, prob: float, beam_idx: int, token: int) -> bool:
        if prob < MIN_BEAM_PROB:
            return False

        text = self.text[beam_idx] + self.itos[token]
        score = self._get_score(text, prob)

        if len(self.beam_heap) == self.max_beam_size:
            if self.beam_heap[0][0] > score - EPS_SCORE:
                return False
            heappop(self.beam_heap)

        heappush(self.beam_heap, (score, (beam_idx, prob, token)))
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

        for _, (b, prob, token) in self.beam_heap:
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
            added_to_results = False

            for i in reversed(range(len(tokens))):
                token_id = int(sort_idx[i].item())
                token_str = self.itos[token_id]
                if not self.is_prefix(check_rest, token_str):
                    continue

                if not added_to_results and self.prediction_complete(text, token_str):
                    added_to_results = True
                    self._add_prediction_before_token(self.probs[b], b, old_state)

                p = (
                    1.0
                    if (check_rest and len(token_str) <= len(check_rest))
                    else float(tokens[token_id].item())
                )
                if not self._add_beam(self.probs[b] * p, b, token_id):
                    break
