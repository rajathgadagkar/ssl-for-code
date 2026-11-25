"""Inference predictor and beam search evaluation utilities."""

from __future__ import annotations

from typing import Any, List, NamedTuple, Optional, Tuple

import torch
from torch import nn

from ssl_for_code.dataset import Tokenizer
from ssl_for_code.evaluate.beam_search import (
    BeamSearchSimple,
    NextWordNewLinePredictionComplete,
    NextWordPredictionComplete,
    PredictionComplete,
)
from ssl_for_code.evaluate.beam_search_lengthy import BeamSearchLengthy
from ssl_for_code.models.state import StateUpdater


class Prediction(NamedTuple):
    prob: float
    text: str
    state: Any


class Predictor:
    """Autoregressive code completion predictor supporting beam search decoding."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        *,
        state_updater: Optional[StateUpdater] = None,
        is_token_by_token: bool = True,
        beam_strategy: str = "simple",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.state_updater = state_updater or StateUpdater()
        self.is_token_by_token = is_token_by_token
        self.beam_strategy = beam_strategy

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _get_predictions(
        self, prompt: torch.Tensor, state: Any
    ) -> Tuple[torch.Tensor, Any]:
        if prompt.shape[0] == 0:
            vocab_size = len(self.tokenizer.itos)
            uniform = prompt.new_ones((prompt.shape[1], vocab_size), dtype=torch.float) / vocab_size
            return uniform.to(self.device), state

        prompt = prompt.to(self.device)
        with torch.no_grad():
            logits, new_state = self.model(prompt, state)

        next_state = self.state_updater(state, new_state)
        probs = torch.softmax(logits[-1], dim=-1)
        return probs, next_state

    def get_next_token(
        self, prompt_text: str, state: Any = None
    ) -> Tuple[str, Any]:
        """Greedily predict the single most likely next subword token."""
        token_ids = self.tokenizer.encode(prompt_text)
        prompt = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(-1)
        probs, next_state = self._get_predictions(prompt, state)
        best_id = int(torch.argmax(probs[0]).item())
        return self.tokenizer.itos[best_id], next_state

    def get_next_word(
        self,
        prompt: torch.Tensor,
        state: Any,
        rest: str,
        probs: Optional[List[float]],
        prediction_complete: PredictionComplete,
        max_beam_size: int = 5,
        max_steps: int = 10,
    ) -> List[Prediction]:
        """Run prefix-constrained beam search to complete the current code token/word."""
        beam_cls = BeamSearchLengthy if self.beam_strategy == "lengthy" else BeamSearchSimple
        beam = beam_cls(
            beam_size=prompt.shape[1],
            prediction_complete=prediction_complete,
            max_beam_size=max_beam_size,
            rest=rest,
            state_updater=self.state_updater,
            probs=probs,
            is_token_by_token=self.is_token_by_token,
            itos=self.tokenizer.itos,
        )

        for _ in range(max_steps):
            next_token, new_state = self._get_predictions(prompt, state)
            beam.update(next_token, new_state, state)
            prompt, state = beam.next_batch(prompt, new_state)
            if prompt is None:
                break

        results = [Prediction(r[0], r[2][0], r[2][1]) for r in beam.result_heap]
        results.sort(key=lambda item: -item.prob)
        return results

    def autocomplete(
        self,
        prefix: str,
        *,
        beam_size: int = 5,
        min_length: int = 15,
        stop_on_newline: bool = False,
    ) -> List[Prediction]:
        """High-level helper returning ranked code completions for a raw code prefix."""
        stripped, token_ids = self.rstrip(prefix)
        rest = prefix[len(stripped) :]
        complete_cls = (
            NextWordNewLinePredictionComplete if stop_on_newline else NextWordPredictionComplete
        )
        completion_rule = complete_cls(rest, min_length)
        prompt = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(-1)
        predictions = self.get_next_word(prompt, None, rest, [1.0], completion_rule, beam_size)
        return [
            Prediction(prob=p.prob, text=p.text[len(rest) :], state=p.state)
            for p in predictions
        ]

    def rstrip(self, prompt: str) -> Tuple[str, List[int]]:
        return self.tokenizer.rstrip(prompt)


__all__ = [
    "Prediction",
    "Predictor",
    "PredictionComplete",
    "NextWordPredictionComplete",
    "NextWordNewLinePredictionComplete",
    "BeamSearchSimple",
    "BeamSearchLengthy",
]
