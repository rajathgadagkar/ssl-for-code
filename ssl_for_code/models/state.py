"""Recurrent hidden-state and Transformer-XL segment-memory managers."""

from __future__ import annotations

from typing import Any, List, Optional

import torch


class StateUpdater:
    """Manages recurrent model state across sequential steps and beam batches."""

    def __call__(self, old_state: Any, new_state: Any) -> Any:
        del old_state
        return new_state

    def get_from_batch(self, state: Any, batch_idx: int) -> Any:
        if state is None:
            return None
        if isinstance(state, torch.Tensor):
            return state[:, batch_idx] if state.ndim >= 2 else state[batch_idx]
        if isinstance(state, tuple):
            return tuple(self.get_from_batch(s, batch_idx) for s in state)
        if isinstance(state, list):
            return [self.get_from_batch(s, batch_idx) for s in state]
        return state

    def make_batch(self, batch: List[Any]) -> Any:
        if not batch or batch[0] is None:
            return None
        first = batch[0]
        if isinstance(first, torch.Tensor):
            dim = 1 if first.ndim >= 2 else 0
            return torch.stack(batch, dim=dim)
        if isinstance(first, tuple):
            return tuple(
                self.make_batch([item[n] for item in batch]) for n in range(len(first))
            )
        if isinstance(first, list):
            return [
                self.make_batch([item[n] for item in batch]) for n in range(len(first))
            ]
        return batch


class MemoryUpdater(StateUpdater):
    """Maintains a sliding window of segment-level memory tensors for Transformer-XL.

    Each layer's memory tensor has shape ``[mem_len, batch_size, d_model]``.
    """

    def __init__(self, mem_len: int = 256) -> None:
        self.mem_len = mem_len

    def __call__(
        self,
        old_mem: Optional[List[torch.Tensor]],
        new_mem: Optional[List[torch.Tensor]],
    ) -> List[torch.Tensor]:
        if self.mem_len == 0 or not new_mem:
            return []

        if old_mem:
            mem = [torch.cat((m, x.detach()), dim=0) for m, x in zip(old_mem, new_mem)]
        else:
            mem = [x.detach() for x in new_mem]

        if mem and mem[0].shape[0] > self.mem_len:
            mem = [m[-self.mem_len :] for m in mem]

        return mem

    def get_from_batch(self, state: Optional[List[torch.Tensor]], batch_idx: int):
        if state is None:
            return None
        return [m[:, batch_idx] for m in state]

    def make_batch(self, batch: List[Optional[List[torch.Tensor]]]):
        if not batch or batch[0] is None:
            return None
        n_layers = len(batch[0])
        return [
            torch.stack([item[layer_idx] for item in batch], dim=1)
            for layer_idx in range(n_layers)
        ]
