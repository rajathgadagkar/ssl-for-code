"""Token-level probability anomaly detection for Python source code."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch

from ssl_for_code.dataset import Tokenizer
from ssl_for_code.evaluate.factory import get_predictor
from ssl_for_code.models.state import StateUpdater


@dataclass
class TokenAnomaly:
    token: str
    probability: float
    severity: str


def classify_probability(prob: float) -> str:
    """Bucket a token's model probability into an anomaly severity tier."""
    if prob > 0.75:
        return "expected"
    if prob > 0.20:
        return "normal"
    if prob > 0.01:
        return "unlikely"
    return "anomaly"


def detect_anomalies(
    tokenizer: Tokenizer,
    text: str,
    model: torch.nn.Module,
    state_updater: StateUpdater,
    *,
    is_token_by_token: bool = True,
    context_window: int = 512,
) -> List[TokenAnomaly]:
    """Score every token in ``text`` under the autoregressive language model."""
    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return []

    device = next(model.parameters()).device
    tensor = torch.tensor(tokens, dtype=torch.long, device=device)
    prompt = tensor[:1].unsqueeze(-1)
    state = None
    results: List[TokenAnomaly] = []

    for i in range(1, len(tensor)):
        with torch.no_grad():
            logits, new_state = model(prompt, state)
        state = state_updater(state, new_state)
        probs = torch.softmax(logits[-1, 0], dim=-1)

        target_id = int(tensor[i].item())
        prob = float(probs[target_id].item())
        results.append(
            TokenAnomaly(
                token=tokenizer.itos[target_id],
                probability=prob,
                severity=classify_probability(prob),
            )
        )

        if is_token_by_token:
            prompt = tensor[i : i + 1].unsqueeze(-1)
        else:
            prompt = tensor[max(0, i + 1 - context_window) : i + 1].unsqueeze(-1)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Score token probabilities to surface code anomalies.")
    parser.add_argument("file", type=Path, help="Python file to inspect.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.01)
    args = parser.parse_args()

    predictor = get_predictor(args.checkpoint)
    findings = detect_anomalies(
        predictor.tokenizer,
        args.file.read_text(encoding="utf-8"),
        predictor.model,
        predictor.state_updater,
        is_token_by_token=predictor.is_token_by_token,
    )
    flagged = [f for f in findings if f.probability < args.threshold]
    print(f"Scanned {len(findings)} tokens; flagged {len(flagged)} low-probability anomalies (< {args.threshold}).")


if __name__ == "__main__":
    main()
