"""Keystroke-saving evaluation benchmark for Python code autocompletion."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from ssl_for_code.evaluate import Predictor
from ssl_for_code.evaluate.factory import get_predictor


@dataclass
class EvaluationReport:
    total_chars: int
    predicted_chars: int
    keystrokes: int
    keystroke_savings: float


def evaluate(
    predictor: Predictor,
    text: str,
    *,
    beam_size: int = 5,
    min_length: int = 5,
) -> EvaluationReport:
    """Simulate character-by-character typing and measure keystroke savings."""
    if len(text) <= 1:
        return EvaluationReport(len(text), 0, len(text), 0.0)

    correct = 0
    i = 0
    keystrokes = 0

    while i + 1 < len(text):
        prefix = text[: i + 1]
        predictions = predictor.autocomplete(
            prefix, beam_size=beam_size, min_length=min_length
        )
        candidate = predictions[0].text if predictions else ""

        if candidate and candidate == text[i + 1 : i + 1 + len(candidate)]:
            correct += len(candidate)
            step = len(candidate)
        else:
            step = 1

        i += step
        keystrokes += 1

    savings = correct / (len(text) - 1)
    return EvaluationReport(
        total_chars=len(text),
        predicted_chars=correct,
        keystrokes=keystrokes,
        keystroke_savings=savings,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate keystroke-saving accuracy on a Python file.")
    parser.add_argument("file", type=Path, help="Path to a Python source file to evaluate.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to model checkpoint.")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam width (default: 5).")
    args = parser.parse_args()

    predictor = get_predictor(args.checkpoint)
    report = evaluate(predictor, args.file.read_text(encoding="utf-8"), beam_size=args.beam_size)
    print(
        f"Evaluated {args.file}: {report.predicted_chars}/{report.total_chars - 1} chars predicted "
        f"({report.keystroke_savings:.2%} keystroke savings, {report.keystrokes} keystrokes)."
    )


if __name__ == "__main__":
    main()
