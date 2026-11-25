"""Self-supervised autoregressive training pipeline for Python code language models."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from ssl_for_code.dataset.dataset import SourceCodeDataset, build_dataloaders, get_tokenizer
from ssl_for_code.models import build_model
from ssl_for_code.models.state import MemoryUpdater, StateUpdater
from ssl_for_code.utils.io import get_data_path, get_device


@dataclass
class TrainConfig:
    model: str = "transformer_xl"
    tokenizer: str = "bpe"
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    rnn_size: int = 512
    rhn_depth: int = 2
    dropout: float = 0.2
    mem_len: int = 256
    batch_size: int = 12
    seq_len: int = 256
    epochs: int = 32
    learning_rate: float = 1.25e-4
    weight_decay: float = 0.01
    grad_norm_clip: float = 1.0
    is_token_by_token: bool = True
    truncate_data: int = 0
    max_steps: Optional[int] = None
    device: str = "auto"
    data_dir: Optional[str] = None
    save_dir: str = "checkpoints"


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    state_updater: StateUpdater,
    device: torch.device,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_norm_clip: float = 1.0,
    max_steps: Optional[int] = None,
) -> Tuple[float, float]:
    """Run one training or validation epoch; return ``(mean_loss, token_accuracy)``."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    state = None

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for step_idx, (src, tgt) in enumerate(loader):
            if max_steps is not None and step_idx >= max_steps:
                break

            src = src.to(device)
            tgt = tgt.to(device)

            logits, new_state = model(src, state)
            state = state_updater(state, new_state)

            vocab_size = logits.size(-1)
            loss = criterion(logits.reshape(-1, vocab_size), tgt.reshape(-1))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_norm_clip)
                optimizer.step()

            n_tok = tgt.numel()
            total_loss += float(loss.item()) * n_tok
            preds = logits.argmax(dim=-1)
            total_correct += int((preds == tgt).sum().item())
            total_tokens += n_tok

    if total_tokens == 0:
        return 0.0, 0.0
    return total_loss / total_tokens, total_correct / total_tokens


@torch.no_grad()
def sample_prompt(
    model: nn.Module,
    tokenizer,
    state_updater: StateUpdater,
    device: torch.device,
    *,
    prompt: str = "def train(",
    steps: int = 25,
    is_token_by_token: bool = True,
) -> str:
    """Greedily sample ``steps`` tokens from ``model`` starting at ``prompt``."""
    model.eval()
    generated = prompt
    current_input = prompt
    state = None

    for _ in range(steps):
        ids = tokenizer.encode(current_input)
        if not ids:
            break
        tensor = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(-1)
        logits, new_state = model(tensor, state)
        state = state_updater(state, new_state)
        next_id = int(logits[-1, 0].argmax(dim=-1).item())
        next_piece = tokenizer.itos[next_id]
        generated += next_piece
        current_input = next_piece if is_token_by_token else generated

    return generated


def train(config: TrainConfig) -> Dict[str, float]:
    """Execute the full training loop and save the best checkpoint."""
    device = get_device(config.device)
    tokenizer = get_tokenizer(config.tokenizer)
    dataset = SourceCodeDataset.from_directory(
        tokenizer,
        data_dir=config.data_dir or get_data_path(),
        truncate_data=config.truncate_data,
    )
    train_loader, valid_loader = build_dataloaders(
        dataset,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        shuffle=False,
    )

    model = build_model(
        config.model,
        n_tokens=tokenizer.n_tokens,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        rnn_size=config.rnn_size,
        rhn_depth=config.rhn_depth,
    ).to(device)

    state_updater: StateUpdater = (
        MemoryUpdater(config.mem_len) if "xl" in config.model.lower() else StateUpdater()
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_valid_loss = float("inf")
    metrics: Dict[str, float] = {}

    for epoch in range(1, config.epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            state_updater,
            device,
            optimizer=optimizer,
            grad_norm_clip=config.grad_norm_clip,
            max_steps=config.max_steps,
        )
        valid_loss, valid_acc = run_epoch(
            model,
            valid_loader,
            criterion,
            state_updater,
            device,
            optimizer=None,
            max_steps=config.max_steps,
        )
        elapsed = time.perf_counter() - t0
        valid_ppl = math.exp(min(valid_loss, 20.0))

        print(
            f"Epoch {epoch:02d}/{config.epochs:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f} ppl={valid_ppl:.2f} | "
            f"{elapsed:.1f}s"
        )

        metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
            "valid_ppl": valid_ppl,
        }

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            ckpt_payload = {
                "config": {**asdict(config), "n_tokens": tokenizer.n_tokens},
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
                "epoch": epoch,
            }
            torch.save(ckpt_payload, save_dir / f"{config.model}_best.pt")

    return metrics


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train a self-supervised Python code language model."
    )
    parser.add_argument(
        "--model",
        default="transformer_xl",
        choices=["transformer_xl", "transformer", "lstm", "rhn"],
    )
    parser.add_argument("--tokenizer", default="bpe", choices=["bpe", "char"])
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mem-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.25e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--truncate-data", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="checkpoints")

    args = parser.parse_args()
    return TrainConfig(
        model=args.model,
        tokenizer=args.tokenizer,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        mem_len=args.mem_len,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        epochs=args.epochs,
        learning_rate=args.lr,
        grad_norm_clip=args.grad_clip,
        truncate_data=args.truncate_data,
        max_steps=args.max_steps,
        device=args.device,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
    )


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
