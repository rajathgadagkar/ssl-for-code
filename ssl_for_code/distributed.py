"""Multi-GPU DistributedDataParallel (DDP) training launcher via ``torchrun``."""

from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ssl_for_code.dataset.dataset import SourceCodeDataset, build_dataloaders, get_tokenizer
from ssl_for_code.models import build_model
from ssl_for_code.models.state import MemoryUpdater, StateUpdater
from ssl_for_code.train import TrainConfig, parse_args, run_epoch
from ssl_for_code.utils.io import get_data_path


def run_distributed(config: TrainConfig) -> None:
    """Initialize process group and train with DistributedDataParallel."""
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=60))

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

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

    base_model = build_model(
        config.model,
        n_tokens=tokenizer.n_tokens,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
    ).to(device)
    ddp_model = DDP(base_model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    state_updater: StateUpdater = (
        MemoryUpdater(config.mem_len) if "xl" in config.model.lower() else StateUpdater()
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        ddp_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    try:
        for epoch in range(1, config.epochs + 1):
            train_loss, train_acc = run_epoch(
                ddp_model,
                train_loader,
                criterion,
                state_updater,
                device,
                optimizer=optimizer,
                grad_norm_clip=config.grad_norm_clip,
                max_steps=config.max_steps,
            )
            valid_loss, valid_acc = run_epoch(
                ddp_model,
                valid_loader,
                criterion,
                state_updater,
                device,
                optimizer=None,
                max_steps=config.max_steps,
            )
            if rank == 0:
                print(
                    f"[DDP] Epoch {epoch:02d}/{config.epochs:02d} | "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                    f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f}"
                )
    finally:
        dist.destroy_process_group()


def main() -> None:
    run_distributed(parse_args())


if __name__ == "__main__":
    main()
