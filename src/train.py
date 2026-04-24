from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import LeNet5
from utils import (
    EpochMetrics,
    evaluate,
    make_run_dir,
    plot_curves,
    save_confusion_matrix,
    save_metrics_jsonl,
    save_prediction_samples,
    set_seed,
)


def build_loaders(data_dir: Path, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_ds = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


@dataclass
class _Tee:
    primary: TextIO
    secondary: TextIO

    def write(self, s: str) -> int:
        n = self.primary.write(s)
        self.secondary.write(s)
        return n

    def flush(self) -> None:
        self.primary.flush()
        self.secondary.flush()

    def isatty(self) -> bool:  # tqdm may query this
        return bool(getattr(self.primary, "isatty", lambda: False)())


def main() -> None:
    parser = argparse.ArgumentParser(description="LeNet-5 on MNIST (train + eval + plots)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    data_dir = Path(args.data_dir)

    run_dir = make_run_dir("runs")
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8", buffering=1) as log_f:
        sys.stdout = _Tee(old_stdout, log_f)
        sys.stderr = _Tee(old_stderr, log_f)
        try:
            print(f"[run_dir] {run_dir}")
            print(f"[device] {device}")

            train_loader, test_loader = build_loaders(data_dir, args.batch_size, args.num_workers)

            model = LeNet5(num_classes=10).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            config = {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
                "data_dir": str(data_dir),
                "device": str(device),
            }
            (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

            rows: list[EpochMetrics] = []
            best_val_acc = -1.0

            for epoch in range(1, args.epochs + 1):
                model.train()
                train_losses: list[float] = []
                train_accs: list[float] = []

                pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", ncols=100, file=sys.stdout)
                for x, y in pbar:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()

                    with torch.inference_mode():
                        acc = (torch.argmax(logits, dim=1) == y).float().mean().item()
                    train_losses.append(loss.item())
                    train_accs.append(acc)
                    pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

                train_loss = float(sum(train_losses) / max(1, len(train_losses)))
                train_acc = float(sum(train_accs) / max(1, len(train_accs)))
                val_loss, val_acc = evaluate(model, test_loader, criterion, device)

                print(
                    f"[epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                )

                rows.append(
                    EpochMetrics(
                        epoch=epoch,
                        train_loss=train_loss,
                        train_acc=train_acc,
                        val_loss=val_loss,
                        val_acc=val_acc,
                    )
                )

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(
                        {"model_state": model.state_dict(), "config": config, "epoch": epoch, "val_acc": val_acc},
                        run_dir / "checkpoints" / "best.pt",
                    )

            save_metrics_jsonl(run_dir / "metrics.jsonl", rows)
            plot_curves(run_dir / "metrics.png", rows)
            save_confusion_matrix(run_dir / "confusion_matrix.png", model, test_loader, device)
            save_prediction_samples(run_dir / "samples.png", model, test_loader, device, max_items=25)

            (run_dir / "model.txt").write_text(str(model), encoding="utf-8")
            print(f"\nDone. Artifacts saved to: {run_dir}")
            print(f"[log] {log_path}")
        except Exception:
            traceback.print_exc()
            raise
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


if __name__ == "__main__":
    main()

