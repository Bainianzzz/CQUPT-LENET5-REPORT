from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_run_dir(root: str | os.PathLike = "runs") -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(root) / ts
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    accs: list[float] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        losses.append(loss.item())
        accs.append(accuracy_from_logits(logits, y))
    return float(np.mean(losses)), float(np.mean(accs))


def save_metrics_jsonl(path: Path, rows: Iterable[EpochMetrics]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")


def plot_curves(out_path: Path, rows: list[EpochMetrics]) -> None:
    epochs = [r.epoch for r in rows]
    train_loss = [r.train_loss for r in rows]
    val_loss = [r.val_loss for r in rows]
    train_acc = [r.train_acc for r in rows]
    val_acc = [r.val_acc for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc, label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


@torch.inference_mode()
def save_confusion_matrix(
    out_path: Path,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list[str] | None = None,
) -> None:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        y_pred.extend(pred)
        y_true.extend(y.numpy().tolist())

    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names if class_names is not None else list(range(10)),
    )
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix (MNIST)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


@torch.inference_mode()
def save_prediction_samples(
    out_path: Path,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_items: int = 25,
) -> None:
    model.eval()
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    ps: list[int] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        xs.append(x.cpu())
        ys.extend(y.numpy().tolist())
        ps.extend(pred.cpu().numpy().tolist())
        if len(ys) >= max_items:
            break

    x_all = torch.cat(xs, dim=0)[:max_items]  # (K,1,28,28)
    ys = ys[:max_items]
    ps = ps[:max_items]

    k = len(ys)
    cols = int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.axis("off")
        if idx >= k:
            continue
        img = x_all[idx, 0].numpy()
        ax.imshow(img, cmap="gray")
        ok = "✓" if ps[idx] == ys[idx] else "✗"
        ax.set_title(f"y={ys[idx]} p={ps[idx]} {ok}", fontsize=10)

    fig.suptitle("Prediction Samples", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

