"""Plotting helpers for training logs and confusion matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(rows: List[Dict], out_path: Path) -> None:
    epochs = [r["epoch"] for r in rows]
    plt.figure(figsize=(8, 5))
    for key, lab in [
        ("train_loss", "train"),
        ("val_loss", "val"),
        ("test_loss", "test"),
    ]:
        if rows and key in rows[0]:
            plt.plot(epochs, [r[key] for r in rows], label=lab)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_acc_curves(rows: List[Dict], out_path: Path) -> None:
    epochs = [r["epoch"] for r in rows]
    plt.figure(figsize=(8, 5))
    for key, lab in [
        ("train_acc", "train"),
        ("val_acc", "val"),
        ("test_acc", "test"),
    ]:
        if rows and key in rows[0]:
            plt.plot(epochs, [r[key] for r in rows], label=lab)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], out_path: Path) -> None:
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
