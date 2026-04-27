"""Classification metrics (JSON-serializable)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    labels = list(range(num_classes))
    acc = float((y_true == y_pred).mean())
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    target_names = class_names if class_names else [str(i) for i in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    per_class: Dict[str, Any] = {}
    for i in labels:
        key = str(i)
        if key in report:
            per_class[key] = {
                "precision": float(report[key]["precision"]),
                "recall": float(report[key]["recall"]),
                "f1": float(report[key]["f1-score"]),
                "support": int(report[key]["support"]),
            }
    per_class_recall = {
        str(i): float(per_class[str(i)]["recall"]) if str(i) in per_class else 0.0 for i in labels
    }

    out = {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "cohen_kappa": kappa,
        "confusion_matrix": cm,
        "per_class": per_class,
        "per_class_recall": per_class_recall,
        "classification_report": report,
    }
    return out


@torch.no_grad()
def gather_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    ys: List[int] = []
    ps: List[int] = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=-1)
        ys.extend(batch["y"].cpu().numpy().tolist())
        ps.extend(pred.cpu().numpy().tolist())
    return np.array(ys), np.array(ps)
