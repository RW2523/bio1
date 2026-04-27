"""Subject-wise train/val/test splits."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def subject_split(
    subjects: Sequence[int],
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> Dict[str, List[int]]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    subs = sorted(set(int(s) for s in subjects))
    rng = random.Random(seed)
    rng.shuffle(subs)
    n = len(subs)
    if n == 1:
        return {"train": list(subs), "val": list(subs), "test": list(subs)}
    if n == 2:
        return {"train": [subs[0]], "val": [subs[1]], "test": [subs[1]]}
    n_train = max(1, int(np.floor(train_ratio * n)))
    n_val = max(1, int(np.floor(val_ratio * n)))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_val = max(1, n_val - 1)
        n_test = n - n_train - n_val
    if n_test < 1:
        n_train = max(1, n_train - 1)
        n_test = n - n_train - n_val
    train = subs[:n_train]
    val = subs[n_train : n_train + n_val]
    test = subs[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def save_split_json(path: Path, split: Dict[str, List[int]], meta: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {"split": split, "meta": meta}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def load_split_json(path: Path) -> Tuple[Dict[str, List[int]], Dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw["split"], raw.get("meta", {})


def mask_for_subjects(subject_tensor: np.ndarray, split_name: str, split: Dict[str, List[int]]) -> np.ndarray:
    allowed = set(split[split_name])
    return np.isin(subject_tensor, list(allowed))


def window_index_split(
    n_windows: int,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random partition of window indices (ignores subject leakage). Secondary comparison."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = np.random.RandomState(seed)
    ix = np.arange(n_windows, dtype=np.int64)
    rng.shuffle(ix)
    n_train = max(1, int(np.floor(train_ratio * n_windows)))
    n_val = max(1, int(np.floor(val_ratio * n_windows)))
    n_test = n_windows - n_train - n_val
    if n_test < 1:
        n_val = max(1, n_val - 1)
        n_test = n_windows - n_train - n_val
    if n_test < 1:
        n_train = max(1, n_train - 1)
        n_test = n_windows - n_train - n_val
    tr = ix[:n_train]
    va = ix[n_train : n_train + n_val]
    te = ix[n_train + n_val :]
    return tr, va, te


def save_window_split_json(
    path: Path,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    meta: Dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "split_mode": "window",
        "train_idx": train_idx.astype(np.int64).tolist(),
        "val_idx": val_idx.astype(np.int64).tolist(),
        "test_idx": test_idx.astype(np.int64).tolist(),
        "meta": meta,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def load_split_payload(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def train_val_test_indices(payload: Dict, subjects: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train/val/test window indices: subject-disjoint (default) or random window split."""
    if "train_idx" in payload:
        return (
            np.asarray(payload["train_idx"], dtype=np.int64),
            np.asarray(payload["val_idx"], dtype=np.int64),
            np.asarray(payload["test_idx"], dtype=np.int64),
        )
    split = payload["split"]
    tr = np.where(np.isin(subjects, split["train"]))[0]
    va = np.where(np.isin(subjects, split["val"]))[0]
    te = np.where(np.isin(subjects, split["test"]))[0]
    return tr, va, te
