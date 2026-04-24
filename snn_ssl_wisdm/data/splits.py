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
