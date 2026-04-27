"""WISDM processed tensor dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from snn_ssl_wisdm.torch_io import torch_load


def load_processed_bundle(path: Path) -> Dict[str, Any]:
    bundle = torch_load(path, map_location="cpu")
    assert "labels" in bundle and "subjects" in bundle
    assert "windows" in bundle or "windows_raw" in bundle
    w = bundle.get("windows_raw") or bundle["windows"]
    assert w.dim() == 3, f"windows must be [N,C,T], got {tuple(w.shape)}"
    return bundle


class WISDMDataset(Dataset):
    """Index windows from a prepared .pt bundle with optional subject filter.

    If the bundle contains ``windows_raw`` and ``norm_subject`` / ``norm_window``,
    apply z-score using training-set statistics only (``norm_mode`` = ``subject`` or
    ``window``). Legacy bundles store pre-normalised ``windows`` only.
    """

    def __init__(
        self,
        bundle_path: Path,
        indices: np.ndarray,
        augment_ssl: bool = False,
        augpred_cfg: Optional[Dict[str, Any]] = None,
        norm_mode: Optional[str] = None,
    ):
        self.bundle = load_processed_bundle(Path(bundle_path))
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment_ssl = augment_ssl
        self.augpred_cfg = augpred_cfg or {}
        self.labels: torch.Tensor = self.bundle["labels"]
        self.subjects: torch.Tensor = self.bundle["subjects"]
        self.weights: Optional[torch.Tensor] = None
        if "sample_weights" in self.bundle:
            self.weights = self.bundle["sample_weights"]

        self._use_raw = "windows_raw" in self.bundle
        self.norm_mode = norm_mode or "subject"
        if self._use_raw:
            self._windows_raw: torch.Tensor = self.bundle["windows_raw"]
            nm = self.bundle.get("norm_subject") if self.norm_mode == "subject" else self.bundle.get("norm_window")
            if nm is None:
                nm = self.bundle["norm_subject"]
            self._mean: torch.Tensor = nm["mean"].float()
            self._std: torch.Tensor = nm["std"].float()
            self.windows = self._windows_raw
        else:
            self._mean = None
            self._std = None
            self.windows: torch.Tensor = self.bundle["windows"]

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int) -> Dict[str, Any]:
        idx = int(self.indices[i])
        if self._mean is not None:
            x = ((self._windows_raw[idx].float() - self._mean) / self._std).clone()
        else:
            x = self.windows[idx].clone()
        y = int(self.labels[idx].item())
        subj = int(self.subjects[idx].item())
        out: Dict[str, Any] = {"x": x, "y": y, "subject": subj, "idx": idx}
        if self.weights is not None:
            out["importance"] = float(self.weights[idx].item())
        return out
