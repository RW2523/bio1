"""Config loading, seeds, device helpers, logging."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def workspace_root() -> Path:
    """Directory containing `snn_ssl_wisdm` package (parent of package dir)."""
    return Path(__file__).resolve().parent.parent


def ensure_spikegpt_src_on_path() -> Path:
    """Prepend SpikeGPT/src so vendored SpikingJelly imports as `spikingjelly` or `src.spikingjelly`."""
    root = workspace_root()
    spike_src = root / "SpikeGPT" / "src"
    if spike_src.is_dir():
        s = str(spike_src)
        if s not in sys.path:
            sys.path.insert(0, s)
    return spike_src


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a mapping at top level.")
    return data


def merge_dict(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in overrides.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def resolve_paths(cfg: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    """Resolve relative paths against workspace root."""
    root = workspace_root()
    cfg = json.loads(json.dumps(cfg))  # deep copy via json (yaml-safe scalars)

    def fix(p: Optional[str]) -> Optional[str]:
        if p is None:
            return None
        pp = Path(p)
        if not pp.is_absolute():
            return str((root / pp).resolve())
        return str(pp)

    if "dataset_root" in cfg:
        cfg["dataset_root"] = fix(cfg["dataset_root"])
    if "processed_path" in cfg:
        cfg["processed_path"] = fix(cfg["processed_path"])
    if "splits_path" in cfg:
        cfg["splits_path"] = fix(cfg["splits_path"])
    if "splits_window_path" in cfg:
        cfg["splits_window_path"] = fix(cfg["splits_window_path"])
    if "output_root" in cfg:
        cfg["output_root"] = fix(cfg["output_root"])
    cfg["_config_path"] = str(config_path.resolve())
    cfg["_workspace_root"] = str(root)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def pick_device(cfg_device: Optional[str]) -> torch.device:
    if cfg_device and cfg_device.lower() != "auto":
        return torch.device(cfg_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_gpu_info(device: torch.device) -> None:
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        mem = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        print(f"CUDA device: {name} ({mem:.1f} GiB total)")
    else:
        print("Running on CPU (no CUDA).")


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
