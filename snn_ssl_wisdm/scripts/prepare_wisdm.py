"""Parse raw WISDM, build windows, normalize, save .pt and split JSON."""

from __future__ import annotations

import argparse
import glob
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from snn_ssl_wisdm.data.splits import save_split_json, subject_split
from snn_ssl_wisdm.train_utils import load_yaml, merge_dict, resolve_paths, set_seed, workspace_root


ACTIVITY_ORDER = "ABCDEFGHIJKLMOPQRS"
ACT_TO_IDX = {c: i for i, c in enumerate(ACTIVITY_ORDER)}


def _parse_line(line: str) -> Optional[Tuple[int, str, float, float, float, float]]:
    line = line.strip()
    if not line:
        return None
    line = line.rstrip(";").strip()
    parts = line.split(",")
    if len(parts) < 6:
        return None
    try:
        sid = int(parts[0])
        act = parts[1].strip()
        ts = float(parts[2])
        x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
    except ValueError:
        return None
    if act not in ACT_TO_IDX:
        return None
    return sid, act, ts, x, y, z


def _read_stream(path: Path) -> List[Tuple[int, str, float, np.ndarray]]:
    rows: List[Tuple[int, str, float, np.ndarray]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = _parse_line(line)
            if p is None:
                continue
            sid, act, ts, x, y, z = p
            rows.append((sid, act, ts, np.array([x, y, z], dtype=np.float32)))
    rows.sort(key=lambda r: r[2])
    return rows


def _interp_gyro_at_times(
    gyro_rows: List[Tuple[int, str, float, np.ndarray]], times: np.ndarray
) -> np.ndarray:
    """Linear interpolate gyro xyz at accel timestamps."""
    if not gyro_rows:
        raise ValueError("empty gyro")
    gt = np.array([r[2] for r in gyro_rows], dtype=np.float64)
    gxyz = np.stack([r[3] for r in gyro_rows], axis=0)
    out = np.zeros((len(times), 3), dtype=np.float32)
    for k in range(3):
        out[:, k] = np.interp(times.astype(np.float64), gt, gxyz[:, k].astype(np.float64)).astype(
            np.float32
        )
    return out


def _resample_window(w: np.ndarray, src_hz: float, dst_hz: float) -> np.ndarray:
    """w: [C, T] -> new length round(T * dst/src) via linear interp."""
    c, t = w.shape
    new_t = max(8, int(round(t * dst_hz / src_hz)))
    x = torch.from_numpy(w).unsqueeze(0)
    y = torch.nn.functional.interpolate(
        x, size=new_t, mode="linear", align_corners=False
    )
    return y.squeeze(0).numpy().astype(np.float32)


def _sliding_windows(
    stream: List[Tuple[int, str, float, np.ndarray]],
    gyro_stream: Optional[List[Tuple[int, str, float, np.ndarray]]],
    window_len: int,
    stride_len: int,
    subject_id: int,
    use_gyro: bool,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not stream:
        return out
    # split by contiguous activity
    segs: List[List[Tuple[int, str, float, np.ndarray]]] = []
    cur: List[Tuple[int, str, float, np.ndarray]] = []
    last_act = None
    for row in stream:
        act = row[1]
        if last_act is not None and act != last_act:
            segs.append(cur)
            cur = []
        cur.append(row)
        last_act = act
    if cur:
        segs.append(cur)

    for seg in segs:
        if len(seg) < window_len:
            continue
        act = seg[0][1]
        label = ACT_TO_IDX[act]
        ts_arr = np.array([r[2] for r in seg], dtype=np.float64)
        xyz = np.stack([r[3] for r in seg], axis=0)
        if use_gyro and gyro_stream is not None:
            gxyz = _interp_gyro_at_times(gyro_stream, ts_arr)
        for start in range(0, len(seg) - window_len + 1, stride_len):
            sl = slice(start, start + window_len)
            acc = xyz[sl].T.copy()
            if use_gyro and gyro_stream is not None:
                gyr = gxyz[sl].T.copy()
                w = np.concatenate([acc, gyr], axis=0)
            else:
                w = acc
            out.append(
                {
                    "window": w.astype(np.float32),
                    "label": label,
                    "subject": subject_id,
                }
            )
    return out


def _discover_files(root: Path, device_type: str, sensor: str) -> List[Path]:
    pat = str(root / "raw" / device_type / sensor / f"data_*_{sensor}_{device_type}.txt")
    files = sorted(glob.glob(pat))
    return [Path(p) for p in files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="snn_ssl_wisdm/configs/default.yaml")
    ap.add_argument("--limit_subjects", type=int, default=None)
    ap.add_argument("--max_windows", type=int, default=None)
    args, _ = ap.parse_known_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (workspace_root() / cfg_path).resolve()
    cfg = resolve_paths(load_yaml(cfg_path), cfg_path)

    cli = {}
    if args.limit_subjects is not None:
        cli["limit_subjects"] = args.limit_subjects
    if args.max_windows is not None:
        cli["max_windows"] = args.max_windows
    cfg = merge_dict(cfg, cli)

    seed = int(cfg["seed"])
    set_seed(seed)
    root = Path(cfg["dataset_root"])
    device_type = cfg["device_type"]
    sensor = cfg["sensor"]
    sample_rate = float(cfg["sample_rate"])
    resample_rate = cfg.get("resample_rate")
    window_sec = float(cfg["window_seconds"])
    stride_sec = float(cfg["stride_seconds"])
    window_len = int(round(window_sec * sample_rate))
    stride_len = int(round(stride_sec * sample_rate))
    if window_len < 32:
        raise ValueError("window too short")
    use_gyro = int(cfg.get("model", {}).get("in_channels", cfg.get("in_channels", 3))) == 6

    accel_files = _discover_files(root, device_type, "accel")
    if not accel_files:
        raise FileNotFoundError(f"No accel files under {root}/raw/{device_type}/accel/")
    random.Random(seed).shuffle(accel_files)
    lim = cfg.get("limit_subjects")
    if lim is not None:
        accel_files = accel_files[: int(lim)]

    all_items: List[Dict[str, Any]] = []
    for apath in accel_files:
        m = re.search(r"data_(\d+)_accel", apath.name)
        if not m:
            continue
        sid = int(m.group(1))
        stream = _read_stream(apath)
        gyro_stream = None
        if use_gyro:
            gpath = root / "raw" / device_type / "gyro" / f"data_{sid}_gyro_{device_type}.txt"
            if gpath.is_file():
                gyro_stream = _read_stream(gpath)
            else:
                print(f"[warn] missing gyro for subject {sid}, skipping gyro merge for this subject")
        wins = _sliding_windows(stream, gyro_stream, window_len, stride_len, sid, use_gyro and gyro_stream)
        all_items.extend(wins)

    maxw = cfg.get("max_windows")
    if maxw is not None and len(all_items) > int(maxw):
        rng = random.Random(seed)
        all_items = rng.sample(all_items, int(maxw))

    if not all_items:
        raise RuntimeError("No windows collected — check paths and parsing.")

    subjects = np.array([it["subject"] for it in all_items], dtype=np.int64)
    labels = np.array([it["label"] for it in all_items], dtype=np.int64)
    windows = np.stack([it["window"] for it in all_items], axis=0)

    if resample_rate is not None:
        dst = float(resample_rate)
        new_w = []
        for i in range(windows.shape[0]):
            new_w.append(_resample_window(windows[i], sample_rate, dst))
        min_t = min(w.shape[1] for w in new_w)
        windows = np.stack([w[:, :min_t] for w in new_w], axis=0)
        sample_rate_eff = dst
        window_len_eff = windows.shape[2]
    else:
        sample_rate_eff = sample_rate
        window_len_eff = window_len

    split = subject_split(subjects.tolist(), seed=seed)
    train_mask = np.isin(subjects, split["train"])
    if train_mask.sum() == 0:
        raise RuntimeError("Train split empty")

    w_train = windows[train_mask]
    mean = w_train.mean(axis=(0, 2), keepdims=True)
    std = w_train.std(axis=(0, 2), keepdims=True) + 1e-6
    windows_n = (windows - mean) / std

    mags = np.linalg.norm(windows_n, axis=1)
    weights = (mags.std(axis=1) + 1e-6).astype(np.float32)

    bundle = {
        "windows": torch.from_numpy(windows_n).float(),
        "labels": torch.from_numpy(labels).long(),
        "subjects": torch.from_numpy(subjects).long(),
        "sample_weights": torch.from_numpy(weights).float(),
        "activity_to_idx": dict(ACT_TO_IDX),
        "idx_to_activity": {v: k for k, v in ACT_TO_IDX.items()},
        "meta": {
            "mean": mean.squeeze().tolist(),
            "std": std.squeeze().tolist(),
            "window_seconds": window_sec,
            "stride_seconds": stride_sec,
            "sample_rate": sample_rate,
            "resample_rate": resample_rate,
            "effective_sample_rate": sample_rate_eff,
            "window_length": int(window_len_eff),
            "channels": int(windows_n.shape[1]),
            "device_type": device_type,
            "sensor_mode": "accel+gyro" if use_gyro else "accel",
            "num_windows": int(windows_n.shape[0]),
            "seed": seed,
        },
    }

    out_pt = Path(cfg["processed_path"])
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(bundle, out_pt, _use_new_zipfile_serialization=True)
    except TypeError:
        torch.save(bundle, out_pt)

    meta_split = {
        "seed": seed,
        "train_ratio": 0.7,
        "val_ratio": 0.1,
        "test_ratio": 0.2,
        "processed_path": str(out_pt),
    }
    save_split_json(Path(cfg["splits_path"]), split, meta_split)
    print(f"Saved {out_pt} ({bundle['windows'].shape[0]} windows, C={bundle['windows'].shape[1]}, T={bundle['windows'].shape[2]})")
    print(f"Splits -> {cfg['splits_path']}")


if __name__ == "__main__":
    main()
