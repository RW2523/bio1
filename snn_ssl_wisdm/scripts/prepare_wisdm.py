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

from snn_ssl_wisdm.data.splits import (
    save_split_json,
    save_window_split_json,
    subject_split,
    window_index_split,
)
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


def _discover_watch_accel_fused(root: Path) -> List[Tuple[int, Path]]:
    pat = str(root / "raw" / "watch" / "accel" / "data_*_accel_watch.txt")
    files = sorted(glob.glob(pat))
    out: List[Tuple[int, Path]] = []
    for p in files:
        m = re.search(r"data_(\d+)_accel_watch", Path(p).name)
        if m:
            out.append((int(m.group(1)), Path(p)))
    return out


def _four_stream_paths(root: Path, sid: int) -> Dict[str, Path]:
    return {
        "watch_accel": root / "raw" / "watch" / "accel" / f"data_{sid}_accel_watch.txt",
        "watch_gyro": root / "raw" / "watch" / "gyro" / f"data_{sid}_gyro_watch.txt",
        "phone_accel": root / "raw" / "phone" / "accel" / f"data_{sid}_accel_phone.txt",
        "phone_gyro": root / "raw" / "phone" / "gyro" / f"data_{sid}_gyro_phone.txt",
    }


def _interp_xyz_on_times(
    stream: List[Tuple[int, str, float, np.ndarray]],
    act: str,
    times_master: np.ndarray,
) -> Optional[np.ndarray]:
    """Interpolate sensor xyz onto times_master; require >=2 samples overlapping segment."""
    if len(times_master) == 0:
        return None
    t0 = float(times_master[0]) - 0.5
    t1 = float(times_master[-1]) + 0.5
    rows = [r for r in stream if r[1] == act and t0 <= r[2] <= t1]
    if len(rows) < 2:
        return None
    ts = np.array([r[2] for r in rows], dtype=np.float64)
    order = np.argsort(ts)
    ts = ts[order]
    xyz = np.stack([rows[i][3] for i in order], axis=0).astype(np.float64)
    out = np.zeros((len(times_master), 3), dtype=np.float32)
    tm = times_master.astype(np.float64)
    for k in range(3):
        out[:, k] = np.interp(tm, ts, xyz[:, k]).astype(np.float32)
    return out


def _fused_segments_to_windows(
    watch_accel: List[Tuple[int, str, float, np.ndarray]],
    watch_gyro: List[Tuple[int, str, float, np.ndarray]],
    phone_accel: List[Tuple[int, str, float, np.ndarray]],
    phone_gyro: List[Tuple[int, str, float, np.ndarray]],
    subject_id: int,
    window_len: int,
    stride_len: int,
) -> List[Dict[str, Any]]:
    """Single-activity segments on watch accel timeline; [12,T] = phone_a, phone_g, watch_a, watch_g."""
    out: List[Dict[str, Any]] = []
    if not watch_accel:
        return out
    segs: List[List[Tuple[int, str, float, np.ndarray]]] = []
    cur: List[Tuple[int, str, float, np.ndarray]] = []
    last_act = None
    for row in watch_accel:
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
        if act not in ACT_TO_IDX:
            continue
        label = ACT_TO_IDX[act]
        ts_arr = np.array([r[2] for r in seg], dtype=np.float64)
        wa = np.stack([r[3] for r in seg], axis=0).astype(np.float32)

        pa = _interp_xyz_on_times(phone_accel, act, ts_arr)
        pg = _interp_xyz_on_times(phone_gyro, act, ts_arr)
        wg = _interp_xyz_on_times(watch_gyro, act, ts_arr)
        if pa is None or pg is None or wg is None:
            continue

        fused = np.concatenate([pa, pg, wa, wg], axis=1)
        assert fused.shape[1] == 12
        L = fused.shape[0]
        for start in range(0, L - window_len + 1, stride_len):
            sl = slice(start, start + window_len)
            w = fused[sl].T.copy()
            out.append({"window": w.astype(np.float32), "label": label, "subject": subject_id})
    return out


def _prepare_fused_12ch(cfg: Dict[str, Any], root: Path, seed: int) -> Dict[str, Any]:
    """Phone+watch accel+gyro aligned on watch timestamps; shape [N,12,128]."""
    sample_rate = float(cfg["sample_rate"])
    window_len = int(round(float(cfg["window_seconds"]) * sample_rate))
    stride_len = int(round(float(cfg["stride_seconds"]) * sample_rate))
    if window_len != 128:
        print(f"[prepare] fused: window_len={window_len} (expected 128 for 6.4s@20Hz)")
    if stride_len != 64:
        print(f"[prepare] fused: stride_len={stride_len} (expected 64 for 50% overlap@20Hz)")

    pairs = _discover_watch_accel_fused(root)
    if not pairs:
        raise FileNotFoundError(f"No watch accel files under {root}/raw/watch/accel/")
    rng = random.Random(seed)
    rng.shuffle(pairs)
    lim = cfg.get("limit_subjects")
    if lim is not None:
        pairs = pairs[: int(lim)]

    all_items: List[Dict[str, Any]] = []
    skipped_subj = 0
    for sid, wpath in pairs:
        paths = _four_stream_paths(root, sid)
        if not all(p.is_file() for p in paths.values()):
            skipped_subj += 1
            continue
        wa = _read_stream(paths["watch_accel"])
        wg = _read_stream(paths["watch_gyro"])
        pa = _read_stream(paths["phone_accel"])
        pg = _read_stream(paths["phone_gyro"])
        wins = _fused_segments_to_windows(wa, wg, pa, pg, sid, window_len, stride_len)
        all_items.extend(wins)

    if skipped_subj:
        print(f"[prepare] fused: skipped {skipped_subj} subjects (missing phone/watch streams)")
    maxw = cfg.get("max_windows")
    if maxw is not None and len(all_items) > int(maxw):
        rng2 = random.Random(seed)
        all_items = rng2.sample(all_items, int(maxw))

    if not all_items:
        raise RuntimeError(
            "No fused windows — need raw/phone|watch/{accel,gyro}/data_{sid}_*_{phone|watch}.txt"
        )

    subjects = np.array([it["subject"] for it in all_items], dtype=np.int64)
    labels = np.array([it["label"] for it in all_items], dtype=np.int64)
    windows_raw = np.stack([it["window"] for it in all_items], axis=0)
    w_t = torch.from_numpy(windows_raw).float()

    split_sub = subject_split(subjects.tolist(), seed=seed)
    tr_sub = np.isin(subjects, split_sub["train"])
    if tr_sub.sum() == 0:
        raise RuntimeError("Subject train split empty")

    mean_s = w_t[tr_sub].mean(dim=(0, 2), keepdim=True)
    std_s = w_t[tr_sub].std(dim=(0, 2), keepdim=True) + 1e-6

    tr_wi, va_wi, te_wi = window_index_split(w_t.shape[0], seed=seed)
    mean_w = w_t[tr_wi].mean(dim=(0, 2), keepdim=True)
    std_w = w_t[tr_wi].std(dim=(0, 2), keepdim=True) + 1e-6

    mags = torch.linalg.norm((w_t - mean_s) / std_s, dim=1)
    weights = (mags.std(dim=1) + 1e-6).float()

    window_sec = float(cfg["window_seconds"])
    stride_sec = float(cfg["stride_seconds"])
    bundle: Dict[str, Any] = {
        "windows_raw": w_t,
        "labels": torch.from_numpy(labels).long(),
        "subjects": torch.from_numpy(subjects).long(),
        "sample_weights": weights,
        "norm_subject": {"mean": mean_s, "std": std_s},
        "norm_window": {"mean": mean_w, "std": std_w},
        "activity_to_idx": dict(ACT_TO_IDX),
        "idx_to_activity": {v: k for k, v in ACT_TO_IDX.items()},
        "meta": {
            "fused_12ch": True,
            "channel_order": ["phone_acc_xyz", "phone_gyro_xyz", "watch_acc_xyz", "watch_gyro_xyz"],
            "window_seconds": window_sec,
            "stride_seconds": stride_sec,
            "sample_rate": sample_rate,
            "window_length": window_len,
            "stride_samples": stride_len,
            "channels": 12,
            "num_windows": int(w_t.shape[0]),
            "seed": seed,
            "single_activity_segments": True,
            "alignment": "watch_accel_timeline_interp_phone_watch_gyro",
        },
    }
    return {"bundle": bundle, "split_sub": split_sub, "tr_wi": tr_wi, "va_wi": va_wi, "te_wi": te_wi}


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
    data_cfg = cfg.get("data") or {}
    if bool(data_cfg.get("fused_12ch", False)):
        res = _prepare_fused_12ch(cfg, root, seed)
        bundle = res["bundle"]
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
            "split_type": "subject_disjoint",
        }
        save_split_json(Path(cfg["splits_path"]), res["split_sub"], meta_split)
        wpath = Path(cfg.get("splits_window_path", "snn_ssl_wisdm/processed/splits_window_seed42.json"))
        save_window_split_json(
            wpath,
            res["tr_wi"],
            res["va_wi"],
            res["te_wi"],
            {"seed": seed, "processed_path": str(out_pt), "split_type": "random_window"},
        )
        shp = bundle["windows_raw"].shape
        print(f"Saved fused 12ch (raw) → {out_pt} ({shp[0]} windows, C={shp[1]}, T={shp[2]})")
        print(f"Subject splits → {cfg['splits_path']}")
        print(f"Window splits  → {wpath}")
        return

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
