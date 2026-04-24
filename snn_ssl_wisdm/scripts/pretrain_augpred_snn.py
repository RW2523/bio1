"""AugPred-style multi-task SSL on WISDM with spiking ResNet."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from snn_ssl_wisdm.amp_utils import amp_scaler_and_autocast
from snn_ssl_wisdm.data.splits import load_split_json, mask_for_subjects
from snn_ssl_wisdm.data.wisdm import WISDMDataset
from snn_ssl_wisdm.models.heads import AugPredHeads
from snn_ssl_wisdm.models.spiking_resnet1d import SpikingResNet1DBackbone
from snn_ssl_wisdm.torch_io import torch_load
from snn_ssl_wisdm.train_utils import (
    load_yaml,
    merge_dict,
    pick_device,
    print_gpu_info,
    resolve_paths,
    set_seed,
    workspace_root,
)
from snn_ssl_wisdm import viz


def _batch_arrow(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B, _, _ = x.shape
    rev = torch.rand(B, device=x.device) < 0.5
    out = x.clone()
    out[rev] = x[rev].flip(-1)
    return out, rev.long()


def _batch_perm(x: torch.Tensor, n_chunks: int, rng: random.Random) -> Tuple[torch.Tensor, torch.Tensor]:
    B, C, T = x.shape
    assert T % n_chunks == 0, f"T={T} not divisible by n_chunks={n_chunks}"
    chunk = T // n_chunks
    chunks = x.view(B, C, n_chunks, chunk)
    y = torch.zeros(B, dtype=torch.long, device=x.device)
    out = torch.empty_like(x)
    for b in range(B):
        if rng.random() < 0.5:
            perm = list(range(n_chunks))
            rng.shuffle(perm)
            out[b] = chunks[b, :, perm, :].reshape(C, T)
            y[b] = 1
        else:
            out[b] = x[b]
            y[b] = 0
    return out, y


def _batch_tw(x: torch.Tensor, strength: float, rng: random.Random) -> Tuple[torch.Tensor, torch.Tensor]:
    B, C, T = x.shape
    y = torch.zeros(B, dtype=torch.long, device=x.device)
    out = torch.empty_like(x)
    for b in range(B):
        if rng.random() < 0.5:
            scale = 1.0 + (rng.random() * 2 - 1) * strength
            new_len = max(8, int(round(T * scale)))
            w = x[b : b + 1]
            warped = F.interpolate(w, size=new_len, mode="linear", align_corners=False)
            back = F.interpolate(warped, size=T, mode="linear", align_corners=False)
            out[b] = back.squeeze(0)
            y[b] = 1
        else:
            out[b] = x[b]
            y[b] = 0
    return out, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit_subjects", type=int, default=None)
    ap.add_argument("--max_windows", type=int, default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (workspace_root() / cfg_path).resolve()
    cfg = resolve_paths(load_yaml(cfg_path), cfg_path)
    cli = {}
    for k, v in [
        ("limit_subjects", args.limit_subjects),
        ("max_windows", args.max_windows),
        ("epochs_pretrain", args.epochs),
    ]:
        if v is not None:
            cli[k] = v
    cfg = merge_dict(cfg, cli)

    seed = int(cfg["seed"])
    set_seed(seed)
    rng_py = random.Random(seed)
    device = pick_device(cfg.get("device"))
    print_gpu_info(device)

    processed = Path(cfg["processed_path"])
    split, _ = load_split_json(Path(cfg["splits_path"]))
    bundle = torch_load(processed, map_location="cpu")
    subjects = bundle["subjects"]
    weights = bundle["sample_weights"]
    meta = bundle.get("meta", {})
    n_chunks = int(cfg.get("augpred", {}).get("num_perm_chunks", 4))
    tw_strength = float(cfg.get("augpred", {}).get("time_warp_strength", 0.15))
    T = int(bundle["windows"].shape[2])
    assert T % n_chunks == 0, f"window length {T} must be divisible by augpred.num_perm_chunks={n_chunks}"

    subj_np = subjects.numpy()
    train_m = mask_for_subjects(subj_np, "train", split)
    train_idx = np.where(train_m)[0]
    val_m = mask_for_subjects(subj_np, "val", split)
    val_idx = np.where(val_m)[0]

    ds_tr = WISDMDataset(processed, train_idx)
    ds_va = WISDMDataset(processed, val_idx)
    w_train = weights[train_idx].float()
    sampler = WeightedRandomSampler(
        w_train, num_samples=len(w_train), replacement=True
    )

    nw = int(cfg["num_workers"])
    bs = int(cfg["batch_size"])
    train_loader = DataLoader(
        ds_tr,
        batch_size=bs,
        sampler=sampler,
        num_workers=nw,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        ds_va, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=device.type == "cuda"
    )

    mcfg = cfg["model"]
    sch = cfg["spiking"]
    backbone = SpikingResNet1DBackbone(
        in_channels=int(mcfg["in_channels"]),
        base_channels=int(mcfg["base_channels"]),
        layers=list(mcfg["layers"]),
        feature_dim=int(mcfg["feature_dim"]),
        timesteps=int(sch["timesteps"]),
        beta=float(sch["beta"]),
        v_threshold=float(sch["threshold"]),
        detach_reset=bool(sch["detach_reset"]),
    ).to(device)
    heads = AugPredHeads(int(mcfg["feature_dim"])).to(device)

    opt = torch.optim.AdamW(
        list(backbone.parameters()) + list(heads.parameters()),
        lr=float(cfg["lr_pretrain"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    amp = bool(cfg.get("amp", True))
    scaler, amp_ctx = amp_scaler_and_autocast(device, amp)
    bce = nn.BCEWithLogitsLoss()
    epochs = int(cfg.get("epochs_pretrain") or 100)
    patience = int(cfg["patience"])
    grad_clip = float(cfg["grad_clip"])
    out_dir = Path(cfg["output_root"]) / "augpred_pretrain_snn"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    rows: List[Dict] = []
    best_val = float("inf")
    stall = 0
    best_sd = None

    for epoch in range(1, epochs + 1):
        backbone.train()
        heads.train()
        tr_tot = 0.0
        tr_n = 0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with amp_ctx():
                xa, ya = _batch_arrow(x)
                zp, yp = _batch_perm(x, n_chunks, rng_py)
                xt, yt = _batch_tw(x, tw_strength, rng_py)
                za = backbone(xa)
                zb = backbone(zp)
                zc = backbone(xt)
                la = bce(heads.head_aot(za).squeeze(-1), ya.float())
                lb = bce(heads.head_perm(zb).squeeze(-1), yp.float())
                lc = bce(heads.head_tw(zc).squeeze(-1), yt.float())
                loss = (la + lb + lc) / 3.0
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(heads.parameters()), grad_clip
                )
            scaler.step(opt)
            scaler.update()
            bs_ = x.shape[0]
            tr_tot += float(loss.item()) * bs_
            tr_n += bs_
        tr_loss = tr_tot / max(1, tr_n)

        backbone.eval()
        heads.eval()
        va_tot = 0.0
        va_n = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device, non_blocking=True)
                with amp_ctx():
                    xa, ya = _batch_arrow(x)
                    xp, yp = _batch_perm(x, n_chunks, rng_py)
                    xt, yt = _batch_tw(x, tw_strength, rng_py)
                    za = backbone(xa)
                    zb = backbone(xp)
                    zc = backbone(xt)
                    la = bce(heads.head_aot(za).squeeze(-1), ya.float())
                    lb = bce(heads.head_perm(zb).squeeze(-1), yp.float())
                    lc = bce(heads.head_tw(zc).squeeze(-1), yt.float())
                    loss = (la + lb + lc) / 3.0
                va_tot += float(loss.item()) * x.shape[0]
                va_n += x.shape[0]
        va_loss = va_tot / max(1, va_n)
        rows.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})
        print(f"epoch {epoch}/{epochs} pretrain train_loss={tr_loss:.4f} val_loss={va_loss:.4f}")
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        if va_loss < best_val - 1e-6:
            best_val = va_loss
            stall = 0
            best_sd = {"backbone": backbone.state_dict(), "heads": heads.state_dict(), "epoch": epoch}
            torch.save({"backbone": best_sd["backbone"]}, out_dir / "best_backbone.pt")
        else:
            stall += 1
            if stall >= patience:
                print(f"Early stopping pretrain at epoch {epoch}")
                break

    if best_sd is not None:
        backbone.load_state_dict(best_sd["backbone"])
        heads.load_state_dict(best_sd["heads"])

    import json

    from snn_ssl_wisdm.train_utils import save_json

    save_json(out_dir / "metrics.json", {"best_val_loss": best_val, "meta": meta})
    if rows:
        viz.plot_loss_curves(rows, out_dir / "loss_curve.png")
    print(f"Saved backbone to {out_dir / 'best_backbone.pt'}")


if __name__ == "__main__":
    main()
