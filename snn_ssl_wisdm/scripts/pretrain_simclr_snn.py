"""SimCLR-style contrastive pretraining (Chen et al., 2020) for spiking ResNet on WISDM.

Two stochastic views per window, NT-Xent on L2-normalised projector outputs.

Training defaults:
  - **Uniform shuffled batches** (not movement-weighted sampling): contrastive negatives
    must be diverse across the dataset; weighted sampling collapses negative diversity.
  - ``drop_last=True`` so every training step has a full batch (stable NT-Xent).
  - **Checkpoint by validation contrastive accuracy** (primary), val loss as tie-breaker.
  - **Separate AdamW weight decay** for backbone vs projector (SimCLR-style: low WD on projector).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from snn_ssl_wisdm.amp_utils import amp_scaler_and_autocast
from snn_ssl_wisdm.data.splits import load_split_payload, train_val_test_indices
from snn_ssl_wisdm.data.wisdm import WISDMDataset
from snn_ssl_wisdm.models.heads import SimCLRProjector
from snn_ssl_wisdm.models.spiking_resnet1d import SpikingResNet1DBackbone
from snn_ssl_wisdm.torch_io import torch_load
from snn_ssl_wisdm.train_utils import (
    load_yaml,
    merge_dict,
    pick_device,
    print_gpu_info,
    resolve_paths,
    save_json,
    set_seed,
    workspace_root,
)
from snn_ssl_wisdm import viz


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric NT-Xent (InfoNCE) for two views of batch size B."""
    B = z1.size(0)
    z1 = F.normalize(z1, dim=1, eps=1e-8)
    z2 = F.normalize(z2, dim=1, eps=1e-8)
    z = torch.cat([z1, z2], dim=0)
    logits = torch.mm(z, z.t()) / temperature
    logits = logits.masked_fill(torch.eye(2 * B, device=z.device, dtype=torch.bool), -1e4)
    labels = torch.cat([torch.arange(B, device=z.device) + B, torch.arange(B, device=z.device)], dim=0)
    return F.cross_entropy(logits, labels)


def nt_xent_accuracy(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> float:
    """Top-1 accuracy of the contrastive classification (diagnostic)."""
    B = z1.size(0)
    z1 = F.normalize(z1, dim=1, eps=1e-8)
    z2 = F.normalize(z2, dim=1, eps=1e-8)
    z = torch.cat([z1, z2], dim=0)
    logits = torch.mm(z, z.t()) / temperature
    logits = logits.masked_fill(torch.eye(2 * B, device=z.device, dtype=torch.bool), -1e4)
    labels = torch.cat([torch.arange(B, device=z.device) + B, torch.arange(B, device=z.device)], dim=0)
    return float((logits.argmax(dim=-1) == labels).float().mean().item())


def simclr_augment(x: torch.Tensor, sc: Dict, training: bool) -> torch.Tensor:
    """Per-sample stochastic augmentations for 1D IMU windows [B, C, T]."""
    out = x.clone()
    B, C, T = out.shape
    device, dtype = out.device, out.dtype

    if not training:
        vn = float(sc.get("val_noise_std", 0.012))
        return out + torch.randn_like(out) * vn

    sc_lo = float(sc.get("aug_scale_lo", 0.75))
    sc_hi = float(sc.get("aug_scale_hi", 1.3))
    s = torch.empty(B, 1, 1, device=device, dtype=dtype).uniform_(sc_lo, sc_hi)
    out = out * s

    noise = float(sc.get("aug_noise_std", 0.08))
    if noise > 0:
        out = out + torch.randn_like(out) * noise

    p_drop = float(sc.get("aug_channel_drop_prob", 0.1))
    if p_drop > 0:
        mask = (torch.rand(B, C, 1, device=device) > p_drop).to(dtype)
        out = out * mask

    p_rev = float(sc.get("aug_reverse_prob", 0.5))
    if p_rev > 0:
        rev = torch.rand(B, device=device) < p_rev
        if rev.any():
            out[rev] = out[rev].flip(-1)

    p_tw = float(sc.get("aug_time_warp_prob", 0.5))
    strength = float(sc.get("aug_time_warp_strength", 0.22))
    if p_tw > 0 and strength > 0:
        tw_m = torch.rand(B, device=device) < p_tw
        idxs = tw_m.nonzero(as_tuple=False).view(-1)
        for b in idxs.tolist():
            fac = 1.0 + (torch.rand(1, device=device, dtype=dtype).item() * 2.0 - 1.0) * strength
            nl = max(8, int(round(T * fac)))
            w = out[b : b + 1]
            w = F.interpolate(w, size=nl, mode="linear", align_corners=False)
            w = F.interpolate(w, size=T, mode="linear", align_corners=False)
            out[b] = w.squeeze(0)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit_subjects", type=int, default=None)
    ap.add_argument("--max_windows", type=int, default=None)
    ap.add_argument("--split", type=str, choices=["subject", "window"], default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (workspace_root() / cfg_path).resolve()
    cfg = resolve_paths(load_yaml(cfg_path), cfg_path)
    cli: Dict = {}
    for k, v in [
        ("limit_subjects", args.limit_subjects),
        ("max_windows", args.max_windows),
        ("epochs_pretrain", args.epochs),
    ]:
        if v is not None:
            cli[k] = v
    cfg = merge_dict(cfg, cli)

    set_seed(int(cfg["seed"]))
    device = pick_device(cfg.get("device"))
    print_gpu_info(device)

    processed = Path(cfg["processed_path"])
    data_cfg = cfg.get("data") or {}
    split_mode = args.split or data_cfg.get("evaluation_split", "subject")
    split_path = (
        Path(cfg["splits_path"])
        if split_mode == "subject"
        else Path(cfg.get("splits_window_path", "snn_ssl_wisdm/processed/splits_window_seed42.json"))
    )
    if not split_path.is_absolute():
        split_path = (workspace_root() / split_path).resolve()
    payload = load_split_payload(split_path)
    bundle = torch_load(processed, map_location="cpu")
    subjects = bundle["subjects"]
    weights = bundle["sample_weights"]
    meta = bundle.get("meta", {})

    sc = cfg.get("simclr", {}) or {}
    temperature = float(sc.get("temperature", 0.15))
    proj_h = int(sc.get("projector_hidden", 2048))
    proj_d = int(sc.get("projector_out", 128))
    use_weighted = bool(sc.get("use_weighted_sampler", False))
    wd_bb = float(sc.get("weight_decay_backbone", cfg.get("weight_decay", 1e-4)))
    wd_proj = float(sc.get("weight_decay_projector", 1e-6))
    eta_min_ratio = float(sc.get("cosine_eta_min_ratio", 0.05))

    subj_np = subjects.numpy()
    train_idx, val_idx, _te = train_val_test_indices(payload, subj_np)
    norm_mode = data_cfg.get("norm_mode")
    if norm_mode is None:
        norm_mode = "window" if split_mode == "window" else "subject"
    print(f"[simclr] split={split_mode} train={len(train_idx)} val={len(val_idx)} norm={norm_mode}")

    ds_tr = WISDMDataset(processed, train_idx, norm_mode=norm_mode)
    ds_va = WISDMDataset(processed, val_idx, norm_mode=norm_mode)

    nw = int(cfg["num_workers"])
    bs = int(cfg["batch_size"])

    if use_weighted:
        w_train = weights[train_idx].float()
        sampler = WeightedRandomSampler(w_train, num_samples=len(w_train), replacement=True)
        train_loader = DataLoader(
            ds_tr,
            batch_size=bs,
            sampler=sampler,
            num_workers=nw,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )
        print("[simclr] WARNING: use_weighted_sampler=true — negatives may be less diverse.")
    else:
        train_loader = DataLoader(
            ds_tr,
            batch_size=bs,
            shuffle=True,
            num_workers=nw,
            pin_memory=device.type == "cuda",
            drop_last=True,
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
        decay_input=bool(sch.get("decay_input", False)),
        soft_reset=bool(sch.get("soft_reset", True)),
    ).to(device)
    projector = SimCLRProjector(
        in_dim=int(mcfg["feature_dim"]),
        hidden_dim=proj_h,
        out_dim=proj_d,
    ).to(device)

    lr = float(cfg.get("lr_simclr", cfg.get("lr_pretrain", 1e-3)))
    opt_groups = [
        {"params": list(backbone.parameters()), "weight_decay": wd_bb},
        {"params": list(projector.parameters()), "weight_decay": wd_proj},
    ]
    if bool(sc.get("use_adamw", False)):
        opt = torch.optim.AdamW(opt_groups, lr=lr)
    else:
        opt = torch.optim.Adam(opt_groups, lr=lr)

    epochs = int(cfg.get("epochs_pretrain") or 300)
    patience = int(cfg.get("patience_pretrain") or cfg.get("patience") or 45)
    min_epochs = int(cfg.get("min_epochs_pretrain") or 50)
    grad_clip = float(cfg["grad_clip"])
    amp_on = bool(cfg.get("amp", True))
    scaler, amp_ctx = amp_scaler_and_autocast(device, amp_on)

    warmup_epochs = min(10, max(1, epochs // 20))

    def lr_lambda(ep: int) -> float:
        if ep < warmup_epochs:
            return (ep + 1) / max(1, warmup_epochs)
        progress = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
        cos = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item())
        return eta_min_ratio + (1.0 - eta_min_ratio) * cos

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    out_dir = Path(cfg["output_root"]) / "simclr_pretrain_snn"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    rows: List[Dict] = []
    # Lexicographic: maximise val_nt_acc, then minimise val_loss (via tuple compare).
    best_key: Tuple[float, float] = (-1.0, float("inf"))
    stall = 0
    best_sd = None

    for epoch in range(1, epochs + 1):
        backbone.train()
        projector.train()
        tr_loss = 0.0
        tr_acc = 0.0
        tr_n = 0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            if x.size(0) < 2:
                continue
            v1 = simclr_augment(x, sc, training=True)
            v2 = simclr_augment(x, sc, training=True)
            opt.zero_grad(set_to_none=True)
            with amp_ctx():
                z1 = projector(backbone(v1))
                z2 = projector(backbone(v2))
                loss = nt_xent_loss(z1, z2, temperature)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(projector.parameters()), grad_clip
                )
            scaler.step(opt)
            scaler.update()
            bsz = x.size(0)
            tr_loss += float(loss.item()) * bsz
            with torch.no_grad():
                tr_acc += nt_xent_accuracy(z1, z2, temperature) * bsz
            tr_n += bsz
        tr_loss /= max(1, tr_n)
        tr_acc /= max(1, tr_n)

        backbone.eval()
        projector.eval()
        va_loss = 0.0
        va_acc = 0.0
        va_n = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device, non_blocking=True)
                if x.size(0) < 2:
                    continue
                with amp_ctx():
                    v1 = simclr_augment(x, sc, training=False)
                    v2 = simclr_augment(x, sc, training=False)
                    z1 = projector(backbone(v1))
                    z2 = projector(backbone(v2))
                    loss = nt_xent_loss(z1, z2, temperature)
                    acc = nt_xent_accuracy(z1, z2, temperature)
                bsz = x.size(0)
                va_loss += float(loss.item()) * bsz
                va_acc += acc * bsz
                va_n += bsz
        va_loss /= max(1, va_n)
        va_acc /= max(1, va_n)

        cur_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else opt.param_groups[0]["lr"]
        scheduler.step()

        rows.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_nt_acc": tr_acc,
                "val_loss": va_loss,
                "val_nt_acc": va_acc,
                "lr": cur_lr,
            }
        )
        print(
            f"epoch {epoch:03d}/{epochs}  simclr  train_loss={tr_loss:.4f} train_nt_acc={tr_acc:.4f}  "
            f"val_loss={va_loss:.4f} val_nt_acc={va_acc:.4f}  lr={cur_lr:.2e}"
        )
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        key = (va_acc, -va_loss)
        old_acc, old_loss = best_key
        best_cmp = (old_acc, -old_loss)
        if key > best_cmp:
            reason = "val_nt_acc" if va_acc > old_acc + 1e-8 else "val_loss_tiebreak"
            best_key = (va_acc, va_loss)
            stall = 0
            best_sd = {
                "backbone": {k: v.cpu() for k, v in backbone.state_dict().items()},
                "projector": {k: v.cpu() for k, v in projector.state_dict().items()},
                "epoch": epoch,
                "val_loss": float(va_loss),
                "val_nt_acc": float(va_acc),
                "selection": reason,
            }
            torch.save({"backbone": best_sd["backbone"]}, out_dir / "best_backbone.pt")
        else:
            stall += 1
            if epoch >= min_epochs and stall >= patience:
                print(
                    f"Early stopping SimCLR at epoch {epoch}  "
                    f"(best_val_nt_acc={best_key[0]:.4f}, best_val_loss={best_key[1]:.5f})"
                )
                break

    if best_sd is not None:
        backbone.load_state_dict(best_sd["backbone"])
        projector.load_state_dict(best_sd["projector"])

    save_json(
        out_dir / "metrics.json",
        {
            "best_val_loss": float(best_key[1]) if best_key[1] < float("inf") else -1.0,
            "best_val_nt_acc": float(best_key[0]),
            "best_epoch": int(best_sd["epoch"]) if best_sd else -1,
            "selection": best_sd.get("selection", "") if best_sd else "",
            "temperature": temperature,
            "projector_hidden": proj_h,
            "projector_out": proj_d,
            "use_weighted_sampler": use_weighted,
            "evaluation_split": split_mode,
            "norm_mode": norm_mode,
            "meta": meta,
        },
    )
    if rows:
        viz.plot_loss_curves(rows, out_dir / "loss_curve.png")
    print(f"[simclr] Saved backbone → {out_dir / 'best_backbone.pt'}")


if __name__ == "__main__":
    main()
