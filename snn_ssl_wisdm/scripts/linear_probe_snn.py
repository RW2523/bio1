"""Linear probe on frozen spiking ResNet (case1 random / case2 pretrained)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from snn_ssl_wisdm.amp_utils import amp_scaler_and_autocast
from snn_ssl_wisdm.data.splits import load_split_json, mask_for_subjects
from snn_ssl_wisdm.data.wisdm import WISDMDataset
from snn_ssl_wisdm.metrics import compute_metrics
from snn_ssl_wisdm.models.heads import LinearClassifierHead
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


class HarModel(nn.Module):
    def __init__(self, backbone: SpikingResNet1DBackbone, head: LinearClassifierHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return self.head(z)


def _class_weights(bundle_labels: torch.Tensor, train_idx: np.ndarray, num_classes: int, device):
    y = bundle_labels[train_idx].numpy()
    cnt = np.bincount(y, minlength=num_classes).astype(np.float32)
    w = 1.0 / (cnt + 1e-6)
    w = w * (num_classes / w.sum())
    return torch.from_numpy(w).float().to(device)


def _run_epoch(model, loader, device, criterion, optimizer, scaler, amp_ctx, train: bool, grad_clip: float):
    if train:
        model.train()
    else:
        model.eval()
    tot_loss = 0.0
    n = 0
    correct = 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with amp_ctx():
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            bs = x.shape[0]
            tot_loss += float(loss.item()) * bs
            correct += int((logits.argmax(-1) == y).sum().item())
            n += bs
    return tot_loss / max(1, n), correct / max(1, n)


@torch.no_grad()
def _eval_loss_acc(model, loader, device, criterion, amp_ctx):
    model.eval()
    tot = 0.0
    n = 0
    correct = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        with amp_ctx():
            logits = model(x)
            loss = criterion(logits, y)
        bs = x.shape[0]
        tot += float(loss.item()) * bs
        correct += int((logits.argmax(-1) == y).sum().item())
        n += bs
    return tot / max(1, n), correct / max(1, n)


@torch.no_grad()
def _predict_all(model, loader, device, amp_ctx):
    model.eval()
    ys: List[int] = []
    ps: List[int] = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        with amp_ctx():
            logits = model(x)
        pred = logits.argmax(-1).cpu().numpy().tolist()
        ys.extend(batch["y"].numpy().tolist())
        ps.extend(pred)
    return np.array(ys), np.array(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--case", type=str, choices=["case1", "case2"], required=True)
    ap.add_argument("--pretrained", type=str, default=None)
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
        ("epochs_linear_probe", args.epochs),
    ]:
        if v is not None:
            cli[k] = v
    cfg = merge_dict(cfg, cli)

    seed = int(cfg["seed"])
    set_seed(seed)
    device = pick_device(cfg.get("device"))
    print_gpu_info(device)

    processed = Path(cfg["processed_path"])
    split, _ = load_split_json(Path(cfg["splits_path"]))
    bundle = torch_load(processed, map_location="cpu")
    windows: torch.Tensor = bundle["windows"]
    labels: torch.Tensor = bundle["labels"]
    subjects: torch.Tensor = bundle["subjects"]
    num_classes = int(cfg["num_classes"])
    assert int(labels.max()) < num_classes, "label out of range"
    print(f"Random baseline accuracy (1 / num_classes): {1.0 / num_classes:.6f}")

    subj_np = subjects.numpy()
    train_m = mask_for_subjects(subj_np, "train", split)
    val_m = mask_for_subjects(subj_np, "val", split)
    test_m = mask_for_subjects(subj_np, "test", split)
    train_idx = np.where(train_m)[0]
    val_idx = np.where(val_m)[0]
    test_idx = np.where(test_m)[0]
    ds_tr = WISDMDataset(processed, train_idx)
    ds_va = WISDMDataset(processed, val_idx)
    ds_te = WISDMDataset(processed, test_idx)

    nw = int(cfg["num_workers"])
    bs = int(cfg["batch_size"])
    train_loader = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=device.type == "cuda")
    val_loader = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=device.type == "cuda")
    test_loader = DataLoader(ds_te, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=device.type == "cuda")

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
    head = LinearClassifierHead(int(mcfg["feature_dim"]), num_classes).to(device)
    model = HarModel(backbone, head).to(device)

    if args.case == "case2":
        if not args.pretrained:
            raise ValueError("case2 requires --pretrained backbone checkpoint")
        ck = torch_load(Path(args.pretrained), map_location=device)
        if isinstance(ck, dict) and "backbone" in ck:
            backbone.load_state_dict(ck["backbone"])
        else:
            backbone.load_state_dict(ck)

    backbone.set_freeze(True)
    for p in head.parameters():
        p.requires_grad = True

    cw = _class_weights(labels, train_idx, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg["lr_head"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    amp = bool(cfg.get("amp", True))
    scaler, amp_ctx = amp_scaler_and_autocast(device, amp)
    epochs = int(cfg.get("epochs_linear_probe") or 50)
    patience = int(cfg["patience"])
    grad_clip = float(cfg["grad_clip"])
    out_dir = Path(cfg["output_root"]) / (
        "case1_random_frozen" if args.case == "case1" else "case2_augpred_frozen"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    best_val = float("inf")
    stall = 0
    best_state = None
    rows = []

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(
            model, train_loader, device, criterion, optimizer, scaler, amp_ctx, True, grad_clip
        )
        va_loss, va_acc = _eval_loss_acc(model, val_loader, device, criterion, amp_ctx)
        te_loss, te_acc = _eval_loss_acc(model, test_loader, device, criterion, amp_ctx)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "test_loss": te_loss,
                "test_acc": te_acc,
            }
        )
        print(
            f"epoch {epoch}/{epochs} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} test_loss={te_loss:.4f} test_acc={te_acc:.4f}"
        )
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        if va_loss < best_val - 1e-6:
            best_val = va_loss
            stall = 0
            best_state = {
                "backbone": backbone.state_dict(),
                "head": head.state_dict(),
                "epoch": epoch,
            }
            torch.save(best_state, out_dir / "best.pt")
        else:
            stall += 1
            if stall >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        backbone.load_state_dict(best_state["backbone"])
        head.load_state_dict(best_state["head"])

    y_true, y_pred = _predict_all(model, test_loader, device, amp_ctx)
    metrics = compute_metrics(y_true, y_pred, num_classes)
    metrics["case"] = args.case
    metrics["random_baseline_acc"] = float(1.0 / num_classes)
    save_json(out_dir / "metrics.json", metrics)
    print(f"Saved metrics to {out_dir / 'metrics.json'}")
    idx2 = bundle.get("idx_to_activity", {})
    names = [idx2.get(i, idx2.get(str(i), str(i))) for i in range(num_classes)]
    viz.plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]), names, out_dir / "confusion_matrix.png"
    )
    if rows:
        viz.plot_loss_curves(rows, out_dir / "loss_curve.png")
        viz.plot_acc_curves(rows, out_dir / "accuracy_curve.png")


if __name__ == "__main__":
    main()
