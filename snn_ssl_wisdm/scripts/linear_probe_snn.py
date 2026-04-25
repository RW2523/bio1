"""Linear probe / fine-tune on spiking ResNet.

Three experimental cases
------------------------
case1 : Randomly-initialized, frozen backbone  + linear head   (baseline)
case2 : AugPred-pretrained,   frozen backbone  + linear head   (SSL benefit)
case3 : AugPred-pretrained,   UNfrozen backbone + MLP head     (full fine-tune → target ≥60 %)

Improvements over v1
--------------------
- Case 3 added: end-to-end fine-tuning with differential LRs (backbone 10× lower).
- Cosine-annealing LR scheduler for all cases.
- WeightedRandomSampler on training set (handles class imbalance).
- Early stopping on validation balanced-accuracy (not loss) for cases 1&2;
  on validation loss for case3 (fine-tuning can over-fit).
- Minimum epochs guard: never stop before 20 epochs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import balanced_accuracy_score

from snn_ssl_wisdm.amp_utils import amp_scaler_and_autocast
from snn_ssl_wisdm.data.splits import load_split_json, mask_for_subjects
from snn_ssl_wisdm.data.wisdm import WISDMDataset
from snn_ssl_wisdm.metrics import compute_metrics
from snn_ssl_wisdm.models.heads import LinearClassifierHead, MLPClassifierHead
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


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class HarModel(nn.Module):
    def __init__(self, backbone: SpikingResNet1DBackbone, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head     = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _class_weights(bundle_labels: torch.Tensor, train_idx: np.ndarray,
                   num_classes: int, device: torch.device) -> torch.Tensor:
    y   = bundle_labels[train_idx].numpy()
    cnt = np.bincount(y, minlength=num_classes).astype(np.float32)
    w   = 1.0 / (cnt + 1e-6)
    w   = w * (num_classes / w.sum())
    return torch.from_numpy(w).float().to(device)


def _sample_weights(bundle_labels: torch.Tensor, train_idx: np.ndarray,
                    num_classes: int) -> torch.Tensor:
    """Per-sample weights for WeightedRandomSampler (inverse class freq)."""
    y   = bundle_labels[train_idx].numpy()
    cnt = np.bincount(y, minlength=num_classes).astype(np.float64)
    freq = cnt / cnt.sum()
    sw  = 1.0 / (freq[y] + 1e-9)
    return torch.from_numpy(sw.astype(np.float32))


def _run_epoch(
    model, loader, device, criterion, optimizer, scaler, amp_ctx,
    train: bool, grad_clip: float,
) -> tuple:
    model.train() if train else model.eval()
    tot_loss = 0.0
    all_y: List[int] = []
    all_p: List[int] = []
    n = 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with amp_ctx():
                logits = model(x)
                loss   = criterion(logits, y)
            if train:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()), grad_clip
                    )
                scaler.step(optimizer)
                scaler.update()
            bs = x.shape[0]
            tot_loss += float(loss.item()) * bs
            all_y.extend(y.cpu().numpy().tolist())
            all_p.extend(logits.argmax(-1).cpu().numpy().tolist())
            n += bs
    bacc = balanced_accuracy_score(all_y, all_p) if len(all_y) > 0 else 0.0
    acc  = np.mean(np.array(all_y) == np.array(all_p)) if len(all_y) > 0 else 0.0
    return tot_loss / max(1, n), float(acc), float(bacc)


@torch.no_grad()
def _predict_all(model, loader, device, amp_ctx) -> tuple:
    model.eval()
    ys: List[int] = []
    ps: List[int] = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        with amp_ctx():
            logits = model(x)
        ps.extend(logits.argmax(-1).cpu().numpy().tolist())
        ys.extend(batch["y"].numpy().tolist())
    return np.array(ys), np.array(ps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",         type=str, required=True)
    ap.add_argument("--case",           type=str, choices=["case1", "case2", "case3"], required=True)
    ap.add_argument("--pretrained",     type=str, default=None,
                    help="Path to best_backbone.pt from AugPred pretrain (required for case2/case3).")
    ap.add_argument("--epochs",         type=int, default=None)
    ap.add_argument("--limit_subjects", type=int, default=None)
    ap.add_argument("--max_windows",    type=int, default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (workspace_root() / cfg_path).resolve()
    cfg = resolve_paths(load_yaml(cfg_path), cfg_path)

    cli: Dict = {}
    for k, v in [
        ("limit_subjects",   args.limit_subjects),
        ("max_windows",      args.max_windows),
    ]:
        if v is not None:
            cli[k] = v
    if args.epochs is not None:
        epochs_key = "epochs_finetune" if args.case == "case3" else "epochs_linear_probe"
        cli[epochs_key] = args.epochs
    cfg = merge_dict(cfg, cli)

    set_seed(int(cfg["seed"]))
    device = pick_device(cfg.get("device"))
    print_gpu_info(device)

    # ── data ──────────────────────────────────────────────────────────────
    processed = Path(cfg["processed_path"])
    split, _  = load_split_json(Path(cfg["splits_path"]))
    bundle    = torch_load(processed, map_location="cpu")
    windows: torch.Tensor = bundle["windows"]
    labels:  torch.Tensor = bundle["labels"]
    subjects: torch.Tensor = bundle["subjects"]
    num_classes = int(cfg["num_classes"])
    assert int(labels.max()) < num_classes, "label out of range"
    print(f"Random-chance baseline: {1.0/num_classes:.4f} ({num_classes} classes)")

    subj_np   = subjects.numpy()
    train_idx = np.where(mask_for_subjects(subj_np, "train", split))[0]
    val_idx   = np.where(mask_for_subjects(subj_np, "val",   split))[0]
    test_idx  = np.where(mask_for_subjects(subj_np, "test",  split))[0]

    ds_tr = WISDMDataset(processed, train_idx)
    ds_va = WISDMDataset(processed, val_idx)
    ds_te = WISDMDataset(processed, test_idx)

    nw  = int(cfg["num_workers"])
    bs  = int(cfg["batch_size"])

    sw_train = _sample_weights(labels, train_idx, num_classes)
    sampler  = WeightedRandomSampler(sw_train, num_samples=len(sw_train), replacement=True)

    train_loader = DataLoader(ds_tr, batch_size=bs, sampler=sampler,
                              num_workers=nw, pin_memory=device.type == "cuda")
    val_loader   = DataLoader(ds_va, batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=device.type == "cuda")
    test_loader  = DataLoader(ds_te, batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=device.type == "cuda")

    # ── model ─────────────────────────────────────────────────────────────
    mcfg = cfg["model"]
    sch  = cfg["spiking"]
    backbone = SpikingResNet1DBackbone(
        in_channels   = int(mcfg["in_channels"]),
        base_channels = int(mcfg["base_channels"]),
        layers        = list(mcfg["layers"]),
        feature_dim   = int(mcfg["feature_dim"]),
        timesteps     = int(sch["timesteps"]),
        beta          = float(sch["beta"]),
        v_threshold   = float(sch["threshold"]),
        detach_reset  = bool(sch["detach_reset"]),
    ).to(device)

    # Load pretrained backbone for case2 and case3
    if args.case in ("case2", "case3"):
        if not args.pretrained:
            raise ValueError(f"--pretrained is required for {args.case}")
        ck = torch_load(Path(args.pretrained), map_location=device)
        sd = ck["backbone"] if isinstance(ck, dict) and "backbone" in ck else ck
        backbone.load_state_dict(sd)
        print(f"[{args.case}] Loaded backbone from {args.pretrained}")

    # Head selection: MLP for case3 (fine-tune), linear for cases 1&2 (probe)
    head_cfg    = cfg.get("head", {})
    feature_dim = int(mcfg["feature_dim"])
    if args.case == "case3":
        head = MLPClassifierHead(
            in_dim     = feature_dim,
            num_classes = num_classes,
            hidden_dim  = int(head_cfg.get("hidden_dim", 256)),
            dropout     = float(head_cfg.get("dropout",    0.3)),
        ).to(device)
    else:
        head = LinearClassifierHead(feature_dim, num_classes).to(device)

    model = HarModel(backbone, head).to(device)

    # Freeze / unfreeze backbone
    if args.case == "case3":
        backbone.set_freeze(False)   # fine-tune everything
        print("[case3] Backbone UNFROZEN — full fine-tuning")
    else:
        backbone.set_freeze(True)    # linear probe
        print(f"[{args.case}] Backbone FROZEN — linear probe")

    # ── optimizer (differential LRs for case3) ────────────────────────────
    cw        = _class_weights(labels, train_idx, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    amp_enabled = bool(cfg.get("amp", True))
    scaler, amp_ctx = amp_scaler_and_autocast(device, amp_enabled)
    grad_clip = float(cfg["grad_clip"])

    if args.case == "case3":
        lr_bb   = float(cfg.get("lr_finetune_backbone", 1e-4))
        lr_head = float(cfg.get("lr_finetune_head",     1e-3))
        param_groups = [
            {"params": backbone.parameters(), "lr": lr_bb},
            {"params": head.parameters(),     "lr": lr_head},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=float(cfg["weight_decay"]))
        epochs    = int(cfg.get("epochs_finetune") or 100)
        patience  = int(cfg.get("patience", 15))
    else:
        lr_head   = float(cfg.get("lr_head", 3e-3))
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr_head,
            weight_decay=float(cfg["weight_decay"]),
        )
        epochs   = int(cfg.get("epochs_linear_probe") or 100)
        patience = int(cfg.get("patience", 15))

    # Cosine annealing with 5-epoch warm-up
    warmup = min(5, epochs // 10)
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / max(1, warmup)
        progress = (ep - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── output dir ────────────────────────────────────────────────────────
    out_name = {
        "case1": "case1_random_frozen",
        "case2": "case2_augpred_frozen",
        "case3": "case3_augpred_finetune",
    }[args.case]
    out_dir  = Path(cfg["output_root"]) / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    min_epochs = 20
    best_val   = -float("inf")   # track val balanced-accuracy (higher = better)
    stall      = 0
    best_state = None
    rows: List[Dict] = []
    log_path = out_dir / "train_log.csv"

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_bacc = _run_epoch(
            model, train_loader, device, criterion, optimizer,
            scaler, amp_ctx, True, grad_clip,
        )
        va_loss, va_acc, va_bacc = _run_epoch(
            model, val_loader,   device, criterion, optimizer,
            scaler, amp_ctx, False, 0.0,
        )
        te_loss, te_acc, te_bacc = _run_epoch(
            model, test_loader,  device, criterion, optimizer,
            scaler, amp_ctx, False, 0.0,
        )
        cur_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]
        scheduler.step()

        rows.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc, "train_bacc": tr_bacc,
            "val_loss":   va_loss, "val_acc":   va_acc, "val_bacc":   va_bacc,
            "test_loss":  te_loss, "test_acc":  te_acc, "test_bacc":  te_bacc,
            "lr": cur_lr,
        })
        print(
            f"epoch {epoch:03d}/{epochs}  [{args.case}]  "
            f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} tr_bacc={tr_bacc:.4f}  "
            f"va_loss={va_loss:.4f} va_acc={va_acc:.4f} va_bacc={va_bacc:.4f}  "
            f"te_acc={te_acc:.4f}  lr={cur_lr:.2e}"
        )
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            dw = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            dw.writeheader()
            dw.writerows(rows)

        # Early stopping on val balanced-accuracy (max)
        if va_bacc > best_val + 1e-5:
            best_val  = va_bacc
            stall     = 0
            best_state = {
                "backbone": {k: v.cpu() for k, v in backbone.state_dict().items()},
                "head":     {k: v.cpu() for k, v in head.state_dict().items()},
                "epoch":    epoch,
            }
            torch.save(best_state, out_dir / "best.pt")
        else:
            stall += 1
            if epoch >= min_epochs and stall >= patience:
                print(f"Early stopping at epoch {epoch}  (best val_bacc={best_val:.4f})")
                break

    # Restore best checkpoint
    if best_state is not None:
        backbone.load_state_dict(best_state["backbone"])
        head.load_state_dict(best_state["head"])

    # ── final evaluation ──────────────────────────────────────────────────
    y_true, y_pred = _predict_all(model, test_loader, device, amp_ctx)
    metrics = compute_metrics(y_true, y_pred, num_classes)
    metrics["case"]              = args.case
    metrics["random_baseline_acc"] = float(1.0 / num_classes)
    metrics["best_val_bacc"]     = float(best_val)
    save_json(out_dir / "metrics.json", metrics)
    print(
        f"\n[{args.case}] Test accuracy={metrics['accuracy']:.4f}  "
        f"balanced_acc={metrics.get('balanced_accuracy', 0):.4f}  "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    print(f"Saved metrics → {out_dir / 'metrics.json'}")

    idx2   = bundle.get("idx_to_activity", {})
    names  = [idx2.get(i, idx2.get(str(i), str(i))) for i in range(num_classes)]
    viz.plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]), names, out_dir / "confusion_matrix.png"
    )
    if rows:
        viz.plot_loss_curves(rows, out_dir / "loss_curve.png")
        viz.plot_acc_curves(rows,  out_dir / "accuracy_curve.png")


if __name__ == "__main__":
    main()
