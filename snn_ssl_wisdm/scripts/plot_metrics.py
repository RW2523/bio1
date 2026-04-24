"""Aggregate plots across output runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snn_ssl_wisdm.train_utils import workspace_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=str, default="outputs")
    args = ap.parse_args()
    root = Path(args.outputs)
    if not root.is_absolute():
        root = (workspace_root() / root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    series = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        logf = sub / "train_log.csv"
        if not logf.is_file():
            continue
        with open(logf, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            rows = list(r)
        if not rows:
            continue
        series.append((sub.name, rows))
    if not series:
        print("No train_log.csv found under", root)
        return
    plt.figure(figsize=(10, 6))
    for name, rows in series:
        epochs = [int(r["epoch"]) for r in rows]
        if "val_loss" in rows[0]:
            vals = [float(r["val_loss"]) for r in rows]
            plt.plot(epochs, vals, label=f"{name} val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    out_png = root / "aggregate_val_loss.png"
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
