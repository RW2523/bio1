"""Load a finished run and print / extend metrics (test set)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from snn_ssl_wisdm.train_utils import workspace_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--run", type=str, required=True, help="Path to run dir (e.g. outputs/case1_random_frozen)")
    args = ap.parse_args()
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = (workspace_root() / run_dir).resolve()
    mpath = run_dir / "metrics.json"
    if not mpath.is_file():
        raise FileNotFoundError(mpath)
    with open(mpath, "r", encoding="utf-8") as f:
        m = json.load(f)
    print(json.dumps(m, indent=2))
    with open(run_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump({"source_metrics": m, "config": args.config}, f, indent=2)


if __name__ == "__main__":
    main()
