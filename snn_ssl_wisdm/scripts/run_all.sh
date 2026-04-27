#!/usr/bin/env bash
# run_all.sh — full snn_ssl_wisdm pipeline
#
# Usage:
#   bash snn_ssl_wisdm/scripts/run_all.sh            # full run
#   bash snn_ssl_wisdm/scripts/run_all.sh --smoke    # quick sanity check
#
# Cases run:
#   Case 1 : random frozen backbone   + linear head  (baseline)
#   Case 2 : SimCLR-pretrained frozen + linear head (SSL benefit)
#   Case 3 : SimCLR-pretrained frozen backbone + MLP head (non-linear probe)
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

SMOKE=0
if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE=1
fi

echo "== Workspace: $ROOT"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

pip install -q -r snn_ssl_wisdm/requirements_unity.txt 2>/dev/null \
    || pip install -r snn_ssl_wisdm/requirements_unity.txt

# ── data preparation ────────────────────────────────────────────────────────
if [[ "$SMOKE" -eq 1 ]]; then
  echo "== Smoke: prepare (4 subjects, max 1000 windows)"
  python -m snn_ssl_wisdm.scripts.prepare_wisdm \
    --config snn_ssl_wisdm/configs/default.yaml \
    --limit_subjects 4 \
    --max_windows 1000
  EP_L=5
  EP_P=5
  EP_F=5
else
  echo "== Full: prepare (all subjects)"
  python -m snn_ssl_wisdm.scripts.prepare_wisdm \
    --config snn_ssl_wisdm/configs/default.yaml
  # Match default.yaml (~60 epochs each stage; override with --epochs)
  EP_L=60
  EP_P=60
  EP_F=60
fi

# ── Case 1: random frozen backbone + linear head ────────────────────────────
echo ""
echo "== Case 1: random frozen backbone + linear head"
python -m snn_ssl_wisdm.scripts.linear_probe_snn \
  --config snn_ssl_wisdm/configs/default.yaml \
  --case case1 \
  --epochs "$EP_L"

# ── SimCLR contrastive pretraining ─────────────────────────────────────────
echo ""
echo "== SimCLR SSL pretraining"
python -m snn_ssl_wisdm.scripts.pretrain_simclr_snn \
  --config snn_ssl_wisdm/configs/default.yaml \
  --epochs "$EP_P"

PRETRAINED="outputs/simclr_pretrain_snn/best_backbone.pt"

# ── Case 2: SimCLR-pretrained frozen backbone + linear head ───────────────
echo ""
echo "== Case 2: SimCLR-pretrained frozen backbone + linear head"
python -m snn_ssl_wisdm.scripts.linear_probe_snn \
  --config snn_ssl_wisdm/configs/default.yaml \
  --case case2 \
  --pretrained "$PRETRAINED" \
  --epochs "$EP_L"

# ── Case 3: SimCLR-pretrained frozen backbone + MLP head ──────────────────
echo ""
echo "== Case 3: frozen backbone + MLP head (SimCLR init, train head only)"
python -m snn_ssl_wisdm.scripts.linear_probe_snn \
  --config snn_ssl_wisdm/configs/default.yaml \
  --case case3 \
  --pretrained "$PRETRAINED" \
  --epochs "$EP_F"

# ── evaluation / plots ───────────────────────────────────────────────────
echo ""
echo "== Evaluate"
python -m snn_ssl_wisdm.scripts.evaluate \
  --config snn_ssl_wisdm/configs/default.yaml \
  --run outputs/case1_random_frozen || true
python -m snn_ssl_wisdm.scripts.evaluate \
  --config snn_ssl_wisdm/configs/default.yaml \
  --run outputs/case2_simclr_frozen || true
python -m snn_ssl_wisdm.scripts.evaluate \
  --config snn_ssl_wisdm/configs/default.yaml \
  --run outputs/case3_simclr_frozen_mlp || true

echo ""
echo "== Plot aggregate"
python -m snn_ssl_wisdm.scripts.plot_metrics --outputs outputs

# ── final comparison table ───────────────────────────────────────────────
python - <<'PY'
import json
from pathlib import Path

root = Path("outputs")

def loadm(p):
    f = root / p / "metrics.json"
    if not f.is_file():
        return None
    with open(f) as fp:
        return json.load(fp)

c1 = loadm("case1_random_frozen")
c2 = loadm("case2_simclr_frozen")
c3 = loadm("case3_simclr_frozen_mlp")

def row(tag, d):
    bacc = d.get("balanced_accuracy", 0)
    return (
        f"  {tag:<12} acc={d['accuracy']:.4f}  bal_acc={bacc:.4f}  "
        f"macro_f1={d['macro_f1']:.4f}  weighted_f1={d['weighted_f1']:.4f}  "
        f"kappa={d['cohen_kappa']:.4f}"
    )

print("\n" + "="*75)
print("   FINAL COMPARISON — test set (SimCLR SSL)")
print("="*75)
if c1: print(row("Case1 (rand-frz)",  c1))
if c2: print(row("Case2 (simclr)",   c2))
if c3: print(row("Case3 (simclr-mlp)", c3))
if c1 and c2:
    delta_acc = c2['accuracy'] - c1['accuracy']
    print(f"\n  SSL benefit (C2-C1): Δacc={delta_acc:+.4f}  Δmacro_f1={c2['macro_f1']-c1['macro_f1']:+.4f}")
if c2 and c3:
    delta_acc = c3['accuracy'] - c2['accuracy']
    print(f"  MLP vs linear (C3-C2): Δacc={delta_acc:+.4f}  Δmacro_f1={c3['macro_f1']-c2['macro_f1']:+.4f}")
if not any([c1, c2, c3]):
    print("  No metrics found — check that the pipeline completed without errors.")
print("="*75)
PY

echo ""
echo "== Done"
