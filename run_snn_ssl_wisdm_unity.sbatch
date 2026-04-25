#!/bin/bash
# =============================================================================
# run_snn_ssl_wisdm_unity.sbatch
# Full SNN + SSL + WISDM pipeline — Unity A100 cluster
#
# Submit from the repo root:
#   sbatch run_snn_ssl_wisdm_unity.sbatch
#
# Pipeline (all 3 cases inlined, no external run_all.sh needed):
#   Step 1  prepare_wisdm     — sliding windows, normalise, subject splits
#   Step 2  Case 1            — random-init   frozen backbone + linear probe
#   Step 3  AugPred pretrain  — 4-task SSL (rotation/AoT/time-warp/scale)
#   Step 4  Case 2            — SSL-pretrained frozen backbone + linear probe
#   Step 5  Case 3            — SSL-pretrained frozen backbone + MLP head (non-linear probe)
#   Step 6  evaluate + plots
#   Step 7  final comparison table
# =============================================================================

# ---------------------------------------------------------------------------
# Slurm directives — walltime and A100 constraint are FIXED; do not change.
# ---------------------------------------------------------------------------
#SBATCH --job-name=wisdm-unity-full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --output=slurm-unity-full-%j.out
#SBATCH --error=slurm-unity-full-%j.err

set -euo pipefail

# =============================================================================
# 0.  Banner
# =============================================================================
echo "============================================================"
echo "Job ID      : ${SLURM_JOB_ID:-none}"
echo "Job name    : ${SLURM_JOB_NAME:-wisdm-unity-full}"
echo "Node        : $(hostname)"
echo "Partition   : ${SLURM_JOB_PARTITION:-unknown}"
echo "Start time  : $(date)"
echo "Working dir : $(pwd)"
echo "============================================================"
echo ""
echo "3-case SNN+SSL+WISDM pipeline"
echo "  Case 1 : random-init frozen ResNet  + linear probe    (baseline)"
echo "  Case 2 : AugPred SSL-pretrained frozen + linear probe (SSL benefit)"
echo "  Case 3 : AugPred SSL-pretrained frozen + MLP head (train head only)"
echo ""
echo "SNN config: beta=0.5 (tau=2), threshold=0.5 → ~16-27% neuron firing rate"
echo "============================================================"

mkdir -p outputs

# =============================================================================
# 1.  Project root & PYTHONPATH
# =============================================================================
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "$PROJECT_ROOT"
echo "PROJECT_ROOT=$PROJECT_ROOT"

# use ${PYTHONPATH:-} so set -u does not fail when PYTHONPATH is unset
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/SpikeGPT/src:$PROJECT_ROOT/ssl-wearables:${PYTHONPATH:-}"

# =============================================================================
# 2.  Load modules (Unity-style; harmless if already loaded / missing)
# =============================================================================
module purge      || true
module load cuda      || true
module load anaconda  || true
module load miniconda || true

# =============================================================================
# 3.  Conda / venv environment
# =============================================================================
CONDA_ENV="${CONDA_ENV:-snn_ssl_wisdm}"

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
        echo "Activating existing conda env: $CONDA_ENV"
        conda activate "$CONDA_ENV"
    else
        echo "Creating conda env: $CONDA_ENV (python=3.10)"
        conda create -y -n "$CONDA_ENV" python=3.10
        conda activate "$CONDA_ENV"
    fi
else
    echo "Conda not found — using python venv."
    python3 -m venv .venv_snn_ssl_wisdm
    source .venv_snn_ssl_wisdm/bin/activate
fi

echo "Python: $(which python)"
python --version

# =============================================================================
# 4.  Dependencies
# =============================================================================
python -m pip install --upgrade pip wheel setuptools

if [ -f "snn_ssl_wisdm/requirements_unity.txt" ]; then
    python -m pip install -r snn_ssl_wisdm/requirements_unity.txt
else
    python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm pyyaml
fi

# CUDA-enabled PyTorch — cu121 wheels cover Turing sm_75+ / Ampere / Ada / Hopper.
echo "Installing PyTorch cu121 wheels …"
python -m pip install --upgrade "torch>=2.1.0" torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121 || \
  echo "WARNING: cu121 wheel install failed; continuing with existing torch."

# =============================================================================
# 5.  GPU preflight — A100 check + kernel probe
# =============================================================================
echo ""
echo "---- GPU preflight ----"
python - <<'PY'
import sys, torch
print("torch:", torch.__version__)
print("cuda  :", torch.version.cuda)
print("avail :", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("ERROR: CUDA unavailable.", file=sys.stderr); sys.exit(2)
name = torch.cuda.get_device_name(0)
cap  = torch.cuda.get_device_capability(0)
print(f"gpu   : {name}")
print(f"cap   : sm_{cap[0]}{cap[1]}")
if cap < (7, 5):
    print(f"ERROR: {name} is sm_{cap[0]}{cap[1]} — need >= sm_75 (Turing/A100).",
          file=sys.stderr); sys.exit(3)
try:
    x = torch.zeros(256, 256, device="cuda")
    torch.nn.functional.relu(x @ x)
except Exception as e:
    print(f"ERROR: CUDA kernel probe failed: {e}", file=sys.stderr); sys.exit(4)
print("CUDA kernel probe: OK")
PY

# =============================================================================
# 6.  SNN backbone preflight — confirm healthy firing rate
# =============================================================================
echo ""
echo "---- SNN backbone preflight ----"
python - <<'PYCHECK'
import sys
sys.path.insert(0, "SpikeGPT/src")
import torch
from snn_ssl_wisdm.models.spiking_resnet1d import SpikingResNet1DBackbone
bb = SpikingResNet1DBackbone(in_channels=3, feature_dim=512, timesteps=4,
                              beta=0.5, v_threshold=0.5)
x  = torch.randn(4, 3, 200)
z  = bb(x)
fr = float((z > 0).float().mean())
if fr < 0.01:
    print(f"ERROR: firing rate {fr:.3f} — backbone is silent. "
          "Check beta/threshold config.", file=sys.stderr); sys.exit(5)
print(f"Firing rate: {fr:.3f}  (healthy range 0.10-0.40)  shape={tuple(z.shape)}")
PYCHECK

# =============================================================================
# 7.  Folder preflight
# =============================================================================
echo ""
echo "---- Folder preflight ----"
test -d SpikeGPT          || { echo "ERROR: Missing SpikeGPT/";           exit 1; }
test -d ssl-wearables     || { echo "ERROR: Missing ssl-wearables/";      exit 1; }
test -d wisdm-dataset     || { echo "ERROR: Missing wisdm-dataset/";      exit 1; }
test -d wisdm-dataset/raw || { echo "ERROR: Missing wisdm-dataset/raw/";  exit 1; }
echo "All required folders present."

# =============================================================================
# Epoch counts for the full run (smoke mode not needed for cluster — use
# --smoke flag from the CLI if you want a quick sanity check locally).
# These match the values in snn_ssl_wisdm/configs/default.yaml.
# =============================================================================
EP_P=300    # epochs_pretrain  (AugPred SSL; matches default.yaml)
EP_L=150    # epochs_linear_probe (Cases 1 & 2)
EP_F=150    # epochs_case3_mlp  (Case 3: frozen backbone + MLP head)
PRETRAINED="outputs/augpred_pretrain_snn/best_backbone.pt"

# =============================================================================
# Step 1 — Prepare WISDM data
# =============================================================================
echo ""
echo "============================================================"
echo "Step 1/7 — Prepare WISDM data  ($(date))"
echo "============================================================"
python -m snn_ssl_wisdm.scripts.prepare_wisdm \
  --config snn_ssl_wisdm/configs/default.yaml

# =============================================================================
# Step 2 — Case 1: randomly-initialised frozen backbone + linear probe
# =============================================================================
echo ""
echo "============================================================"
echo "Step 2/7 — Case 1: random-init frozen backbone + linear probe  ($(date))"
echo "============================================================"
python -m snn_ssl_wisdm.scripts.linear_probe_snn \
  --config  snn_ssl_wisdm/configs/default.yaml \
  --case    case1 \
  --epochs  "$EP_L"

# =============================================================================
# Step 3 — AugPred self-supervised pretraining
#   4 pretext tasks: rotation (4-class CE), arrow-of-time, time-warp, amplitude-scale
#   Cosine-annealing LR, patience=30, min_epochs=40
# =============================================================================
echo ""
echo "============================================================"
echo "Step 3/7 — AugPred SSL pretraining  ($(date))"
echo "============================================================"
python -m snn_ssl_wisdm.scripts.pretrain_augpred_snn \
  --config  snn_ssl_wisdm/configs/default.yaml \
  --epochs  "$EP_P"

# =============================================================================
# Step 4 — Case 2: SSL-pretrained frozen backbone + linear probe
# =============================================================================
echo ""
echo "============================================================"
echo "Step 4/7 — Case 2: SSL-pretrained frozen backbone + linear probe  ($(date))"
echo "============================================================"
python -m snn_ssl_wisdm.scripts.linear_probe_snn \
  --config     snn_ssl_wisdm/configs/default.yaml \
  --case       case2 \
  --pretrained "$PRETRAINED" \
  --epochs     "$EP_L"

# =============================================================================
# Step 5 — Case 3: SSL-pretrained frozen backbone + MLP head (non-linear probe)
#   Only the MLP classifier is trained; backbone stays frozen (like Cases 1–2).
# =============================================================================
echo ""
echo "============================================================"
echo "Step 5/7 — Case 3: frozen backbone + MLP head (SSL init)  ($(date))"
echo "============================================================"
python -m snn_ssl_wisdm.scripts.linear_probe_snn \
  --config     snn_ssl_wisdm/configs/default.yaml \
  --case       case3 \
  --pretrained "$PRETRAINED" \
  --epochs     "$EP_F"

# =============================================================================
# Step 6 — Evaluate + plots
# =============================================================================
echo ""
echo "============================================================"
echo "Step 6/7 — Evaluate + plots  ($(date))"
echo "============================================================"
python -m snn_ssl_wisdm.scripts.evaluate \
  --config snn_ssl_wisdm/configs/default.yaml \
  --run outputs/case1_random_frozen       || true
python -m snn_ssl_wisdm.scripts.evaluate \
  --config snn_ssl_wisdm/configs/default.yaml \
  --run outputs/case2_augpred_frozen      || true
python -m snn_ssl_wisdm.scripts.evaluate \
  --config snn_ssl_wisdm/configs/default.yaml \
  --run outputs/case3_augpred_frozen_mlp   || true

python -m snn_ssl_wisdm.scripts.plot_metrics --outputs outputs

# =============================================================================
# Step 7 — Final comparison table
# =============================================================================
echo ""
echo "============================================================"
echo "Step 7/7 — Final comparison table  ($(date))"
echo "============================================================"
python - <<'PYFINAL'
import json
from pathlib import Path

root = Path("outputs")

def loadm(rel):
    f = root / rel / "metrics.json"
    return json.load(open(f)) if f.is_file() else None

c1 = loadm("case1_random_frozen")
c2 = loadm("case2_augpred_frozen")
c3 = loadm("case3_augpred_frozen_mlp")
pt = loadm("augpred_pretrain_snn")

SEP = "=" * 76

def row(tag, d):
    bacc = d.get("balanced_accuracy", 0.0)
    return (f"  {tag:<26}  acc={d['accuracy']:.4f}  bal_acc={bacc:.4f}  "
            f"macro_f1={d['macro_f1']:.4f}  weighted_f1={d['weighted_f1']:.4f}  "
            f"kappa={d['cohen_kappa']:.4f}")

print(f"\n{SEP}")
print("  FINAL RESULTS — test set")
print(SEP)
if c1: print(row("Case 1  rand-frozen  (baseline)", c1))
if c2: print(row("Case 2  ssl-frozen   (linear probe)", c2))
if c3: print(row("Case 3  ssl-frozen-mlp (MLP probe)", c3))

if pt:
    print(f"\n  AugPred pretrain  best_val_loss={pt.get('best_val_loss','?'):.5f}  "
          f"best_epoch={pt.get('best_epoch','?')}")

if c1 and c2:
    print(f"\n  SSL benefit   (C2 − C1) : "
          f"Δacc={c2['accuracy']-c1['accuracy']:+.4f}  "
          f"Δmacro_f1={c2['macro_f1']-c1['macro_f1']:+.4f}")
if c2 and c3:
    print(f"  MLP vs linear (C3 − C2) : "
          f"Δacc={c3['accuracy']-c2['accuracy']:+.4f}  "
          f"Δmacro_f1={c3['macro_f1']-c2['macro_f1']:+.4f}")

if c3:
    print(f"\n  Case 3 (frozen backbone + MLP) test accuracy = {c3['accuracy']:.4f}")

if not any([c1, c2, c3]):
    print("  No metrics.json found — check pipeline completed without errors.")

print(SEP)
PYFINAL

# =============================================================================
# Done
# =============================================================================
echo ""
echo "============================================================"
echo "Output artefacts:"
find outputs -maxdepth 3 \( -name "*.json" -o -name "*.png" -o -name "*.pt" -o -name "*.csv" \) | sort
echo ""
echo "End time: $(date)"
echo "============================================================"
