#!/bin/bash
# ---------------------------------------------------------------------------
# Slurm header (Unity reference — keep walltime + A100 constraint stable unless
# site policy changes). PyTorch cu121 wheels need Turing+ / A100-class GPUs.
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
# Full pipeline: omit --qos=short (short QoS is for small priority-boosted jobs;
# can confuse long runs). For a one-off interactive boost on a *short* test, add
# on sbatch line: --qos=short (see Unity docs).
#SBATCH --output=slurm-unity-full-%j.out
#SBATCH --error=slurm-unity-full-%j.err

set -euo pipefail

echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Job name: ${SLURM_JOB_NAME:-wisdm-unity-full}"
echo "Node: $(hostname)"
echo "Slurm partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "============================================================"

mkdir -p outputs

# ---------------------------------------------------------------------
# Set project root.
# Recommended usage:
#   cd /path/to/bio1
#   sbatch run_snn_ssl_wisdm_unity.sbatch
# ---------------------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "$PROJECT_ROOT"

echo "PROJECT_ROOT=$PROJECT_ROOT"

# ---------------------------------------------------------------------
# Load modules. Adjust these if Unity uses different module names.
# ---------------------------------------------------------------------
module purge || true

# Common Unity-style options. These may need small edits on your cluster.
module load cuda || true
module load anaconda || true
module load miniconda || true

# ---------------------------------------------------------------------
# Conda environment.
# If you already have an environment, submit with:
#   sbatch --export=ALL,CONDA_ENV=snn_ssl_wisdm run_snn_ssl_wisdm_unity.sbatch
# ---------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-snn_ssl_wisdm}"

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"

    if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
        echo "Activating existing conda env: $CONDA_ENV"
        conda activate "$CONDA_ENV"
    else
        echo "Creating conda env: $CONDA_ENV"
        conda create -y -n "$CONDA_ENV" python=3.10
        conda activate "$CONDA_ENV"
    fi
else
    echo "Conda not found. Falling back to python venv."
    python3 -m venv .venv_snn_ssl_wisdm
    source .venv_snn_ssl_wisdm/bin/activate
fi

echo "Python: $(which python)"
python --version

# ---------------------------------------------------------------------
# Install dependencies.
# Install a CUDA-enabled PyTorch explicitly (default pip "torch" can be CPU-only
# or omit sm_52 Maxwell anyway — old GPUs still fail at kernel launch).
# ---------------------------------------------------------------------
python -m pip install --upgrade pip wheel setuptools

if [ -f "snn_ssl_wisdm/requirements_unity.txt" ]; then
    python -m pip install -r snn_ssl_wisdm/requirements_unity.txt
else
    echo "requirements_unity.txt not found yet. Installing baseline dependencies."
    python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm pyyaml
fi

echo "Installing PyTorch with CUDA 12.1 wheels (Turing sm_75+ / Ampere / Ada / Hopper-class GPUs)."
python -m pip install --upgrade "torch>=2.1.0" "torchvision" "torchaudio" \
  --index-url https://download.pytorch.org/whl/cu121 || {
    echo "WARNING: cu121 wheel install failed; continuing with whatever torch is present."
  }

# Make local packages visible.
# NOTE: use ${PYTHONPATH:-} because `set -u` makes an unset PYTHONPATH an error on some clusters.
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/SpikeGPT/src:$PROJECT_ROOT/ssl-wearables:${PYTHONPATH:-}"

echo "CUDA / GPU preflight:"
python - <<'PY'
import sys
import torch

print("torch:", torch.__version__)
print("cuda built:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available to PyTorch on this node.", file=sys.stderr)
    sys.exit(2)

name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print("gpu:", name)
print("compute capability:", cap)

# Official manylinux cu121 wheels target sm_75+ (Turing) and newer; Maxwell (5.x) is unsupported.
min_major, min_minor = 7, 5
if cap < (min_major, min_minor):
    print(
        f"ERROR: GPU {name} is sm_{cap[0]}{cap[1]}. "
        f"This PyTorch build needs compute capability >= {min_major}.{min_minor} "
        f"(e.g. RTX 20-series+, A100, L40S). Re-submit on an A100/L40S partition, "
        f"or use an older PyTorch+cuda build that still ships Maxwell kernels.",
        file=sys.stderr,
    )
    sys.exit(3)

# One tiny kernel to catch "no kernel image" mismatches early.
try:
    x = torch.zeros(256, 256, device="cuda")
    torch.nn.functional.relu(x @ x)
except Exception as e:
    print("ERROR: CUDA kernel probe failed:", e, file=sys.stderr)
    sys.exit(4)
print("CUDA kernel probe: OK")
PY

# ---------------------------------------------------------------------
# Basic preflight.
# ---------------------------------------------------------------------
echo "Preflight folder check:"
test -d SpikeGPT || { echo "Missing SpikeGPT/"; exit 1; }
test -d ssl-wearables || { echo "Missing ssl-wearables/"; exit 1; }
test -d wisdm-dataset || { echo "Missing wisdm-dataset/"; exit 1; }
test -d wisdm-dataset/raw || { echo "Missing wisdm-dataset/raw/"; exit 1; }

# ---------------------------------------------------------------------
# Run full pipeline.
# The Cursor-created script should accept --smoke for quick tests,
# and no flag for the full run.
# ---------------------------------------------------------------------
chmod +x snn_ssl_wisdm/scripts/run_all.sh

echo "============================================================"
echo "Running full SNN SSL WISDM pipeline"
echo "============================================================"

bash snn_ssl_wisdm/scripts/run_all.sh

echo "============================================================"
echo "Final output files:"
find outputs -maxdepth 3 -type f | sort
echo "End time: $(date)"
echo "============================================================"
