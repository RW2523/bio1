#!/bin/bash
#SBATCH --job-name=snn_ssl_wisdm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Job name: ${SLURM_JOB_NAME:-snn_ssl_wisdm}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "============================================================"

mkdir -p logs outputs

# ---------------------------------------------------------------------
# Set project root.
# Recommended usage:
#   cd /path/to/new_b
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
# ---------------------------------------------------------------------
python -m pip install --upgrade pip wheel setuptools

if [ -f "snn_ssl_wisdm/requirements_unity.txt" ]; then
    python -m pip install -r snn_ssl_wisdm/requirements_unity.txt
else
    echo "requirements_unity.txt not found yet. Installing baseline dependencies."
    python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm pyyaml
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || \
    python -m pip install torch torchvision torchaudio
fi

# Make local packages visible.
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/SpikeGPT/src:$PROJECT_ROOT/ssl-wearables:$PYTHONPATH"

echo "CUDA check:"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
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
