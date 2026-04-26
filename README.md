# bio1

Monorepo layout:

- **`snn_ssl_wisdm/`** — Spiking ResNet + **SimCLR** SSL + WISDM frozen probing ([`snn_ssl_wisdm/README.md`](snn_ssl_wisdm/README.md)); AugPred script remains optional.
- **`SpikeGPT/`** — SpikingJelly / reference SNN code (vendored for imports).
- **`ssl-wearables/`** — Original Oxford SSL HAR codebase (reference / PYTHONPATH).
- **`wisdm-dataset/`** — WISDM raw and ARFF data ([dataset README](wisdm-dataset/README.txt)).
- **`run_snn_ssl_wisdm_unity.sbatch`** — Slurm job for Unity: **`partition=gpu`**, **`--constraint=a100`**, **`--time=04:00:00`**, **`--gpus=1`** (fixed in the script header unless Unity policy changes). Stdout/stderr: **`slurm-unity-full-<jobid>.out`** / **`.err`** in the submit directory.

Quick start (from this directory):

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
pip install -r snn_ssl_wisdm/requirements_unity.txt
bash snn_ssl_wisdm/scripts/run_all.sh --smoke   # or full run without --smoke
```

## Running on Unity (Slurm)

Use these on a **Unity login node** after the repo is on the cluster (clone or `rsync`).

### Submit the job

```bash
cd /path/to/bio1
sbatch run_snn_ssl_wisdm_unity.sbatch
```

Replace `/path/to/bio1` with your real project path on Unity. Slurm prints a job id (e.g. `Submitted batch job 12345678`). Logs are written as **`slurm-unity-full-<jobid>.out`** and **`slurm-unity-full-<jobid>.err`** in the directory you submitted from.

### Watch the job (optional)

```bash
squeue -u "$USER"
```

### Read stdout and stderr

```bash
ls -lt slurm-unity-full-*.out slurm-unity-full-*.err
tail -f slurm-unity-full-<JOBID>.out
tail -f slurm-unity-full-<JOBID>.err
```

Replace `<JOBID>` with the numeric id from `sbatch` (same id in both filenames).

### Optional: conda env name or project root

```bash
sbatch --export=ALL,CONDA_ENV=my_env_name run_snn_ssl_wisdm_unity.sbatch
```

```bash
PROJECT_ROOT=/absolute/path/to/bio1 sbatch run_snn_ssl_wisdm_unity.sbatch
```

### Quick debug run (smoke) on the cluster

In `run_snn_ssl_wisdm_unity.sbatch`, change the pipeline line to:

```bash
bash snn_ssl_wisdm/scripts/run_all.sh --smoke
```

instead of `bash snn_ssl_wisdm/scripts/run_all.sh` (no flag), then submit again with `sbatch`.

### GPU / PyTorch (avoid “no kernel image is available for the device”)

Recent **PyTorch CUDA wheels** (e.g. cu121) ship GPU kernels for **Turing (sm_75) and newer** (Ampere, Ada, Hopper). They do **not** support very old cards such as **Maxwell** (e.g. GTX TITAN X, sm_52). If Slurm schedules such a node, training can fail after `prepare` with:

`torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device`

**Fix (recommended):** request **A100** nodes via Slurm (this repo uses **`--partition=gpu`** + **`--constraint=a100`**). The sbatch script also runs an early **compute-capability + CUDA matmul probe** so the job **fails fast** with a clear message instead of dying mid–linear probe.

**Wall clock:** the header uses **`#SBATCH --time=04:00:00`** as the project default. If a full run needs longer, increase **only** `--time` in agreement with Unity policy (keep **`--constraint=a100`** unless admins say otherwise).

The sbatch script also **`pip install`s PyTorch with `--index-url https://download.pytorch.org/whl/cu121`** after `requirements_unity.txt` so the job gets a **CUDA-enabled** build on the node.

### Cluster-specific notes

- If **`sbatch`** is not found, connect via SSH to Unity’s Slurm login node per your site documentation.
- If the job is **rejected** for partition or GPU settings, edit the active `#SBATCH` lines at the **top** of `run_snn_ssl_wisdm_unity.sbatch` (`--partition`, `--gres`, `--cpus-per-task`, `--mem`) to match `sinfo` / your site docs.
- **`module load`** lines in the sbatch file may need names that match your Unity software stack; adjust if modules fail.
