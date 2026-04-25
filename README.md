# bio1

Monorepo layout:

- **`snn_ssl_wisdm/`** — Spiking ResNet + AugPred SSL + WISDM linear probing ([`snn_ssl_wisdm/README.md`](snn_ssl_wisdm/README.md)).
- **`SpikeGPT/`** — SpikingJelly / reference SNN code (vendored for imports).
- **`ssl-wearables/`** — Original Oxford SSL HAR codebase (reference / PYTHONPATH).
- **`wisdm-dataset/`** — WISDM raw and ARFF data ([dataset README](wisdm-dataset/README.txt)).
- **`run_snn_ssl_wisdm_unity.sbatch`** — Slurm job for Unity clusters (defaults to **A100** partition on UMass Unity to avoid old-GPU + PyTorch wheel mismatches).

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
mkdir -p logs
sbatch run_snn_ssl_wisdm_unity.sbatch
```

Replace `/path/to/bio1` with your real project path on Unity. Slurm prints a job id (e.g. `Submitted batch job 12345678`).

### Watch the job (optional)

```bash
squeue -u "$USER"
```

### Read stdout and stderr

```bash
ls -lt logs/
tail -f logs/snn_ssl_wisdm_<JOBID>.out
tail -f logs/snn_ssl_wisdm_<JOBID>.err
```

Substitute `<JOBID>` with the id from `sbatch`.

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

**Fix (recommended):** request a **newer GPU partition** (e.g. UMass Unity [`superpod-a100`](https://docs.unity.rc.umass.edu/documentation/cluster_specs/partitions) or `gpupod-l40s`). The repo’s `run_snn_ssl_wisdm_unity.sbatch` defaults to **`#SBATCH --partition=superpod-a100`** and **`#SBATCH --gres=gpu:1`**, and runs an early **compute-capability + CUDA matmul probe** so the job **fails fast** with a clear message instead of dying mid–linear probe.

**If your Unity is not UMass:** edit the top of `run_snn_ssl_wisdm_unity.sbatch` — comment the `superpod-a100` lines and uncomment one of the **alternate** blocks (e.g. Cambridge CSD3 **`-p ampere`**) or ask your admins for the correct **partition / `--gres`** syntax.

The sbatch script also **`pip install`s PyTorch with `--index-url https://download.pytorch.org/whl/cu121`** after `requirements_unity.txt` so the job gets a **CUDA-enabled** build on the node.

### Cluster-specific notes

- If **`sbatch`** is not found, connect via SSH to Unity’s Slurm login node per your site documentation.
- If the job is **rejected** for partition or GPU settings, edit the active `#SBATCH` lines at the **top** of `run_snn_ssl_wisdm_unity.sbatch` (`--partition`, `--gres`, `--cpus-per-task`, `--mem`) to match `sinfo` / your site docs.
- **`module load`** lines in the sbatch file may need names that match your Unity software stack; adjust if modules fail.
