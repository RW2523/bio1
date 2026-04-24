# bio1

Monorepo layout:

- **`snn_ssl_wisdm/`** — Spiking ResNet + AugPred SSL + WISDM linear probing ([`snn_ssl_wisdm/README.md`](snn_ssl_wisdm/README.md)).
- **`SpikeGPT/`** — SpikingJelly / reference SNN code (vendored for imports).
- **`ssl-wearables/`** — Original Oxford SSL HAR codebase (reference / PYTHONPATH).
- **`wisdm-dataset/`** — WISDM raw and ARFF data ([dataset README](wisdm-dataset/README.txt)).
- **`run_snn_ssl_wisdm_unity.sbatch`** — Slurm job for Unity clusters.

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

### Cluster-specific notes

- If **`sbatch`** is not found, connect via SSH to Unity’s Slurm login node per your site documentation.
- If the job is **rejected** for partition or GPU settings, edit the `#SBATCH` lines in `run_snn_ssl_wisdm_unity.sbatch` (e.g. `--partition`, `--gres`) to match `sinfo` or Unity’s current policy.
- **`module load`** lines in the sbatch file may need names that match your Unity software stack; adjust if modules fail.
