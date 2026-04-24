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
