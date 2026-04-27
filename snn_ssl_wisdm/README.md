# snn_ssl_wisdm

Integration pipeline: **WISDM** raw sensor windows → **spiking 1D ResNet** (Conv1d blocks, SpikingJelly-style LIF with **soft reset** and **DC / decay_input=false** drive) → optional **SimCLR** pretrain → **frozen-backbone** linear / MLP probes.

Default data path (**`data.fused_12ch: true`** in `configs/default.yaml`): **12 channels** = phone accel xyz + phone gyro xyz + watch accel xyz + watch gyro xyz, aligned on the **watch accel timeline** (6.4 s @ 20 Hz → **128** samples, stride **64**). Windows are taken only inside **single-activity** segments. The `.pt` bundle stores **raw** windows plus **subject-train** and **window-train** normalisation tensors; `WISDMDataset` applies z-score per `norm_mode` / split. **Subject-disjoint** splits are primary (`splits_path`); **random window** indices for paper-style comparison live in `splits_window_path` — train with `--split window` (outputs go to `*_window` subdirs for probes).

The older **AugPred** multi-task script (`pretrain_augpred_snn.py`) is still available if you switch the pipeline back and set checkpoint paths accordingly.

## Layout

- `configs/default.yaml` — hyperparameters (`simclr:` block, probe settings, paths).
- `scripts/prepare_wisdm.py` — parse `wisdm-dataset/raw/...`, sliding windows, normalization, `.pt` cache + split JSON.
- `scripts/pretrain_simclr_snn.py` — SimCLR SSL; saves `outputs/simclr_pretrain_snn/best_backbone.pt` (backbone only; projector discarded for downstream).
- `scripts/pretrain_augpred_snn.py` — optional AugPred multi-task SSL (legacy).
- `scripts/linear_probe_snn.py` — frozen backbone + trainable head: `case1` random init, `case2` / `case3` load SimCLR backbone.
- `scripts/evaluate.py` — print `metrics.json` for a run directory.
- `scripts/plot_metrics.py` — aggregate loss curves under `outputs/`.
- `scripts/run_all.sh` — full or smoke end-to-end.

## Environment

From the workspace root (`new_b`), which contains `SpikeGPT/`, `wisdm-dataset/`, and `snn_ssl_wisdm/`:

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
pip install -r snn_ssl_wisdm/requirements_unity.txt
```

SpikingJelly is imported from `SpikeGPT/src` automatically (`train_utils.ensure_spikegpt_src_on_path`). If that fails, a minimal fallback LIF is used.

## CLI (from workspace root)

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python -m snn_ssl_wisdm.scripts.prepare_wisdm --config snn_ssl_wisdm/configs/default.yaml

python -m snn_ssl_wisdm.scripts.linear_probe_snn --config snn_ssl_wisdm/configs/default.yaml --case case1

python -m snn_ssl_wisdm.scripts.pretrain_simclr_snn --config snn_ssl_wisdm/configs/default.yaml

python -m snn_ssl_wisdm.scripts.linear_probe_snn --config snn_ssl_wisdm/configs/default.yaml --case case2 --pretrained outputs/simclr_pretrain_snn/best_backbone.pt

python -m snn_ssl_wisdm.scripts.linear_probe_snn --config snn_ssl_wisdm/configs/default.yaml --case case3 --pretrained outputs/simclr_pretrain_snn/best_backbone.pt

python -m snn_ssl_wisdm.scripts.evaluate --config snn_ssl_wisdm/configs/default.yaml --run outputs/case1_random_frozen

python -m snn_ssl_wisdm.scripts.plot_metrics --outputs outputs
```

## Smoke test

```bash
bash snn_ssl_wisdm/scripts/run_all.sh --smoke
```

## Full run

```bash
bash snn_ssl_wisdm/scripts/run_all.sh
```

## Notes

- Default sensor: **watch + accel**, 20 Hz, 10 s windows (200 samples), stride 5 s.
- Set `model.in_channels: 6` in YAML to concatenate **watch accel + watch gyro** (gyro linearly interpolated onto accel timestamps).
- `model.feature_dim` must equal `base_channels * 8` for the default `[2,2,2,2]` ResNet layout (64 → 512).
- After SimCLR, probe outputs are written to `outputs/case2_simclr_frozen/` and `outputs/case3_simclr_frozen_mlp/`.
