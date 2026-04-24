# snn_ssl_wisdm

Integration pipeline: **WISDM** raw sensor windows → **spiking 1D ResNet** (SpikingJelly-style LIF from `SpikeGPT/src/spikingjelly`) → **linear probing** or **AugPred self-supervised pretraining** (arrow-of-time, permutation, time-warp).

## Layout

- `configs/default.yaml` — hyperparameters and paths.
- `scripts/prepare_wisdm.py` — parse `wisdm-dataset/raw/...`, sliding windows, normalization, `.pt` cache + split JSON.
- `scripts/pretrain_augpred_snn.py` — multi-task SSL; saves `outputs/augpred_pretrain_snn/best_backbone.pt`.
- `scripts/linear_probe_snn.py` — frozen backbone + trainable linear head (`case1` random init, `case2` load pretrained backbone).
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

python -m snn_ssl_wisdm.scripts.pretrain_augpred_snn --config snn_ssl_wisdm/configs/default.yaml

python -m snn_ssl_wisdm.scripts.linear_probe_snn --config snn_ssl_wisdm/configs/default.yaml --case case2 --pretrained outputs/augpred_pretrain_snn/best_backbone.pt

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
