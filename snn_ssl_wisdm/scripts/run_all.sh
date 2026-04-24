#!/usr/bin/env bash
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

pip install -q -r snn_ssl_wisdm/requirements_unity.txt 2>/dev/null || pip install -r snn_ssl_wisdm/requirements_unity.txt

if [[ "$SMOKE" -eq 1 ]]; then
  echo "== Smoke: prepare (4 subjects, max 1000 windows)"
  python -m snn_ssl_wisdm.scripts.prepare_wisdm \
    --config snn_ssl_wisdm/configs/default.yaml \
    --limit_subjects 4 \
    --max_windows 1000
  EP_L=2
  EP_P=2
else
  echo "== Full: prepare (all subjects)"
  python -m snn_ssl_wisdm.scripts.prepare_wisdm --config snn_ssl_wisdm/configs/default.yaml
  EP_L=50
  EP_P=100
fi

echo "== Case 1 linear probe"
python -m snn_ssl_wisdm.scripts.linear_probe_snn \
  --config snn_ssl_wisdm/configs/default.yaml \
  --case case1 \
  --epochs "$EP_L"

echo "== AugPred pretrain"
python -m snn_ssl_wisdm.scripts.pretrain_augpred_snn \
  --config snn_ssl_wisdm/configs/default.yaml \
  --epochs "$EP_P"

echo "== Case 2 linear probe"
python -m snn_ssl_wisdm.scripts.linear_probe_snn \
  --config snn_ssl_wisdm/configs/default.yaml \
  --case case2 \
  --pretrained outputs/augpred_pretrain_snn/best_backbone.pt \
  --epochs "$EP_L"

echo "== Evaluate"
python -m snn_ssl_wisdm.scripts.evaluate --config snn_ssl_wisdm/configs/default.yaml --run outputs/case1_random_frozen || true
python -m snn_ssl_wisdm.scripts.evaluate --config snn_ssl_wisdm/configs/default.yaml --run outputs/case2_augpred_frozen || true

echo "== Plot aggregate"
python -m snn_ssl_wisdm.scripts.plot_metrics --outputs outputs

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
c2 = loadm("case2_augpred_frozen")
print("\n=== Final comparison (test set) ===")
if c1 and c2:
    def row(tag, d):
        return f"{tag:8}  acc={d['accuracy']:.4f}  macro_f1={d['macro_f1']:.4f}  weighted_f1={d['weighted_f1']:.4f}  kappa={d['cohen_kappa']:.4f}"
    print(row("Case1", c1))
    print(row("Case2", c2))
    print(f"Delta    acc={c2['accuracy']-c1['accuracy']:+.4f}  macro_f1={c2['macro_f1']-c1['macro_f1']:+.4f}  "
          f"weighted_f1={c2['weighted_f1']-c1['weighted_f1']:+.4f}  kappa={c2['cohen_kappa']-c1['cohen_kappa']:+.4f}")
else:
    print("Missing metrics.json for one or both cases.")
PY

echo "== Done"
