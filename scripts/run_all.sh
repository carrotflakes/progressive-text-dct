#!/bin/bash
# End-to-end Phase 2 + Phase 3: train all four variants, then evaluate.
# Usage: scripts/run_all.sh [STEPS]   (STEPS overrides config train.steps)
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
mkdir -p runs results

STEPS=${1:-}
EXTRA=()
if [ -n "$STEPS" ]; then EXTRA=(--steps "$STEPS"); fi

for v in main enc0 b2 b3; do
  echo "=== [$(date +%H:%M:%S)] training $v ==="
  .venv/bin/python src/train.py --config config.yaml --variant "$v" "${EXTRA[@]}" \
      2>&1 | tee "runs/train_${v}.log"
  cp "runs/${v}/train_log.csv" "results/train_${v}.csv" 2>/dev/null || true
done

echo "=== [$(date +%H:%M:%S)] evaluating ==="
.venv/bin/python src/evaluate.py --config config.yaml 2>&1 | tee runs/eval.log
echo "=== [$(date +%H:%M:%S)] all done. See results/ ==="
