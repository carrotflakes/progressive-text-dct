#!/bin/bash
# Phase 2: train all four variants sequentially.
# Usage: scripts/run_train.sh [STEPS]
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
STEPS=${1:-}
EXTRA=()
if [ -n "$STEPS" ]; then EXTRA=(--steps "$STEPS"); fi
for v in main enc0 b2 b3; do
  echo "=== training $v ==="
  .venv/bin/python src/train.py --config config.yaml --variant "$v" "${EXTRA[@]}"
done
