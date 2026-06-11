#!/bin/bash
# Phase 2: train all four variants sequentially (resumable).
# Usage: scripts/run_train.sh [STEPS]
# Designed to run inside tmux: scripts/tmux_train.sh
set -e
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
STEPS=${1:-}
EXTRA=(--resume)
if [ -n "$STEPS" ]; then EXTRA+=(--steps "$STEPS"); fi
mkdir -p runs results
for v in main b2 b3 b4; do
  echo "=== [$(date +%H:%M:%S)] training $v ==="
  PYTHONPATH=src $PY src/train.py --config config.yaml --variant "$v" "${EXTRA[@]}" \
    2>&1 | tee -a "runs/train_${v}.log"
  cp "runs/${v}/train_log.csv" "results/train_${v}.csv" || true
done
echo "=== [$(date +%H:%M:%S)] all variants done ==="
