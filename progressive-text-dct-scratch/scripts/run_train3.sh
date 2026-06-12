#!/bin/bash
# task3 Phase 2: train A (enc_dct), B (enc_latent), C (emb_dct) with Muon.
# Usage: scripts/run_train3.sh [STEPS]   (default 60000, set in this script)
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
STEPS=${1:-60000}
mkdir -p runs results
for v in enc_dct enc_latent emb_dct; do
  echo "=== [$(date +%H:%M:%S)] training $v (muon, $STEPS steps) ==="
  PYTHONPATH=src $PY src/train.py --config config.yaml --variant "$v" \
    --opt muon --steps "$STEPS" --resume 2>&1 | tee -a "runs/train_${v}.log"
  cp "runs/${v}/train_log.csv" "results/task3_train_${v}.csv" || true
done
echo "=== [$(date +%H:%M:%S)] task3 training done ==="
