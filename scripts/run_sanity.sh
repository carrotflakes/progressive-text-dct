#!/bin/bash
# Phase 1 sanity check: overfit 100 samples at K=K_max, expect ~100% token acc
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
.venv/bin/python src/train.py --config config.yaml --variant main --sanity "$@"
