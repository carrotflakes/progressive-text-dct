#!/bin/bash
# Phase 3: evaluate all variants + B1 on the test set
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
.venv/bin/python src/evaluate.py --config config.yaml "$@"
