#!/bin/bash
# Phase 3: evaluate all variants + B1, spectrum analysis, qualitative samples
set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."
PYTHONPATH=src ../.venv/bin/python src/evaluate.py --config config.yaml "$@"
