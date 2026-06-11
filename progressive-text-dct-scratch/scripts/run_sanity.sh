#!/bin/bash
# Phase 1 sanity: (1) reversibility unit tests, (2) 100-sample overfit at K=64
set -e
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
echo "=== stage 1: unit tests (incl. DCT reversibility) ==="
PYTHONPATH=src $PY src/test_pipeline.py
echo "=== stage 2: 100-sample overfit (K=K_max) ==="
PYTHONPATH=src $PY src/train.py --config config.yaml --variant main --sanity "$@"
