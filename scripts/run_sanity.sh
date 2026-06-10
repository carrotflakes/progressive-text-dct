#!/bin/bash
# Phase 1 sanity check: overfit 100 samples at K=K_max, expect ~100% token acc
set -e
cd "$(dirname "$0")/.."
.venv/bin/python src/train.py --config config.yaml --variant main --sanity "$@"
