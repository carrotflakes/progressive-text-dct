#!/bin/bash
# Phase 3: evaluate all variants + B1 on the test set
set -e
cd "$(dirname "$0")/.."
.venv/bin/python src/evaluate.py --config config.yaml "$@"
