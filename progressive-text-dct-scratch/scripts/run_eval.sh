#!/bin/bash
# Phase 3: evaluate all variants + B1, spectrum analysis, qualitative samples
set -e
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
PYTHONPATH=src ../.venv/bin/python src/evaluate.py --config config.yaml "$@"
