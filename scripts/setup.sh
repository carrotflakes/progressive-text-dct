#!/bin/bash
# One-shot environment setup for a fresh machine (e.g. RunPod A6000 pod).
# Creates a venv, installs deps, pre-downloads the base model, and builds the
# tokenized data cache so the first training run starts immediately.
set -e
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
echo "=== creating venv (.venv) with $PY ==="
$PY -m venv .venv
.venv/bin/pip install -U pip wheel
.venv/bin/pip install -r requirements.txt

echo "=== pre-downloading base model + building data cache ==="
.venv/bin/python - <<'PY'
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys, os
sys.path.insert(0, "src")
from data import prepare_chunks
cfg = yaml.safe_load(open("config.yaml"))
name = cfg["model"]["base_model"]
tok = AutoTokenizer.from_pretrained(name)
AutoModelForCausalLM.from_pretrained(name)  # cache weights
prepare_chunks(cfg, tok)  # cache token chunks
print("setup complete")
PY
echo "=== done. Run: scripts/run_sanity.sh  then  scripts/run_all.sh ==="
