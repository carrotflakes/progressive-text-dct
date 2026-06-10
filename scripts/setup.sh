#!/bin/bash
# One-shot environment setup for a fresh machine (e.g. RunPod GPU pod).
# Reuses the machine's pre-installed (CUDA-matched) torch via a
# --system-site-packages venv, installs the remaining deps, pre-downloads the
# base model, and builds the tokenized data cache.
set -e
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
echo "=== creating venv (.venv, --system-site-packages) with $PY ==="
$PY -m venv --system-site-packages .venv
.venv/bin/pip install -U pip wheel

# Reuse pre-installed torch if present (RunPod pods ship a working one). Only
# install torch from PyPI if the machine genuinely has none.
if .venv/bin/python -c "import torch" 2>/dev/null; then
  echo "=== reusing pre-installed torch: $(.venv/bin/python -c 'import torch; print(torch.__version__)') ==="
else
  echo "=== no torch found; installing default torch wheel (check CUDA match!) ==="
  .venv/bin/pip install torch
fi

echo "=== installing remaining deps (requirements.txt, no torch) ==="
.venv/bin/pip install -r requirements.txt

echo "=== verifying torch + CUDA ==="
.venv/bin/python -c "import torch; print('torch', torch.__version__, '| cuda build', torch.version.cuda, '| available', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA not available — check driver/torch match'"

echo "=== pre-downloading base model + building data cache ==="
.venv/bin/python - <<'PY'
import yaml, sys
from transformers import AutoTokenizer, AutoModelForCausalLM
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
