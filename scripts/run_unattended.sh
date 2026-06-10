#!/bin/bash
# Unattended run for a RunPod pod: setup -> sanity -> train+eval -> push -> self-stop.
# Billing stops automatically when the job finishes (the pod removes/stops itself).
#
# Usage on the pod:
#   export GITHUB_TOKEN=ghp_xxx        # PAT with 'repo' (or contents:write) scope
#   export STOP_MODE=remove            # remove (terminate, no more billing) | stop (keep disk)
#   bash scripts/run_unattended.sh [STEPS]
#
# Safety: `set -e` means any failure aborts BEFORE the self-stop line, so a
# broken run leaves the pod ALIVE for debugging (you only pay while it lives).
# Results are pushed BEFORE stopping, so a successful run loses nothing.
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."

STEPS=${1:-}
STOP_MODE=${STOP_MODE:-remove}

# --- 1. environment + data ---
if [ ! -x .venv/bin/python ]; then
  PYTHON=${PYTHON:-python3} bash scripts/setup.sh
fi

# --- 2. sanity (fast, catches setup breakage before spending hours) ---
bash scripts/run_sanity.sh

# --- 3. train all variants + evaluate ---
bash scripts/run_all.sh "$STEPS"

# --- 4. push results back to GitHub (needs write auth) ---
if [ -n "$GITHUB_TOKEN" ]; then
  git config user.name  "${GIT_NAME:-runpod-bot}"
  git config user.email "${GIT_EMAIL:-runpod@example.com}"
  git remote set-url origin \
    "https://x-access-token:${GITHUB_TOKEN}@github.com/carrotflakes/progressive-text-dct.git"
  git add -A results report.md runs/*.log 2>/dev/null || git add -A results report.md
  git commit -m "RunPod results (steps=${STEPS:-default})" || echo "nothing to commit"
  git push origin main
  echo "results pushed."
else
  echo "WARNING: GITHUB_TOKEN unset -> results NOT pushed. Download results/ manually"
  echo "         BEFORE the pod is removed, or re-run with a token."
fi

# --- 5. self-stop to halt billing ---
if command -v runpodctl >/dev/null 2>&1 && [ -n "$RUNPOD_POD_ID" ]; then
  echo "stopping pod $RUNPOD_POD_ID (mode=$STOP_MODE) ..."
  if [ "$STOP_MODE" = "stop" ]; then
    runpodctl stop pod "$RUNPOD_POD_ID"     # keeps the disk; GPU billing stops
  else
    runpodctl remove pod "$RUNPOD_POD_ID"   # terminates; all billing stops
  fi
else
  echo "runpodctl / RUNPOD_POD_ID not available; stop the pod manually."
fi
