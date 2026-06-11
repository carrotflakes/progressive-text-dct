#!/bin/bash
# Launch Phase 2 training inside a tmux session (spec requirement).
# Usage: scripts/tmux_train.sh [STEPS]; attach with: tmux attach -t dct-scratch
cd "$(dirname "$0")/.."
SESSION=dct-scratch
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session $SESSION already exists; attach with: tmux attach -t $SESSION"
  exit 1
fi
tmux new-session -d -s "$SESSION" "bash scripts/run_train.sh $1 2>&1 | tee runs/train_all.log"
echo "started tmux session '$SESSION' (logs: runs/train_all.log)"
