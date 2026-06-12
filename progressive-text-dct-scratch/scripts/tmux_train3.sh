#!/bin/bash
# Launch task3 training inside tmux. Attach: tmux attach -t dct-task3
cd "$(dirname "$0")/.."
SESSION=dct-task3
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session $SESSION already exists; attach with: tmux attach -t $SESSION"
  exit 1
fi
tmux new-session -d -s "$SESSION" "bash scripts/run_train3.sh $1 2>&1 | tee runs/train_task3.log"
echo "started tmux session '$SESSION' (logs: runs/train_task3.log)"
