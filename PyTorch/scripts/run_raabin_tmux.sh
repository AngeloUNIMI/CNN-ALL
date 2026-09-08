#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/raabin_wbc.yaml}"
SESSION="${2:-raabin-pytorch}"
LOG="${3:-raabin_pytorch.log}"

COMMAND="cnn-all run --config '$CONFIG' 2>&1 | tee '$LOG'"
tmux new-session -d -s "$SESSION" "$COMMAND"
echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
echo "Log: $LOG"
