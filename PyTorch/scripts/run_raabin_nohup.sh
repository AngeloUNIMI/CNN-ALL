#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/raabin_wbc.yaml}"
LOG="${2:-raabin_pytorch.log}"

nohup cnn-all run --config "$CONFIG" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "${LOG}.pid"
echo "Started PID $PID"
echo "Log: $LOG"
echo "Monitor with: tail -f $LOG"
