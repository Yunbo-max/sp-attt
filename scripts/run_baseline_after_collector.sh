#!/usr/bin/env bash
set -euo pipefail

collector_pid="${1:?usage: $0 COLLECTOR_PID}"
while kill -0 "${collector_pid}" 2>/dev/null; do
  sleep 300
done

exec python3 scripts/run_alfworld_matrix.py \
  --methods react ttt attt --seeds 0 --episodes 140 --split valid_seen \
  --max-new-tokens 16 --output-dir results/alfworld_baselines --resume
