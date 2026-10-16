#!/usr/bin/env bash
# Start llama-server with the pinned config. Override per run, e.g.:
#   CTX=16384 PARALLEL=8 bash setup/start_server.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source setup/config.env
# Note: each slot gets CTX / PARALLEL tokens of context.
exec third_party/llama.cpp/build/bin/llama-server \
  -m "models/$MODEL_FILE" \
  -c "$CTX" --parallel "$PARALLEL" -ngl "$NGL" \
  --host 0.0.0.0 --port "$SERVER_PORT" --metrics
