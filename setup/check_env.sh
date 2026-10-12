#!/usr/bin/env bash
# Print the environment fingerprint. Run on any machine used for experiments
# and save the output with results (scripts will also write it to run_config.json).
# Usage: bash setup/check_env.sh [> results/<run_id>/env.txt]
cd "$(dirname "$0")/.."
source setup/config.env

echo "date:            $(date -Iseconds)"
echo "repo_commit:     $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet 2>/dev/null || echo ' (dirty)')"
echo "os:              $(. /etc/os-release 2>/dev/null; echo "$PRETTY_NAME") / kernel $(uname -r)"
grep -qi microsoft /proc/version 2>/dev/null && echo "wsl:             yes"
echo "cpu:             $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
echo "ram_total:       $(free -g | awk '/Mem:/{print $2}') GB"
if command -v nvidia-smi >/dev/null; then
  echo "gpu:             $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
else
  echo "gpu:             none (nvidia-smi not found)"
fi
echo "cuda_toolkit:    $(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' || echo 'not found')"
echo "python:          $(python3 --version 2>&1)"
LC=third_party/llama.cpp
echo "llama_cpp:       $(git -C $LC rev-parse --short HEAD 2>/dev/null || echo 'not built') (pinned: $LLAMA_CPP_COMMIT)"
if [ -f "models/$MODEL_FILE" ]; then
  echo "model:           $MODEL_FILE sha256=$(sha256sum "models/$MODEL_FILE" | cut -c1-16)"
else
  echo "model:           $MODEL_FILE (not downloaded)"
fi
echo "server_defaults: CTX=$CTX PARALLEL=$PARALLEL NGL=$NGL"
