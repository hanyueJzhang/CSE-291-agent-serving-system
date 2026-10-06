#!/usr/bin/env bash
# Clone llama.cpp at the pinned commit and build llama-server.
# Usage: bash setup/build_llama.sh [--cpu]   (default: CUDA build)
set -euo pipefail
cd "$(dirname "$0")/.."
source setup/config.env

DIR=third_party/llama.cpp
if [ ! -d "$DIR" ]; then
  git clone "$LLAMA_CPP_REPO" "$DIR"
fi
cd "$DIR"
git fetch --all --quiet
if [ "$LLAMA_CPP_COMMIT" != "TODO_PIN_COMMIT" ]; then
  git checkout "$LLAMA_CPP_COMMIT"
else
  echo "WARNING: LLAMA_CPP_COMMIT not pinned yet; building $(git rev-parse HEAD)"
fi

CUDA_FLAG="-DGGML_CUDA=ON"
[ "${1:-}" = "--cpu" ] && CUDA_FLAG=""
cmake -B build $CUDA_FLAG
cmake --build build --config Release -j
echo "Built llama.cpp at commit $(git rev-parse HEAD)"
