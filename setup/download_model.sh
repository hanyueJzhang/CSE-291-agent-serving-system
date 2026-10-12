#!/usr/bin/env bash
# Download the pinned GGUF model into models/ (gitignored).
set -euo pipefail
cd "$(dirname "$0")/.."
source setup/config.env
mkdir -p models
if [ -f "models/$MODEL_FILE" ]; then echo "models/$MODEL_FILE exists"; exit 0; fi
pip install -q -r requirements.txt
hf download "$MODEL_HF_REPO" "$MODEL_FILE" --local-dir models
