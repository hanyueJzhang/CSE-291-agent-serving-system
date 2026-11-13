#!/usr/bin/env bash
# Aggregate decode throughput vs batch size on the pinned model (Phase 2 check:
# does a bigger decode batch pay on the 3080?). Stop llama-server first, it
# holds the GPU memory:  tmux kill-session -t llm
#
#   bash scripts/batched_bench.sh | tee results/batched_bench.txt
#
# Each of B sequences gets its own PP-token prompt and generates TG tokens, in
# one unified KV buffer. S_TG = B*TG / T_TG is the aggregate decode speed; W1's
# median task peak context is ~2.7K tokens, so PP=2048 is representative.
set -euo pipefail
cd "$(dirname "$0")/.."
source setup/config.env
exec third_party/llama.cpp/build/bin/llama-batched-bench \
  -m "models/$MODEL_FILE" -ngl "$NGL" -c 24576 -b 2048 -ub 512 -kvu \
  -npp 2048 -ntg 128 -npl 1,2,4,6,8
