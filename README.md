# CSE 291 — Memory-Aware Agent Edge Serving

Course project (Option 1). We run a local LLM runtime (llama.cpp `llama-server`) on a memory-bounded machine, drive it with multi-step agent workloads, and study admission/scheduling policies.

- **Phase 1:** FIFO + fixed-concurrency baseline, mixed workloads, request- and task-level metrics, bottleneck analysis.
- **Phase 2:** one targeted memory-aware scheduling policy, compared against FIFO and a tuned fixed-concurrency baseline.

## Pinned setup

| item | value |
|---|---|
| llama.cpp commit | TODO (see `setup/config.env`) |
| model | TODO (draft: Qwen2.5-3B-Instruct) |
| quantization | TODO (draft: Q4_K_M) |
| dev/run machine | ROG laptop, RTX 4080 Laptop (12 GB), WSL2 Ubuntu |

## Layout

```
setup/      build llama.cpp, download model, start server (pinned in config.env)
workload/   agent task definitions (tasks/*.json) and the workload runner
scheduler/  FIFO baseline, later the memory-aware policy
scripts/    experiment drivers (sweep concurrency, context length, ...)
analysis/   plotting and analysis
results/    measurement data (format: results/SCHEMA.md)
report/     write-ups
models/     GGUF files (gitignored)
```

## Quick start (WSL2 / Linux with NVIDIA GPU)

Prereqs: NVIDIA driver on Windows (do not install a Linux driver inside WSL), then in Ubuntu: CUDA Toolkit, `cmake`, `git`, `build-essential`, Python 3.10+. Check `nvidia-smi` works.

```bash
git clone <this repo> && cd <repo>
bash setup/build_llama.sh          # add --cpu for a CPU-only build
bash setup/download_model.sh
bash setup/start_server.sh         # override e.g. CTX=16384 PARALLEL=8 bash setup/start_server.sh

# sanity check (another terminal)
curl localhost:8080/health
curl localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hi"}]}'
```

## Reproducing results

TODO once the runner exists: one command per experiment, e.g. `python scripts/run_experiment.py --workload workload/tasks/example_mix.json --scheduler fifo --concurrency 4`.

## Team workflow

See [CONTRIBUTING.md](CONTRIBUTING.md).
