# CSE 291 — Memory-Aware Agent Edge Serving

Course project (Option 1). We run a local LLM runtime (llama.cpp `llama-server`) on a memory-bounded machine, drive it with multi-step agent workloads, and study admission/scheduling policies.

- **Phase 1:** FIFO + fixed-concurrency baseline, mixed workloads, request- and task-level metrics, bottleneck analysis.
- **Phase 2:** one targeted memory-aware scheduling policy, compared against FIFO and a tuned fixed-concurrency baseline.

## Pinned setup

| item | value |
|---|---|
| llama.cpp commit | `972b5fc` (see `setup/config.env`) |
| model | Llama-3.1-8B-Instruct (GGUF from `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`) |
| quantization | Q4_K_M |
| run machine | Desktop, RTX 3080 (10 GB), Ryzen 7 7700X, 32 GB RAM, Win11 + WSL2 Ubuntu 24.04, CUDA 13.3 |

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
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash setup/build_llama.sh          # add --cpu for a CPU-only build
bash setup/download_model.sh
bash setup/start_server.sh         # override e.g. CTX=16384 PARALLEL=8 CACHE_RAM=0 bash setup/start_server.sh

# sanity check (another terminal)
curl localhost:8080/health
curl localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hi"}]}'
```

## Phase 1 report figures

Generate all comparative Phase 1 figures from the checked-in summaries:

```bash
python -m analysis.reporting.cli analysis/report_specs/phase1.json
```

PNG and PDF outputs are written to `report/figures/phase1/`. The report loader
uses `summary.json` for measurements and `analysis/report_specs/phase1_runs.json`
only to fill dimensions that older runs did not record. New sweep plans should
put explicit scalar metadata in their `dimensions` objects; the sweep runner
preserves it in `run_config.json`, `summary.json`, and `dim_*` columns of
`sweep.csv`.

## Keeping environments consistent

Only the machine that produces reported numbers must match exactly; dev machines just need to run the code.

- **Pinned in git:** llama.cpp commit, model file + quant, server defaults (`setup/config.env`), Python deps (`requirements.txt`, later an exact `requirements.lock`).
- **Recorded, not pinned:** OS, NVIDIA driver, CUDA toolkit, GPU, RAM. Run `bash setup/check_env.sh` and keep its output with each result set.
- Changing anything pinned requires a PR the whole team agrees on, since it invalidates earlier results.

## Reproducing results

One command per run, against the llama-server that is already running:

```bash
python -m scripts.run_experiment --workload workload/tasks/example_mix.json --scheduler fifo --concurrency 4
# sweep points without new JSON files: --num-tasks N --rate R --seed S
# prompt-cache ablation: --no-cache-prompt
```

Then summarize the run (latency percentiles, throughput, prompt-cache reuse, memory peaks; definitions in [analysis/METRICS.md](analysis/METRICS.md)):

```bash
python -m analysis.summarize results/<run_id> --plots          # writes summary.json + PNGs into the run folder
python -m analysis.summarize results/example_mix_fifo_c* --table results/sweep.csv   # one row per run
```

For a set of runs, write a plan (e.g. `scripts/sweeps/phase1_baseline.json`) and let the sweep restart llama-server before each run, so no run inherits another's cache or memory:

```bash
python -m scripts.sweep scripts/sweeps/phase1_baseline.json --dry-run   # show what will run
tmux new -s sweep 'python -m scripts.sweep scripts/sweeps/phase1_baseline.json'
# results/phase1_baseline/<run_id>/ per run, results/phase1_baseline/sweep.csv for all;
# rerun the same command to resume after an interruption
```

## Team workflow

See [CONTRIBUTING.md](CONTRIBUTING.md).
