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

## Phase 2 slot diagnostics

`requests.csv` records which llama-server slot served each request (`slot_id`,
read from the `__verbose` field that `"verbose": true` adds to the stream). Two
offline analyses use it (both also work, minus the per-slot columns, on Phase 1
runs without `slot_id`):

```bash
python -m scripts.sweep scripts/sweeps/phase2_diag.json      # W1 0.95X, c = 4 and 5, seed 42 (~40 min)
python -m analysis.slot_replay results/phase2_diag/w1_fifo_c4_l95 results/phase2_diag/w1_fifo_c5_l95
python -m analysis.slot_gaps   results/phase2_diag/w1_fifo_c4_l95 results/phase2_diag/w1_fifo_c5_l95
```

`slot_replay` replays the measured request timeline with different slot
choices (llama.cpp's prefix-then-LRU, plain LRU, dead-first, dead-first with an
oracle victim, plus a hold-threshold table) and reports how much carried agent
history each would keep. `slot_gaps` checks whether c = 5 beats c = 4 by hiding
dispatch gaps or by better slot choice.

## Phase 2 adaptive admission scheduler

`--scheduler adaptive` (`scheduler/adaptive.py`) tunes the in-flight limit
online instead of taking a fixed `--concurrency` (which becomes the starting
limit). After each window of completions it estimates how many requests waited
inside llama-server (Little's law on first token - prefill - dispatch), raises
the limit while slots go idle and work is held back, and lowers it when a
hidden server queue builds up, so it settles near "all slots busy + one
waiting" on any slot count. With `--overload-wait-s S`, ready follow-up steps
go first while some first step has waited S seconds. With `--aging-s A`, the
order outside overload is continuation-first with A-second aging (as in
`--scheduler agent`). With `--adapt-startup-trim`, until the limit first holds
a request back it is lowered to one above the peak number in flight, so a
too-high starting limit does not survive to the first burst. The limit over time is in
`run_config.json` (`scheduler_summary`, `scheduler_trace`).

```bash
python -m scripts.sweep scripts/sweeps/phase2_adaptive.json      # 13 runs, ~3.5 h; compare commands in its _comment
python -m scripts.sweep scripts/sweeps/phase2_adaptive_cf.json   # adaptive + continuation-first, 3 runs, ~45 min
python -m scripts.sweep scripts/sweeps/phase2_adaptive_trim.json # startup trim, start c5 and c10, 6 runs, ~1.5 h
```

## Phase 2 lease scheduler

`--scheduler lease` (`scheduler/lease.py`) admits whole tasks instead of single
requests: a task leases one llama-server slot from its first step to its last,
and every request is pinned to its slot with `id_slot`. `--concurrency` is the
number of slots to lease (at most the server's). While a task waits for a tool,
its slot is held for up to `--hold-max-s` seconds (default 2) if waits after
that kind of step are usually that short, so the follow-up finds its history in
place and prefills only the appended tokens; with longer tool waits the slot is
given up at once and the policy is FIFO with client-side slot choice. New tasks
are admitted in arrival order to slots whose task has finished.

The 2 s default comes from `analysis.prefill_stall`, which measures how much one
slot's prefill slows the other slots' decoding and how much an idle slot costs:

```bash
python -m analysis.prefill_stall results/phase2_diag/w1_fifo_c5_l95 results/phase2_stress/w1_fifo_c5_l115_s4*
```

`run_config.json` records the counts (`scheduler_summary`: holds, follow-ups
that found their history, live histories overwritten, `pin_mismatches` if
llama-server served a request from another slot) and each overwritten history
(`scheduler_trace`).

```bash
python -m scripts.sweep scripts/sweeps/phase2_lease_quick.json   # FIFO c = 5 vs lease, seed 42, 2 runs, ~30 min
python -m scripts.sweep scripts/sweeps/phase2_lease.json         # vs FIFO c = 5 and the 5-slot + host-cache main line, 12 runs, ~3 h
python -m scripts.sweep scripts/sweeps/phase2_lease_loads.json   # 1.15X, 0.75X and the synthetic mix, 18 runs, ~4 h
python -m scripts.sweep scripts/sweeps/phase2_lease_tools.json   # no-hold ablation and 1.5x tool waits, 9 runs, ~2.3 h
python -m scripts.sweep scripts/sweeps/phase2_lease_kvu.json     # leases on an 8-slot shared KV pool vs the main line, 12 runs, ~3 h
python -m scripts.sweep scripts/sweeps/phase2_lease_soft.json    # 8-slot pool, lease up to 8 but new tasks past 6 only after a 2 s wait, 12 runs, ~3 h
python -m scripts.sweep scripts/sweeps/phase2_lease_stress.json  # the lease configs at 1.15X overload, 12 runs, ~3.5 h
```

Compare commands are in each plan's `_comment`.

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
