# Metric definitions (`analysis/summarize.py`)

Every number `summarize.py` reports, how it is computed, where its inputs come from, and what it does not mean.
Column names refer to `results/SCHEMA.md`.

## Where the inputs come from

All times use one monotonic clock in the client process: seconds since the run started (`RunClock`, reset just before the first task is released).

| column | set where | meaning |
|---|---|---|
| request `t_arrive` | `scheduler/base.py`, when `admit()` is called | the step became ready: task arrival, previous LLM step done, or tool done |
| `t_dispatch` | `scheduler/base.py`, when the scheduler grants a slot | the HTTP request is sent right after this |
| `t_first_token` | `workload/llama_client.py`, first streamed chunk with non-empty content | client receive time |
| request `t_finish` | `llama_client.py`, after the stream ends | client receive time |
| `prompt_tokens` | server `usage.prompt_tokens` | whole prompt incl. chat template and salt, **including** cached tokens |
| `cached_tokens` | server `usage.prompt_tokens_details.cached_tokens` (`timings.cache_n`) | prompt tokens whose KV was reused instead of recomputed |
| `output_tokens` | server `usage.completion_tokens` | = `max_output_tokens`, because the client sets `ignore_eos` |
| `prefill_ms` / `decode_ms` | server `timings.prompt_ms` / `timings.predicted_ms` | server compute time for uncached prompt tokens / for generation |
| task `t_arrive` | `workload/runner.py`, start of `run_task` | after the Poisson arrival wait |
| task `t_finish` | `runner.py`, after the last step or the failing step | |
| `total_queue_s`, `total_tool_s` | `runner.py` | sum of the task's request `queue_s`; measured tool sleep time |
| memory columns | `workload/sampler.py`, every `--sample-interval-s` (0.5 s) | see "Memory" below |

## Request metrics (one value per `ok` request)

| metric | formula | notes |
|---|---|---|
| queue | `t_dispatch − t_arrive` | time waiting in **our** scheduler only |
| TTFT incl. queue | `t_first_token − t_arrive` | what the agent waits for until output starts |
| TTFT server | `t_first_token − t_dispatch` | prefill of the uncached prompt, plus waiting for the server's batch loop while other slots decode, plus localhost HTTP (~1 ms) |
| end-to-end | `t_finish − t_arrive` | |
| decode ms/token | `decode_ms / output_tokens` | same as llama.cpp's own `predicted_per_token_ms`. Under concurrency it includes slowdown from sharing batches, which is intended. The first token is produced by the prefill, so the true per-step time is `n/(n−1)` times larger (< 1 % at 128 tokens). |

Each is reported as mean, P50, P95, P99 and max.

## Task metrics (one value per `ok` task)

| metric | formula | notes |
|---|---|---|
| latency | `t_finish − t_arrive` | **includes simulated tool time** |
| latency − tool | `t_finish − t_arrive − total_tool_s` | tool delays are fixed by the workload seed, not by the scheduler; use this to compare schedulers |
| queue | `total_queue_s` (mean shown) | |

Reported overall and per `task_type`.

## By step (context growth)

`ok` requests grouped by `<task_type>/step<step_idx>`; `step_idx` counts tool steps, so agent LLM steps are `step0`, `step2`, `step4`. Each group reports prompt tokens, uncached tokens, prefill, server TTFT and decode ms/token.

## Throughput

`makespan = max(task t_finish) − min(task t_arrive)` over **all** tasks, then

- tasks/s = `#ok tasks / makespan`, requests/s = `#ok requests / makespan`
- output tokens/s = `Σ output_tokens (ok requests) / makespan`
- uncached prompt tokens/s = `Σ (prompt_tokens − cached_tokens) (ok requests) / makespan`: prompt tokens the server actually computed. Cache hits are excluded because they cost no prefill.

**Caveat:** when the server keeps up with arrivals, these just equal the arrival rate. In the first `example_mix` run the last task arrived at 112.6 s and the run ended at 114.5 s, so tasks/s ≈ 50 / 114.5 says nothing about capacity. To measure capacity, raise `--rate` until the queue grows.

## Prompt cache

- all requests: `Σ cached_tokens / Σ prompt_tokens`
- follow-up steps: same sum over requests with `step_idx > 0`, i.e. the 2nd and later LLM steps of agent tasks. This shows whether an agent's carried context was still in a slot when its next step arrived.

First steps also get up to 41 cached tokens (measured in #11) from the shared chat template and salt, so the all-requests ratio is slightly inflated. The follow-up ratio is the one to compare.

The ratio shows that reuse happened, not what it saved. For that, prefill is reported separately for first steps (`step_idx = 0`) and follow-up steps (`step_idx > 0`):

| metric | formula | notes |
|---|---|---|
| prefill | `prefill_ms` (server `timings.prompt_ms`) | server time to process the uncached prompt tokens. Unlike server TTFT it excludes waiting for the batch loop, so it isolates the cost reuse saves. |
| uncached tokens | `prompt_tokens − cached_tokens` | the tokens that prefill had to compute |

To answer "does reuse matter", run the same configuration with and without `--no-cache-prompt` and compare follow-up prefill (and TTFT). The summary prints `prompt cache on/OFF` so the two runs cannot be confused.

### Avoidable re-prefill (`analysis/reprefill.py`)

Bounds how much GPU time re-processing displaced context costs. For each follow-up, `new = max(prompt − (prev_prompt + prev_output), 0)` tokens are newly appended; the rest of the uncached tokens were already computed once. Their share of `prefill_ms` is the avoidable re-prefill. Totals are reported as a share of makespan and of GPU-busy time (makespan × mean `gpu_util_pct`); both are upper bounds because prefill can overlap other slots' decoding.

    python -m analysis.reprefill results/mooncake_baseline/w1_fifo_c4_l{75,75_cram8g,95,95_cram8g}

## Memory

| metric | formula | notes |
|---|---|---|
| GPU mem peak | `max(gpu_mem_used_mb)` | NVML **device-wide** used memory, including other processes (under WSL, Windows shares the GPU). Flat, because llama.cpp allocates the whole KV cache at startup. |
| GPU util mean | `mean(gpu_util_pct)` | |
| server RSS peak | `max(server_rss_mb)` | grows with `--cache-ram`; restart the server before each run |
| slot tokens peak | `max(slot_tokens_total)` | `Σ n_prompt_tokens` over `/slots`: KV **occupied** (idle slots keep old prompts), not KV needed |
| deferred max | `max(server_requests_deferred)` | > 0 means requests queued inside llama-server, invisible to `queue`; rerun |
| active slot tokens | `Σ prompt_tokens` of requests with `t_dispatch ≤ t < t_finish` at each sample | KV used by requests in flight |
| idle slot tokens (peak, mean) | `max(slot_tokens_total − active, 0)` | KV held by idle slots, e.g. during tool calls. Assumes an idle slot's `n_prompt_tokens` in `/slots` stays non-zero; check once on the pinned llama.cpp |

## Failures

| metric | counted from |
|---|---|
| requests error / timeout | `requests.csv` `status` = `error` / `timeout` |
| tasks failed / timeout | `tasks.csv` `status` = `failed` / `timeout`; a task ends at its first failed request |
| top errors | the 3 most frequent `error` messages of failed requests (first 200 characters), with counts |

### Context overflow (`analysis/overflow.py`)

llama.cpp reserves KV at startup, so too much context never runs the GPU out of memory. It fails in one of two ways, both seen in `phase1_overflow`: a prompt that does not fit the slot is rejected with HTTP 400 (`status = error`; p1536 at step 2, p2048 at step 0), and a prompt that fits but whose output does not is cut at the slot limit while still reporting `status = ok` (every p1024 task's step 4). The second is invisible to `summarize.py`; `overflow.py` counts outputs shorter than requested as truncated. With `ignore_eos` every complete response has exactly the requested length, so `overflow.py` regenerates the requested lengths from `run_config.json` and counts `output_tokens < requested` as truncated. All 46 Phase 1 runs show 0 truncated outputs.

    python -m analysis.overflow results/phase1_overflow/*/

## Comparison table (`--table`)

One row per run with the setup (scheduler, concurrency, rate, seed, server slots, context per slot, prompt cache on/off), the failure counts, and the main values above. The setup columns tell runs of a sweep apart without parsing `run_id`.

## Statistics

- Percentiles use linear interpolation between closest ranks (numpy's default): rank `r = (n − 1) · p / 100`, value `x[⌊r⌋] + (x[⌈r⌉] − x[⌊r⌋]) · (r − ⌊r⌋)`.
- **Small samples:** with `n` values, P99 lies between the two largest values once `n < 100`. One `example_mix` run has 68 requests and 50 tasks (only 9 of them agent tasks), so its P99 is effectively the maximum and one outlier moves it. Report P95, use more tasks (`--num-tasks`), or repeat with several `--seed`s.
- Only `ok` requests and tasks enter latencies and throughput. If any failed, the remaining latencies are survivor-biased, so check the counts first.
