# Mooncake trace workload

Turns the Mooncake FAST'25 toolagent trace into tasks in the same format as
`generate_tasks`, so the runner, schedulers, sweep and summarize work unchanged.

| File | What it is |
|---|---|
| `toolagent_trace.jsonl` | The trace (from `setup/download_trace.sh`). 23,608 requests, 0–3537 s. Fields: `timestamp` (ms), `input_length`, `output_length`, `hash_ids` (one per 512-token block, prefix-chained). |
| `mooncake.py` | The generator, plus a dry run that prints the stats. |
| `words.txt` | 1024 words. Each one is a single Llama 3 token after a space. |
| `check_words.py` | Checks `words.txt` against the running llama-server's `/tokenize`. |

The workloads are `workload/tasks/mooncake_w{1,2,3}.json`, one per 900 s window.

## Rules

1. **Window.** Take the requests with `start_s <= timestamp < end_s`.
2. **One request = one task.** Families are never merged into multi-step tasks.
3. **step0 prompt.** Concatenate `B(hash_id)` for the request's blocks. `B(id)` is 128 words drawn from `words.txt` with RNG seed `id`. The input is scaled by 1/4: `ceil(input_length / 4)` tokens, with the last block cut short. Equal hash IDs give equal text, so prefixes shared in the trace are shared token-for-token. The prompt contains no task ID.
4. **step0 output** = `output_length`, not scaled.
5. **Follow-ups** come from `task_types` (the agent_3step follow-ups): tool 0.5–2 s, LLM +300/128, tool 1 s, LLM +300/256, with `carry_context` on both LLM steps. step0 has no `carry_context`.
6. **Arrival** = `timestamp/1000 - start_s + U[0, 3)`. The trace is quantized to 3 s.
7. **Fit filter.** Drop any task whose `estimate_peak_context` is over 0.95 × `fit_ctx_per_slot` (6144). That is about 90% kept.
8. **Families.** A template is a hash shared by at least 100 requests in the whole trace. There are three template prefixes: `[0]`, `[46..57]` and `[74..78]`. A request's family is its first non-template hash, meaning requests with the same template and the same next block.
9. **Load control.** Shuffle the window's families with `seed`, then add whole families until there are at least `arrival.num_tasks` tasks. This can overshoot by one family. A smaller N gives a subset of a larger N. Each task's jitter and tool delays come from its own RNG (`seed:line`), so a task is identical in every set.
10. **Task fields.** `id` is `task-<trace line>`. `family` is the family ID (for analysis only).

## Theoretical prefix hit

Go through the chosen tasks in arrival order. A step0 block counts as a hit if an
earlier task already sent the same hash. This assumes an unbounded cache, and
counts only step0 sharing across tasks, not the follow-ups reusing their own
step0. With 1/10 of a window sampled by family, it is about 58–66% (uniform
sampling gives about 51–56%). Without the fit filter it would be 44–57%, because
the long requests the filter drops are mostly unique blocks.

## Commands

```bash
python -m workload.traces.mooncake workload/tasks/mooncake_w1.json --fraction 0.1   # dry run
python -m workload.traces.check_words --url http://localhost:8080                  # needs a running server
python -m scripts.sweep scripts/sweeps/mooncake_calib.json --dry-run               # show what will run
tmux new -s sweep 'python -m scripts.sweep scripts/sweeps/mooncake_calib.json'
```

## Progress

Done:

- [x] Generator and workloads `mooncake_w{1,2,3}.json`.
- [x] `check_words` on the 3080 server (24576/4): 1024 words, 200 seams, 50 prompts, OK.
- [x] Dry run of the calibration set (W1, N=550):
  - 5462 requests in the window, 4866 fit at 6144 tok/slot (89.1%).
  - 550 tasks in 488 families: 39 have more than one member, the largest has 7.
  - Mean rate 0.611 tasks/s, 27–49 tasks per minute.
  - step0 prompt p50 1547 / p99 3980, output p50 25 / p99 655, peak context p50 2658 / max 5818.
  - Theoretical step0 prefix hit: 63.9% of blocks, 67.1% of tokens.

To do:

- [ ] c=1 check: run W1 with `--concurrency 1 --num-tasks 20` and compare step0
      `cached_tokens` with the dry run's theoretical hit. Expect a bit less than
      theory, because there are 4 slots instead of an unbounded cache, plus about
      41 template/salt tokens per request.
- [ ] Calibration run `mooncake_calib`. X = `tasks/s` of `w1_fifo_c4_n550`,
      expected about 0.33–0.46. Also check there are no failures or timeouts.
- [ ] Set the load levels N = 0.5X, 0.75X and 0.95X × 900. Write the FIFO plan for
      W1/W2/W3 × 3 loads, all at 24576/4 with `CACHE_RAM=0`.
- [ ] Run it, then summarize it into `sweep.csv`.


## Phase 2 TODO

- If the policy depends on tool waits (context retention, prioritizing ready agent
  steps), add a control run with longer tool delays. Use W1 at the middle load
  (0.75X), with both tool delays set to `{"dist": "fixed", "value": 5.0}`, and run
  FIFO against the policy. Keep the same N.
  - Phase 1 showed tool delay barely changes throughput, but follow-up KV reuse
    drops from 41% (0.5 s) to 6% (5 s).
  - The trace's own delays are synthetic (0.5–2 s and 1 s), so short delays alone
    could misjudge the policy.
  - This needs only a copy of the workload JSON with the delays changed, no code.
