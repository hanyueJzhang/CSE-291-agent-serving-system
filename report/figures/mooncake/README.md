# Mooncake baseline figures

Generate both figures and the exact-value tables from the repository root:

```bash
python -m analysis.reporting.cli analysis/report_specs/mooncake.json
```

This writes:

- `load_scaling.{png,pdf}`
- `host_cache_comparison.{png,pdf}`
- `summary.md`
- `summary_load.csv`
- `summary_host_cache.csv`

The legacy manifest in `analysis/report_specs/mooncake_runs.json` records the
load level, trace window, and host-cache size because these runs predate the
explicit `dimensions` metadata added to the sweep runner.

## Load scaling

The calibration run measured a saturated FIFO capacity of approximately
0.330 tasks/s. L50, L75, and L95 offer 50%, 75%, and 95% of that capacity. The
figure reports the mean over three non-overlapping Mooncake trace windows and
shades the full W1–W3 range.

As load rises from L50 to L95, mean task-latency P95 increases from 18.6 s to
42.8 s. Step-0 shared-prefix reuse falls from 42.5% to 28.3%, and same-task
follow-up reuse falls from 73.5% to 33.0%. Follow-up prefill P50 consequently
rises from 97 ms to 307 ms. This supports the interpretation that increased
interleaving replaces useful prompt state and creates more recomputation and
queueing near capacity.

## Host-cache comparison

The host-cache experiment was run only on W1, so it is shown separately and
must not be presented as a three-window average. At L95, enabling an 8 GiB
host prompt cache lowers task-latency P95 from 38.5 s to 28.7 s and raises
follow-up reuse from 35.2% to 80.1%. Peak server RSS rises from about 0.8 GiB
to 8.6 GiB.

This demonstrates a real latency/recomputation versus host-memory tradeoff.
It does not by itself prove that host cache increases maximum throughput:
these runs use fixed offered loads and all tasks already complete successfully.

## Metric definitions

- **Task-latency P95:** 95th percentile from task arrival through all model and
  tool steps to task completion, including scheduler queueing.
- **Step-0 reuse:** token-weighted cached prompt tokens divided by prompt tokens
  for the trace-derived first request. It mainly captures shared prefixes
  between tasks.
- **Follow-up reuse:** the same token-weighted ratio for steps 2 and 4. It
  measures whether a task's carried context remains reusable after tool waits.
- **Follow-up prefill P50:** median server time spent processing the uncached
  portion of steps 2 and 4; it excludes scheduler waiting, tool delay, and
  output decoding.
