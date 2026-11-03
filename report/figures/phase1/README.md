# Phase 1 figure guide

## Reproduce the complete figure set

Run the reporting CLI from the repository root:

```bash
python -m analysis.reporting.cli analysis/report_specs/phase1.json
```

The report specification selects the result runs, validates the controlled
variables, and generates both PNG and PDF versions of all five figures in this
directory:

```text
capacity.{png,pdf}
concurrency.{png,pdf}
context_growth.{png,pdf}
tool_delay.{png,pdf}
host_cache.{png,pdf}
```

The legacy manifest at `analysis/report_specs/phase1_runs.json` supplies
experiment dimensions that the original Phase 1 runs did not record. Future
runs preserve these dimensions directly in `run_config.json`, `summary.json`,
and `sweep.csv`, so their report specifications do not need filename parsing.

These four main figures answer different baseline questions:

| Figure | Question |
|---|---|
| [Capacity](capacity.pdf) | How much offered work can the fixed FIFO system sustain? |
| [Concurrency](concurrency.pdf) | How many model requests should FIFO admit at once? |
| [Context growth](context_growth.pdf) | How does agent context size affect KV pressure and completed work? |
| [Tool delay](tool_delay.pdf) | Does an agent's context survive while it waits for tools? |

## 1. Capacity and saturation

**Why we need it:** Establish the system's sustainable task rate and identify the dominant bottleneck before designing a new scheduler.

**What it shows:** Completed throughput follows offered load at low rates, but plateaus near 1 task/s. Beyond that point, tasks arrive faster than the server completes them, so FIFO waiting grows rapidly. At 4 offered tasks/s, task P95 latency is about 146 s and accumulated queue-time P95 is about 135 s.

**Conclusion:** The baseline is capacity-limited near 1 task/s, and overload latency is dominated by waiting in our FIFO scheduler. Admission control is therefore relevant.

## 2. Scheduler concurrency

**Why we need it:** Select a tuned fixed-concurrency FIFO baseline. Phase 2 should not claim an improvement by comparing against an underconfigured baseline.

**What it shows:** On a four-slot llama.cpp server, increasing scheduler concurrency from 1 to 4 raises throughput and sharply reduces task P95 latency. Increasing it from 4 to 8 provides almost no additional benefit; the black X indicates that excess requests instead queue internally in llama.cpp.

**Conclusion:** Use scheduler concurrency 4 with four server slots. Lower concurrency underuses the server, while higher concurrency moves queueing outside our scheduler's control.

## 3. Initial context size

**Why we need it:** Fixed request counts do not capture memory pressure. Four short-context requests and four long-context requests have very different KV-cache footprints.

**What it shows:** Only initial prompt size changes (512, 1,024, 2,048, or 4,096 tokens); the agent steps, appended tokens, arrival rate, scheduler, and server configuration remain controlled. From the smallest to largest context, peak KV-slot occupancy rises from about 29% to 87%, follow-up prefill P50 rises from 221 ms to 1,157 ms, throughput falls from 0.323 to 0.201 tasks/s, and task P95 latency rises from about 22 s to 231 s.

**Conclusion:** Concurrency alone is not enough for admission decisions. Larger retained contexts consume more KV-slot capacity, reduce completed work, and increase tail latency, motivating context- or memory-aware admission.

## 4. Tool wait duration

**Why we need it:** Agent tasks release the model while using tools, then return with a dependent follow-up request. Phase 1 must determine whether context remains reusable during these idle periods.

**What it shows:** Only the fixed delay of each tool call changes (0.5, 2, 5, or 10 s). As the delay grows, follow-up prompt tokens served from cache fall from about 41% to 4%, while follow-up prefill P50 rises from about 106 ms to 304 ms.

**Conclusion:** Longer tool waits make an agent's previous context less likely to survive in the server slots. Prioritizing ready follow-up steps or otherwise considering context reuse may reduce repeated prompt processing.

## Supporting result: host prompt cache

[Host prompt cache](host_cache.pdf) is supporting or appendix evidence rather than a core figure. Additional host cache improves follow-up reuse and prefill time at moderate load, but the measured end-to-end task throughput and P95 latency do not materially improve. An 8 GiB cache also raises peak server RSS from about 0.8 GiB to 8.8 GiB. This result helps justify focusing Phase 2 on admission and scheduling instead of a large fixed host-cache allocation.

## Supporting result: avoidable re-prefill

[Re-prefill](reprefill.pdf) is generated separately because it aggregates seeds 42-44:

```bash
python -m analysis.reporting.reprefill_figure
```

Left: the share of GPU-busy time spent re-prefilling history that a follow-up had already processed (`analysis/reprefill.py`), for W1 at 0.50X/0.75X/0.95X and the output-scaled control, with the host cache off and at 8 GiB. Right: task P95 for the same runs. Bars are seed means, whiskers min-max, dots individual seeds; W1 0.75X with the cache is seed 42 only. Re-prefill grows with load (4% to 19%) and roughly doubles when outputs are scaled with inputs (37%); the cache cuts it to 1-6%. Tail latency varies much more across seeds than re-prefill does, because near saturation small differences in sampled load are amplified by queueing.

## Overall Phase 1 finding

The fixed FIFO baseline works best when scheduler concurrency matches server parallelism, but it still degrades when offered load exceeds capacity, when agent contexts become large, or when tool waits prevent follow-up context reuse. These measurements motivate a Phase 2 policy that considers current load, estimated context footprint, and whether a ready step can reuse retained context.

The numerical results are specific to the measured workload, model, hardware, and llama.cpp configuration; they should not be treated as universal capacity limits.
