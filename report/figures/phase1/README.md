# Phase 1 figure guide

These three figures answer different baseline questions:

| Figure | Question |
|---|---|
| [Capacity](capacity.pdf) | How much offered work can the fixed FIFO system sustain? |
| [Concurrency](concurrency.pdf) | How many model requests should FIFO admit at once? |
| [Context growth](context_growth.pdf) | How does agent context size affect KV pressure and completed work? |

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

## Overall Phase 1 finding

The fixed FIFO baseline works best when scheduler concurrency matches server parallelism, but it still degrades when offered load exceeds capacity or when agent contexts become large. These measurements motivate a Phase 2 policy that considers both current load and estimated context footprint.

The numerical results are specific to the measured workload, model, hardware, and llama.cpp configuration; they should not be treated as universal capacity limits.
