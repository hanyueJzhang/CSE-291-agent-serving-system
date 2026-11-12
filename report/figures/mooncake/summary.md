# Mooncake FIFO baseline summary

Load levels are fractions of the calibrated capacity X = 0.330 tasks/s. Baseline values are means across W1–W3; the bracketed latency values show the full window range.

| Load | Nominal offered rate | Completed throughput | Task latency P95 | Step-0 reuse | Follow-up reuse | Follow-up prefill P50 | Mean GPU util. |
|---|---:|---:|---:|---:|---:|---:|---:|
| L50 | 0.165 tasks/s | 0.165 tasks/s | 18.6 s [17.6–19.5] | 42.5% | 73.5% | 97 ms | 61.6% |
| L75 | 0.247 tasks/s | 0.246 tasks/s | 27.9 s [25.2–30.8] | 37.6% | 51.8% | 140 ms | 78.0% |
| L95 | 0.314 tasks/s | 0.309 tasks/s | 42.8 s [34.2–55.6] | 28.3% | 33.0% | 307 ms | 89.3% |

## Host-cache comparison (W1 only)

| Load | Host cache | Task latency P95 | Step-0 reuse | Follow-up reuse | Follow-up prefill P50 | Mean GPU util. | Peak server RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| L75 | Off | 27.6 s | 39.4% | 56.7% | 109 ms | 78.4% | 0.8 GiB |
| L75 | 8 GiB | 23.8 s | 63.2% | 81.9% | 106 ms | 71.5% | 8.8 GiB |
| L95 | Off | 38.5 s | 29.4% | 35.2% | 278 ms | 88.9% | 0.8 GiB |
| L95 | 8 GiB | 28.7 s | 61.4% | 80.1% | 108 ms | 78.4% | 8.6 GiB |

Step-0 reuse measures shared-prefix reuse between trace-derived tasks. Follow-up reuse measures whether a task's carried context survives its synthetic tool waits.
