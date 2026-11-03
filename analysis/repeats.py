"""Mean and spread over seeds for the runs the Phase 1 report cites.

    python -m analysis.repeats              # table
    python -m analysis.repeats --json out.json

Seed 42 is the original run (mooncake_baseline, phase1_baseline, mooncake_out4);
seeds 43 and 44 are in results/phase1_repeats/<run_id>_s<seed>. For each
configuration the table gives mean ± sample standard deviation (n = 3) and the
per-seed values in seed order. Avoidable re-prefill comes from
analysis.reprefill.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

from .reprefill import analyze as reprefill

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
REPEATS = RESULTS / "phase1_repeats"
SEEDS = (42, 43, 44)

# label -> (seed-42 folder, run_id used in phase1_repeats)
CONFIGS: dict[str, tuple[str, str]] = {
    "W1 0.50X": ("mooncake_baseline/w1_fifo_c4_l50", "w1_fifo_c4_l50"),
    "W1 0.75X": ("mooncake_baseline/w1_fifo_c4_l75", "w1_fifo_c4_l75"),
    "W1 0.95X": ("mooncake_baseline/w1_fifo_c4_l95", "w1_fifo_c4_l95"),
    "W1 0.95X + 8 GiB cache": ("mooncake_baseline/w1_fifo_c4_l95_cram8g", "w1_fifo_c4_l95_cram8g"),
    "synthetic 1 task/s": ("phase1_baseline/rate_c4_r1", "rate_c4_r1"),
    "synthetic 4 tasks/s": ("phase1_baseline/rate_c4_r4", "rate_c4_r4"),
    "W1 out/4, 550 tasks": ("mooncake_out4/w1out4_fifo_c4_n550", "w1out4_fifo_c4_n550"),
    "W1 out/4, 550 tasks + 8 GiB cache": ("mooncake_out4/w1out4_fifo_c4_n550_cram8g", "w1out4_fifo_c4_n550_cram8g"),
}

METRICS: dict[str, Callable[[dict[str, Any], Path], float]] = {
    "tasks/s": lambda s, d: s["throughput"]["tasks_per_s"],
    "task P95 (s)": lambda s, d: s["tasks"]["latency_s"]["p95"],
    "task P99 (s)": lambda s, d: s["tasks"]["latency_s"]["p99"],
    "mean queue (s)": lambda s, d: s["tasks"]["queue_s"]["mean"],
    "follow-up cached (%)": lambda s, d: 100 * s["prompt_cache"]["cached_ratio_followup_steps"],
    "GPU util (%)": lambda s, d: s["memory"]["gpu_util_mean_pct"],
    "re-prefill (% busy)": lambda s, d: reprefill(d)["share_of_gpu_busy_pct"],
}


def run_dirs(seed42: str, run_id: str) -> list[Path]:
    return [RESULTS / seed42] + [REPEATS / f"{run_id}_s{s}" for s in SEEDS[1:]]


def collect() -> dict[str, dict[str, list[float]]]:
    table: dict[str, dict[str, list[float]]] = {}
    for label, (seed42, run_id) in CONFIGS.items():
        dirs = run_dirs(seed42, run_id)
        missing = [str(d) for d in dirs if not (d / "summary.json").is_file()]
        if missing:
            print(f"skipping {label}: missing {', '.join(missing)}", file=sys.stderr)
            continue
        summaries = [json.loads((d / "summary.json").read_text(encoding="utf-8")) for d in dirs]
        table[label] = {m: [f(s, d) for s, d in zip(summaries, dirs)] for m, f in METRICS.items()}
    return table


def fmt(values: list[float]) -> str:
    mean, sd = statistics.mean(values), statistics.stdev(values)
    digits = 3 if abs(mean) < 2 else 1
    seeds = "/".join(f"{v:.{digits}f}" for v in values)
    return f"{mean:.{digits}f} ± {sd:.{digits}f} ({seeds})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="also write raw per-seed values as JSON")
    args = parser.parse_args(argv)

    table = collect()
    for label, metrics in table.items():
        print(label)
        for name, values in metrics.items():
            print(f"  {name:<22} {fmt(values)}")
    if args.json:
        args.json.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
