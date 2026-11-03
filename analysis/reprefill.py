"""Bound the cost of re-prefilling displaced agent context.

    python -m analysis.reprefill results/mooncake_baseline/w1_fifo_c4_l75 \
        results/mooncake_baseline/w1_fifo_c4_l95 ...

For each follow-up request (step > 0) the history it carries is the previous
step's prompt plus its output. Tokens beyond that history are newly appended;
uncached tokens beyond the new ones were already processed once and had to be
recomputed because the context was displaced. Their share of the request's
prefill time is attributed to "avoidable re-prefill":

    new         = max(prompt - (prev_prompt + prev_output), 0)
    uncached    = prompt - cached
    avoidable   = prefill_ms * max(uncached - new, 0) / uncached

Per run we report the total, its share of the makespan (wall time) and of
GPU-busy time (makespan x mean sampled gpu_util_pct). Per-request prefill can
overlap other slots' decoding, so both shares are upper bounds.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def _follow_up_pairs(rows: list[dict[str, str]]):
    by_task: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            by_task.setdefault(row["task_id"], []).append(row)
    for steps in by_task.values():
        steps.sort(key=lambda r: int(r["step_idx"]))
        yield from zip(steps, steps[1:])


def avoidable_ms(prev: dict[str, str], cur: dict[str, str]) -> tuple[float, int]:
    """Return (avoidable prefill ms, avoidable tokens) for one follow-up."""
    prompt = int(cur["prompt_tokens"])
    uncached = prompt - int(cur["cached_tokens"])
    history = int(prev["prompt_tokens"]) + int(prev["output_tokens"])
    new = max(prompt - history, 0)
    tokens = max(uncached - new, 0)
    if uncached <= 0:
        return 0.0, 0
    return float(cur["prefill_ms"]) * tokens / uncached, tokens


def analyze(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "requests.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with (run_dir / "memory.csv").open(newline="", encoding="utf-8") as f:
        util = [float(r["gpu_util_pct"]) for r in csv.DictReader(f) if r.get("gpu_util_pct")]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    avoid_ms = follow_up_ms = 0.0
    avoid_tokens = follow_ups = 0
    for prev, cur in _follow_up_pairs(rows):
        ms, tokens = avoidable_ms(prev, cur)
        avoid_ms += ms
        avoid_tokens += tokens
        follow_up_ms += float(cur["prefill_ms"])
        follow_ups += 1

    makespan_s = float(summary["makespan_s"])
    mean_util = sum(util) / len(util) if util else float("nan")
    busy_s = makespan_s * mean_util / 100
    avoid_s = avoid_ms / 1000
    return {
        "run": run_dir.name,
        "follow_ups": follow_ups,
        "makespan_s": makespan_s,
        "mean_gpu_util_pct": mean_util,
        "follow_up_prefill_s": follow_up_ms / 1000,
        "avoidable_prefill_s": avoid_s,
        "avoidable_tokens_per_follow_up": avoid_tokens / follow_ups if follow_ups else 0.0,
        "share_of_wall_pct": 100 * avoid_s / makespan_s if makespan_s else float("nan"),
        "share_of_gpu_busy_pct": 100 * avoid_s / busy_s if busy_s else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="run folders under results/")
    parser.add_argument("--json", type=Path, help="also write the rows as JSON")
    args = parser.parse_args(argv)

    runs = [d for d in args.runs if (d / "requests.csv").is_file()]
    for d in args.runs:
        if d not in runs:
            print(f"skipping {d}: no requests.csv (not a run folder)", file=sys.stderr)
    results = [analyze(d) for d in runs]
    header = f"{'run':28s} {'util%':>6} {'makespan_s':>10} {'avoid_s':>8} {'%wall':>6} {'%busy':>6} {'tok/fu':>7}"
    print(header)
    for r in results:
        print(
            f"{r['run']:28s} {r['mean_gpu_util_pct']:6.1f} {r['makespan_s']:10.0f} "
            f"{r['avoidable_prefill_s']:8.1f} {r['share_of_wall_pct']:6.1f} "
            f"{r['share_of_gpu_busy_pct']:6.1f} {r['avoidable_tokens_per_follow_up']:7.0f}"
        )
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
