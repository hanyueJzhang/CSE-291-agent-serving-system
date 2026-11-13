"""Server time per request that is neither prefill nor decode.

    python -m analysis.server_overhead results/cache_budget/w1_fifo_c5_l95_cram1g ...

For each successful request, the time from client dispatch to finish minus
llama.cpp's own prefill_ms + decode_ms is time the request spent in the server
doing something else: waiting in llama.cpp's internal queue (when the client
admits more requests than there are slots), HTTP and tokenization, and host
prompt-cache work (saving, searching and restoring KV). Comparing runs with
the same concurrency and tasks isolates the cache's share.

Per run we report this non-compute time (p50, mean, p95), the median decode
speed, and the summed prefill + decode time across requests.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def non_compute_ms(row: dict[str, str]) -> float:
    in_server_ms = (float(row["t_finish"]) - float(row["t_dispatch"])) * 1000
    return in_server_ms - float(row["prefill_ms"]) - float(row["decode_ms"])


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(q * len(ordered)), len(ordered) - 1)]


def analyze(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "requests.csv").open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") == "ok"]
    gaps = [non_compute_ms(r) for r in rows]
    decode = [
        float(r["decode_ms"]) / int(r["output_tokens"])
        for r in rows
        if int(r["output_tokens"]) > 1
    ]
    compute_s = sum(float(r["prefill_ms"]) + float(r["decode_ms"]) for r in rows) / 1000
    return {
        "run": run_dir.name,
        "requests": len(rows),
        "non_compute_p50_ms": statistics.median(gaps),
        "non_compute_mean_ms": statistics.fmean(gaps),
        "non_compute_p95_ms": _pct(gaps, 0.95),
        "decode_ms_per_token_p50": statistics.median(decode),
        "compute_sum_s": compute_s,
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
    print(f"{'run':30s} {'p50_ms':>7} {'mean_ms':>8} {'p95_ms':>7} {'dec_ms/tok':>10} {'compute_s':>9}")
    for r in results:
        print(
            f"{r['run']:30s} {r['non_compute_p50_ms']:7.0f} {r['non_compute_mean_ms']:8.0f} "
            f"{r['non_compute_p95_ms']:7.0f} {r['decode_ms_per_token_p50']:10.2f} {r['compute_sum_s']:9.0f}"
        )
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
