"""How much does one slot's prefill slow the other slots' decoding?

    python -m analysis.prefill_stall results/phase2_diag/w1_fifo_c5_l95 results/phase2_stress/w1_fifo_c5_l115_s42
    python -m analysis.prefill_stall <runs...> --history 2000 --json out.json

llama-server puts the prompt tokens of a slot that is prefilling in the same
batch as the next token of every decoding slot, so a decoding slot gets one
token per prefill batch instead of one per few milliseconds. This script
measures that from requests.csv alone and prices an idle slot against a
re-prefill (the motivation for scheduler/lease.py).

Per request, from the client timestamps (as in analysis/slot_trace.py):

    prefill  = [t_first_token - prefill_ms / 1000, t_first_token]
    decode   = [t_first_token, t_finish]
    n_busy   = mean number of requests holding a slot during its decode (itself included)
    overlap  = seconds of its decode during which another request was prefilling

Per run it reports the share of decode time that overlaps a prefill, and the
median decode ms/token of requests with almost no overlap, by n_busy. Over all
runs given it fits

    decode_s = output_tokens * (a + b * n_busy) + stall * overlap

stall = 1 would mean decoding stops completely during another slot's prefill.

Break-even hold: with n slots, holding one slot idle for t seconds while work
waits loses t * (rate(n) - rate(n-1)) tokens, where rate(k) = k / (ms/token at
k busy). Re-prefilling a history of H tokens takes H / prefill rate seconds,
during which the other n-1 slots lose `stall` of their output and the
follow-up's own slot produces nothing. The hold pays off up to the t where
the two are equal. Rates are pooled medians of few samples, so read the
result as "about 2 s", not to the decimal.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MIN_OUTPUT = 64             # shorter decodes are too noisy per token
CLEAN_OVERLAP = 0.02        # "stall-free": under 2% of the decode overlaps a prefill


@dataclass
class Sample:
    output: int
    n_busy: float
    overlap_s: float
    decode_s: float
    duration_s: float

    @property
    def ms_per_token(self) -> float:
        return 1000 * self.decode_s / self.output


def _union_length(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, float("-inf")
    for a, b in sorted(intervals):
        if b > end:
            total += b - max(a, end)
            end = b
    return total


def load_samples(run_dir: str | Path) -> tuple[list[Sample], float]:
    """Decode samples of one run and its prefill rate (uncached prompt tokens per prefill second)."""
    with (Path(run_dir) / "requests.csv").open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") == "ok" and r.get("t_first_token")]
    reqs = []
    for r in rows:
        first, finish = float(r["t_first_token"]), float(r["t_finish"])
        prefill_s = float(r["prefill_ms"] or 0) / 1000
        reqs.append((first - prefill_s, first, finish, int(float(r["output_tokens"] or 0)),
                     float(r["decode_ms"] or 0) / 1000,
                     max(int(float(r["prompt_tokens"] or 0)) - int(float(r["cached_tokens"] or 0)), 0)))
    samples = []
    for i, (_, first, finish, output, decode_s, _) in enumerate(reqs):
        duration = finish - first
        if output < MIN_OUTPUT or duration <= 0 or decode_s <= 0:
            continue
        busy, prefills = duration, []
        for j, (o_start, o_first, o_finish, *_rest) in enumerate(reqs):
            if j == i:
                continue
            busy += max(min(o_finish, finish) - max(o_start, first), 0.0)
            a, b = max(o_start, first), min(o_first, finish)
            if b > a:
                prefills.append((a, b))
        samples.append(Sample(output, busy / duration, _union_length(prefills), decode_s, duration))
    prefill_s = sum(first - start for start, first, *_ in reqs)
    uncached = sum(r[5] for r in reqs)
    return samples, (uncached / prefill_s if prefill_s > 0 else float("nan"))


def stall_free_ms(samples: list[Sample]) -> dict[int, tuple[float, int]]:
    """n_busy (rounded) -> (median ms/token, sample count) over requests with almost no overlap."""
    groups: dict[int, list[float]] = {}
    for s in samples:
        if s.overlap_s < CLEAN_OVERLAP * s.duration_s:
            groups.setdefault(round(s.n_busy), []).append(s.ms_per_token)
    return {n: (statistics.median(v), len(v)) for n, v in sorted(groups.items())}


def _solve(a: list[list[float]], y: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting."""
    n = len(y)
    m = [row[:] + [y[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        if abs(m[col][col]) < 1e-12:
            raise ValueError("singular fit (need requests with different n_busy and overlap)")
        for r in range(n):
            if r != col:
                k = m[r][col] / m[col][col]
                m[r] = [x - k * p for x, p in zip(m[r], m[col])]
    return [m[i][n] / m[i][i] for i in range(n)]


def fit(samples: list[Sample]) -> dict[str, float]:
    """Least squares for decode_s = output * (a + b * n_busy) + stall * overlap."""
    xs = [(s.output, s.output * s.n_busy, s.overlap_s) for s in samples]
    ys = [s.decode_s for s in samples]
    ata = [[sum(x[i] * x[j] for x in xs) for j in range(3)] for i in range(3)]
    aty = [sum(x[i] * y for x, y in zip(xs, ys)) for i in range(3)]
    a, b, stall = _solve(ata, aty)
    mean = sum(ys) / len(ys)
    ss_res = sum((y - (a * x[0] + b * x[1] + stall * x[2])) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - mean) ** 2 for y in ys)
    return {"base_ms_per_token": 1000 * a, "ms_per_token_per_busy_slot": 1000 * b, "stall": stall,
            "r2": 1 - ss_res / ss_tot if ss_tot else float("nan")}


def break_even_s(ms: dict[int, float], n: int, history: int, prefill_rate: float, stall: float) -> float | None:
    """Idle seconds of one of n slots that cost as much output as re-prefilling `history` tokens."""
    if n < 2 or n not in ms or n - 1 not in ms:
        return None
    rate = lambda k: 1000 * k / ms[k]
    gained_by_slot = rate(n) - rate(n - 1)
    if gained_by_slot <= 0:
        return None
    prefill_s = history / prefill_rate
    lost = prefill_s * (stall * rate(n) * (n - 1) / n + rate(n) / n)
    return lost / gained_by_slot


def analyze(run_dirs: list[str], history: int) -> dict[str, Any]:
    runs, pooled, rates = [], [], []
    for run_dir in run_dirs:
        samples, prefill_rate = load_samples(run_dir)
        decode = sum(s.duration_s for s in samples)
        runs.append({
            "run": str(run_dir), "samples": len(samples), "prefill_tokens_per_s": prefill_rate,
            "overlap_share": sum(s.overlap_s for s in samples) / decode if decode else float("nan"),
            "stall_free_ms_per_token": {n: {"median": m, "n": c} for n, (m, c) in stall_free_ms(samples).items()},
        })
        pooled += samples
        rates.append(prefill_rate)
    out: dict[str, Any] = {"runs": runs}
    if pooled:
        out["fit"] = fit(pooled)
        ms = {n: m for n, (m, _) in stall_free_ms(pooled).items()}
        prefill_rate = statistics.median(rates)
        out["pooled"] = {
            "stall_free_ms_per_token": ms, "prefill_tokens_per_s": prefill_rate, "history_tokens": history,
            "tokens_per_s": {n: 1000 * n / m for n, m in ms.items()},
            "break_even_hold_s": {n: break_even_s(ms, n, history, prefill_rate, out["fit"]["stall"]) for n in ms},
        }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run folders containing requests.csv")
    p.add_argument("--history", type=int, default=2000, help="history tokens a displaced follow-up re-prefills")
    p.add_argument("--json", help="also write the numbers to this file")
    args = p.parse_args(argv)
    result = analyze(args.runs, args.history)

    for run in result["runs"]:
        by_busy = "  ".join(f"{n}: {v['median']:.1f} ({v['n']})" for n, v in run["stall_free_ms_per_token"].items())
        print(f"{run['run']}\n  decode time overlapping another slot's prefill: {100 * run['overlap_share']:.1f}%"
              f"   prefill {run['prefill_tokens_per_s']:.0f} tok/s\n  stall-free ms/token by busy slots (n): {by_busy}")
    if "fit" in result:
        f, pooled = result["fit"], result["pooled"]
        print(f"\nall runs: decode_s = out * ({f['base_ms_per_token']:.2f} + {f['ms_per_token_per_busy_slot']:.2f}"
              f" * n_busy) ms + {f['stall']:.2f} * overlap_s   (R2 {f['r2']:.3f})")
        print(f"  busy  ms/token  tokens/s  break-even hold for a {pooled['history_tokens']}-token history")
        for n, ms in pooled["stall_free_ms_per_token"].items():
            hold = pooled["break_even_hold_s"][n]
            print(f"  {n:>4}  {ms:8.1f}  {pooled['tokens_per_s'][n]:8.0f}  "
                  + (f"{hold:.1f} s" if hold is not None else "-"))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
