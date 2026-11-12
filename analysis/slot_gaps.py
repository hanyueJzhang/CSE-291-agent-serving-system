"""Where does c = 5 beat c = 4? Slot idle gaps and follow-up slot affinity.

    python -m analysis.slot_gaps results/phase2_diag/w1_fifo_c4_l95 results/phase2_diag/w1_fifo_c5_l95

Two candidate explanations (proposal 1.6 / 2.5 step 3):

1. Hidden dispatch gaps. With concurrency = slots, a freed slot sits idle
   until the client sees the response finish, the scheduler admits the next
   request and llama-server receives and tokenizes it. With one extra request
   already waiting inside llama-server, the slot is refilled at once.
   Measured as idle slot-seconds while some request was ready (arrived but not
   yet holding a slot), and, with slot_id, as the per-slot handoff gap
   (next start - previous end on the same slot, when the next request was
   already waiting at the previous end).

2. Better slot choice. The extra request is placed by llama-server, which
   picks the free slot with the longest prompt-prefix match. Measured as how
   often a follow-up lands on its previous step's slot (needs slot_id) and
   the share of carried history the server actually reused.

Slot hold times come from analysis/slot_trace.py (start = first token - prefill).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .slot_trace import Run, load_run, percentile


def idle_while_ready(run: Run) -> dict[str, float]:
    """Integrate free slots and ready requests over time (no slot_id needed)."""
    events: list[tuple[float, int, int]] = []     # (time, d_busy, d_ready)
    for r in run.requests:
        events.append((r.t_arrive, 0, +1))
        events.append((r.start, +1, -1))
        events.append((r.end, -1, 0))
    events.sort(key=lambda e: (e[0], -e[1]))      # at equal times, finish before start
    busy = ready = 0
    t_prev = events[0][0] if events else 0.0
    idle_ready = idle_total = busy_total = 0.0
    for t, d_busy, d_ready in events:
        dt = t - t_prev
        if dt > 0:
            free = max(run.n_slots - busy, 0)
            idle_total += free * dt
            busy_total += min(busy, run.n_slots) * dt
            idle_ready += min(free, ready) * dt
        busy += d_busy
        ready += d_ready
        t_prev = t
    span = run.n_slots * run.makespan
    return {
        "slot_busy_pct": 100 * busy_total / span if span else float("nan"),
        "idle_slot_s": idle_total,
        "idle_slot_s_while_ready": idle_ready,
        "idle_while_ready_pct_of_slot_time": 100 * idle_ready / span if span else float("nan"),
    }


def handoff_gaps(run: Run) -> list[float]:
    """Per-slot gaps where the next request was already ready when the slot freed."""
    by_slot: dict[int, list] = {}
    for r in run.requests:
        by_slot.setdefault(r.slot_id, []).append(r)
    gaps = []
    for reqs in by_slot.values():
        reqs.sort(key=lambda r: r.start)
        for prev, nxt in zip(reqs, reqs[1:]):
            if nxt.t_arrive <= prev.end:
                gaps.append(max(nxt.start - prev.end, 0.0))
    return gaps


def analyze(run_dir: Path) -> dict[str, Any]:
    run = load_run(run_dir, prefixes=False)
    reqs = run.requests
    fus = [r for r in reqs if r.is_followup]
    history = sum(r.carried for r in fus)
    server_wait = [r.start - r.t_dispatch for r in reqs]
    out: dict[str, Any] = {
        "run": run_dir.name,
        "slots": run.n_slots,
        "concurrency": ((run.config.get("scheduler") or {}).get("max_concurrency")),
        "requests": len(reqs),
        "makespan_s": run.makespan,
        **idle_while_ready(run),
        "server_wait_mean_s": sum(server_wait) / len(server_wait) if server_wait else float("nan"),
        "server_wait_p95_s": percentile(server_wait, 95),
        "history_reused_pct": 100 * sum(min(r.cached, r.carried) for r in fus) / history if history else float("nan"),
        "has_slot_id": run.has_slots,
    }
    if run.has_slots:
        gaps = handoff_gaps(run)
        same = [r for r in fus if r.slot_id == r.prev.slot_id]
        out.update({
            "handoffs": len(gaps),
            "handoff_gap_mean_ms": 1000 * sum(gaps) / len(gaps) if gaps else float("nan"),
            "handoff_gap_p50_ms": 1000 * percentile(gaps, 50),
            "handoff_gap_p95_ms": 1000 * percentile(gaps, 95),
            "follow_ups_on_own_slot_pct": 100 * len(same) / len(fus) if fus else float("nan"),
            "own_slot_history_reused_pct": (
                100 * sum(min(r.cached, r.carried) for r in same) / sum(r.carried for r in same)
                if same else float("nan")),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="run folders under results/")
    parser.add_argument("--json", type=Path, help="also write the rows as JSON")
    args = parser.parse_args(argv)
    rows = []
    for run_dir in args.runs:
        if not (run_dir / "requests.csv").is_file():
            print(f"skipping {run_dir}: no requests.csv", file=sys.stderr)
            continue
        rows.append(analyze(run_dir))

    fields = [
        ("concurrency", "c", "{}"),
        ("slot_busy_pct", "busy%", "{:.1f}"),
        ("idle_slot_s_while_ready", "idle|ready s", "{:.1f}"),
        ("idle_while_ready_pct_of_slot_time", "idle|ready %", "{:.2f}"),
        ("server_wait_mean_s", "srv wait s", "{:.2f}"),
        ("history_reused_pct", "hist reuse%", "{:.1f}"),
        ("handoff_gap_p50_ms", "gap p50 ms", "{:.0f}"),
        ("handoff_gap_p95_ms", "gap p95 ms", "{:.0f}"),
        ("follow_ups_on_own_slot_pct", "own slot%", "{:.1f}"),
    ]
    print(f"{'run':28s} " + " ".join(f"{h:>12s}" for _, h, _ in fields))
    for r in rows:
        cells = []
        for key, _, fmt in fields:
            v = r.get(key)
            cells.append(f"{'-' if v is None else fmt.format(v):>12s}")
        print(f"{r['run']:28s} " + " ".join(cells))
    if any(not r["has_slot_id"] for r in rows):
        print("(gap and own-slot columns need slot_id in requests.csv, i.e. a Phase 2 run)")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
