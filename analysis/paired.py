"""Paired multi-seed comparison of policies against a baseline.

    python -m analysis.paired results/phase2_cf --baseline fifo_c5
    python -m analysis.paired results/phase2_cf results/cache_budget --baseline fifo_c5 --json out.json

Phase 1 showed that seeds differ far more than some policy effects (W1 0.95X
c = 4 task P95: 38 / 70 / 82 s for seeds 42-44), so a policy is only compared
with the baseline on the *same* workload instance: same workload, load and
seed. For every metric we report, per policy, the per-seed relative change
versus the baseline run of that seed, and their mean ± sample std. Raw values
per seed are kept in the JSON.

Runs are grouped by their run_config.json dimensions: the policy label is the
`policy` dimension (or --label-key), pairing uses `seed` plus every other
dimension except the label and those in --ignore (default: experiment, phase,
cache_ram_mb, concurrency, which vary with the policy).

Metrics (ok tasks/requests only, plus failure counts):
  task_p50_s, task_p95_s     task latency (arrival to end, includes tool time)
  first_ttft_p95_s           a task's arrival to its first token (new-task
                             starvation check for continuation-first)
  first_queue_mean_s         wait of first steps from ready to holding a slot
                             (client queue + llama-server queue when c > slots)
  followup_queue_mean_s      the same for follow-up steps
  followup_cached_pct        cached share of follow-up prompt tokens
  avoidable_reprefill_s      re-prefill of already-computed history (analysis.reprefill)
  throughput_tasks_s         ok tasks / makespan
  tasks_not_ok               failed + timeout + rejected
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from .reprefill import avoidable_ms
from .slot_trace import percentile

DEFAULT_IGNORE = ("experiment", "phase", "cache_ram_mb", "concurrency")
METRICS = [
    # key, label, lower is better
    ("task_p50_s", "task P50 s", True),
    ("task_p95_s", "task P95 s", True),
    ("first_ttft_p95_s", "new-task TTFT P95 s", True),
    ("first_queue_mean_s", "1st-step queue s", True),
    ("followup_queue_mean_s", "follow-up queue s", True),
    ("followup_cached_pct", "follow-up cached %", False),
    ("avoidable_reprefill_s", "avoidable re-prefill s", True),
    ("throughput_tasks_s", "tasks/s", False),
    ("tasks_not_ok", "tasks not ok", True),
]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(v: str | None) -> float | None:
    return float(v) if v not in (None, "") else None


def run_metrics(run_dir: Path) -> dict[str, float]:
    tasks = _read(run_dir / "tasks.csv")
    reqs = _read(run_dir / "requests.csv")
    ok_tasks = [t for t in tasks if t["status"] == "ok"]
    ok_reqs = [r for r in reqs if r["status"] == "ok"]
    task_arrive = {t["task_id"]: _f(t["t_arrive"]) for t in tasks}

    latency = [_f(t["t_finish"]) - _f(t["t_arrive"]) for t in ok_tasks]
    by_task: dict[str, list[dict[str, str]]] = {}
    for r in ok_reqs:
        by_task.setdefault(r["task_id"], []).append(r)
    for steps in by_task.values():
        steps.sort(key=lambda r: int(r["step_idx"]))

    firsts = [s[0] for s in by_task.values()]
    followups = [r for s in by_task.values() for r in s[1:]]
    ttft = [_f(r["t_first_token"]) - task_arrive[r["task_id"]] for r in firsts
            if _f(r["t_first_token"]) is not None and task_arrive.get(r["task_id"]) is not None]

    def queue(rows):
        # Ready until the request actually got a slot: the client queue plus any
        # wait inside llama-server (c > slots), estimated as in analysis/slot_trace.
        q = []
        for r in rows:
            start = _f(r["t_dispatch"])
            first, prefill = _f(r.get("t_first_token")), _f(r.get("prefill_ms")) or 0.0
            if first is not None:
                start = max(start, first - prefill / 1000)
            q.append(start - _f(r["t_arrive"]))
        return sum(q) / len(q) if q else float("nan")

    fu_prompt = sum(int(r["prompt_tokens"] or 0) for r in followups)
    fu_cached = sum(int(r["cached_tokens"] or 0) for r in followups)
    avoid = 0.0
    for steps in by_task.values():
        for prev, cur in zip(steps, steps[1:]):
            avoid += avoidable_ms(prev, cur)[0]

    starts = [_f(t["t_arrive"]) for t in tasks if _f(t["t_arrive"]) is not None]
    ends = [_f(t["t_finish"]) for t in tasks if _f(t["t_finish"]) is not None]
    makespan = max(ends) - min(starts) if starts and ends else float("nan")
    return {
        "task_p50_s": percentile(latency, 50),
        "task_p95_s": percentile(latency, 95),
        "first_ttft_p95_s": percentile(ttft, 95),
        "first_queue_mean_s": queue(firsts),
        "followup_queue_mean_s": queue(followups),
        "followup_cached_pct": 100 * fu_cached / fu_prompt if fu_prompt else float("nan"),
        "avoidable_reprefill_s": avoid / 1000,
        "throughput_tasks_s": len(ok_tasks) / makespan if makespan else float("nan"),
        "tasks_not_ok": float(len(tasks) - len(ok_tasks)),
    }


def find_runs(roots: list[Path], exclude: list[str] | None = None) -> list[Path]:
    """Run folders under roots; folder names matching an `exclude` glob are left out."""
    runs = []
    for root in roots:
        if (root / "requests.csv").is_file():
            runs.append(root)
            continue
        runs.extend(sorted(p.parent for p in root.glob("*/requests.csv")
                           if ".incomplete-" not in p.parent.name))
    return [r for r in runs if not any(fnmatch.fnmatch(r.name, pat) for pat in exclude or [])]


def compare(run_dirs: list[Path], baseline: str, label_key: str = "policy",
            ignore: tuple[str, ...] = DEFAULT_IGNORE) -> dict[str, Any]:
    by_key: dict[tuple, dict[str, tuple[Path, dict[str, float]]]] = {}
    skipped = []
    for run_dir in run_dirs:
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        dims = dict(config.get("dimensions") or {})
        label = dims.pop(label_key, None)
        if label is None:
            skipped.append(f"{run_dir}: no '{label_key}' dimension")
            continue
        label = str(label)
        for k in ignore:
            dims.pop(k, None)
        key = tuple(sorted(dims.items()))
        if label in by_key.get(key, {}):
            skipped.append(f"{run_dir}: duplicate ({label}, {dict(key)}), keeping {by_key[key][label][0]}")
            continue
        by_key.setdefault(key, {})[label] = (run_dir, run_metrics(run_dir))

    policies: dict[str, dict[str, Any]] = {}
    for key, runs in sorted(by_key.items()):
        if baseline not in runs:
            skipped.extend(f"{path}: no '{baseline}' run with {dict(key)}" for path, _ in runs.values())
            continue
        if len(runs) == 1:
            continue                     # baseline alone: nothing to pair with
        base_path, base = runs[baseline]
        for label, (path, metrics) in runs.items():
            entry = policies.setdefault(label, {"pairs": []})
            entry["pairs"].append({
                "instance": dict(key), "run": str(path), "baseline_run": str(base_path),
                "metrics": metrics, "baseline": base,
                "rel_change_pct": {m: _rel(metrics[m], base[m]) for m, _, _ in METRICS},
            })
    for entry in policies.values():
        entry["n"] = len(entry["pairs"])
        entry["summary"] = {}
        for m, _, _ in METRICS:
            rel = [p["rel_change_pct"][m] for p in entry["pairs"] if p["rel_change_pct"][m] is not None]
            entry["summary"][m] = {
                "mean_rel_pct": statistics.mean(rel) if rel else None,
                "std_rel_pct": statistics.stdev(rel) if len(rel) > 1 else None,
                "per_instance_rel_pct": rel,
                "mean_value": statistics.mean(p["metrics"][m] for p in entry["pairs"]),
            }
    return {"baseline": baseline, "label_key": label_key, "policies": policies, "skipped": skipped}


def _rel(value: float, base: float) -> float | None:
    if base in (0, None) or value is None or base != base or value != value:
        return None
    return 100 * (value - base) / base


def print_table(result: dict[str, Any]) -> None:
    policies = result["policies"]
    print(f"baseline: {result['baseline']}   (change vs baseline on the same instance: "
          "mean ± std [per seed])")
    for m, label, lower in METRICS:
        print(f"\n{label}{'' if lower else '  (higher is better)'}")
        for name, entry in sorted(policies.items(), key=lambda kv: (kv[0] != result["baseline"], kv[0])):
            s = entry["summary"][m]
            if name == result["baseline"]:
                print(f"  {name:22s} n={entry['n']}  value {s['mean_value']:.3g}")
                continue
            mean, std = s["mean_rel_pct"], s["std_rel_pct"]
            per = ", ".join(f"{v:+.0f}" for v in s["per_instance_rel_pct"])
            cell = "n/a" if mean is None else f"{mean:+6.1f}%" + ("" if std is None else f" ± {std:4.1f}")
            print(f"  {name:22s} n={entry['n']}  {cell:16s} [{per}]  value {s['mean_value']:.3g}")
    for note in result["skipped"]:
        print(f"skipped: {note}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="+", type=Path, help="run folders, or sweep folders containing them")
    parser.add_argument("--baseline", required=True, help="label of the baseline policy, e.g. fifo_c5")
    parser.add_argument("--label-key", default="policy", help="dimension that names the policy")
    parser.add_argument("--ignore", nargs="*", default=list(DEFAULT_IGNORE),
                        help="dimensions that do not define the workload instance")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="run folder name globs to leave out, e.g. 'w1_cf_*'")
    parser.add_argument("--json", type=Path, help="also write everything as JSON")
    args = parser.parse_args(argv)
    result = compare(find_runs(args.roots, args.exclude), args.baseline, args.label_key, tuple(args.ignore))
    print_table(result)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
