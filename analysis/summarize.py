"""Summarize run folders written by scripts/run_experiment.py.

    python -m analysis.summarize results/<run_id>
    python -m analysis.summarize results/example_mix_fifo_c* --table results/sweep.csv --plots

For each run, prints latency percentiles, throughput, prompt-cache reuse and
memory peaks, and writes results/<run_id>/summary.json. With several runs it
also prints one comparison row per run (and writes it as CSV with --table).
--plots writes latency_cdf.png and memory.png into each run folder.

Only requests and tasks with status "ok" enter latency and throughput numbers;
failures are counted separately. Times are seconds unless the name says _ms.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

PERCENTILES = (50, 95, 99)


# ---- reading -------------------------------------------------------------------

def _num(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_csv(path: Path) -> list[dict[str, Any]]:
    """Rows with numeric columns parsed to float and blanks to None."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    text_columns = {"run_id", "task_id", "task_type", "status", "error"}
    return [
        {k: (v if k in text_columns else _num(v)) for k, v in row.items()}
        for row in rows
    ]


def load_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    config_path = run_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    return {
        "dir": run_dir,
        "config": config,
        "requests": read_csv(run_dir / "requests.csv"),
        "tasks": read_csv(run_dir / "tasks.csv"),
        "memory": read_csv(run_dir / "memory.csv"),
    }


# ---- statistics ----------------------------------------------------------------

def percentile(values: list[float], p: float) -> float | None:
    """Linear interpolation between closest ranks (numpy's default)."""
    if not values:
        return None
    xs = sorted(values)
    rank = (len(xs) - 1) * p / 100
    lo, hi = math.floor(rank), math.ceil(rank)
    return xs[lo] + (xs[hi] - xs[lo]) * (rank - lo)


def describe(values: Iterable[float | None]) -> dict[str, Any]:
    xs = [v for v in values if v is not None]
    out: dict[str, Any] = {"n": len(xs), "mean": sum(xs) / len(xs) if xs else None}
    for p in PERCENTILES:
        out[f"p{p}"] = percentile(xs, p)
    out["max"] = max(xs) if xs else None
    return out


def _diff(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


# ---- per-run summary -----------------------------------------------------------

def request_latencies(req: dict[str, Any]) -> dict[str, float | None]:
    out_tokens = req.get("output_tokens")
    decode_ms = req.get("decode_ms")
    return {
        "queue_s": _diff(req.get("t_dispatch"), req.get("t_arrive")),
        # what the agent waits for: queueing + prefill
        "ttft_s": _diff(req.get("t_first_token"), req.get("t_arrive")),
        # server-side only: prefill after dispatch
        "server_ttft_s": _diff(req.get("t_first_token"), req.get("t_dispatch")),
        "e2e_s": _diff(req.get("t_finish"), req.get("t_arrive")),
        "decode_ms_per_token": decode_ms / out_tokens if decode_ms is not None and out_tokens else None,
    }


def summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    config = run["config"]
    requests, tasks, memory = run["requests"], run["tasks"], run["memory"]
    ok_requests = [r for r in requests if r.get("status") == "ok"]
    ok_tasks = [t for t in tasks if t.get("status") == "ok"]

    lat = [request_latencies(r) for r in ok_requests]
    request_stats = {key: describe(row[key] for row in lat) for key in lat[0]} if lat else {}

    task_latency = [_diff(t.get("t_finish"), t.get("t_arrive")) for t in ok_tasks]
    by_type: dict[str, Any] = {}
    for task_type in sorted({t["task_type"] for t in ok_tasks}):
        of_type = [t for t in ok_tasks if t["task_type"] == task_type]
        by_type[task_type] = {
            "latency_s": describe(_diff(t["t_finish"], t["t_arrive"]) for t in of_type),
            "queue_s": describe(t.get("total_queue_s") for t in of_type),
            "tool_s": describe(t.get("total_tool_s") for t in of_type),
        }

    # Makespan: first task arrival to last task end, over all tasks.
    starts = [t["t_arrive"] for t in tasks if t.get("t_arrive") is not None]
    ends = [t["t_finish"] for t in tasks if t.get("t_finish") is not None]
    makespan = max(ends) - min(starts) if starts and ends else None
    out_tokens = sum(r.get("output_tokens") or 0 for r in ok_requests)
    prompt_tokens = sum(r.get("prompt_tokens") or 0 for r in ok_requests)

    def per_s(count: float) -> float | None:
        return count / makespan if makespan else None

    def cache_ratio(rows: list[dict[str, Any]]) -> float | None:
        prompt = sum(r.get("prompt_tokens") or 0 for r in rows)
        cached = sum(r.get("cached_tokens") or 0 for r in rows)
        return cached / prompt if prompt else None

    def peak(column: str) -> float | None:
        values = [m[column] for m in memory if m.get(column) is not None]
        return max(values) if values else None

    gpu_util = [m["gpu_util_pct"] for m in memory if m.get("gpu_util_pct") is not None]
    workload = config.get("workload") or {}
    scheduler = config.get("scheduler") or {}

    return {
        "run_id": config.get("run_id", run["dir"].name),
        "scheduler": scheduler.get("name"),
        "concurrency": scheduler.get("max_concurrency"),
        "workload": workload.get("name"),
        "rate_per_s": (workload.get("arrival") or {}).get("rate_per_s"),
        "seed": workload.get("seed"),
        "server_slots": (config.get("server") or {}).get("total_slots"),
        "ctx_per_slot": (config.get("server") or {}).get("n_ctx_per_slot"),
        "counts": {
            "tasks": len(tasks),
            "tasks_ok": len(ok_tasks),
            "requests": len(requests),
            "requests_ok": len(ok_requests),
            "requests_failed": len(requests) - len(ok_requests),
        },
        "makespan_s": makespan,
        "throughput": {
            "tasks_per_s": per_s(len(ok_tasks)),
            "requests_per_s": per_s(len(ok_requests)),
            "output_tokens_per_s": per_s(out_tokens),
            "prompt_tokens_per_s": per_s(prompt_tokens),
        },
        "requests": request_stats,
        "tasks": {"latency_s": describe(task_latency), "by_type": by_type},
        "prompt_cache": {
            "cached_ratio_all": cache_ratio(ok_requests),
            # Follow-up steps of agent tasks: did the carried context survive?
            "cached_ratio_followup_steps": cache_ratio(
                [r for r in ok_requests if (r.get("step_idx") or 0) > 0]),
        },
        "memory": {
            "gpu_mem_peak_mb": peak("gpu_mem_used_mb"),
            "gpu_util_mean_pct": sum(gpu_util) / len(gpu_util) if gpu_util else None,
            "server_rss_peak_mb": peak("server_rss_mb"),
            "slot_tokens_peak": peak("slot_tokens_total"),
            "server_requests_deferred_max": peak("server_requests_deferred"),
            "samples": len(memory),
        },
        "warnings": list(config.get("warnings") or []),
    }


# ---- output --------------------------------------------------------------------

def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_summary(s: dict[str, Any]) -> str:
    c = s["counts"]
    lines = [
        f"{s['run_id']}",
        f"  setup     {s['scheduler']} c={s['concurrency']}  workload {s['workload']}  "
        f"rate {s['rate_per_s']}/s  seed {s['seed']}  slots {s['server_slots']} x {s['ctx_per_slot']} tok",
        f"  counts    tasks ok {c['tasks_ok']}/{c['tasks']}   requests ok {c['requests_ok']}/{c['requests']}",
        f"  makespan  {_fmt(s['makespan_s'], 1)} s   "
        f"tasks/s {_fmt(s['throughput']['tasks_per_s'], 3)}   "
        f"out tok/s {_fmt(s['throughput']['output_tokens_per_s'], 1)}",
        "",
        f"  {'request':<22}{'mean':>9}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}",
    ]
    for key, label in [
        ("queue_s", "queue (s)"),
        ("ttft_s", "TTFT incl. queue (s)"),
        ("server_ttft_s", "TTFT server (s)"),
        ("e2e_s", "end-to-end (s)"),
        ("decode_ms_per_token", "decode (ms/token)"),
    ]:
        d = s["requests"].get(key)
        if d:
            lines.append(f"  {label:<22}" + "".join(f"{_fmt(d[k]):>9}" for k in ("mean", "p50", "p95", "p99", "max")))
    lines += ["", f"  {'task latency (s)':<22}{'n':>5}{'mean':>9}{'p50':>9}{'p95':>9}{'p99':>9}{'queue':>9}"]
    rows = [("all", s["tasks"]["latency_s"], None)] + [
        (t, v["latency_s"], v["queue_s"]) for t, v in s["tasks"]["by_type"].items()
    ]
    for name, d, q in rows:
        lines.append(
            f"  {name:<22}{d['n']:>5}" + "".join(f"{_fmt(d[k]):>9}" for k in ("mean", "p50", "p95", "p99"))
            + f"{_fmt(q['mean'] if q else None):>9}"
        )
    pc, m = s["prompt_cache"], s["memory"]
    lines += [
        "",
        f"  cache     cached/prompt tokens: all {_fmt(pc['cached_ratio_all'], 3)}   "
        f"follow-up steps {_fmt(pc['cached_ratio_followup_steps'], 3)}",
        f"  memory    gpu peak {_fmt(m['gpu_mem_peak_mb'], 0)} MB   gpu util mean {_fmt(m['gpu_util_mean_pct'], 0)}%   "
        f"rss peak {_fmt(m['server_rss_peak_mb'], 0)} MB   slot tokens peak {_fmt(m['slot_tokens_peak'], 0)}",
    ]
    if m["server_requests_deferred_max"]:
        lines.append("  CHECK     server_requests_deferred > 0: requests queued inside llama-server; rerun")
    for w in s["warnings"]:
        lines.append(f"  WARNING   {w}")
    return "\n".join(lines)


TABLE_COLUMNS = [
    ("run_id", lambda s: s["run_id"]),
    ("scheduler", lambda s: s["scheduler"]),
    ("concurrency", lambda s: s["concurrency"]),
    ("rate_per_s", lambda s: s["rate_per_s"]),
    ("tasks_ok", lambda s: s["counts"]["tasks_ok"]),
    ("tasks", lambda s: s["counts"]["tasks"]),
    ("makespan_s", lambda s: s["makespan_s"]),
    ("tasks_per_s", lambda s: s["throughput"]["tasks_per_s"]),
    ("output_tokens_per_s", lambda s: s["throughput"]["output_tokens_per_s"]),
    ("ttft_p50_s", lambda s: s["requests"].get("ttft_s", {}).get("p50")),
    ("ttft_p99_s", lambda s: s["requests"].get("ttft_s", {}).get("p99")),
    ("queue_p99_s", lambda s: s["requests"].get("queue_s", {}).get("p99")),
    ("task_latency_p50_s", lambda s: s["tasks"]["latency_s"]["p50"]),
    ("task_latency_p99_s", lambda s: s["tasks"]["latency_s"]["p99"]),
    ("cached_ratio_followup", lambda s: s["prompt_cache"]["cached_ratio_followup_steps"]),
    ("gpu_mem_peak_mb", lambda s: s["memory"]["gpu_mem_peak_mb"]),
    ("server_rss_peak_mb", lambda s: s["memory"]["server_rss_peak_mb"]),
    ("slot_tokens_peak", lambda s: s["memory"]["slot_tokens_peak"]),
]


def table_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{name: get(s) for name, get in TABLE_COLUMNS} for s in summaries]


def write_table(path: Path, summaries: list[dict[str, Any]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[name for name, _ in TABLE_COLUMNS])
        writer.writeheader()
        for row in table_rows(summaries):
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


def plot_run(run: dict[str, Any]) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir, written = run["dir"], []
    ok = [request_latencies(r) for r in run["requests"] if r.get("status") == "ok"]
    if ok:
        fig, ax = plt.subplots(figsize=(6, 4))
        for key, label in [("ttft_s", "TTFT incl. queue"), ("queue_s", "queue"), ("e2e_s", "request end-to-end")]:
            xs = sorted(v[key] for v in ok if v[key] is not None)
            if xs:
                ax.plot(xs, [(i + 1) / len(xs) for i in range(len(xs))], label=label)
        ax.set_xlabel("seconds")
        ax.set_ylabel("fraction of requests")
        ax.set_title(run["dir"].name, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        written.append(out_dir / "latency_cdf.png")
        fig.savefig(written[-1], dpi=120)
        plt.close(fig)

    memory = run["memory"]
    if memory:
        panels = [("gpu_mem_used_mb", "GPU mem (MB)"), ("server_rss_mb", "server RSS (MB)"),
                  ("slot_tokens_total", "slot tokens")]
        fig, axes = plt.subplots(len(panels), 1, figsize=(6, 6), sharex=True)
        for ax, (column, label) in zip(axes, panels):
            pts = [(m["t"], m[column]) for m in memory if m.get("t") is not None and m.get(column) is not None]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts])
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("seconds since run start")
        axes[0].set_title(run["dir"].name, fontsize=9)
        fig.tight_layout()
        written.append(out_dir / "memory.png")
        fig.savefig(written[-1], dpi=120)
        plt.close(fig)
    return written


# ---- CLI -----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path, help="run folders, e.g. results/<run_id>")
    p.add_argument("--table", type=Path, help="write one comparison row per run to this CSV")
    p.add_argument("--plots", action="store_true", help="write latency_cdf.png and memory.png per run")
    args = p.parse_args(argv)

    summaries = []
    for run_dir in args.runs:
        if not (run_dir / "requests.csv").exists():
            print(f"skipping {run_dir}: no requests.csv", file=sys.stderr)
            continue
        run = load_run(run_dir)
        summary = summarize_run(run)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(format_summary(summary) + "\n")
        if args.plots:
            for path in plot_run(run):
                print(f"  wrote {path}")
        summaries.append(summary)

    if not summaries:
        return 1
    if len(summaries) > 1:
        keys = ["run_id", "concurrency", "tasks_ok", "tasks_per_s", "ttft_p50_s", "ttft_p99_s",
                "task_latency_p99_s", "gpu_mem_peak_mb", "server_rss_peak_mb"]
        width = {k: max(len(k), 8) for k in keys}
        width["run_id"] = max(len(s["run_id"]) for s in summaries)
        print("  ".join(f"{k:<{width[k]}}" if k == "run_id" else f"{k:>{width[k]}}" for k in keys))
        for row in table_rows(summaries):
            print("  ".join(f"{row[k]:<{width[k]}}" if k == "run_id" else f"{_fmt(row[k]):>{width[k]}}"
                            for k in keys))
    if args.table:
        write_table(args.table, summaries)
        print(f"\nwrote {args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
