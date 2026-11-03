"""Classify how requests fail when a task's context outgrows its slot.

    python -m analysis.overflow results/phase1_overflow/*/

llama.cpp reserves KV memory at startup, so too much context does not run the
GPU out of memory. It shows up in one of two ways instead:

- error:     prompt + max_tokens does not fit the slot. The pinned llama.cpp
             (972b5fc) checks this up front and rejects the request with
             HTTP 400 "request (N tokens) exceeds the available context size";
             this is what phase1_overflow observed.
- truncated: a response shorter than requested. With ignore_eos every complete
             response has exactly the requested length, so output_tokens <
             requested means generation stopped early (e.g. a runtime that
             truncates instead of rejecting). It would still report status
             "ok", so summarize.py would count it as a success; this script
             is the check.

The requested output length comes from regenerating the run's tasks from the
workload stored in run_config.json. Rows are per run and LLM step index.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from workload.generator import generate_tasks


def requested_outputs(config: dict[str, Any]) -> dict[tuple[str, int], int]:
    tasks = generate_tasks(config["workload"])
    return {
        (task["id"], idx): step["max_output_tokens"]
        for task in tasks
        for idx, step in enumerate(task["steps"])
        if step.get("kind") == "llm"
    }


def classify(row: dict[str, str], requested: int | None) -> str:
    if row["status"] != "ok":
        return row["status"]          # error / timeout
    if requested is not None and int(row["output_tokens"]) < requested:
        return "truncated"
    return "ok"


def analyze(run_dir: Path) -> dict[str, Any]:
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    requested = requested_outputs(config)
    with (run_dir / "requests.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with (run_dir / "tasks.csv").open(newline="", encoding="utf-8") as f:
        task_status = Counter(r["status"] for r in csv.DictReader(f))

    by_step: dict[int, Counter] = {}
    errors: Counter = Counter()
    truncated_tasks: set[str] = set()
    for row in rows:
        step = int(row["step_idx"])
        kind = classify(row, requested.get((row["task_id"], step)))
        by_step.setdefault(step, Counter())[kind] += 1
        if kind == "truncated":
            truncated_tasks.add(row["task_id"])
        if row.get("error"):
            errors[row["error"][:120]] += 1
    return {
        "run": run_dir.name,
        "tasks": dict(task_status),
        "tasks_with_truncated_output": len(truncated_tasks),
        "requests_by_step": {s: dict(c) for s, c in sorted(by_step.items())},
        "top_errors": errors.most_common(3),
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
    for r in results:
        print(f"{r['run']}: tasks {r['tasks']}, tasks with truncated output {r['tasks_with_truncated_output']}")
        for step, counts in r["requests_by_step"].items():
            print(f"  step {step}: {counts}")
        for message, n in r["top_errors"]:
            print(f"  {n} x {message}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
