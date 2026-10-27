"""Mooncake toolagent trace as a workload: one trace request = one task.

A trace workload JSON has a "trace" section instead of a rate:

    "trace": {"path": "workload/traces/toolagent_trace.jsonl",
              "window_s": [0, 900], "fit_ctx_per_slot": 6144},
    "arrival": {"process": "trace", "num_tasks": 210},
    "task_types": [{"type": "mooncake_agent", "steps": [
        {"kind": "llm", "from_trace": true}, ...follow-up steps...]}]

step0 comes from the trace request: its prompt is the concatenation of B(hash_id)
blocks, input scaled by 1/INPUT_SCALE (the last block is cut to the remaining
tokens), output length kept as is. B(id) is BLOCK_WORDS single-token words drawn
with seed id, so equal hash IDs give equal text and shared prefixes stay shared.
The prompt carries no task ID for the same reason. Later steps come from the
template; step0 has no carry_context.

Load is set by sampling whole families (first non-template hash): drop tasks that
don't fit fit_ctx_per_slot, shuffle the window's families with the workload seed,
and add whole families until num_tasks is reached (it may overshoot by one
family). A smaller num_tasks therefore gives a subset of a larger one. Each task's
jitter and tool delays come from its own seeded RNG, so a task is identical in
every set that contains it.

Dry run (no server needed):

    python -m workload.traces.mooncake workload/tasks/mooncake_w1.json --fraction 0.1
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..generator import concrete_steps, estimate_peak_context, load_workload

REPO_ROOT = Path(__file__).resolve().parents[2]
WORDS_PATH = Path(__file__).resolve().parent / "words.txt"

TRACE_BLOCK_TOKENS = 512        # Mooncake hash_ids cover 512-token blocks
INPUT_SCALE = 4                 # prompt tokens = trace input_length / 4
BLOCK_WORDS = TRACE_BLOCK_TOKENS // INPUT_SCALE
TEMPLATE_MIN_REQUESTS = 100     # a hash shared by this many requests is a template
JITTER_S = 3.0                  # trace timestamps are quantized to 3 s
FIT_MARGIN = 0.95               # same margin as the run_experiment context warning


@lru_cache(maxsize=None)
def _words() -> tuple[str, ...]:
    return tuple(WORDS_PATH.read_text(encoding="utf-8").split())


@lru_cache(maxsize=None)
def block_text(hash_id: int) -> str:
    """B(hash_id): BLOCK_WORDS words, each one Llama 3 token after a space."""
    return " ".join(random.Random(hash_id).choices(_words(), k=BLOCK_WORDS))


def scaled_input_tokens(input_length: int) -> int:
    return max(1, math.ceil(input_length / INPUT_SCALE))


def trace_prompt(hash_ids: list[int], num_tokens: int) -> str:
    num_blocks = math.ceil(num_tokens / BLOCK_WORDS)
    blocks = [block_text(h) for h in hash_ids[:num_blocks]]
    tail = num_tokens - BLOCK_WORDS * (num_blocks - 1)
    blocks[-1] = " ".join(blocks[-1].split()[:tail])
    return " ".join(blocks)


@lru_cache(maxsize=None)
def _load_trace(path: str) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Trace rows plus each row's family ID (first non-template hash)."""
    with open(path, encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    counts = Counter(h for row in rows for h in set(row["hash_ids"]))
    templates = {h for h, n in counts.items() if n >= TEMPLATE_MIN_REQUESTS}
    families = {}
    for line, row in enumerate(rows):
        own = [h for h in row["hash_ids"] if h not in templates]
        families[line] = own[0] if own else -1 - line   # all-template: its own family
    return rows, families


def load_trace(trace: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[int, int]]:
    path = Path(trace["path"])
    if not path.is_absolute():
        path = REPO_ROOT / path
    return _load_trace(str(path))


def _follow_up_template(workload: dict[str, Any]) -> dict[str, Any]:
    task_types = workload["task_types"]
    if len(task_types) != 1 or not task_types[0]["steps"][0].get("from_trace"):
        raise ValueError("a trace workload needs exactly one task type whose "
                         'first step is {"kind": "llm", "from_trace": true}')
    return task_types[0]


def _make_task(workload: dict[str, Any], line: int, row: dict[str, Any], family: int) -> dict[str, Any]:
    template = _follow_up_template(workload)
    start_s = workload["trace"]["window_s"][0]
    rng = random.Random(f"{workload['seed']}:{line}")
    task_id = f"task-{line:05d}"
    prompt_tokens = scaled_input_tokens(row["input_length"])
    step0 = {
        "kind": "llm",
        "prompt": trace_prompt(row["hash_ids"], prompt_tokens),
        "prompt_tokens": prompt_tokens,
        "max_output_tokens": row["output_length"],
    }
    arrival_s = row["timestamp"] / 1000 - start_s + rng.uniform(0, JITTER_S)
    return {
        "id": task_id,
        "type": template["type"],
        "family": family,
        "arrival_s": arrival_s,
        "steps": [step0, *concrete_steps(task_id, template["steps"][1:], rng, first_step_idx=1)],
    }


def window_tasks(workload: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """All tasks of the window that fit fit_ctx_per_slot, and the window's request count."""
    trace = workload["trace"]
    rows, families = load_trace(trace)
    start_s, end_s = trace["window_s"]
    limit = FIT_MARGIN * trace["fit_ctx_per_slot"]
    in_window = [line for line, row in enumerate(rows)
                 if start_s * 1000 <= row["timestamp"] < end_s * 1000]
    tasks = [_make_task(workload, line, rows[line], families[line]) for line in in_window]
    return [t for t in tasks if estimate_peak_context(t) <= limit], len(in_window)


def sample_families(tasks: list[dict[str, Any]], num_tasks: int, seed: int) -> list[dict[str, Any]]:
    """Whole families in seeded random order until num_tasks is reached."""
    by_family: dict[int, list[dict[str, Any]]] = {}
    for task in tasks:
        by_family.setdefault(task["family"], []).append(task)
    order = list(by_family)
    random.Random(seed).shuffle(order)
    chosen: list[dict[str, Any]] = []
    for family in order:
        if len(chosen) >= num_tasks:
            break
        chosen.extend(by_family[family])
    return sorted(chosen, key=lambda t: (t["arrival_s"], t["id"]))


def generate_trace_tasks(workload: dict[str, Any]) -> list[dict[str, Any]]:
    tasks, _ = window_tasks(workload)
    return sample_families(tasks, workload["arrival"]["num_tasks"], workload["seed"])


# ---- dry run -------------------------------------------------------------------

def theoretical_prefix_hit(workload: dict[str, Any], tasks: list[dict[str, Any]]) -> dict[str, float]:
    """step0 prefix reuse with an unbounded cache, in arrival order: the share of
    blocks (and of scaled prompt tokens) whose hash an earlier task already sent."""
    rows, _ = load_trace(workload["trace"])
    seen: set[int] = set()
    blocks = hit_blocks = tokens = hit_tokens = 0
    for task in tasks:
        row = rows[int(task["id"].split("-")[1])]
        n_tokens = task["steps"][0]["prompt_tokens"]
        for i, h in enumerate(row["hash_ids"][:math.ceil(n_tokens / BLOCK_WORDS)]):
            size = min(BLOCK_WORDS, n_tokens - i * BLOCK_WORDS)
            blocks += 1
            tokens += size
            if h in seen:
                hit_blocks += 1
                hit_tokens += size
            seen.add(h)
    return {"block": hit_blocks / blocks if blocks else 0.0,
            "token": hit_tokens / tokens if tokens else 0.0}


def dry_run(workload: dict[str, Any]) -> str:
    from analysis.summarize import describe

    candidates, in_window = window_tasks(workload)
    tasks = sample_families(candidates, workload["arrival"]["num_tasks"], workload["seed"])
    start_s, end_s = workload["trace"]["window_s"]
    families = Counter(t["family"] for t in tasks)
    per_minute = Counter(int(t["arrival_s"] // 60) for t in tasks)
    minutes = math.ceil((end_s - start_s) / 60)
    hit = theoretical_prefix_hit(workload, tasks)

    def dist(name: str, values: list[float]) -> str:
        d = describe(values)
        return (f"  {name:<18} p50 {d['p50']:>6.0f}  p95 {d['p95']:>6.0f}  "
                f"p99 {d['p99']:>6.0f}  max {d['max']:>6.0f}  mean {d['mean']:>7.1f}")

    lines = [
        f"{workload['name']}  window [{start_s}, {end_s})s  seed {workload['seed']}",
        f"  requests in window {in_window}, fit {len(candidates)} "
        f"({len(candidates) / in_window:.1%}) at {workload['trace']['fit_ctx_per_slot']} tok/slot",
        f"  tasks {len(tasks)} (target {workload['arrival']['num_tasks']}, "
        f"{len(tasks) / len(candidates):.1%} of fitting)   families {len(families)}, "
        f"multi-member {sum(n > 1 for n in families.values())}, largest {max(families.values(), default=0)}",
        f"  mean rate {len(tasks) / (end_s - start_s):.3f} tasks/s",
        "  tasks per minute " + " ".join(str(per_minute.get(m, 0)) for m in range(minutes)),
        dist("step0 prompt tok", [t["steps"][0]["prompt_tokens"] for t in tasks]),
        dist("step0 output tok", [t["steps"][0]["max_output_tokens"] for t in tasks]),
        dist("peak context tok", [estimate_peak_context(t) for t in tasks]),
        f"  theoretical step0 prefix hit  blocks {hit['block']:.1%}  tokens {hit['token']:.1%}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("workload", help="trace workload JSON, e.g. workload/tasks/mooncake_w1.json")
    p.add_argument("--num-tasks", type=int, help="override arrival.num_tasks")
    p.add_argument("--fraction", type=float, help="num_tasks = this share of the fitting tasks")
    p.add_argument("--seed", type=int, help="override the workload seed")
    args = p.parse_args(argv)

    workload = load_workload(args.workload)
    if args.seed is not None:
        workload["seed"] = args.seed
    if args.num_tasks is not None:
        workload["arrival"]["num_tasks"] = args.num_tasks
    if args.fraction is not None:
        workload["arrival"]["num_tasks"] = round(args.fraction * len(window_tasks(workload)[0]))
    print(dry_run(workload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
