"""Per-request slot timeline of one run, shared by slot_replay and slot_gaps.

Needs requests.csv with slot_id (Phase 2 runs). The time a request held its
slot is estimated from the client's timestamps:

    start = max(t_dispatch, t_first_token - prefill_ms / 1000)
    end   = t_finish

With concurrency <= slots, start ~= t_dispatch. With one more admitted request
than slots, the extra one waits inside llama-server until a slot frees; the
first-token time minus its prefill recovers when it actually got the slot.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _f(value: str | None) -> float | None:
    return float(value) if value not in (None, "") else None


def _i(value: str | None) -> int | None:
    return int(float(value)) if value not in (None, "") else None


@dataclass(eq=False)
class Req:
    task_id: str
    step_idx: int
    t_arrive: float
    t_dispatch: float
    start: float
    end: float
    prompt: int
    cached: int
    output: int
    prefill_ms: float
    slot_id: int | None
    # Filled by load_run.
    prev: "Req | None" = None
    next: "Req | None" = None
    carried: int = 0              # tokens of history this request extends (0 for a first step)

    @property
    def is_followup(self) -> bool:
        return self.prev is not None and self.carried > 0

    @property
    def held(self) -> int:
        """Tokens left in the slot when the request finishes."""
        return self.prompt + self.output


@dataclass
class Run:
    path: Path
    n_slots: int
    requests: list[Req]                     # ok requests, sorted by start
    config: dict[str, Any] = field(default_factory=dict)
    step0_words: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def makespan(self) -> float:
        return max((r.end for r in self.requests), default=0.0)

    @property
    def has_slots(self) -> bool:
        return bool(self.requests) and all(r.slot_id is not None for r in self.requests)

    def prefill_ms_per_token(self) -> float:
        """Mean prefill time per uncached prompt token over the run."""
        ms = sum(r.prefill_ms for r in self.requests)
        tokens = sum(max(r.prompt - r.cached, 0) for r in self.requests)
        return ms / tokens if tokens else 0.0

    def shared_prefix(self, a: str, b: str) -> int:
        """Approximate common token prefix of two tasks' contexts (their step-0 prompts).

        Mooncake prompts are one-token words, so words ~= tokens; synthetic
        prompts start with the task ID and share nothing. The per-run salt and
        chat-template header are a few tokens common to all and are ignored.
        """
        if a == b:
            return 0
        wa, wb = self.step0_words.get(a), self.step0_words.get(b)
        if not wa or not wb:
            return 0
        n = 0
        for x, y in zip(wa, wb):
            if x != y:
                break
            n += 1
        return n


def _step0_words(config: dict[str, Any], notes: list[str]) -> dict[str, list[str]]:
    workload = config.get("workload")
    if not workload:
        return {}
    try:
        from workload.generator import generate_tasks
        tasks = generate_tasks(workload)
    except Exception as exc:  # e.g. trace file not downloaded
        notes.append(f"could not regenerate tasks for cross-task prefixes ({exc!r}); assuming none")
        return {}
    words = {}
    for task in tasks:
        first = next((s for s in task["steps"] if s["kind"] == "llm"), None)
        if first is not None:
            words[task["id"]] = first["prompt"].split()
    return words


def load_run(run_dir: str | Path, *, prefixes: bool = True) -> Run:
    run_dir = Path(run_dir)
    config = {}
    if (run_dir / "run_config.json").is_file():
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    with (run_dir / "requests.csv").open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") == "ok"]

    reqs = []
    for row in rows:
        t_dispatch = _f(row["t_dispatch"]) or 0.0
        first = _f(row.get("t_first_token"))
        prefill_ms = _f(row.get("prefill_ms")) or 0.0
        start = t_dispatch if first is None else max(t_dispatch, first - prefill_ms / 1000)
        reqs.append(Req(
            task_id=row["task_id"], step_idx=int(row["step_idx"]),
            t_arrive=_f(row["t_arrive"]) or t_dispatch, t_dispatch=t_dispatch,
            start=start, end=_f(row["t_finish"]) or start,
            prompt=_i(row["prompt_tokens"]) or 0, cached=_i(row["cached_tokens"]) or 0,
            output=_i(row["output_tokens"]) or 0, prefill_ms=prefill_ms,
            slot_id=_i(row.get("slot_id")),
        ))

    by_task: dict[str, list[Req]] = {}
    for r in reqs:
        by_task.setdefault(r.task_id, []).append(r)
    for steps in by_task.values():
        steps.sort(key=lambda r: r.step_idx)
        for prev, cur in zip(steps, steps[1:]):
            cur.prev, prev.next = prev, cur
            history = prev.prompt + prev.output
            # A step that carries context has a prompt at least as long as the history.
            cur.carried = history if cur.prompt >= history else 0

    reqs.sort(key=lambda r: (r.start, r.t_dispatch))
    server = config.get("server") or {}
    n_slots = server.get("total_slots") or (max((r.slot_id for r in reqs if r.slot_id is not None), default=-1) + 1)
    notes: list[str] = []
    words = _step0_words(config, notes) if prefixes else {}
    return Run(path=run_dir, n_slots=int(n_slots or 1), requests=reqs, config=config,
               step0_words=words, notes=notes)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * q / 100
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)
