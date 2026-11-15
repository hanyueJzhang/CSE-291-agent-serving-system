"""Phase 2 scheduler: slot leases (task-level admission, slots held over short tool waits).

Why: on W1 without a host cache, a follow-up usually finds its slot taken by
another task (30-40% of follow-up history reused at 0.95X) and re-prefills its
whole context. That prefill does not only delay the follow-up: llama-server
puts the prompt tokens in the same batch as every other slot's next token, so
the other slots almost stop meanwhile. analysis/prefill_stall.py measures both
sides on seven W1 runs (four slots, FIFO c = 5, no host cache, 0.75X-1.15X):

- each second of a request's decode that overlaps another slot's prefill
  adds 0.93 s to it (R2 0.97), and 24-33% of decode time at 0.95X-1.15X
  overlaps one;
- without that stall a token takes about 8.4 / 10.2 / 13.0 / 13.3 ms with 1-4
  slots busy: four busy slots emit ~300 tokens/s, three ~230.

So an idle slot costs little and a re-prefill costs every slot. Priced in
lost output tokens, keeping one of four slots idle breaks even with
re-prefilling a 2,000-token history after about 1.9 s. (analysis/slot_replay's
hold table compares idle slot-seconds with prefill seconds one to one, which
makes holding look like a loss.)

Policy: a task leases one slot from its first step to its last. Every request
is pinned to a slot (Request.slot -> llama-server id_slot), so at most one
request per slot is in flight and nothing queues inside llama-server.

- A follow-up goes straight back to the slot held for it, ahead of waiting
  first steps. It prefills only the newly appended tokens and the task queues
  once, not once per step.
- Everything else is admitted in arrival order to a free slot that is not
  held: one whose task has finished first (the one sharing the longest prompt
  prefix with the request, then least recently used), otherwise one whose task
  gave it up.
- Tool waits: when a step finishes and the task has LLM steps left, the slot
  is held for hold_max_s if the usual wait after this kind of step is at most
  hold_max_s (running mean of observed waits per task type and step; held the
  first time, when nothing is known). Otherwise it is given up at once. A held
  slot whose owner has not returned after hold_max_s is given up too. The
  history stays in a given-up slot until another request overwrites it.

hold_max_s is the break-even idle time above (default 2 s; W1 tool waits are
0.5-2 s). The five-slot runs are too noisy to give their own value. With
longer tool waits no slot is held and the policy is FIFO with client-side
slot choice; hold_max_s = 0 gives that always.

Assumes a task's later LLM steps carry its context (true of every workload in
workload/tasks); a step that starts a new context still returns to the task's
slot, where the hold bought it nothing.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any, Sequence

from .base import Request, Scheduler

WAIT_ALPHA = 0.2            # weight of the newest observed tool wait in the running mean


@dataclass
class _Slot:
    idx: int
    busy: Request | None = None
    owner: str | None = None             # task whose context the slot holds
    prompt: str = ""                     # that context's first message, for prefix matching
    live: bool = False                   # owner has LLM steps left, so it will want this context again
    hold_until: float | None = None      # idle and reserved for the owner until then
    last_used: float = -1.0


def _shared_prefix(a: str, b: str) -> int:
    """Length of the common prefix of two strings."""
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


class LeaseScheduler(Scheduler):
    name = "lease"

    def __init__(self, max_concurrency: int, *, hold_max_s: float = 2.0, **kwargs):
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if not math.isfinite(hold_max_s) or hold_max_s < 0:
            raise ValueError("hold_max_s must be finite and >= 0")
        super().__init__(**kwargs)
        self.max_concurrency = max_concurrency      # slots 0..n-1 of llama-server
        self.hold_max_s = hold_max_s
        self._slots = [_Slot(i) for i in range(max_concurrency)]
        self._wait: dict[tuple[str, int], float] = {}                 # (task type, step) -> usual wait after it
        self._left: dict[str, tuple[float, tuple[str, int]]] = {}     # task -> (when its last step finished, key)
        self._failed: set[Request] = set()
        self._counts = dict.fromkeys(
            ("holds", "given_up_at_once", "followups_kept", "followups_displaced",
             "live_evictions", "pin_mismatches"), 0)
        self.trace: list[dict[str, Any]] = []       # one row per overwritten live history

    # ---- admission -----------------------------------------------------------

    def _held(self, slot: _Slot, now: float) -> bool:
        return slot.live and slot.hold_until is not None and now < slot.hold_until

    def _home(self, req: Request) -> _Slot | None:
        """The idle slot that still holds this task's context, if any."""
        for slot in self._slots:
            if slot.busy is None and slot.owner == req.task_id:
                return slot
        return None

    def pick_next(self, waiting: Sequence[Request], running: Sequence[Request]) -> Request | None:
        now = self._clock()
        head = waiting[0]
        home = self._home(head)
        if home is not None:
            return self._assign(head, home, now)
        free = [s for s in self._slots if s.busy is None and not self._held(s, now)]
        if free:
            # Finished owners first (longest shared prompt prefix, then LRU), so a
            # live task's history is overwritten only when nothing else is free.
            slot = min(free, key=lambda s: (s.live, -_shared_prefix(s.prompt, head.task_prompt), s.last_used))
            return self._assign(head, slot, now)
        # The head must wait. A later step behind it may still go back to its task's
        # slot: that slot is held, so nobody ahead of it could have used it.
        for req in waiting[1:]:
            home = self._home(req)
            if home is not None:
                return self._assign(req, home, now)
        return None

    def _assign(self, req: Request, slot: _Slot, now: float) -> Request:
        kept = slot.owner == req.task_id
        if req.is_followup:
            self._counts["followups_kept" if kept else "followups_displaced"] += 1
        if slot.live and not kept:
            self._counts["live_evictions"] += 1
            self.trace.append({"t": now, "event": "evict", "slot": slot.idx, "owner": slot.owner,
                               "by": req.task_id, "idle_s": now - slot.last_used})
        for other in self._slots:           # a stale copy of this task's context elsewhere
            if other is not slot and other.owner == req.task_id:
                other.live, other.hold_until = False, None
        slot.busy, slot.owner, slot.prompt = req, req.task_id, req.task_prompt
        slot.live, slot.hold_until = True, None
        req.slot = slot.idx
        return req

    # ---- runner-facing API -------------------------------------------------

    async def acquire(self, req: Request) -> None:
        left = self._left.pop(req.task_id, None)
        if left is not None:
            t_left, key = left
            wait = self._clock() - t_left
            usual = self._wait.get(key)
            self._wait[key] = wait if usual is None else usual + WAIT_ALPHA * (wait - usual)
        await super().acquire(req)

    def observe(self, req: Request, completion: Any) -> None:
        if getattr(completion, "status", "ok") != "ok":
            self._failed.add(req)           # the runner ends the task here
        served = getattr(completion, "slot_id", None)
        if served is not None and req.slot is not None and served != req.slot:
            self._counts["pin_mismatches"] += 1

    def release(self, req: Request) -> None:
        if req not in self._running:
            raise ValueError(f"request {req.task_id}/{req.step_idx} is not running")
        now = self._clock()
        slot = self._slots[req.slot]
        slot.busy, slot.last_used, slot.hold_until = None, now, None
        failed = req in self._failed
        self._failed.discard(req)
        if failed or req.remaining_llm_steps <= 0:
            slot.live = False
        else:
            key = (req.task_type, req.step_idx)
            self._left[req.task_id] = (now, key)
            usual = self._wait.get(key)
            if self.hold_max_s > 0 and (usual is None or usual <= self.hold_max_s):
                self._counts["holds"] += 1
                slot.hold_until = now + self.hold_max_s
                self._wake_when_hold_ends(slot)
            else:
                self._counts["given_up_at_once"] += 1
        super().release(req)

    # ---- hold expiry -----------------------------------------------------------

    def _wake_when_hold_ends(self, slot: _Slot) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return      # no loop: the hold still lapses by the clock at the next arrival or completion
        loop.call_later(self.hold_max_s, self._end_hold, slot, slot.hold_until)

    def _end_hold(self, slot: _Slot, hold_until: float) -> None:
        if slot.busy is None and slot.hold_until == hold_until:     # still the same hold
            slot.hold_until = None
            self._dispatch()

    # ---- reporting -----------------------------------------------------------

    def config(self) -> dict:
        return {**super().config(), "max_concurrency": self.max_concurrency, "hold_max_s": self.hold_max_s}

    def summary(self) -> dict:
        return {
            **self._counts,
            "usual_wait_s": {f"{task_type}/{step}": round(wait, 3)
                             for (task_type, step), wait in sorted(self._wait.items())},
        }
