"""Phase 2 agent-aware scheduler (proposal 2.1): continuation-first with aging.

Same admission cap as FIFO; only the order changes. A ready follow-up step
(its tool call has returned) goes ahead of a task's first step, because the
task has already used the GPU and is one or two steps from done: this is close
to "least remaining work first" and turns a three-step agent's three queue
waits into roughly one. Within a class, arrival order.

Aging: a first step that has waited aging_s seconds joins the follow-up class
(ordered by arrival with them), so new tasks cannot starve under a steady
stream of follow-ups. aging_s=None disables aging.

continuation_first=False gives plain FIFO, so the same class can serve as the
"off" point of the ablation.
"""
from __future__ import annotations

from typing import Callable, Sequence

from .base import Request, Scheduler


class AgentScheduler(Scheduler):
    name = "agent"

    def __init__(
        self,
        max_concurrency: int,
        *,
        continuation_first: bool = True,
        aging_s: float | None = None,
        **kwargs,
    ):
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if aging_s is not None and aging_s < 0:
            raise ValueError("aging_s must be >= 0 or None")
        super().__init__(**kwargs)
        self.max_concurrency = max_concurrency
        self.continuation_first = continuation_first
        self.aging_s = aging_s

    def priority_class(self, req: Request, now: float) -> int:
        """0 = goes first (follow-up, or aged first step), 1 = first step."""
        if not self.continuation_first or req.is_followup:
            return 0
        if self.aging_s is not None and req.t_arrive is not None and now - req.t_arrive >= self.aging_s:
            return 0
        return 1

    def pick_next(self, waiting: Sequence[Request], running: Sequence[Request]) -> Request | None:
        if len(running) >= self.max_concurrency:
            return None
        if not self.continuation_first:
            return waiting[0]
        now = self._clock()
        # waiting is in arrival order, so min() keeps arrival order within a class.
        return min(waiting, key=lambda r: self.priority_class(r, now))

    def config(self) -> dict:
        return {
            **super().config(),
            "max_concurrency": self.max_concurrency,
            "continuation_first": self.continuation_first,
            "aging_s": self.aging_s,
        }
