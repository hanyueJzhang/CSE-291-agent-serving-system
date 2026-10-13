"""Phase 1 baseline: strict FIFO with a fixed concurrency cap."""
from __future__ import annotations

from typing import Sequence

from .base import Request, Scheduler


class FIFOScheduler(Scheduler):
    """Admit requests in arrival order, at most `max_concurrency` in flight.
    """

    name = "fifo"

    def __init__(self, max_concurrency: int, **kwargs):
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        super().__init__(**kwargs)
        self.max_concurrency = max_concurrency

    def pick_next(self, waiting: Sequence[Request], running: Sequence[Request]) -> Request | None:
        if len(running) < self.max_concurrency:
            return waiting[0]
        return None

    def config(self) -> dict:
        return {**super().config(), "max_concurrency": self.max_concurrency}
