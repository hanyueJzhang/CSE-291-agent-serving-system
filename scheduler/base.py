"""Scheduler interface"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Sequence


@dataclass(eq=False)
class Request:
    """One LLM call (one agent step) as seen by the scheduler."""
    task_id: str
    step_idx: int
    task_type: str = ""
    prompt_tokens: int = 0
    max_output_tokens: int = 0
    payload: Any = None

    # Set by the scheduler, in seconds on the scheduler's clock.
    t_arrive: float | None = None
    t_dispatch: float | None = None
    t_finish: float | None = None

    @property
    def queue_s(self) -> float | None:
        if self.t_arrive is None or self.t_dispatch is None:
            return None
        return self.t_dispatch - self.t_arrive


class Scheduler:
    """Base class: queue mechanics, timestamps, cancellation. Policies override pick_next()."""

    name = "base"

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        # Pass the runner's clock so all CSV timestamps agree.
        self._clock = clock
        self._waiting: list[Request] = []   # arrival order
        self._running: list[Request] = []   # admission order
        self._grants: dict[Request, asyncio.Future] = {}

    # ---- policy hook -------------------------------------------------------

    def pick_next(self, waiting: Sequence[Request], running: Sequence[Request]) -> Request | None:
        """Return a request from `waiting` to admit now, or None to admit nothing yet.

        Called after every arrival and every completion, repeatedly until it
        returns None. `waiting` is in arrival order. Do not mutate the lists.
        """
        raise NotImplementedError

    def config(self) -> dict:
        """Policy name and parameters, for run_config.json."""
        return {"name": self.name}

    # ---- runner-facing API -------------------------------------------------

    @asynccontextmanager
    async def admit(self, req: Request) -> AsyncIterator[Request]:
        """Wait for admission, run the body, then release."""
        await self.acquire(req)
        try:
            yield req
        finally:
            self.release(req)

    async def acquire(self, req: Request) -> None:
        """Enqueue `req` and wait until the policy admits it. Prefer admit()."""
        if req in self._grants or req in self._running:
            raise ValueError(f"request {req.task_id}/{req.step_idx} already submitted")
        if req.t_arrive is None:
            req.t_arrive = self._clock()
        future = asyncio.get_running_loop().create_future()
        self._grants[req] = future
        self._waiting.append(req)
        self._dispatch()
        try:
            await future
        except asyncio.CancelledError:
            # Cancelled while waiting, or right after being admitted.
            self._grants.pop(req, None)
            if req in self._waiting:
                self._waiting.remove(req)
            elif req in self._running:
                self.release(req)
            raise

    def release(self, req: Request) -> None:
        """Mark an admitted request finished and admit whatever the policy picks next."""
        if req not in self._running:
            raise ValueError(f"request {req.task_id}/{req.step_idx} is not running")
        self._running.remove(req)
        req.t_finish = self._clock()
        self._dispatch()

    # ---- observability -----------------------------------------------------

    @property
    def queue_len(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

    # ---- internals ---------------------------------------------------------

    def _dispatch(self) -> None:
        # Drop waiters whose task was cancelled but hasn't run its except block yet.
        self._waiting = [r for r in self._waiting if not self._grants[r].cancelled()]
        while self._waiting:
            req = self.pick_next(self._waiting, self._running)
            if req is None:
                break
            if req not in self._waiting:
                raise RuntimeError(f"{type(self).__name__}.pick_next returned a request that is not waiting")
            self._waiting.remove(req)
            self._running.append(req)
            req.t_dispatch = self._clock()
            self._grants.pop(req).set_result(None)
