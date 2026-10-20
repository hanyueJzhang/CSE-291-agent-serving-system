"""Small test doubles used while developing the workload runner."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable

from scheduler import Request
from workload.llama_client import Completion


class FakeModelClient:
    """Record model calls and return a deterministic response."""

    def __init__(self, response: str = "Fake model response"):
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def generate(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        return self.response

    async def complete(self, messages: list[dict[str, str]], max_tokens: int) -> Completion:
        return Completion(text=await self.generate(messages=messages, max_tokens=max_tokens))


class ImmediateScheduler:
    """Admit every request immediately; suitable only as a runner test double."""

    name = "immediate"

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._num_running = 0

    @asynccontextmanager
    async def admit(self, request: Request) -> AsyncGenerator[Request, None]:
        request.t_arrive = self._clock()
        request.t_dispatch = request.t_arrive
        self._num_running += 1
        try:
            yield request
        finally:
            self._num_running -= 1
            request.t_finish = self._clock()

    @property
    def queue_len(self) -> int:
        return 0

    @property
    def num_running(self) -> int:
        return self._num_running

    def config(self) -> dict[str, str]:
        return {"name": self.name}
