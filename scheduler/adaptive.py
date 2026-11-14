"""FIFO admission with bounded, windowed concurrency probes.

Only saturated windows are compared. This is an experimental controller, not
an estimator of free GPU memory or a predictor of future agent steps.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Sequence

from .base import Request, Scheduler


class AdaptiveScheduler(Scheduler):
    name = "adaptive"

    def __init__(
        self,
        max_concurrency: int,
        *,
        min_concurrency: int = 1,
        initial_concurrency: int | None = None,
        window_s: float = 15.0,
        min_completions: int = 8,
        backlog_fraction: float = 0.5,
        improvement: float = 0.05,
        throughput_tolerance: float = 0.03,
        latency_tolerance: float = 0.20,
        cooldown_s: float = 30.0,
        max_cooldown_s: float = 120.0,
        **kwargs,
    ):
        initial = max_concurrency if initial_concurrency is None else initial_concurrency
        if any(type(value) is not int for value in (min_concurrency, initial, max_concurrency, min_completions)):
            raise ValueError("concurrency bounds and min_completions must be integers")
        if not 1 <= min_concurrency <= initial <= max_concurrency:
            raise ValueError("expected 1 <= min_concurrency <= initial_concurrency <= max_concurrency")
        if not math.isfinite(window_s) or window_s <= 0 or min_completions < 1:
            raise ValueError("window_s must be positive and finite; min_completions must be >= 1")
        if not (math.isfinite(cooldown_s) and math.isfinite(max_cooldown_s)
                and 0 <= cooldown_s <= max_cooldown_s):
            raise ValueError("expected finite 0 <= cooldown_s <= max_cooldown_s")
        for key, value in {
            "backlog_fraction": backlog_fraction,
            "improvement": improvement,
            "throughput_tolerance": throughput_tolerance,
            "latency_tolerance": latency_tolerance,
        }.items():
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{key} must be between 0 and 1")
        super().__init__(**kwargs)
        self.max_concurrency = max_concurrency  # immutable hard upper bound
        self.min_concurrency = min_concurrency
        self.initial_concurrency = initial
        self.current_concurrency = initial
        self.window_s = window_s
        self.min_completions = min_completions
        self.backlog_fraction = backlog_fraction
        self.improvement = improvement
        self.throughput_tolerance = throughput_tolerance
        self.latency_tolerance = latency_tolerance
        self.cooldown_s = cooldown_s
        self.max_cooldown_s = max_cooldown_s
        self._next_probe_at = self._clock()
        self._next_retry_cooldown_s = cooldown_s
        self.events: list[dict[str, Any]] = []
        self._feedback: dict[Request, tuple[str, int | None]] = {}
        self._settling: set[Request] = set()
        self._reference: dict[str, Any] | None = None
        self._probe_from: int | None = None
        self._direction = -1
        self._backlogged = False
        self._reset_window(self._clock())

    def config(self) -> dict:
        return {
            **super().config(),
            "max_concurrency": self.max_concurrency,
            "min_concurrency": self.min_concurrency,
            "initial_concurrency": self.initial_concurrency,
            "window_s": self.window_s,
            "min_completions": self.min_completions,
            "backlog_fraction": self.backlog_fraction,
            "improvement": self.improvement,
            "throughput_tolerance": self.throughput_tolerance,
            "latency_tolerance": self.latency_tolerance,
            "cooldown_s": self.cooldown_s,
            "max_cooldown_s": self.max_cooldown_s,
        }

    def pick_next(self, waiting: Sequence[Request], running: Sequence[Request]) -> Request | None:
        return waiting[0] if len(running) < self.current_concurrency else None

    async def acquire(self, req: Request) -> None:
        try:
            await super().acquire(req)
        except asyncio.CancelledError:
            # Update backlog accounting even when the last queued waiter leaves.
            self._dispatch()
            raise

    def on_completion(self, req: Request, *, status: str, output_tokens: int | None) -> None:
        """Receive actual results before release; cancelled calls have no result."""
        if req not in self._running:
            raise ValueError("completion feedback requires a running request")
        self._feedback[req] = (status, output_tokens)

    def finish(self) -> None:
        """Record the partial final window without accepting an unfinished probe."""
        if self._running or self._waiting:
            raise RuntimeError("finish requires a drained scheduler")
        self._account()
        self._finish_window(self._clock(), "run_end", valid=False)

    def _reset_window(self, now: float) -> None:
        self._window_start = self._last_account = now
        self._backlog_s = 0.0
        self._samples: list[tuple[int, float, float]] = []
        self._errors = 0
        self._missing_usage = 0

    def _account(self) -> None:
        now = self._clock()
        if self._backlogged:
            self._backlog_s += now - self._last_account
        self._last_account = now

    def _dispatch(self) -> None:
        self._account()
        super()._dispatch()
        self._backlogged = bool(self._waiting) and len(self._running) >= self.current_concurrency

    def release(self, req: Request) -> None:
        if req not in self._running:
            raise ValueError(f"request {req.task_id}/{req.step_idx} is not running")
        self._account()
        feedback = self._feedback.pop(req, None)
        was_settling = bool(self._settling)
        self._settling.discard(req)
        if not was_settling and feedback is not None:
            status, tokens = feedback
            if status != "ok":
                self._errors += 1
            elif tokens is None or tokens <= 0:
                self._missing_usage += 1
            else:
                now = self._clock()
                self._samples.append((tokens, now - req.t_arrive, now - req.t_dispatch))
        super().release(req)
        now = self._clock()
        if feedback is None:
            # Cancellation/exception is not a fast successful completion.
            self._finish_window(now, "interrupted", valid=False)
        elif was_settling:
            if not self._settling:
                self._reset_window(now)
        elif now - self._window_start >= self.window_s:
            self._evaluate(now)

    def _evaluate(self, now: float) -> None:
        duration = now - self._window_start
        reason = "measured"
        if self._errors:
            reason = "request_errors"
        elif self._missing_usage:
            reason = "missing_usage"
        elif len(self._samples) < self.min_completions:
            reason = "insufficient_samples"
        elif self._backlog_s / duration < self.backlog_fraction:
            reason = "insufficient_backlog"
        self._finish_window(now, reason, valid=reason == "measured")

    def _finish_window(self, now: float, reason: str, *, valid: bool) -> None:
        duration = now - self._window_start
        n = len(self._samples)
        observed = {
            "t": now,
            "duration_s": duration,
            "concurrency": self.current_concurrency,
            "completions": n,
            "errors": self._errors,
            "missing_usage": self._missing_usage,
            "backlog_fraction": self._backlog_s / duration if duration else 0.0,
            "output_tokens_per_s": sum(s[0] for s in self._samples) / duration if duration else 0.0,
            "mean_request_latency_s": sum(s[1] for s in self._samples) / n if n else None,
            "mean_service_s": sum(s[2] for s in self._samples) / n if n else None,
            "reason": reason,
        }
        target = self.current_concurrency
        action = "hold"
        if not valid:
            if self._probe_from is not None:
                target = self._probe_from
                action = "abort_probe"
            self._reference = None
            self._probe_from = None
        elif self._probe_from is None and now < self._next_probe_at:
            action = "cooldown"
        elif self._probe_from is None:
            target += self._direction
            if not self.min_concurrency <= target <= self.max_concurrency:
                self._direction *= -1
                target = self.current_concurrency + self._direction
            if self.min_concurrency <= target <= self.max_concurrency:
                self._reference = observed
                self._probe_from = self.current_concurrency
                action = "start_probe"
            else:
                target = self.current_concurrency
        else:
            ref = self._reference
            rate, old_rate = observed["output_tokens_per_s"], ref["output_tokens_per_s"]
            latency, old_latency = observed["mean_request_latency_s"], ref["mean_request_latency_s"]
            faster = rate >= old_rate * (1 + self.improvement)
            latency_ok = latency <= old_latency * (1 + self.latency_tolerance)
            cheaper = (
                self.current_concurrency < self._probe_from
                and rate >= old_rate * (1 - self.throughput_tolerance)
                and latency <= old_latency * (1 - self.improvement)
            )
            if (faster and latency_ok) or cheaper:
                action = "accept_probe"
            else:
                target = self._probe_from
                self._direction *= -1
                action = "reject_probe"
            self._probe_from = None
            self._reference = None
        if action in {"accept_probe", "reject_probe", "abort_probe"} and reason != "run_end":
            if action == "accept_probe":
                delay = self.cooldown_s
                self._next_retry_cooldown_s = self.cooldown_s
            else:
                delay = self._next_retry_cooldown_s
                self._next_retry_cooldown_s = min(self.max_cooldown_s, delay * 2)
            self._next_probe_at = now + delay
        observed.update(
            action=action,
            next_concurrency=target,
            next_probe_at=self._next_probe_at,
            next_retry_cooldown_s=self._next_retry_cooldown_s,
        )
        self.events.append(observed)
        if target != self.current_concurrency:
            self.current_concurrency = target
            # Let requests dispatched at the old cap drain without preemption.
            self._settling = set(self._running)
            self._dispatch()
        self._reset_window(now)
