"""Phase 2 scheduler: adaptive admission (concurrency limit found online) plus
overload protection.

Why: Phase 2 showed that the admission cap is the scheduling knob latency is
most sensitive to, and that its best value moves with the server: on four
6144-token slots c = 5 was best (c = 4 starves the GPU between requests,
c = 8 stretches decode), on five slots c = 6. In every case the best cap was
"every slot busy, plus about one request already waiting inside
llama-server", so a freed slot is refilled at once without a hidden queue
building up in the server. A fixed cap has to be re-tuned for every server
configuration; this scheduler finds it from the responses it already gets.

Signal: for each finished request the server-side start is estimated as in
analysis/slot_trace.py (first token - prefill time), so its wait inside
llama-server is start - dispatch. Over a window of recent completions,
Little's law gives the average number of requests waiting inside the server:

    Q = sum(server waits) / window duration

Control (evaluated every window/2 completions once the window is full; after
a change the window is cleared, so each limit is judged on its own data):

    Q > target + band                     -> limit - 1   (hidden server queue)
    Q < target - band and client backlog  -> limit + 1   (slots go idle while we hold work)
    otherwise                             -> keep

Defaults (target 0.75, band 0.5) come from the Phase 1 W1 runs on four slots:
over a whole run Q was 0.02-0.15 at c = slots (0.15 with host-cache restores),
0.45-0.83 at c = slots + 1, 0.74 at slots + 2 and 1.28 at slots + 4; inside a
saturated window the values are higher. So the band keeps slots + 1, raises
c = slots and lowers anything well above slots + 1.

"Client backlog" means the cap actually held a request back since the last
evaluation; without it a low Q only means the load is light, and raising the
limit would change nothing.

Overload protection (overload_wait_s): when the oldest waiting first step has
waited at least overload_wait_s, the queue is no longer a transient burst. The
scheduler then admits ready follow-up steps ahead of first steps, so tasks
already in progress finish instead of all tasks slowing down together. It
switches back to FIFO once no first step has waited overload_wait_s / 2. At
0.95X load this rarely triggers, which avoids the new-task starvation that
always-on continuation-first showed in Phase 2.
"""
from __future__ import annotations

from collections import deque
from statistics import median
from typing import Any, Sequence

from .base import Request, Scheduler


class AdaptiveScheduler(Scheduler):
    name = "adaptive"

    def __init__(
        self,
        max_concurrency: int,
        *,
        min_concurrency: int = 1,
        max_limit: int | None = None,
        target_server_queue: float = 0.75,
        band: float = 0.5,
        window: int = 16,
        overload_wait_s: float | None = None,
        **kwargs,
    ):
        if max_concurrency < 1 or min_concurrency < 1:
            raise ValueError("concurrency limits must be >= 1")
        if window < 2:
            raise ValueError("window must be >= 2")
        if band <= 0 or target_server_queue - band < 0:
            raise ValueError("need band > 0 and target_server_queue >= band")
        if overload_wait_s is not None and overload_wait_s <= 0:
            raise ValueError("overload_wait_s must be > 0 or None")
        super().__init__(**kwargs)
        self.initial_limit = max_concurrency
        self.limit = max_concurrency
        self.min_limit = min_concurrency
        self.max_limit = max_limit if max_limit is not None else max(4 * max_concurrency, 16)
        self.target = target_server_queue
        self.band = band
        self.window = window
        self.overload_wait_s = overload_wait_s

        self._obs: deque[tuple[float, float, float | None]] = deque(maxlen=window)  # (t_dispatch, wait, tpot)
        self._since_eval = 0
        self._held_back = False
        self.overload = False
        self.trace: list[dict[str, Any]] = []        # one row per evaluation or mode switch

    # ---- ordering and admission -------------------------------------------

    def _update_overload(self, waiting: Sequence[Request], now: float) -> None:
        if self.overload_wait_s is None:
            return
        firsts = [now - r.t_arrive for r in waiting if not r.is_followup and r.t_arrive is not None]
        oldest = max(firsts, default=0.0)
        if not self.overload and oldest >= self.overload_wait_s:
            self.overload = True
            self.trace.append({"t": now, "event": "overload_on", "limit": self.limit, "oldest_first_wait_s": oldest})
        elif self.overload and oldest < self.overload_wait_s / 2:
            self.overload = False
            self.trace.append({"t": now, "event": "overload_off", "limit": self.limit, "oldest_first_wait_s": oldest})

    def pick_next(self, waiting: Sequence[Request], running: Sequence[Request]) -> Request | None:
        now = self._clock()
        self._update_overload(waiting, now)
        if len(running) >= self.limit:
            self._held_back = True
            return None
        if self.overload:
            return min(waiting, key=lambda r: 0 if r.is_followup else 1)   # stable: arrival order within class
        return waiting[0]

    # ---- feedback -----------------------------------------------------------

    def observe(self, req: Request, completion: Any) -> None:
        if getattr(completion, "status", "ok") != "ok" or req.t_dispatch is None:
            return
        first = getattr(completion, "t_first_token", None)
        if first is None:
            return
        prefill_s = (getattr(completion, "prefill_ms", None) or 0.0) / 1000
        wait = max(first - prefill_s - req.t_dispatch, 0.0)
        out, decode_ms = getattr(completion, "output_tokens", None), getattr(completion, "decode_ms", None)
        tpot = decode_ms / out if out and decode_ms is not None else None
        self._obs.append((req.t_dispatch, wait, tpot))
        self._since_eval += 1
        if len(self._obs) == self.window and self._since_eval >= self.window // 2:
            self._evaluate()

    def server_queue_estimate(self) -> float | None:
        """Average number of requests waiting inside llama-server over the window."""
        if len(self._obs) < 2:
            return None
        span = self._clock() - self._obs[0][0]
        return sum(w for _, w, _ in self._obs) / span if span > 0 else None

    def _evaluate(self) -> None:
        q = self.server_queue_estimate()
        if q is None:
            return
        old = self.limit
        if q > self.target + self.band and self.limit > self.min_limit:
            self.limit -= 1
        elif q < self.target - self.band and self._held_back and self.limit < self.max_limit:
            self.limit += 1
        tpots = [t for _, _, t in self._obs if t is not None]
        self.trace.append({
            "t": self._clock(), "event": "eval", "limit": self.limit, "prev_limit": old,
            "server_queue": q, "held_back": self._held_back, "running": self.num_running,
            "waiting": self.queue_len, "tpot_ms_p50": median(tpots) if tpots else None,
        })
        self._since_eval = 0
        self._held_back = False
        if self.limit != old:
            self._obs.clear()            # judge the new limit on its own completions only
        if self.limit > old:
            self._dispatch()

    def config(self) -> dict:
        return {
            **super().config(),
            "max_concurrency": self.initial_limit,
            "min_concurrency": self.min_limit,
            "max_limit": self.max_limit,
            "target_server_queue": self.target,
            "band": self.band,
            "window": self.window,
            "overload_wait_s": self.overload_wait_s,
        }

    def summary(self) -> dict:
        evals = [r for r in self.trace if r["event"] == "eval"]
        return {
            "final_limit": self.limit,
            "limits_seen": sorted({r["limit"] for r in evals} | {self.initial_limit}),
            "evaluations": len(evals),
            "overload_switches": sum(r["event"].startswith("overload") for r in self.trace),
        }
