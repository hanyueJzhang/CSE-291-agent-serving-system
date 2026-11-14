import asyncio
import unittest

from scheduler import AdaptiveScheduler, FIFOScheduler, Request, make_scheduler
from tests.test_scheduler import FakeClock, settle


class AdaptiveConfigTests(unittest.TestCase):
    def test_factory_and_bounds(self):
        scheduler = make_scheduler("adaptive", max_concurrency=6, min_concurrency=4,
                                   initial_concurrency=5)
        self.assertIsInstance(scheduler, AdaptiveScheduler)
        self.assertEqual(scheduler.current_concurrency, 5)
        self.assertEqual(scheduler.config()["max_concurrency"], 6)
        self.assertEqual(scheduler.config()["initial_concurrency"], 5)
        self.assertEqual(scheduler.config()["cooldown_s"], 30.0)
        self.assertEqual(scheduler.config()["max_cooldown_s"], 120.0)
        self.assertEqual(FIFOScheduler(5).config(), {"name": "fifo", "max_concurrency": 5})

    def test_invalid_config_is_rejected(self):
        for params in [
            {"max_concurrency": 0},
            {"min_concurrency": 0},
            {"min_concurrency": 4},
            {"initial_concurrency": 4},
            {"window_s": 0},
            {"window_s": float("nan")},
            {"min_completions": 0},
            {"max_concurrency": 2.5},
            {"backlog_fraction": 0},
            {"improvement": float("inf")},
            {"cooldown_s": -1},
            {"cooldown_s": float("nan")},
            {"max_cooldown_s": float("inf")},
            {"cooldown_s": 40, "max_cooldown_s": 30},
        ]:
            with self.subTest(params=params), self.assertRaises(ValueError):
                AdaptiveScheduler(**{"max_concurrency": 3, **params})


class AdaptiveAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def load(self, *, n=40, **params):
        self.clock = FakeClock()
        self.scheduler = AdaptiveScheduler(
            **{"max_concurrency": 3, "initial_concurrency": 2, "min_concurrency": 1,
               "window_s": 2.0, "min_completions": 2, "cooldown_s": 0.0, **params}, clock=self.clock,
        )
        # All requests have already waited a long time; this avoids a growing
        # startup queue dominating the latency guard in throughput-probe tests.
        self.requests = [Request(str(i), 0, t_arrive=-100.0) for i in range(n)]
        self.workers = [asyncio.create_task(self.scheduler.acquire(r)) for r in self.requests]
        await settle()
        self.addAsyncCleanup(self.cleanup, self.scheduler, self.workers)
        return self.scheduler

    async def cleanup(self, scheduler=None, workers=None):
        scheduler = self.scheduler if scheduler is None else scheduler
        workers = self.workers if workers is None else workers
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        for req in list(scheduler._running):
            scheduler.release(req)

    async def complete(self, *, dt=1.0, tokens=10, status="ok", req=None):
        req = req or self.scheduler._running[0]
        self.clock.now += dt
        self.scheduler.on_completion(req, status=status, output_tokens=tokens)
        self.scheduler.release(req)
        await settle()

    async def drain_transition(self):
        for req in list(self.scheduler._settling):
            await self.complete(dt=0.0, req=req)
        self.assertFalse(self.scheduler._settling)

    async def test_rejects_lower_cap_when_goodput_falls_and_does_not_preempt(self):
        scheduler = await self.load()
        await self.complete()
        await self.complete()
        self.assertEqual(scheduler.current_concurrency, 1)
        # Lowering the cap does not cancel the two already-dispatched requests.
        self.assertEqual(scheduler.num_running, 2)
        self.assertEqual(scheduler.events[-1]["action"], "start_probe")
        await self.drain_transition()
        self.assertEqual(scheduler.num_running, 1)
        await self.complete(tokens=5)
        await self.complete(tokens=5)
        self.assertEqual(scheduler.current_concurrency, 2)
        self.assertEqual(scheduler.events[-1]["action"], "reject_probe")
        self.assertEqual(scheduler.events[-1]["output_tokens_per_s"], 5.0)
        self.assertEqual(scheduler.config()["initial_concurrency"], 2)

    async def test_accepts_higher_cap_when_goodput_improves(self):
        scheduler = await self.load(min_concurrency=2)
        await self.complete()
        await self.complete()
        self.assertEqual(scheduler.current_concurrency, 3)
        await self.drain_transition()
        for _ in range(4):
            await self.complete(dt=0.5, tokens=20)
        self.assertEqual(scheduler.current_concurrency, 3)
        self.assertEqual(scheduler.events[-1]["action"], "accept_probe")

    async def test_increase_is_rejected_if_latency_regresses(self):
        scheduler = await self.load(min_concurrency=2)
        await self.complete()
        await self.complete()
        await self.drain_transition()
        for req in scheduler._running:
            req.t_arrive = -1000
        await self.complete(tokens=100)
        await self.complete(tokens=100)
        self.assertEqual(scheduler.current_concurrency, 2)
        self.assertEqual(scheduler.events[-1]["action"], "reject_probe")

    async def test_unchanged_goodput_can_accept_lower_cap_with_lower_latency(self):
        scheduler = await self.load()
        await self.complete()
        await self.complete()
        await self.drain_transition()
        for _ in range(2):
            req = scheduler._running[0]
            req.t_arrive = self.clock.now
            await self.complete(req=req)
        self.assertEqual(scheduler.current_concurrency, 1)
        self.assertEqual(scheduler.events[-1]["action"], "accept_probe")

    async def test_low_load_does_not_trigger_probes(self):
        scheduler = await self.load(n=2)
        await self.complete()
        await self.complete()
        self.assertEqual(scheduler.current_concurrency, 2)
        self.assertEqual(scheduler.events[-1]["reason"], "insufficient_backlog")

    async def test_missing_usage_and_errors_cannot_win_a_probe(self):
        for status, tokens in [("error", 1000), ("timeout", None), ("ok", None)]:
            with self.subTest(status=status, tokens=tokens):
                scheduler = await self.load(min_concurrency=2)
                await self.complete()
                await self.complete()
                await self.drain_transition()
                await self.complete(status=status, tokens=tokens)
                await self.complete(tokens=1000)
                self.assertEqual(scheduler.current_concurrency, 2)
                self.assertEqual(scheduler.events[-1]["action"], "abort_probe")
                await self.cleanup()

    async def test_fifo_order_and_fixed_bounds(self):
        scheduler = await self.load(n=10, min_concurrency=2, max_concurrency=2)
        order = []
        while scheduler._running:
            order.append(scheduler._running[0].task_id)
            await self.complete()
        self.assertEqual(order, [str(i) for i in range(10)])
        self.assertEqual(scheduler.current_concurrency, 2)
        scheduler.finish()
        self.assertEqual(scheduler.events[-1]["reason"], "run_end")

    async def test_cancellation_does_not_leak_capacity_or_count_as_success(self):
        scheduler = await self.load(n=5)
        self.workers[2].cancel()
        await settle()
        self.assertEqual(scheduler.queue_len, 2)
        scheduler.release(self.requests[0])  # exception/cancellation without feedback
        await settle()
        self.assertEqual(scheduler.num_running, 2)
        self.assertEqual(scheduler.events[-1]["reason"], "interrupted")
        self.assertEqual(scheduler.events[-1]["completions"], 0)

    async def test_finish_reverts_an_unmeasured_probe(self):
        scheduler = await self.load(n=4)
        await self.complete()
        await self.complete()
        self.assertEqual(scheduler.current_concurrency, 1)
        await self.drain_transition()
        scheduler.finish()
        self.assertEqual(scheduler.current_concurrency, 2)
        self.assertEqual(scheduler.events[-1]["action"], "abort_probe")

    async def test_cancelling_last_waiter_stops_backlog_accounting(self):
        scheduler = await self.load(n=3)
        self.clock.now = 0.5
        self.workers[2].cancel()
        await settle()
        await self.complete(dt=1.5)
        await self.complete(dt=0.0)
        self.assertEqual(scheduler.current_concurrency, 2)
        self.assertAlmostEqual(scheduler.events[0]["backlog_fraction"], 0.25)

    async def test_rejections_back_off_and_success_resets_cooldown(self):
        scheduler = await self.load(n=100, cooldown_s=4.0, max_cooldown_s=8.0)
        await self.complete()
        await self.complete()  # t=2: probe c=1
        await self.drain_transition()
        await self.complete(tokens=5)
        await self.complete(tokens=5)  # t=4: reject, back to c=2
        self.assertEqual(scheduler.events[-1]["action"], "reject_probe")
        self.assertEqual(scheduler.events[-1]["next_probe_at"], 8.0)
        self.assertEqual(scheduler.events[-1]["next_retry_cooldown_s"], 8.0)
        await self.drain_transition()
        await self.complete()
        await self.complete()  # t=6: measure, but hold during cooldown
        self.assertEqual(scheduler.events[-1]["action"], "cooldown")
        self.assertEqual(scheduler.current_concurrency, 2)
        await self.complete()
        await self.complete()  # t=8: a fresh reference can start probe c=3
        self.assertEqual(scheduler.events[-1]["action"], "start_probe")
        self.assertEqual(scheduler.current_concurrency, 3)
        await self.drain_transition()
        await self.complete()
        await self.complete()  # t=10: no throughput gain; second rejection
        self.assertEqual(scheduler.events[-1]["action"], "reject_probe")
        self.assertEqual(scheduler.events[-1]["next_probe_at"], 18.0)
        self.assertEqual(scheduler.events[-1]["next_retry_cooldown_s"], 8.0)
        await self.drain_transition()
        while self.clock.now < 18.0:
            await self.complete()
        self.assertEqual(scheduler.events[-1]["action"], "start_probe")
        self.assertEqual(scheduler.current_concurrency, 1)
        await self.drain_transition()
        for _ in range(2):
            req = scheduler._running[0]
            req.t_arrive = self.clock.now
            await self.complete(req=req)
        self.assertEqual(scheduler.events[-1]["action"], "accept_probe")
        self.assertEqual(scheduler.events[-1]["next_probe_at"], 24.0)
        self.assertEqual(scheduler.events[-1]["next_retry_cooldown_s"], 4.0)

    async def test_accepted_cap_is_held_and_cooldown_still_collects_measurements(self):
        scheduler = await self.load(min_concurrency=2, cooldown_s=10.0)
        await self.complete()
        await self.complete()
        await self.drain_transition()
        for _ in range(4):
            await self.complete(dt=0.5, tokens=20)
        self.assertEqual(scheduler.events[-1]["action"], "accept_probe")
        deadline = scheduler.events[-1]["next_probe_at"]
        for _ in range(4):
            await self.complete()
        self.assertEqual(scheduler.current_concurrency, 3)
        self.assertEqual(scheduler.events[-1]["action"], "cooldown")
        self.assertEqual(scheduler.events[-1]["completions"], 2)
        self.assertEqual(scheduler.events[-1]["output_tokens_per_s"], 10.0)
        self.assertEqual(scheduler.events[-1]["next_probe_at"], deadline)

    async def test_invalid_probe_backs_off_but_invalid_cooldown_window_does_not_extend_it(self):
        scheduler = await self.load(cooldown_s=4.0)
        await self.complete()
        await self.complete()
        await self.drain_transition()
        await self.complete(status="error")
        await self.complete()
        self.assertEqual(scheduler.events[-1]["action"], "abort_probe")
        deadline = scheduler.events[-1]["next_probe_at"]
        await self.drain_transition()
        await self.complete(status="error")
        await self.complete()
        self.assertEqual(scheduler.events[-1]["reason"], "request_errors")
        self.assertEqual(scheduler.events[-1]["next_probe_at"], deadline)
        self.assertEqual(scheduler.current_concurrency, 2)


if __name__ == "__main__":
    unittest.main()
