import asyncio
import time
import unittest
from types import SimpleNamespace

from scheduler import AdaptiveScheduler, Request, make_scheduler
from tests.test_scheduler import FakeClock, settle


def req(task_id, followup=False, step_idx=0):
    return Request(task_id=task_id, step_idx=step_idx, is_followup=followup)


def done(first, prefill_ms=0.0, out=10, decode_ms=100.0):
    return SimpleNamespace(status="ok", t_first_token=first, prefill_ms=prefill_ms,
                           output_tokens=out, decode_ms=decode_ms)


class SlotServer:
    """llama-server stand-in: `slots` requests run at once, the rest wait inside it."""

    def __init__(self, slots, service_s, clock=time.monotonic):
        self.sem = asyncio.Semaphore(slots)
        self.service_s = service_s
        self.clock = clock

    async def complete(self):
        async with self.sem:
            start = self.clock()
            await asyncio.sleep(self.service_s)
            return done(first=start + 0.001, prefill_ms=1.0)


async def drive(scheduler, server, n_requests, arrive_every_s=0.0):
    async def one(i):
        if arrive_every_s:
            await asyncio.sleep(i * arrive_every_s)
        r = req(f"t{i}")
        async with scheduler.admit(r):
            c = await server.complete()
            scheduler.observe(r, c)
    await asyncio.gather(*(one(i) for i in range(n_requests)))


class ControlLawTests(unittest.IsolatedAsyncioTestCase):
    async def _run_window(self, s, clock, waits, held_back):
        """Feed one full window of completions with the given server waits."""
        for i, w in enumerate(waits):
            r = req(f"x{i}")
            await s.acquire(r)
            s._held_back = s._held_back or held_back
            clock.now += 1.0
            s.observe(r, done(first=r.t_dispatch + w))
            s.release(r)

    async def test_raises_limit_when_slots_idle_and_work_held_back(self):
        clock = FakeClock()
        s = AdaptiveScheduler(2, window=4, clock=clock)
        await self._run_window(s, clock, [0.0] * 4, held_back=True)
        self.assertEqual(s.limit, 3)

    async def test_keeps_limit_without_backlog(self):
        clock = FakeClock()
        s = AdaptiveScheduler(2, window=4, clock=clock)
        await self._run_window(s, clock, [0.0] * 4, held_back=False)
        self.assertEqual(s.limit, 2)

    async def test_lowers_limit_on_hidden_server_queue(self):
        clock = FakeClock()
        s = AdaptiveScheduler(6, window=4, clock=clock)
        # Each request waited 3 s inside the server over a ~4 s window: Q ~ 3.
        await self._run_window(s, clock, [3.0] * 4, held_back=True)
        self.assertEqual(s.limit, 5)
        self.assertGreater(s.trace[-1]["server_queue"], 1.5)

    async def test_failed_requests_are_ignored(self):
        s = AdaptiveScheduler(2, window=2)
        r = req("a")
        await s.acquire(r)
        s.observe(r, SimpleNamespace(status="error", t_first_token=None))
        self.assertEqual(len(s._obs), 0)

    async def test_overload_mode_puts_followups_first_then_switches_off(self):
        clock = FakeClock()
        s = AdaptiveScheduler(1, overload_wait_s=10.0, clock=clock)
        blocker = req("blk")
        await s.acquire(blocker)
        new, fu = req("n"), req("a", followup=True, step_idx=2)
        t_new, t_fu = asyncio.create_task(s.acquire(new)), asyncio.create_task(s.acquire(fu))
        await settle()
        clock.now = 11.0                         # new task has waited 11 s: overload
        s.release(blocker)
        await settle()
        self.assertTrue(t_fu.done())
        self.assertFalse(t_new.done())
        self.assertTrue(s.overload)
        s.release(fu)
        await settle()
        self.assertTrue(t_new.done())
        later = req("later")
        clock.now = 12.0
        t_later = asyncio.create_task(s.acquire(later))   # only a fresh first step waits now
        await settle()
        s.release(new)
        await settle()
        self.assertFalse(s.overload)
        self.assertTrue(t_later.done())

    def test_bad_params(self):
        with self.assertRaises(ValueError):
            AdaptiveScheduler(0)
        with self.assertRaises(ValueError):
            AdaptiveScheduler(2, target_server_queue=0.2, band=0.5)
        self.assertEqual(make_scheduler("adaptive", max_concurrency=3).limit, 3)


class ConvergenceTests(unittest.IsolatedAsyncioTestCase):
    """Against a server with fixed slots, the limit settles at slots + about one."""

    async def test_converges_up_from_too_low(self):
        s = AdaptiveScheduler(1, window=8)
        await drive(s, SlotServer(slots=3, service_s=0.02), n_requests=300)
        self.assertIn(s.limit, (4, 5))
        self.assertGreater(s.summary()["evaluations"], 10)

    async def test_converges_down_from_too_high(self):
        s = AdaptiveScheduler(10, window=8)
        await drive(s, SlotServer(slots=3, service_s=0.02), n_requests=300)
        self.assertIn(s.limit, (4, 5))


if __name__ == "__main__":
    unittest.main()
