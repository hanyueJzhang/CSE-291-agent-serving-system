import asyncio
import unittest

from scheduler import LeaseScheduler, Request, make_scheduler
from tests.fakes import FakeModelClient
from tests.test_runner import generate_test_tasks
from tests.test_scheduler import FakeClock, settle
from workload.llama_client import Completion
from workload.runner import run_task


def first(task_id, more_steps=1, prompt=""):
    return Request(task_id=task_id, step_idx=0, task_type="agent",
                   remaining_llm_steps=more_steps, task_prompt=prompt)


def first_peak(task_id, peak, more_steps=1):
    return Request(task_id=task_id, step_idx=0, task_type="agent",
                   remaining_llm_steps=more_steps, peak_context=peak)


def followup(task_id, step_idx=2, more_steps=0):
    return Request(task_id=task_id, step_idx=step_idx, task_type="agent",
                   is_followup=True, remaining_llm_steps=more_steps)


class LeaseSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_request_per_slot(self):
        s = LeaseScheduler(2)
        reqs = [first(f"t{i}") for i in range(3)]
        tasks = [asyncio.create_task(s.acquire(r)) for r in reqs]
        await settle()
        self.assertEqual([t.done() for t in tasks], [True, True, False])
        self.assertEqual({reqs[0].slot, reqs[1].slot}, {0, 1})
        self.assertIsNone(reqs[2].slot)

    async def test_followup_returns_to_its_held_slot_ahead_of_new_task(self):
        clock = FakeClock()
        s = LeaseScheduler(1, hold_max_s=2.0, clock=clock)
        a0 = first("A")
        await s.acquire(a0)
        new = first("N", more_steps=0)
        t_new = asyncio.create_task(s.acquire(new))
        await settle()
        s.release(a0)                       # A goes to its tool; the slot is held
        await settle()
        self.assertFalse(t_new.done())

        clock.now = 1.0
        a1 = followup("A")
        await asyncio.wait_for(s.acquire(a1), 1)
        self.assertEqual(a1.slot, a0.slot)
        self.assertFalse(t_new.done())
        s.release(a1)                       # A is done: the slot is free at once
        await settle()
        self.assertTrue(t_new.done())
        self.assertEqual(s.summary()["followups_kept"], 1)
        self.assertEqual(s.summary()["live_evictions"], 0)

    async def test_hold_lapses_after_hold_max_s(self):
        clock = FakeClock()
        s = LeaseScheduler(1, hold_max_s=2.0, clock=clock)
        a0 = first("A")
        await s.acquire(a0)
        new = first("N", more_steps=0)
        t_new = asyncio.create_task(s.acquire(new))
        await settle()
        s.release(a0)
        clock.now = 2.5                     # A has not come back
        a1 = followup("A")
        t_a1 = asyncio.create_task(s.acquire(a1))
        await settle()
        self.assertTrue(t_new.done())       # arrival order: N takes the lapsed slot
        self.assertFalse(t_a1.done())
        s.release(new)
        await settle()
        self.assertTrue(t_a1.done())
        self.assertEqual(s.summary()["followups_displaced"], 1)
        self.assertEqual(s.summary()["live_evictions"], 1)
        self.assertEqual(s.trace[0]["owner"], "A")

    async def test_hold_lapses_without_another_event(self):
        s = LeaseScheduler(1, hold_max_s=0.02)      # real clock: the timer must wake the queue
        a0 = first("A")
        await s.acquire(a0)
        t_new = asyncio.create_task(s.acquire(first("N", more_steps=0)))
        await settle()
        s.release(a0)
        await settle()
        self.assertFalse(t_new.done())
        await asyncio.sleep(0.1)
        self.assertTrue(t_new.done())

    async def test_long_usual_wait_gives_the_slot_up_at_once(self):
        clock = FakeClock()
        s = LeaseScheduler(1, hold_max_s=2.0, clock=clock)
        a0 = first("A")
        await s.acquire(a0)
        s.release(a0)
        clock.now = 5.0                     # A's tool took 5 s; nobody wanted the slot meanwhile
        a1 = followup("A")
        await asyncio.wait_for(s.acquire(a1), 1)
        s.release(a1)
        self.assertEqual(s.summary()["usual_wait_s"], {"agent/0": 5.0})
        self.assertEqual(s.summary()["followups_kept"], 1)

        b0 = first("B")
        await s.acquire(b0)
        t_new = asyncio.create_task(s.acquire(first("N", more_steps=0)))
        await settle()
        s.release(b0)                       # same kind of step: expect ~5 s, more than hold_max_s
        await settle()
        self.assertTrue(t_new.done())
        self.assertEqual(s.summary()["given_up_at_once"], 1)

    async def test_new_task_takes_finished_owner_with_longest_shared_prefix(self):
        s = LeaseScheduler(3, hold_max_s=0)             # never hold: slots differ only by their owner
        live = first("L", prompt="template-2 live")
        done1 = first("D1", more_steps=0, prompt="template-1 x")
        done2 = first("D2", more_steps=0, prompt="template-2 y")
        for r in (live, done1, done2):
            await s.acquire(r)
        for r in (done2, done1, live):
            s.release(r)

        new = first("N", more_steps=0, prompt="template-2 z")
        await s.acquire(new)
        self.assertEqual(new.slot, done2.slot)          # finished owner, same template
        other = first("M", more_steps=0, prompt="unrelated")
        await s.acquire(other)
        self.assertEqual(other.slot, done1.slot)        # finished owner before L's live history
        self.assertEqual(s.summary()["live_evictions"], 0)
        last = first("K", more_steps=0)
        await s.acquire(last)
        self.assertEqual(last.slot, live.slot)
        self.assertEqual(s.summary()["live_evictions"], 1)

    async def test_given_up_history_is_reused_if_nobody_took_the_slot(self):
        s = LeaseScheduler(2, hold_max_s=0)
        a0 = first("A")
        await s.acquire(a0)
        s.release(a0)
        b0 = first("B", more_steps=0)
        await s.acquire(b0)                 # takes the empty slot, not A's
        a1 = followup("A")
        await s.acquire(a1)
        self.assertNotEqual(b0.slot, a0.slot)
        self.assertEqual(a1.slot, a0.slot)
        self.assertEqual(s.summary()["followups_kept"], 1)

    async def test_failed_request_ends_the_lease(self):
        clock = FakeClock()
        s = LeaseScheduler(1, hold_max_s=2.0, clock=clock)
        a0 = first("A")
        await s.acquire(a0)
        t_new = asyncio.create_task(s.acquire(first("N", more_steps=0)))
        await settle()
        s.observe(a0, Completion(status="error"))
        s.release(a0)
        await settle()
        self.assertTrue(t_new.done())
        self.assertEqual(s.summary()["holds"], 0)

    async def test_counts_requests_served_by_another_slot(self):
        s = LeaseScheduler(2)
        r = first("A", more_steps=0)
        await s.acquire(r)
        s.observe(r, Completion(slot_id=r.slot))
        s.observe(r, Completion(slot_id=r.slot + 1))
        self.assertEqual(s.summary()["pin_mismatches"], 1)

    async def test_soft_limit_holds_new_task_back_until_overflow_wait(self):
        clock = FakeClock()
        s = LeaseScheduler(3, soft_limit=2, overflow_wait_s=2.0, clock=clock)
        a, b = first("A"), first("B")
        await s.acquire(a)
        await s.acquire(b)
        c = first("C")
        t_c = asyncio.create_task(s.acquire(c))
        await settle()
        self.assertFalse(t_c.done())                # a third slot is free, but two are leased
        clock.now = 2.0                             # C has waited overflow_wait_s
        d = first("D")
        t_d = asyncio.create_task(s.acquire(d))     # any event re-checks the head
        await settle()
        self.assertTrue(t_c.done())
        self.assertFalse(t_d.done())                # no free slot left
        self.assertEqual(s.summary()["soft_held_back"], 1)
        self.assertEqual(s.summary()["overflow_admits"], 1)

    async def test_soft_limit_admits_when_a_lease_ends(self):
        clock = FakeClock()
        s = LeaseScheduler(3, soft_limit=2, overflow_wait_s=5.0, clock=clock)
        a, b = first("A", more_steps=0), first("B")
        await s.acquire(a)
        await s.acquire(b)
        c = first("C")
        t_c = asyncio.create_task(s.acquire(c))
        await settle()
        self.assertFalse(t_c.done())
        s.release(a)                                # A is done: one lease left
        await settle()
        self.assertTrue(t_c.done())
        self.assertEqual(s.summary()["overflow_admits"], 0)

    async def test_soft_limit_never_holds_back_followups(self):
        clock = FakeClock()
        s = LeaseScheduler(2, soft_limit=1, hold_max_s=0.0, clock=clock)
        a0 = first("A")
        await s.acquire(a0)
        b1 = followup("B")                          # its slot was lost: it needs any free slot
        await asyncio.wait_for(s.acquire(b1), 1)
        self.assertEqual(b1.slot, 1)

    async def test_overflow_timer_wakes_the_queue(self):
        s = LeaseScheduler(2, soft_limit=1, overflow_wait_s=0.02)     # real clock
        await s.acquire(first("A"))
        t_b = asyncio.create_task(s.acquire(first("B")))
        await settle()
        self.assertFalse(t_b.done())
        await asyncio.sleep(0.1)
        self.assertTrue(t_b.done())

    async def test_decode_limit_ignores_held_slots(self):
        clock = FakeClock()
        s = LeaseScheduler(3, decode_limit=2, hold_max_s=2.0, clock=clock)
        a, b = first("A"), first("B")
        await s.acquire(a)
        await s.acquire(b)
        c = first("C")
        t_c = asyncio.create_task(s.acquire(c))
        await settle()
        self.assertFalse(t_c.done())                # two decoding: the limit binds
        s.release(a)                                # A goes to its tool; its slot is held
        await settle()
        self.assertTrue(t_c.done())                 # one decoding: C takes the third slot
        self.assertNotEqual(c.slot, a.slot)
        a1 = followup("A")
        await asyncio.wait_for(s.acquire(a1), 1)    # a follow-up always returns, even past the limit
        self.assertEqual(a1.slot, a.slot)
        self.assertEqual(s.summary()["decode_held_back"], 1)

    async def test_kv_budget_holds_new_task_until_a_task_ends(self):
        clock = FakeClock()
        s = LeaseScheduler(3, kv_budget_tokens=10_000, hold_max_s=2.0, clock=clock)
        a = first_peak("A", 6000, more_steps=0)
        await s.acquire(a)
        b = first_peak("B", 5000)
        t_b = asyncio.create_task(s.acquire(b))
        await settle()
        self.assertFalse(t_b.done())                # 6000 + 5000 > 10000
        s.release(a)                                # A is done: its history no longer counts
        await settle()
        self.assertTrue(t_b.done())
        self.assertEqual(s.summary()["kv_held_back"], 1)

    def test_soft_limit_params_are_checked(self):
        with self.assertRaises(ValueError):
            LeaseScheduler(2, soft_limit=3)
        with self.assertRaises(ValueError):
            LeaseScheduler(2, soft_limit=0)
        with self.assertRaises(ValueError):
            LeaseScheduler(2, soft_limit=1, overflow_wait_s=-1)
        with self.assertRaises(ValueError):
            LeaseScheduler(2, decode_limit=3)
        with self.assertRaises(ValueError):
            LeaseScheduler(2, kv_budget_tokens=0)

    def test_config_and_registry(self):
        s = make_scheduler("lease", max_concurrency=4, hold_max_s=3.5)
        self.assertIsInstance(s, LeaseScheduler)
        self.assertEqual(s.config(), {"name": "lease", "max_concurrency": 4, "hold_max_s": 3.5,
                                      "soft_limit": None, "overflow_wait_s": 2.0,
                                      "decode_limit": None, "kv_budget_tokens": None})
        for bad in (dict(max_concurrency=0), dict(max_concurrency=2, hold_max_s=-1),
                    dict(max_concurrency=2, hold_max_s=float("inf"))):
            with self.subTest(bad), self.assertRaises(ValueError):
                LeaseScheduler(**bad)


class PinRecordingClient(FakeModelClient):
    def __init__(self):
        super().__init__()
        self.pins = []

    async def complete(self, messages, max_tokens, *, id_slot=None):
        self.pins.append(id_slot)
        return Completion(text="ok", slot_id=id_slot)


class RunnerPinningTests(unittest.IsolatedAsyncioTestCase):
    async def test_runner_pins_every_step_of_a_task_to_its_slot(self):
        task = generate_test_tasks("one_agent_task")[0]
        client = PinRecordingClient()
        scheduler = LeaseScheduler(2)

        async def no_sleep(_):
            return None

        result = await run_task(task, scheduler, client, sleep=no_sleep)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(client.pins, [0, 0])
        self.assertEqual([r.task_prompt for r in result["requests"]], [task["steps"][0]["prompt"]] * 2)
        self.assertEqual(scheduler.summary()["followups_kept"], 1)
        self.assertEqual(scheduler.summary()["pin_mismatches"], 0)


if __name__ == "__main__":
    unittest.main()
