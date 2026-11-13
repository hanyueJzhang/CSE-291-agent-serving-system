import asyncio
import unittest

from scheduler import AgentScheduler, Request, make_scheduler
from tests.fakes import FakeModelClient, ImmediateScheduler
from tests.test_runner import generate_test_tasks
from tests.test_scheduler import FakeClock, settle
from workload.runner import run_task


def req(task_id, followup=False, step_idx=0):
    return Request(task_id=task_id, step_idx=step_idx, is_followup=followup)


class AgentSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def _fill(self, scheduler, blocker_count):
        blockers = [req(f"blk{i}") for i in range(blocker_count)]
        for b in blockers:
            await scheduler.acquire(b)
        return blockers

    async def test_followups_go_before_first_steps(self):
        clock = FakeClock()
        s = AgentScheduler(1, clock=clock)
        (blocker,) = await self._fill(s, 1)
        new1, fu, new2 = req("n1"), req("a", followup=True, step_idx=2), req("n2")
        tasks = [asyncio.create_task(s.acquire(r)) for r in (new1, fu, new2)]
        await settle()
        s.release(blocker)
        await settle()
        self.assertEqual([t.done() for t in tasks], [False, True, False])
        s.release(fu)
        await settle()
        self.assertEqual([t.done() for t in tasks], [True, True, False])   # then arrival order

    async def test_aging_promotes_long_waiting_first_step(self):
        clock = FakeClock()
        s = AgentScheduler(1, aging_s=10.0, clock=clock)
        (blocker,) = await self._fill(s, 1)
        old = req("old")
        t_old = asyncio.create_task(s.acquire(old))
        await settle()
        clock.now = 11.0                       # old has waited 11 s
        fu = req("a", followup=True, step_idx=2)
        t_fu = asyncio.create_task(s.acquire(fu))
        await settle()
        s.release(blocker)
        await settle()
        # Aged first step ranks with follow-ups and arrived earlier.
        self.assertTrue(t_old.done())
        self.assertFalse(t_fu.done())
        s.release(old)
        await settle()
        self.assertTrue(t_fu.done())

    async def test_off_switch_is_fifo(self):
        s = AgentScheduler(1, continuation_first=False)
        (blocker,) = await self._fill(s, 1)
        new, fu = req("n"), req("a", followup=True, step_idx=2)
        tasks = [asyncio.create_task(s.acquire(r)) for r in (new, fu)]
        await settle()
        s.release(blocker)
        await settle()
        self.assertEqual([t.done() for t in tasks], [True, False])

    async def test_respects_cap(self):
        s = AgentScheduler(2)
        reqs = [req(f"t{i}", followup=bool(i % 2)) for i in range(4)]
        tasks = [asyncio.create_task(s.acquire(r)) for r in reqs]
        await settle()
        self.assertEqual(sum(t.done() for t in tasks), 2)
        self.assertEqual(s.num_running, 2)

    def test_config_and_registry(self):
        s = make_scheduler("agent", max_concurrency=5, aging_s=15.0)
        self.assertIsInstance(s, AgentScheduler)
        self.assertEqual(s.config(), {"name": "agent", "max_concurrency": 5,
                                      "continuation_first": True, "aging_s": 15.0})
        with self.assertRaises(ValueError):
            AgentScheduler(0)
        with self.assertRaises(ValueError):
            AgentScheduler(1, aging_s=-1)


class RunnerAgentFieldsTests(unittest.IsolatedAsyncioTestCase):
    async def test_requests_carry_agent_context(self):
        task = generate_test_tasks("one_agent_task")[0]

        async def no_sleep(_):
            return None

        result = await run_task(task, ImmediateScheduler(), FakeModelClient(), sleep=no_sleep)
        first, second = result["requests"]
        self.assertFalse(first.is_followup)
        self.assertEqual(first.history_tokens, 0)
        self.assertEqual(first.remaining_llm_steps, 1)
        self.assertTrue(second.is_followup)
        self.assertGreater(second.history_tokens, 3 + 64)       # prompt + output + template
        self.assertEqual(second.remaining_llm_steps, 0)
        self.assertEqual(first.peak_context, second.peak_context)
        self.assertGreater(first.peak_context, 0)

    async def test_admission_check_rejects_without_running(self):
        task = generate_test_tasks("one_agent_task")[0]
        client = FakeModelClient()
        result = await run_task(task, ImmediateScheduler(), client, max_context=50)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(client.calls, [])
        self.assertEqual(result["requests"], [])
        self.assertEqual(result["num_steps"], 0)

    async def test_admission_check_passes_fitting_task(self):
        task = generate_test_tasks("one_short_task")[0]

        result = await run_task(task, ImmediateScheduler(), FakeModelClient(), max_context=10_000)
        self.assertEqual(result["status"], "ok")


if __name__ == "__main__":
    unittest.main()
