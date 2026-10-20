import unittest

from scheduler import FIFOScheduler
from tests.fakes import FakeModelClient, ImmediateScheduler
from workload.llama_client import Completion
from workload.runner import run_task


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def agent_task():
    return {
        "id": "task-0000",
        "type": "agent",
        "arrival_s": 0.0,
        "steps": [
            {"kind": "llm", "prompt": "a", "prompt_tokens": 1, "max_output_tokens": 4},
            {"kind": "tool", "delay_s": 2.0, "result": "r"},
            {"kind": "llm", "prompt": "b", "prompt_tokens": 1, "max_output_tokens": 4, "carry_context": True},
        ],
    }


class ScriptedClient:
    """complete() returns the scripted Completions in order, advancing the clock."""

    def __init__(self, clock, completions, step_s=1.0):
        self.clock = clock
        self.completions = list(completions)
        self.step_s = step_s
        self.calls = 0

    async def complete(self, messages, max_tokens):
        self.calls += 1
        self.clock.now += self.step_s
        completion = self.completions.pop(0)
        completion.t_finish = self.clock.now
        return completion


class RunTaskMetricsTests(unittest.IsolatedAsyncioTestCase):
    async def test_records_task_timings(self):
        clock = FakeClock()
        clock.now = 5.0

        async def fake_sleep(delay):
            clock.now += delay

        client = ScriptedClient(clock, [Completion(text="x", prompt_tokens=10), Completion(text="y")])
        scheduler = FIFOScheduler(max_concurrency=1, clock=clock)

        result = await run_task(agent_task(), scheduler, client, sleep=fake_sleep, clock=clock)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["t_arrive"], 5.0)
        self.assertEqual(result["t_finish"], 5.0 + 1.0 + 2.0 + 1.0)
        self.assertEqual(result["total_tool_s"], 2.0)
        self.assertEqual(result["total_queue_s"], 0.0)
        self.assertEqual(result["num_steps"], 3)
        self.assertEqual(len(result["completions"]), 2)
        self.assertEqual(result["completions"][0].prompt_tokens, 10)

    async def test_failed_request_ends_task(self):
        clock = FakeClock()

        async def fake_sleep(delay):
            raise AssertionError("tool step should not run after a failure")

        client = ScriptedClient(clock, [Completion(status="error", error="HTTP 400: too long")])

        result = await run_task(agent_task(), ImmediateScheduler(clock), client, sleep=fake_sleep, clock=clock)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["completions"][0].error, "HTTP 400: too long")
        self.assertEqual(client.calls, 1)
        self.assertEqual(result["num_steps"], 0)
        self.assertEqual(len(result["requests"]), 1)
        self.assertEqual(result["responses"], [])

    async def test_timed_out_request_marks_task_timeout(self):
        clock = FakeClock()

        async def fake_sleep(delay):
            clock.now += delay

        client = ScriptedClient(clock, [Completion(text="x"), Completion(status="timeout", error="slow")])

        result = await run_task(agent_task(), ImmediateScheduler(clock), client, sleep=fake_sleep, clock=clock)

        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["num_steps"], 2)
        self.assertEqual(len(result["requests"]), 2)

    async def test_fake_client_goes_through_complete(self):
        result = await run_task(agent_task(), ImmediateScheduler(), FakeModelClient(), sleep=_no_sleep)

        self.assertEqual(result["status"], "ok")
        self.assertEqual([c.text for c in result["completions"]], ["Fake model response"] * 2)

    async def test_client_without_complete_is_rejected(self):
        class GenerateOnlyClient:
            async def generate(self, messages, max_tokens):
                return "text"

        with self.assertRaises(AttributeError):
            await run_task(agent_task(), ImmediateScheduler(), GenerateOnlyClient(), sleep=_no_sleep)


async def _no_sleep(delay):
    return None


if __name__ == "__main__":
    unittest.main()
