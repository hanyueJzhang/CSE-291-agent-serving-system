import asyncio
from pathlib import Path
import unittest

from scheduler import FIFOScheduler
from tests.fake_llama_server import NORMAL_STREAM, FakeLlamaServer
from workload import generator as workload_generator
from workload.llama_client import LlamaClient
from workload.runner import run_workload


TEST_WORKLOADS = Path(__file__).resolve().parent / "fixtures" / "workloads.json"


class ControlledHandler:
    """llama-server chat handler that holds chosen prompts until released.

    Records the prompts it receives and how many requests it serves at once.
    """

    def __init__(self, held_prompts, hold_until_active=1):
        self.held_prompts = set(held_prompts)
        self.hold_until_active = hold_until_active
        self.prompts = []
        self.num_active = 0
        self.max_active = 0
        self.all_held_started = asyncio.Event()
        self.release = asyncio.Event()
        self._respond = FakeLlamaServer.stream(NORMAL_STREAM)

    async def __call__(self, request):
        prompt = (await request.json())["messages"][-1]["content"]
        self.prompts.append(prompt)
        self.num_active += 1
        self.max_active = max(self.max_active, self.num_active)
        try:
            if prompt in self.held_prompts:
                if self.num_active == self.hold_until_active:
                    self.all_held_started.set()
                await self.release.wait()
            return await self._respond(request)
        finally:
            self.num_active -= 1


class FIFOIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Runner + FIFO scheduler + LlamaClient against a fake llama-server."""

    async def asyncSetUp(self):
        self.fake = await FakeLlamaServer().start()
        self.client = LlamaClient(self.fake.url, timeout_s=5.0)

    async def asyncTearDown(self):
        await self.client.close()
        await self.fake.close()

    def generate(self, name):
        workloads = workload_generator.load_workload(TEST_WORKLOADS)
        return workload_generator.generate_tasks(workloads[name])

    async def wait_for_queued_request(self, scheduler):
        while scheduler.queue_len == 0:
            await asyncio.sleep(0.001)

    async def test_runner_respects_fifo_concurrency_limit(self):
        tasks = self.generate("two_short_tasks")
        first_prompt = tasks[0]["steps"][0]["prompt"]
        second_prompt = tasks[1]["steps"][0]["prompt"]

        scheduler = FIFOScheduler(max_concurrency=1)
        handler = ControlledHandler(held_prompts=[first_prompt], hold_until_active=1)
        self.fake.handler = handler

        async def skip_arrival_wait(delay):
            await asyncio.sleep(0)

        workload_run = asyncio.create_task(
            run_workload(tasks, scheduler, self.client, sleep=skip_arrival_wait)
        )

        await asyncio.wait_for(handler.all_held_started.wait(), timeout=2)
        await asyncio.wait_for(self.wait_for_queued_request(scheduler), timeout=2)

        self.assertEqual(scheduler.num_running, 1)
        self.assertEqual(scheduler.queue_len, 1)
        self.assertEqual(handler.prompts, [first_prompt])

        handler.release.set()
        results = await asyncio.wait_for(workload_run, timeout=2)

        self.assertEqual(handler.prompts, [first_prompt, second_prompt])
        self.assertEqual(handler.max_active, 1)
        self.assertEqual(
            [result["task_id"] for result in results],
            ["task-0000", "task-0001"],
        )
        self.assertTrue(all(result["status"] == "ok" for result in results))
        self.assertEqual(scheduler.num_running, 0)
        self.assertEqual(scheduler.queue_len, 0)

        # The second request waited in our queue while the first was held.
        second_request = results[1]["requests"][0]
        self.assertGreater(second_request.queue_s, 0)
        self.assertEqual(results[1]["completions"][0].prompt_tokens, 30)

    async def test_runner_uses_two_fifo_slots(self):
        tasks = self.generate("three_short_tasks")
        prompts = [task["steps"][0]["prompt"] for task in tasks]

        scheduler = FIFOScheduler(max_concurrency=2)
        handler = ControlledHandler(held_prompts=prompts[:2], hold_until_active=2)
        self.fake.handler = handler

        async def skip_arrival_wait(delay):
            await asyncio.sleep(0)

        workload_run = asyncio.create_task(
            run_workload(tasks, scheduler, self.client, sleep=skip_arrival_wait)
        )

        await asyncio.wait_for(handler.all_held_started.wait(), timeout=2)
        await asyncio.wait_for(self.wait_for_queued_request(scheduler), timeout=2)

        self.assertEqual(scheduler.num_running, 2)
        self.assertEqual(scheduler.queue_len, 1)
        # The first two may reach the server in either order.
        self.assertEqual(sorted(handler.prompts), sorted(prompts[:2]))

        handler.release.set()
        results = await asyncio.wait_for(workload_run, timeout=2)

        self.assertEqual(sorted(handler.prompts), sorted(prompts))
        self.assertEqual(handler.prompts[-1], prompts[2])
        self.assertEqual(handler.max_active, 2)
        self.assertEqual(
            [result["task_id"] for result in results],
            ["task-0000", "task-0001", "task-0002"],
        )
        self.assertEqual(scheduler.num_running, 0)
        self.assertEqual(scheduler.queue_len, 0)


if __name__ == "__main__":
    unittest.main()
