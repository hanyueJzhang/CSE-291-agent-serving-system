import asyncio
from pathlib import Path
import unittest

from scheduler import FIFOScheduler
from workload import generator as workload_generator
from workload.runner import run_workload


TEST_WORKLOADS = Path(__file__).resolve().parent / "fixtures" / "workloads.json"


class FIFOIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_runner_respects_fifo_concurrency_limit(self):
        workloads = workload_generator.load_workload(TEST_WORKLOADS)
        tasks = workload_generator.generate_tasks(workloads["two_short_tasks"])
        first_prompt = tasks[0]["steps"][0]["prompt"]
        second_prompt = tasks[1]["steps"][0]["prompt"]

        scheduler = FIFOScheduler(max_concurrency=1)
        first_started = asyncio.Event()
        release_first = asyncio.Event()

        class ControlledModelClient:
            def __init__(self):
                self.prompts = []
                self.num_active = 0
                self.max_active = 0

            async def generate(self, messages, max_tokens):
                prompt = messages[-1]["content"]
                self.prompts.append(prompt)
                self.num_active += 1
                self.max_active = max(self.max_active, self.num_active)
                try:
                    if prompt == first_prompt:
                        first_started.set()
                        await release_first.wait()
                    return "Fake model response"
                finally:
                    self.num_active -= 1

        async def skip_arrival_wait(delay):
            await asyncio.sleep(0)

        async def wait_for_queued_request():
            while scheduler.queue_len == 0:
                await asyncio.sleep(0)

        model_client = ControlledModelClient()
        workload_run = asyncio.create_task(
            run_workload(
                tasks,
                scheduler,
                model_client,
                sleep=skip_arrival_wait,
            )
        )

        await asyncio.wait_for(first_started.wait(), timeout=0.5)
        await asyncio.wait_for(wait_for_queued_request(), timeout=0.5)

        self.assertEqual(scheduler.num_running, 1)
        self.assertEqual(scheduler.queue_len, 1)
        self.assertEqual(model_client.prompts, [first_prompt])

        release_first.set()
        results = await asyncio.wait_for(workload_run, timeout=0.5)

        self.assertEqual(model_client.prompts, [first_prompt, second_prompt])
        self.assertEqual(model_client.max_active, 1)
        self.assertEqual(
            [result["task_id"] for result in results],
            ["task-0000", "task-0001"],
        )
        self.assertEqual(scheduler.num_running, 0)
        self.assertEqual(scheduler.queue_len, 0)

    async def test_runner_uses_two_fifo_slots(self):
        workloads = workload_generator.load_workload(TEST_WORKLOADS)
        tasks = workload_generator.generate_tasks(workloads["three_short_tasks"])
        prompts = [task["steps"][0]["prompt"] for task in tasks]

        scheduler = FIFOScheduler(max_concurrency=2)
        two_started = asyncio.Event()
        release_first_two = asyncio.Event()

        class ControlledModelClient:
            def __init__(self):
                self.prompts = []
                self.num_active = 0
                self.max_active = 0

            async def generate(self, messages, max_tokens):
                prompt = messages[-1]["content"]
                self.prompts.append(prompt)
                self.num_active += 1
                self.max_active = max(self.max_active, self.num_active)
                if self.num_active == 2:
                    two_started.set()
                try:
                    if prompt in prompts[:2]:
                        await release_first_two.wait()
                    return "Fake model response"
                finally:
                    self.num_active -= 1

        async def skip_arrival_wait(delay):
            await asyncio.sleep(0)

        async def wait_for_queued_request():
            while scheduler.queue_len == 0:
                await asyncio.sleep(0)

        model_client = ControlledModelClient()
        workload_run = asyncio.create_task(
            run_workload(
                tasks,
                scheduler,
                model_client,
                sleep=skip_arrival_wait,
            )
        )

        await asyncio.wait_for(two_started.wait(), timeout=0.5)
        await asyncio.wait_for(wait_for_queued_request(), timeout=0.5)

        self.assertEqual(scheduler.num_running, 2)
        self.assertEqual(scheduler.queue_len, 1)
        self.assertEqual(model_client.prompts, prompts[:2])

        release_first_two.set()
        results = await asyncio.wait_for(workload_run, timeout=0.5)

        self.assertEqual(model_client.prompts, prompts)
        self.assertEqual(model_client.max_active, 2)
        self.assertEqual(
            [result["task_id"] for result in results],
            ["task-0000", "task-0001", "task-0002"],
        )
        self.assertEqual(scheduler.num_running, 0)
        self.assertEqual(scheduler.queue_len, 0)


if __name__ == "__main__":
    unittest.main()
