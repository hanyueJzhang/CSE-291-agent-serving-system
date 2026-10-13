import asyncio
from pathlib import Path
import unittest

from tests.fakes import FakeModelClient, ImmediateScheduler
from workload import generator as workload_generator
from workload import runner as workload_runner
from workload.runner import run_task


TEST_WORKLOADS = Path(__file__).resolve().parent / "fixtures" / "workloads.json"


def generate_test_tasks(name):
    workloads = workload_generator.load_workload(TEST_WORKLOADS)
    return workload_generator.generate_tasks(workloads[name])


class RunTaskTests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_one_short_task(self):
        task = generate_test_tasks("one_short_task")[0]
        scheduler = ImmediateScheduler()
        model_client = FakeModelClient()

        result = await run_task(task, scheduler, model_client)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["responses"], ["Fake model response"])
        self.assertEqual(len(model_client.calls), 1)
        self.assertEqual(model_client.calls[0]["max_tokens"], 64)
        self.assertEqual(
            model_client.calls[0]["messages"],
            [{"role": "user", "content": task["steps"][0]["prompt"]}],
        )

        request = result["requests"][0]
        self.assertEqual(request.task_id, "task-0000")
        self.assertIsNotNone(request.t_arrive)
        self.assertEqual(request.t_dispatch, request.t_arrive)
        self.assertIsNotNone(request.t_finish)
        self.assertEqual(scheduler.num_running, 0)

    async def test_runs_tool_between_llm_steps_and_retains_context(self):
        task = generate_test_tasks("one_agent_task")[0]
        scheduler = ImmediateScheduler()
        tool_delays = []
        running_counts_during_tool = []
        running_counts_during_model = []

        class SchedulerObservingModelClient(FakeModelClient):
            async def generate(self, messages, max_tokens):
                running_counts_during_model.append(scheduler.num_running)
                return await super().generate(messages, max_tokens)

        model_client = SchedulerObservingModelClient()

        async def fake_sleep(delay):
            tool_delays.append(delay)
            running_counts_during_tool.append(scheduler.num_running)

        result = await run_task(
            task,
            scheduler,
            model_client,
            sleep=fake_sleep,
        )

        self.assertEqual(tool_delays, [0.25])
        self.assertEqual(running_counts_during_tool, [0])
        self.assertEqual(running_counts_during_model, [1, 1])
        self.assertEqual(len(model_client.calls), 2)
        self.assertEqual([request.step_idx for request in result["requests"]], [0, 2])
        self.assertEqual(
            model_client.calls[1]["messages"],
            [
                {"role": "user", "content": task["steps"][0]["prompt"]},
                {"role": "assistant", "content": "Fake model response"},
                {"role": "tool", "content": task["steps"][1]["result"]},
                {"role": "user", "content": task["steps"][2]["prompt"]},
            ],
        )
        self.assertEqual(result["responses"], ["Fake model response"] * 2)
        self.assertEqual(scheduler.num_running, 0)

    async def test_another_agent_progresses_during_tool_wait(self):
        tasks = generate_test_tasks("mixed_concurrent_tasks")
        task_a, task_b = tasks
        self.assertEqual([task["type"] for task in tasks], ["agent_2request", "short_chat"])

        a_first_prompt = task_a["steps"][0]["prompt"]
        a_second_prompt = task_a["steps"][2]["prompt"]
        b_prompt = task_b["steps"][0]["prompt"]
        arrival_delays = {task["arrival_s"] for task in tasks}
        scheduler = ImmediateScheduler()
        a_is_waiting = asyncio.Event()
        b_has_progressed = asyncio.Event()
        events = []

        async def controlled_sleep(delay):
            if delay in arrival_delays:
                return
            events.append("A:tool-start")
            a_is_waiting.set()
            await b_has_progressed.wait()
            events.append("A:tool-finish")

        class CoordinatedModelClient(FakeModelClient):
            async def generate(self, messages, max_tokens):
                latest_prompt = messages[-1]["content"]
                if latest_prompt == a_first_prompt:
                    events.append("A:model-1")
                elif latest_prompt == b_prompt:
                    await a_is_waiting.wait()
                    events.append("B:model-1")
                    b_has_progressed.set()
                elif latest_prompt == a_second_prompt:
                    events.append("A:model-2")
                return await super().generate(messages, max_tokens)

        results = await asyncio.wait_for(
            workload_runner.run_workload(
                tasks,
                scheduler,
                CoordinatedModelClient(),
                sleep=controlled_sleep,
            ),
            timeout=0.5,
        )

        self.assertEqual(
            events,
            [
                "A:model-1",
                "A:tool-start",
                "B:model-1",
                "A:tool-finish",
                "A:model-2",
            ],
        )
        self.assertEqual(
            [result["task_id"] for result in results],
            ["task-0000", "task-0001"],
        )
        self.assertEqual(scheduler.num_running, 0)

    async def test_concurrent_agents_keep_separate_contexts(self):
        tasks = generate_test_tasks("two_agent_tasks")
        task_a, task_b = tasks

        async def yield_to_other_agent(delay):
            await asyncio.sleep(0)

        scheduler = ImmediateScheduler()
        model_client = FakeModelClient()
        results = await workload_runner.run_workload(
            tasks,
            scheduler,
            model_client,
            sleep=yield_to_other_agent,
        )

        calls_by_prompt = {
            call["messages"][-1]["content"]: call["messages"]
            for call in model_client.calls
        }
        self.assertEqual(
            calls_by_prompt[task_a["steps"][2]["prompt"]],
            [
                {"role": "user", "content": task_a["steps"][0]["prompt"]},
                {"role": "assistant", "content": "Fake model response"},
                {"role": "tool", "content": task_a["steps"][1]["result"]},
                {"role": "user", "content": task_a["steps"][2]["prompt"]},
            ],
        )
        self.assertEqual(
            calls_by_prompt[task_b["steps"][2]["prompt"]],
            [
                {"role": "user", "content": task_b["steps"][0]["prompt"]},
                {"role": "assistant", "content": "Fake model response"},
                {"role": "tool", "content": task_b["steps"][1]["result"]},
                {"role": "user", "content": task_b["steps"][2]["prompt"]},
            ],
        )
        self.assertEqual(
            [result["task_id"] for result in results],
            ["task-0000", "task-0001"],
        )
        self.assertEqual(scheduler.num_running, 0)


if __name__ == "__main__":
    unittest.main()
