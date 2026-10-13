import asyncio
import unittest

from tests.fakes import FakeModelClient, ImmediateScheduler
from workload import runner as workload_runner
from workload.runner import run_task


class RunTaskTests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_one_short_task(self):
        task = {
            "id": "task-1",
            "type": "short_chat",
            "steps": [
                {
                    "kind": "llm",
                    "prompt": "Hello",
                    "prompt_tokens": 256,
                    "max_output_tokens": 64,
                }
            ],
        }
        scheduler = ImmediateScheduler()
        model_client = FakeModelClient()

        result = await run_task(task, scheduler, model_client)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["responses"], ["Fake model response"])
        self.assertEqual(len(model_client.calls), 1)
        self.assertEqual(model_client.calls[0]["max_tokens"], 64)
        self.assertEqual(
            model_client.calls[0]["messages"],
            [{"role": "user", "content": "Hello"}],
        )

        request = result["requests"][0]
        self.assertEqual(request.task_id, "task-1")
        self.assertIsNotNone(request.t_arrive)
        self.assertEqual(request.t_dispatch, request.t_arrive)
        self.assertIsNotNone(request.t_finish)
        self.assertEqual(scheduler.num_running, 0)

    async def test_runs_tool_between_llm_steps_and_retains_context(self):
        task = {
            "id": "task-2",
            "type": "agent_2request",
            "steps": [
                {
                    "kind": "llm",
                    "prompt": "Find information about topic X.",
                    "prompt_tokens": 128,
                    "max_output_tokens": 64,
                },
                {
                    "kind": "tool",
                    "delay_s": {"dist": "fixed", "value": 0.25},
                    "result": "Simulated tool result about topic X.",
                },
                {
                    "kind": "llm",
                    "prompt": "Summarize the tool result.",
                    "prompt_tokens": 256,
                    "max_output_tokens": 64,
                    "carry_context": True,
                },
            ],
        }
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
                {"role": "user", "content": "Find information about topic X."},
                {"role": "assistant", "content": "Fake model response"},
                {"role": "tool", "content": "Simulated tool result about topic X."},
                {"role": "user", "content": "Summarize the tool result."},
            ],
        )
        self.assertEqual(result["responses"], ["Fake model response"] * 2)
        self.assertEqual(scheduler.num_running, 0)

    async def test_another_agent_progresses_during_tool_wait(self):
        task_a = {
            "id": "task-a",
            "type": "agent_2request",
            "steps": [
                {
                    "kind": "llm",
                    "prompt": "A first prompt",
                    "prompt_tokens": 64,
                    "max_output_tokens": 32,
                },
                {
                    "kind": "tool",
                    "delay_s": {"dist": "fixed", "value": 1.0},
                    "result": "A tool result",
                },
                {
                    "kind": "llm",
                    "prompt": "A second prompt",
                    "prompt_tokens": 128,
                    "max_output_tokens": 32,
                    "carry_context": True,
                },
            ],
        }
        task_b = {
            "id": "task-b",
            "type": "short_chat",
            "steps": [
                {
                    "kind": "llm",
                    "prompt": "B prompt",
                    "prompt_tokens": 64,
                    "max_output_tokens": 32,
                }
            ],
        }
        scheduler = ImmediateScheduler()
        a_is_waiting = asyncio.Event()
        b_has_progressed = asyncio.Event()
        events = []

        async def controlled_sleep(delay):
            events.append("A:tool-start")
            a_is_waiting.set()
            await b_has_progressed.wait()
            events.append("A:tool-finish")

        class CoordinatedModelClient(FakeModelClient):
            async def generate(self, messages, max_tokens):
                latest_prompt = messages[-1]["content"]
                if latest_prompt == "A first prompt":
                    events.append("A:model-1")
                elif latest_prompt == "B prompt":
                    await a_is_waiting.wait()
                    events.append("B:model-1")
                    b_has_progressed.set()
                elif latest_prompt == "A second prompt":
                    events.append("A:model-2")
                return await super().generate(messages, max_tokens)

        results = await asyncio.wait_for(
            workload_runner.run_tasks(
                [task_a, task_b],
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
        self.assertEqual([result["task_id"] for result in results], ["task-a", "task-b"])
        self.assertEqual(scheduler.num_running, 0)

    async def test_concurrent_agents_keep_separate_contexts(self):
        task_a = {
            "id": "task-a",
            "type": "agent_2request",
            "steps": [
                {
                    "kind": "llm",
                    "prompt": "task-a first prompt",
                    "prompt_tokens": 64,
                    "max_output_tokens": 32,
                },
                {
                    "kind": "tool",
                    "delay_s": {"dist": "fixed", "value": 0.0},
                    "result": "task-a tool result",
                },
                {
                    "kind": "llm",
                    "prompt": "task-a second prompt",
                    "prompt_tokens": 128,
                    "max_output_tokens": 32,
                    "carry_context": True,
                },
            ],
        }
        task_b = {
            "id": "task-b",
            "type": "agent_2request",
            "steps": [
                {
                    "kind": "llm",
                    "prompt": "task-b first prompt",
                    "prompt_tokens": 64,
                    "max_output_tokens": 32,
                },
                {
                    "kind": "tool",
                    "delay_s": {"dist": "fixed", "value": 0.0},
                    "result": "task-b tool result",
                },
                {
                    "kind": "llm",
                    "prompt": "task-b second prompt",
                    "prompt_tokens": 128,
                    "max_output_tokens": 32,
                    "carry_context": True,
                },
            ],
        }

        async def yield_to_other_agent(delay):
            await asyncio.sleep(0)

        scheduler = ImmediateScheduler()
        model_client = FakeModelClient()
        results = await workload_runner.run_tasks(
            [task_a, task_b],
            scheduler,
            model_client,
            sleep=yield_to_other_agent,
        )

        calls_by_prompt = {
            call["messages"][-1]["content"]: call["messages"]
            for call in model_client.calls
        }
        self.assertEqual(
            calls_by_prompt["task-a second prompt"],
            [
                {"role": "user", "content": "task-a first prompt"},
                {"role": "assistant", "content": "Fake model response"},
                {"role": "tool", "content": "task-a tool result"},
                {"role": "user", "content": "task-a second prompt"},
            ],
        )
        self.assertEqual(
            calls_by_prompt["task-b second prompt"],
            [
                {"role": "user", "content": "task-b first prompt"},
                {"role": "assistant", "content": "Fake model response"},
                {"role": "tool", "content": "task-b tool result"},
                {"role": "user", "content": "task-b second prompt"},
            ],
        )
        self.assertEqual([result["task_id"] for result in results], ["task-a", "task-b"])
        self.assertEqual(scheduler.num_running, 0)


if __name__ == "__main__":
    unittest.main()
