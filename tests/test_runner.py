import unittest

from tests.fakes import FakeModelClient, ImmediateScheduler
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


if __name__ == "__main__":
    unittest.main()
