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


if __name__ == "__main__":
    unittest.main()
