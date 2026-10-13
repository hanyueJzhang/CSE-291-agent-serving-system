from pathlib import Path
import unittest

from tests.fakes import FakeModelClient, ImmediateScheduler
from workload import generator as workload_generator
from workload import runner as workload_runner


EXAMPLE_WORKLOAD = (
    Path(__file__).resolve().parents[1] / "workload" / "tasks" / "example_mix.json"
)
TEST_WORKLOADS = Path(__file__).resolve().parent / "fixtures" / "workloads.json"


def load_test_workload(name):
    return workload_generator.load_workload(TEST_WORKLOADS)[name]


class WorkloadGeneratorTests(unittest.TestCase):
    def test_loads_example_workload(self):
        workload = workload_generator.load_workload(EXAMPLE_WORKLOAD)

        self.assertEqual(workload["name"], "example_mix")
        self.assertEqual(workload["seed"], 42)
        self.assertEqual(workload["arrival"]["num_tasks"], 50)
        self.assertEqual(len(workload["task_types"]), 3)

    def test_generation_is_reproducible(self):
        workload = workload_generator.load_workload(EXAMPLE_WORKLOAD)

        first = workload_generator.generate_tasks(workload)
        second = workload_generator.generate_tasks(workload)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 50)
        self.assertEqual(
            [task["id"] for task in first],
            [f"task-{index:04d}" for index in range(50)],
        )
        arrivals = [task["arrival_s"] for task in first]
        self.assertEqual(arrivals, sorted(arrivals))

    def test_materializes_prompts_and_tool_delays(self):
        workload = load_test_workload("generator_features")

        task = workload_generator.generate_tasks(workload)[0]

        self.assertEqual(task["id"], "task-0000")
        self.assertEqual(task["type"], "agent_2request")
        self.assertGreaterEqual(task["arrival_s"], 0.0)
        self.assertEqual(len(task["steps"][0]["prompt"].split()), 3)
        self.assertEqual(task["steps"][2]["prompt_tokens"], 2)
        self.assertEqual(len(task["steps"][2]["prompt"].split()), 2)
        self.assertGreaterEqual(task["steps"][1]["delay_s"], 0.5)
        self.assertLessEqual(task["steps"][1]["delay_s"], 2.0)
        self.assertEqual(task["steps"][3]["delay_s"], 1.0)


class GeneratedTaskIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_workload_waits_for_generated_arrival_times(self):
        workload = load_test_workload("two_short_tasks")
        tasks = workload_generator.generate_tasks(workload)
        arrival_waits = []

        async def fake_sleep(delay):
            arrival_waits.append(delay)

        results = await workload_runner.run_workload(
            tasks,
            ImmediateScheduler(),
            FakeModelClient(),
            sleep=fake_sleep,
        )

        self.assertEqual(arrival_waits, [task["arrival_s"] for task in tasks])
        self.assertEqual([result["task_id"] for result in results], ["task-0000", "task-0001"])


if __name__ == "__main__":
    unittest.main()
