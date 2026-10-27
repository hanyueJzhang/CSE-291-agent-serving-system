from pathlib import Path
import unittest

from tests.fakes import FakeModelClient, ImmediateScheduler
from workload import generator as workload_generator
from workload import runner as workload_runner
from workload.traces import mooncake as trace


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


TRACE_WORKLOAD = (
    Path(__file__).resolve().parents[1] / "workload" / "tasks" / "mooncake_w2.json"
)


def load_trace_workload(num_tasks):
    workload = workload_generator.load_workload(TRACE_WORKLOAD)
    workload["arrival"]["num_tasks"] = num_tasks
    return workload


class TraceGeneratorTests(unittest.TestCase):
    def test_generation_is_reproducible(self):
        first = workload_generator.generate_tasks(load_trace_workload(120))
        second = workload_generator.generate_tasks(load_trace_workload(120))

        self.assertEqual(first, second)
        self.assertGreaterEqual(len(first), 120)
        self.assertEqual(len({task["id"] for task in first}), len(first))
        arrivals = [task["arrival_s"] for task in first]
        self.assertEqual(arrivals, sorted(arrivals))

    def test_step_shapes(self):
        task = workload_generator.generate_tasks(load_trace_workload(20))[0]

        self.assertEqual([s["kind"] for s in task["steps"]], ["llm", "tool", "llm", "tool", "llm"])
        step0 = task["steps"][0]
        self.assertNotIn("carry_context", step0)
        self.assertEqual(len(step0["prompt"].split()), step0["prompt_tokens"])
        self.assertNotIn(task["id"], step0["prompt"])
        self.assertTrue(task["steps"][2]["carry_context"])
        self.assertTrue(task["steps"][4]["carry_context"])
        self.assertIsInstance(task["family"], int)

    def test_same_hash_gives_same_text(self):
        self.assertEqual(trace.block_text(46), trace.block_text(46))
        self.assertNotEqual(trace.block_text(46), trace.block_text(47))
        self.assertEqual(len(trace.block_text(46).split()), trace.BLOCK_WORDS)

        rows, _ = trace.load_trace(load_trace_workload(1)["trace"])
        a, b = [r for r in rows if r["hash_ids"][:2] == [46, 47] and len(r["hash_ids"]) > 3][:2]
        shared = trace.BLOCK_WORDS * 2
        prompt_a = trace.trace_prompt(a["hash_ids"], trace.scaled_input_tokens(a["input_length"]))
        prompt_b = trace.trace_prompt(b["hash_ids"], trace.scaled_input_tokens(b["input_length"]))
        self.assertEqual(prompt_a.split()[:shared], prompt_b.split()[:shared])
        self.assertTrue(prompt_a.startswith(trace.block_text(46) + " " + trace.block_text(47)))
        self.assertNotEqual(prompt_a.split()[shared:], prompt_b.split()[shared:])

    def test_families_are_kept_or_dropped_whole(self):
        workload = load_trace_workload(300)
        candidates, _ = trace.window_tasks(workload)
        chosen = workload_generator.generate_tasks(workload)
        chosen_ids = {task["id"] for task in chosen}
        chosen_families = {task["family"] for task in chosen}

        for task in candidates:
            self.assertEqual(task["id"] in chosen_ids, task["family"] in chosen_families)

    def test_lower_load_is_subset_of_higher_load(self):
        low = workload_generator.generate_tasks(load_trace_workload(100))
        high = workload_generator.generate_tasks(load_trace_workload(300))

        self.assertLess(len(low), len(high))
        high_by_id = {task["id"]: task for task in high}
        for task in low:
            self.assertEqual(high_by_id.get(task["id"]), task)

    def test_arrivals_fall_in_window(self):
        workload = load_trace_workload(10_000)
        start_s, end_s = workload["trace"]["window_s"]
        rows, _ = trace.load_trace(workload["trace"])
        tasks = workload_generator.generate_tasks(workload)

        for task in tasks:
            offset = rows[int(task["id"].split("-")[1])]["timestamp"] / 1000 - start_s
            self.assertGreaterEqual(task["arrival_s"], offset)
            self.assertLess(task["arrival_s"], offset + trace.JITTER_S)
            self.assertGreaterEqual(task["arrival_s"], 0.0)
            self.assertLess(task["arrival_s"], end_s - start_s + trace.JITTER_S)

    def test_drops_tasks_that_do_not_fit(self):
        workload = load_trace_workload(10_000)
        limit = trace.FIT_MARGIN * workload["trace"]["fit_ctx_per_slot"]
        candidates, in_window = trace.window_tasks(workload)

        self.assertLess(len(candidates), in_window)
        self.assertTrue(all(
            workload_generator.estimate_peak_context(t) <= limit for t in candidates
        ))


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
