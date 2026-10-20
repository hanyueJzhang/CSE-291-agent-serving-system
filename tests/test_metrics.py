import csv
import tempfile
import unittest
from pathlib import Path

from scheduler import Request
from workload.llama_client import Completion
from workload.metrics import (
    REQUEST_COLUMNS, TASK_COLUMNS, request_rows, task_rows, write_csv,
)


def make_result(task_id, task_type, status, t_arrive, t_finish, requests):
    reqs, comps = [], []
    for step_idx, (t_req_arrive, t_dispatch, t_end, comp) in enumerate(requests):
        req = Request(task_id=task_id, step_idx=step_idx, task_type=task_type,
                      prompt_tokens=100, max_output_tokens=16)
        req.t_arrive, req.t_dispatch, req.t_finish = t_req_arrive, t_dispatch, t_end
        comp.t_finish = t_end
        reqs.append(req)
        comps.append(comp)
    return {
        "task_id": task_id, "task_type": task_type, "status": status,
        "requests": reqs, "completions": comps,
        "t_arrive": t_arrive, "t_finish": t_finish,
        "num_steps": len(reqs), "total_queue_s": sum(r.queue_s for r in reqs), "total_tool_s": 0.0,
    }


RESULTS = [
    make_result("t0", "short", "ok", 0.0, 2.0, [
        (0.0, 0.5, 2.0, Completion(text="a", prompt_tokens=100, cached_tokens=0, output_tokens=16, t_first_token=1.0)),
    ]),
    make_result("t1", "agent", "ok", 1.0, 5.0, [
        (1.0, 1.0, 2.0, Completion(text="b", prompt_tokens=100, cached_tokens=20, output_tokens=16, t_first_token=1.5)),
        (3.0, 4.0, 5.0, Completion(text="c", prompt_tokens=200, cached_tokens=130, output_tokens=16, t_first_token=4.5)),
    ]),
    make_result("t2", "short", "failed", 2.0, 3.0, [
        (2.0, 2.5, 3.0, Completion(status="error", error="HTTP 400")),
    ]),
]


class RowsTests(unittest.TestCase):
    def test_rows(self):
        reqs = request_rows("run1", RESULTS)
        tasks = task_rows("run1", RESULTS)

        self.assertEqual(len(reqs), 4)
        self.assertEqual(reqs[2]["step_idx"], 1)
        self.assertEqual(reqs[2]["cached_tokens"], 130)
        self.assertEqual(reqs[2]["t_first_token"], 4.5)
        self.assertEqual(reqs[3]["status"], "error")
        self.assertEqual(set(reqs[0]), set(REQUEST_COLUMNS))
        self.assertEqual(set(tasks[0]), set(TASK_COLUMNS))
        self.assertEqual(tasks[1]["total_queue_s"], 1.0)

    def test_write_csv_blanks_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "requests.csv"
            write_csv(path, REQUEST_COLUMNS, request_rows("run1", RESULTS))
            with path.open() as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[3]["prompt_tokens"], "")
        self.assertEqual(rows[3]["error"], "HTTP 400")


if __name__ == "__main__":
    unittest.main()
