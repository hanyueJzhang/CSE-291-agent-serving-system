import asyncio
import unittest

from scheduler import FIFOScheduler, Request, Scheduler, make_scheduler


class FakeClock:
    """Manually advanced clock so timestamps can be checked exactly."""

    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


def make_request(task_id, step_idx=0, **fields):
    return Request(task_id=task_id, step_idx=step_idx, **fields)


async def settle():
    """Let every runnable task run until it blocks again."""
    for _ in range(3):
        await asyncio.sleep(0)


class FIFOConfigTests(unittest.TestCase):
    def test_rejects_non_positive_concurrency(self):
        for bad in (0, -1):
            with self.subTest(max_concurrency=bad):
                with self.assertRaises(ValueError):
                    FIFOScheduler(max_concurrency=bad)

    def test_config_reports_name_and_cap(self):
        self.assertEqual(
            FIFOScheduler(max_concurrency=3).config(),
            {"name": "fifo", "max_concurrency": 3},
        )

    def test_make_scheduler_builds_by_name(self):
        scheduler = make_scheduler("fifo", max_concurrency=2)

        self.assertIsInstance(scheduler, FIFOScheduler)
        self.assertEqual(scheduler.max_concurrency, 2)

    def test_make_scheduler_rejects_unknown_name(self):
        with self.assertRaisesRegex(ValueError, "unknown scheduler 'nope'"):
            make_scheduler("nope")

    def test_queue_s_needs_both_timestamps(self):
        request = make_request("task-0000")
        self.assertIsNone(request.queue_s)

        request.t_arrive = 1.0
        self.assertIsNone(request.queue_s)

        request.t_dispatch = 3.5
        self.assertEqual(request.queue_s, 2.5)


class FIFOAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_admits_up_to_cap_and_queues_the_rest(self):
        scheduler = FIFOScheduler(max_concurrency=2)
        requests = [make_request(f"task-{i:04d}") for i in range(4)]
        tasks = [asyncio.create_task(scheduler.acquire(r)) for r in requests]
        await settle()

        self.assertEqual([t.done() for t in tasks], [True, True, False, False])
        self.assertEqual((scheduler.num_running, scheduler.queue_len), (2, 2))

        scheduler.release(requests[0])
        await settle()
        self.assertEqual([t.done() for t in tasks], [True, True, True, False])
        self.assertEqual((scheduler.num_running, scheduler.queue_len), (2, 1))

        scheduler.release(requests[1])
        scheduler.release(requests[2])
        await settle()
        self.assertTrue(tasks[3].done())

        scheduler.release(requests[3])
        self.assertEqual((scheduler.num_running, scheduler.queue_len), (0, 0))

    async def test_admits_in_arrival_order(self):
        scheduler = FIFOScheduler(max_concurrency=1)
        admitted = []

        async def worker(request):
            async with scheduler.admit(request):
                admitted.append(request.task_id)
                await asyncio.sleep(0)

        arrival = ["task-0003", "task-0000", "task-0004", "task-0001", "task-0002"]
        await asyncio.gather(*(worker(make_request(t)) for t in arrival))

        self.assertEqual(admitted, arrival)

    async def test_never_exceeds_cap_under_load(self):
        scheduler = FIFOScheduler(max_concurrency=3)
        in_flight = 0
        peak = 0
        admitted = []

        async def worker(i):
            nonlocal in_flight, peak
            async with scheduler.admit(make_request(f"task-{i:04d}")):
                admitted.append(i)
                in_flight += 1
                peak = max(peak, in_flight)
                for _ in range(i % 4):
                    await asyncio.sleep(0)
                in_flight -= 1

        await asyncio.gather(*(worker(i) for i in range(20)))

        self.assertEqual(peak, 3)
        self.assertEqual(admitted, list(range(20)))
        self.assertEqual((scheduler.num_running, scheduler.queue_len), (0, 0))

    async def test_records_timestamps_on_scheduler_clock(self):
        clock = FakeClock(10.0)
        scheduler = FIFOScheduler(max_concurrency=1, clock=clock)
        first, second = make_request("task-0000"), make_request("task-0001")

        await scheduler.acquire(first)
        clock.now = 11.0
        waiter = asyncio.create_task(scheduler.acquire(second))
        await settle()
        clock.now = 13.0
        scheduler.release(first)
        await waiter
        clock.now = 14.5
        scheduler.release(second)

        self.assertEqual((first.t_arrive, first.t_dispatch, first.t_finish), (10.0, 10.0, 13.0))
        self.assertEqual((second.t_arrive, second.t_dispatch, second.t_finish), (11.0, 13.0, 14.5))
        self.assertEqual(first.queue_s, 0.0)
        self.assertEqual(second.queue_s, 2.0)

    async def test_keeps_preset_arrival_time(self):
        scheduler = FIFOScheduler(max_concurrency=1, clock=FakeClock(5.0))
        request = make_request("task-0000", t_arrive=2.0)

        async with scheduler.admit(request):
            pass

        self.assertEqual(request.t_arrive, 2.0)
        self.assertEqual(request.queue_s, 3.0)

    async def test_admit_releases_slot_when_body_raises(self):
        scheduler = FIFOScheduler(max_concurrency=1)
        request = make_request("task-0000")

        with self.assertRaises(RuntimeError):
            async with scheduler.admit(request):
                raise RuntimeError("model call failed")

        self.assertEqual(scheduler.num_running, 0)
        self.assertIsNotNone(request.t_finish)
        async with scheduler.admit(make_request("task-0001")):
            self.assertEqual(scheduler.num_running, 1)


class MisuseTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_duplicate_submission(self):
        scheduler = FIFOScheduler(max_concurrency=1)
        running, waiting = make_request("task-0000"), make_request("task-0001")
        await scheduler.acquire(running)
        waiter = asyncio.create_task(scheduler.acquire(waiting))
        await settle()

        for request in (running, waiting):
            with self.subTest(task_id=request.task_id):
                with self.assertRaisesRegex(ValueError, "already submitted"):
                    await scheduler.acquire(request)
        self.assertEqual((scheduler.num_running, scheduler.queue_len), (1, 1))

        scheduler.release(running)
        await waiter
        scheduler.release(waiting)

    async def test_release_requires_running_request(self):
        scheduler = FIFOScheduler(max_concurrency=1)
        request = make_request("task-0000")

        with self.assertRaisesRegex(ValueError, "not running"):
            scheduler.release(request)

        async with scheduler.admit(request):
            pass
        with self.assertRaisesRegex(ValueError, "not running"):
            scheduler.release(request)


class CancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_waiter_leaves_queue(self):
        scheduler = FIFOScheduler(max_concurrency=1)
        holder, quitter, next_up = (make_request(f"task-{i:04d}") for i in range(3))
        await scheduler.acquire(holder)
        quitter_task = asyncio.create_task(scheduler.acquire(quitter))
        next_task = asyncio.create_task(scheduler.acquire(next_up))
        await settle()
        self.assertEqual(scheduler.queue_len, 2)

        quitter_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await quitter_task
        self.assertEqual(scheduler.queue_len, 1)

        scheduler.release(holder)
        await next_task
        self.assertIsNone(quitter.t_dispatch)
        self.assertEqual((scheduler.num_running, scheduler.queue_len), (1, 0))
        scheduler.release(next_up)

    async def test_release_skips_waiter_whose_cancel_is_pending(self):
        scheduler = FIFOScheduler(max_concurrency=1)
        holder, quitter, next_up = (make_request(f"task-{i:04d}") for i in range(3))
        await scheduler.acquire(holder)
        quitter_task = asyncio.create_task(scheduler.acquire(quitter))
        next_task = asyncio.create_task(scheduler.acquire(next_up))
        await settle()

        # The quitter's grant future is cancelled, but its cleanup has not run yet.
        quitter_task.cancel()
        scheduler.release(holder)

        await next_task
        with self.assertRaises(asyncio.CancelledError):
            await quitter_task
        self.assertIsNone(quitter.t_dispatch)
        self.assertEqual((scheduler.num_running, scheduler.queue_len), (1, 0))
        scheduler.release(next_up)

    async def test_waiter_cancelled_after_grant_returns_its_slot(self):
        scheduler = FIFOScheduler(max_concurrency=1)
        holder, granted, later = (make_request(f"task-{i:04d}") for i in range(3))
        await scheduler.acquire(holder)
        granted_task = asyncio.create_task(scheduler.acquire(granted))
        await settle()

        # Admits `granted`, but its task has not resumed to see the grant yet.
        scheduler.release(holder)
        self.assertEqual(scheduler.num_running, 1)
        granted_task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await granted_task
        self.assertIsNotNone(granted.t_dispatch)
        self.assertIsNotNone(granted.t_finish)
        self.assertEqual(scheduler.num_running, 0)

        await asyncio.wait_for(scheduler.acquire(later), timeout=0.5)
        self.assertEqual(scheduler.num_running, 1)
        scheduler.release(later)


class PolicyHookTests(unittest.IsolatedAsyncioTestCase):
    async def test_base_scheduler_requires_pick_next(self):
        with self.assertRaises(NotImplementedError):
            await Scheduler().acquire(make_request("task-0000"))

    async def test_rejects_pick_next_returning_unknown_request(self):
        class BadScheduler(Scheduler):
            def pick_next(self, waiting, running):
                return make_request("stranger")

        with self.assertRaisesRegex(RuntimeError, "not waiting"):
            await BadScheduler().acquire(make_request("task-0000"))

    async def test_admits_whatever_the_policy_picks(self):
        class ShortestPromptFirst(Scheduler):
            def pick_next(self, waiting, running):
                if running:
                    return None
                return min(waiting, key=lambda r: r.prompt_tokens)

        scheduler = ShortestPromptFirst()
        holder = make_request("holder")
        await scheduler.acquire(holder)
        requests = [
            make_request(f"task-{i:04d}", prompt_tokens=tokens)
            for i, tokens in enumerate([300, 100, 200])
        ]
        tasks = [asyncio.create_task(scheduler.acquire(r)) for r in requests]
        await settle()

        admitted = []
        current = holder
        for _ in requests:
            scheduler.release(current)
            await settle()
            (current,) = [r for r, t in zip(requests, tasks) if t.done() and r.t_finish is None]
            admitted.append(current.prompt_tokens)
        scheduler.release(current)

        self.assertEqual(admitted, [100, 200, 300])


if __name__ == "__main__":
    unittest.main()
