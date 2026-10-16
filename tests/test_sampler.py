import asyncio
import unittest

from tests.fake_llama_server import FakeLlamaServer
from workload.sampler import MemorySampler, _port_arg, parse_prometheus


class FakeProcess:
    class _Mem:
        rss = 512 * 2**20

    def memory_info(self):
        return self._Mem()


class ParseTests(unittest.TestCase):
    def test_parse_prometheus(self):
        text = (
            "# HELP llamacpp:requests_processing x\n"
            "# TYPE llamacpp:requests_processing gauge\n"
            "llamacpp:requests_processing 2\n"
            "llamacpp:prompt_tokens_total 1.5e+03\n"
            "other_metric 7\n"
        )
        self.assertEqual(parse_prometheus(text), {"requests_processing": 2.0, "prompt_tokens_total": 1500.0})

    def test_port_arg(self):
        self.assertEqual(_port_arg(["llama-server", "--port", "9000"]), "9000")
        self.assertEqual(_port_arg(["llama-server", "--port=9001"]), "9001")
        self.assertEqual(_port_arg(["llama-server", "-m", "x.gguf"]), "8080")


class MemorySamplerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fake = await FakeLlamaServer(total_slots=2).start()

    async def asyncTearDown(self):
        await self.fake.close()

    async def test_sample_reads_all_sources(self):
        self.fake.active = 1
        sampler = MemorySampler(
            clock=lambda: 1.25, base_url=self.fake.url,
            server_process=FakeProcess(), use_gpu=False,
        )
        await sampler.start()
        rows = await sampler.stop()

        row = rows[-1]
        self.assertEqual(row["t"], 1.25)
        self.assertEqual(row["server_rss_mb"], 512.0)
        self.assertEqual(row["slot_tokens_total"], 100)
        self.assertEqual(row["server_requests_deferred"], 0.0)
        self.assertNotIn("gpu_mem_used_mb", row)
        self.assertEqual(sampler.notes, [])

    async def test_samples_periodically_until_stopped(self):
        sampler = MemorySampler(clock=asyncio.get_running_loop().time, interval_s=0.01,
                                server_process=FakeProcess(), use_gpu=False)
        await sampler.start()
        await asyncio.sleep(0.1)
        rows = await sampler.stop()
        self.assertGreaterEqual(len(rows), 4)
        self.assertEqual(rows, sorted(rows, key=lambda r: r["t"]))

    async def test_missing_metrics_endpoint_is_noted_and_skipped(self):
        self.fake.metrics_enabled = False
        sampler = MemorySampler(clock=lambda: 0.0, base_url=self.fake.url, use_gpu=False)
        await sampler.start()
        await sampler.sample()
        rows = await sampler.stop()

        self.assertIsNone(rows[-1].get("server_requests_deferred"))
        self.assertIn("slot_tokens_total", rows[-1])
        self.assertEqual(sum("/metrics unavailable" in n for n in sampler.notes), 1)
        self.assertTrue(any("process not found" in n for n in sampler.notes))


if __name__ == "__main__":
    unittest.main()
