import asyncio
import socket
import unittest

from aiohttp import web

from tests.fake_llama_server import (
    NORMAL_STREAM, USAGE_CHUNK, FakeLlamaServer, chunk, finish_chunk_with_slot, sse,
)
from workload.llama_client import LlamaClient, LlamaClientError


class LlamaClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fake = await FakeLlamaServer().start()
        self.client = LlamaClient(self.fake.url, timeout_s=2.0)

    async def asyncTearDown(self):
        await self.client.close()
        await self.fake.close()

    async def test_streams_text_and_reads_server_metrics(self):
        messages = [{"role": "user", "content": "hi"}]
        result = await self.client.complete(messages, max_tokens=2)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.text, "Hello")
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(result.prompt_tokens, 30)
        self.assertEqual(result.cached_tokens, 12)
        self.assertEqual(result.output_tokens, 2)
        self.assertEqual(result.prefill_ms, 45.5)
        self.assertEqual(result.decode_ms, 20.25)
        self.assertLessEqual(result.t_start, result.t_first_token)
        self.assertLessEqual(result.t_first_token, result.t_finish)

        body = self.fake.bodies[0]
        self.assertEqual(body["messages"], messages)
        self.assertEqual(body["max_tokens"], 2)
        self.assertTrue(body["stream"])
        self.assertEqual(body["stream_options"], {"include_usage": True})
        self.assertTrue(body["ignore_eos"])
        self.assertTrue(body["cache_prompt"])
        self.assertEqual(body["temperature"], 0.0)

    async def test_asks_for_and_reads_the_serving_slot(self):
        self.fake.handler = FakeLlamaServer.stream([
            sse(chunk("Hi")),
            sse(finish_chunk_with_slot(3)),
            sse(USAGE_CHUNK),
            b"data: [DONE]\n\n",
        ])
        result = await self.client.complete([{"role": "user", "content": "hi"}], max_tokens=2)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.slot_id, 3)
        self.assertEqual(result.text, "Hi")
        self.assertIs(self.fake.bodies[0]["verbose"], True)
        self.assertNotIn("id_slot", self.fake.bodies[0])

    async def test_slot_id_blank_when_server_sends_no_verbose(self):
        result = await self.client.complete([{"role": "user", "content": "hi"}], max_tokens=2)
        self.assertEqual(result.status, "ok")
        self.assertIsNone(result.slot_id)

    async def test_pins_slot_and_can_skip_verbose(self):
        client = LlamaClient(self.fake.url, timeout_s=2.0, report_slot=False)
        try:
            await client.complete([{"role": "user", "content": "hi"}], max_tokens=2, id_slot=2)
        finally:
            await client.close()
        body = self.fake.bodies[0]
        self.assertEqual(body["id_slot"], 2)
        self.assertNotIn("verbose", body)

    async def test_generate_returns_text(self):
        text = await self.client.generate([{"role": "user", "content": "hi"}], max_tokens=2)
        self.assertEqual(text, "Hello")

    async def test_handles_sse_lines_split_across_network_chunks(self):
        raw = b"".join(NORMAL_STREAM)
        pieces = [raw[i:i + 7] for i in range(0, len(raw), 7)]
        self.fake.handler = FakeLlamaServer.stream(pieces)

        result = await self.client.complete([{"role": "user", "content": "hi"}], max_tokens=2)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.text, "Hello")
        self.assertEqual(result.prompt_tokens, 30)

    async def test_back_to_back_requests_when_server_drops_connections(self):
        # llama-server closes a connection after one streamed response without
        # sending "Connection: close". A client that reuses the pooled connection
        # then fails instantly with ServerDisconnectedError (seen on the 3080).
        body = b"".join(NORMAL_STREAM)
        response = (
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
        )

        async def one_request_per_connection(reader, writer):
            await reader.readuntil(b"\r\n\r\n")
            writer.write(response)
            await writer.drain()
            try:
                # A second request on this connection gets no answer: just close.
                await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=1)
            except (asyncio.IncompleteReadError, asyncio.TimeoutError, ConnectionError):
                pass
            writer.close()

        server = await asyncio.start_server(one_request_per_connection, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = LlamaClient(f"http://127.0.0.1:{port}")
        messages = [{"role": "user", "content": "hi"}]
        try:
            results = [await client.complete(messages, max_tokens=2) for _ in range(3)]
        finally:
            await client.close()
            server.close()
            await server.wait_closed()

        self.assertEqual([r.status for r in results], ["ok", "ok", "ok"], [r.error for r in results])

    async def test_falls_back_to_timings_without_usage(self):
        timings_only = {k: v for k, v in USAGE_CHUNK.items() if k != "usage"}
        self.fake.handler = FakeLlamaServer.stream([
            sse(chunk("Hi")),
            sse(chunk(finish_reason="stop")),
            sse(timings_only),
            b"data: [DONE]\n\n",
        ])

        result = await self.client.complete([{"role": "user", "content": "hi"}], max_tokens=2)

        self.assertEqual(result.prompt_tokens, 30)   # prompt_n + cache_n
        self.assertEqual(result.cached_tokens, 12)
        self.assertEqual(result.output_tokens, 2)

    async def test_http_error_is_reported_not_raised(self):
        async def reject(request):
            return web.json_response(
                {"error": {"code": 400, "message": "request (3000 tokens) exceeds the available context size (2048 tokens)"}},
                status=400,
            )

        self.fake.handler = reject

        result = await self.client.complete([{"role": "user", "content": "hi"}], max_tokens=2)

        self.assertEqual(result.status, "error")
        self.assertIn("HTTP 400", result.error)
        self.assertIn("exceeds the available context size", result.error)
        self.assertIsNotNone(result.t_finish)

        with self.assertRaises(LlamaClientError) as ctx:
            await self.client.generate([{"role": "user", "content": "hi"}], max_tokens=2)
        self.assertEqual(ctx.exception.completion.status, "error")

    async def test_error_event_mid_stream(self):
        self.fake.handler = FakeLlamaServer.stream([
            sse(chunk("partial")),
            sse({"error": {"code": 500, "message": "Context size has been exceeded."}}),
        ])

        result = await self.client.complete([{"role": "user", "content": "hi"}], max_tokens=2)

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error, "Context size has been exceeded.")
        self.assertEqual(result.text, "partial")

    async def test_stream_without_finish_reason_is_an_error(self):
        self.fake.handler = FakeLlamaServer.stream([sse(chunk("cut off"))])

        result = await self.client.complete([{"role": "user", "content": "hi"}], max_tokens=2)

        self.assertEqual(result.status, "error")
        self.assertIn("finish_reason", result.error)

    async def test_timeout(self):
        self.fake.handler = FakeLlamaServer.stream(NORMAL_STREAM, delay_s=0.2)
        client = LlamaClient(self.fake.url, timeout_s=0.3)
        try:
            result = await client.complete([{"role": "user", "content": "hi"}], max_tokens=2)
            self.assertEqual(result.status, "timeout")
            with self.assertRaises(LlamaClientError):
                await client.generate([{"role": "user", "content": "hi"}], max_tokens=2)
        finally:
            await client.close()

    async def test_connection_refused_is_an_error(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        client = LlamaClient(f"http://127.0.0.1:{port}")
        try:
            result = await client.complete([{"role": "user", "content": "hi"}], max_tokens=2)
            self.assertEqual(result.status, "error")
        finally:
            await client.close()

    async def test_wait_ready_polls_until_healthy(self):
        self.fake.health_statuses = [503, 503, 200]
        await self.client.wait_ready(timeout_s=5.0, poll_s=0.01)
        self.assertEqual(self.fake.health_statuses, [200])

    async def test_wait_ready_times_out(self):
        self.fake.health_statuses = [503]
        with self.assertRaises(TimeoutError):
            await self.client.wait_ready(timeout_s=0.05, poll_s=0.01)

    async def test_sampling_options_are_sent(self):
        client = LlamaClient(self.fake.url, ignore_eos=False, cache_prompt=False, temperature=0.7, seed=3)
        try:
            await client.complete([{"role": "user", "content": "hi"}], max_tokens=5)
        finally:
            await client.close()

        body = self.fake.bodies[-1]
        self.assertFalse(body["ignore_eos"])
        self.assertFalse(body["cache_prompt"])
        self.assertEqual(body["temperature"], 0.7)
        self.assertEqual(body["seed"], 3)


if __name__ == "__main__":
    unittest.main()
