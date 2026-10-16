"""A minimal stand-in for llama-server, shaped like the pinned commit's responses."""

from __future__ import annotations

import asyncio
import json

from aiohttp import web
from aiohttp.test_utils import TestServer


def sse(event):
    return f"data: {json.dumps(event)}\n\n".encode()


def chunk(content=None, finish_reason=None):
    delta = {"content": content} if content is not None else {}
    return {
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


# Final chunk with stream_options.include_usage.
USAGE_CHUNK = {
    "object": "chat.completion.chunk",
    "choices": [],
    "usage": {
        "completion_tokens": 2,
        "prompt_tokens": 30,
        "total_tokens": 32,
        "prompt_tokens_details": {"cached_tokens": 12},
    },
    "timings": {
        "cache_n": 12,
        "prompt_n": 18,
        "prompt_ms": 45.5,
        "predicted_n": 2,
        "predicted_ms": 20.25,
    },
}

NORMAL_STREAM = [
    sse(chunk("Hel")),
    sse(chunk("lo")),
    sse(chunk(finish_reason="length")),
    sse(USAGE_CHUNK),
    b"data: [DONE]\n\n",
]


class FakeLlamaServer:
    """Serves /v1/chat/completions, /health, /props, /slots and /metrics.

    Set `handler` to change the chat response, `health_statuses` to script
    /health, and `metrics_enabled = False` to mimic a server without --metrics.
    """

    def __init__(self, total_slots=2, n_ctx=2048):
        self.bodies = []
        self.handler = self.stream(NORMAL_STREAM)
        self.health_statuses = [200]
        self.total_slots = total_slots
        self.n_ctx = n_ctx
        self.metrics_enabled = True
        self.active = 0
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._chat)
        app.router.add_get("/health", self._health)
        app.router.add_get("/props", self._props)
        app.router.add_get("/slots", self._slots)
        app.router.add_get("/metrics", self._metrics)
        self.server = TestServer(app)

    async def start(self):
        await self.server.start_server()
        return self

    async def close(self):
        await self.server.close()

    @property
    def url(self):
        return str(self.server.make_url("")).rstrip("/")

    async def _chat(self, request):
        self.bodies.append(await request.json())
        self.active += 1
        try:
            return await self.handler(request)
        finally:
            self.active -= 1

    async def _health(self, request):
        status = self.health_statuses.pop(0) if len(self.health_statuses) > 1 else self.health_statuses[0]
        return web.json_response({"status": "ok" if status == 200 else "loading"}, status=status)

    async def _props(self, request):
        return web.json_response({
            "default_generation_settings": {"n_ctx": self.n_ctx, "params": {}},
            "total_slots": self.total_slots,
            "model_path": "models/fake.gguf",
            "build_info": "b0-fake",
            "endpoint_slots": True,
            "endpoint_metrics": self.metrics_enabled,
            "chat_template": "{{ messages }}",
        })

    async def _slots(self, request):
        return web.json_response([
            {
                "id": i,
                "n_ctx": self.n_ctx,
                "is_processing": i < self.active,
                "n_prompt_tokens": 100 * (i + 1) if i < self.active else 0,
            }
            for i in range(self.total_slots)
        ])

    async def _metrics(self, request):
        if not self.metrics_enabled:
            return web.json_response(
                {"error": {"code": 501, "message": "This server does not support metrics endpoint."}},
                status=501,
            )
        return web.Response(
            text=(
                "# HELP llamacpp:requests_processing Number of requests processing.\n"
                "# TYPE llamacpp:requests_processing gauge\n"
                f"llamacpp:requests_processing {self.active}\n"
                "# TYPE llamacpp:requests_deferred gauge\n"
                "llamacpp:requests_deferred 0\n"
            ),
            content_type="text/plain",
        )

    @staticmethod
    def stream(pieces, delay_s=0.0):
        async def handler(request):
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            for piece in pieces:
                if delay_s:
                    await asyncio.sleep(delay_s)
                await resp.write(piece)
            await resp.write_eof()
            return resp

        return handler
