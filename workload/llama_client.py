"""Async client for llama-server's OpenAI-compatible chat endpoint.

Streams every request so we can timestamp the first token, and reads the
server-reported token counts and timings from the final chunks. Field names
match the pinned llama.cpp commit (see setup/config.env):

- the last chunk carries `timings`: cache_n, prompt_n, prompt_ms, predicted_n, predicted_ms
- with stream_options.include_usage, a final chunk with empty `choices` carries
  `usage`: prompt_tokens, completion_tokens, prompt_tokens_details.cached_tokens
- errors arrive either as a non-200 response or as a `data: {"error": ...}` event
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

import aiohttp


class LlamaClientError(RuntimeError):
    """Raised by generate() when a request does not produce usable output."""

    def __init__(self, completion: "Completion"):
        super().__init__(f"{completion.status}: {completion.error}")
        self.completion = completion


@dataclass
class Completion:
    """Result and measurements of one chat request."""

    text: str = ""
    status: str = "ok"            # ok / error / timeout (results/SCHEMA.md)
    finish_reason: str | None = None   # "length" = hit max_tokens, "stop" = model stopped
    error: str | None = None

    # Client clock (the runner's clock), seconds.
    t_start: float | None = None
    t_first_token: float | None = None
    t_finish: float | None = None

    # Server-reported.
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    output_tokens: int | None = None
    prefill_ms: float | None = None
    decode_ms: float | None = None

    @property
    def ttft_s(self) -> float | None:
        if self.t_start is None or self.t_first_token is None:
            return None
        return self.t_first_token - self.t_start


class LlamaClient:
    """Send chat requests to one llama-server instance.

    Defaults aim at controlled measurements: greedy sampling with a fixed seed,
    and ignore_eos so every response runs to exactly max_tokens.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        *,
        timeout_s: float = 300.0,
        temperature: float = 0.0,
        seed: int = 0,
        ignore_eos: bool = True,
        cache_prompt: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.seed = seed
        self.ignore_eos = ignore_eos
        self.cache_prompt = cache_prompt
        self._clock = clock
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "LlamaClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    def config(self) -> dict:
        """Client parameters, for run_config.json."""
        return {
            "base_url": self.base_url,
            "timeout_s": self.timeout_s,
            "temperature": self.temperature,
            "seed": self.seed,
            "ignore_eos": self.ignore_eos,
            "cache_prompt": self.cache_prompt,
        }

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            # No pool cap: the scheduler decides how many requests are in flight.
            # No session-wide timeout either: complete() enforces timeout_s per request.
            connector = aiohttp.TCPConnector(limit=0)
            self._session = aiohttp.ClientSession(
                connector=connector, timeout=aiohttp.ClientTimeout(total=None)
            )
        return self._session

    # ---- health ------------------------------------------------------------

    async def wait_ready(self, timeout_s: float = 120.0, poll_s: float = 1.0) -> None:
        """Wait until /health returns 200 (it returns 503 while the model loads)."""
        deadline = self._clock() + timeout_s
        last_error = "no response"
        while True:
            try:
                async with self._get_session().get(
                    f"{self.base_url}/health", timeout=aiohttp.ClientTimeout(total=poll_s * 5)
                ) as resp:
                    if resp.status == 200:
                        return
                    last_error = f"HTTP {resp.status}"
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = repr(exc)
            if self._clock() >= deadline:
                raise TimeoutError(f"llama-server at {self.base_url} not ready: {last_error}")
            await asyncio.sleep(poll_s)

    # ---- chat --------------------------------------------------------------

    async def complete(self, messages: list[dict[str, str]], max_tokens: int) -> Completion:
        """Run one chat request. Never raises for server errors or timeouts;
        they are reported through Completion.status instead."""
        body = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
            "ignore_eos": self.ignore_eos,
            "cache_prompt": self.cache_prompt,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        result = Completion(t_start=self._clock())
        try:
            await asyncio.wait_for(self._stream(body, result), timeout=self.timeout_s)
        except asyncio.TimeoutError:
            result.status = "timeout"
            result.error = f"no complete response within {self.timeout_s}s"
        except aiohttp.ClientError as exc:
            result.status = "error"
            result.error = repr(exc)
        result.t_finish = self._clock()
        return result

    async def generate(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        """Return the response text; raise LlamaClientError on error or timeout."""
        completion = await self.complete(messages, max_tokens)
        if completion.status in ("error", "timeout"):
            raise LlamaClientError(completion)
        return completion.text

    async def _stream(self, body: dict[str, Any], result: Completion) -> None:
        url = f"{self.base_url}/v1/chat/completions"
        async with self._get_session().post(url, json=body) as resp:
            if resp.status != 200:
                result.status = "error"
                result.error = f"HTTP {resp.status}: {_error_message(await resp.text())}"
                return

            parts: list[str] = []
            # SSE lines can be split across network chunks; readline() reassembles them.
            async for line in resp.content:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[len(b"data:"):].strip()
                if data == b"[DONE]":
                    break
                event = json.loads(data)
                if "error" in event:
                    result.status = "error"
                    result.error = _error_message(event)
                    break
                self._apply_event(event, result, parts)

            result.text = "".join(parts)
            if result.status == "ok" and result.finish_reason is None:
                result.status = "error"
                result.error = "stream ended without a finish_reason"

    def _apply_event(self, event: dict[str, Any], result: Completion, parts: list[str]) -> None:
        for choice in event.get("choices") or []:
            content = (choice.get("delta") or {}).get("content")
            if content:
                if result.t_first_token is None:
                    result.t_first_token = self._clock()
                parts.append(content)
            if choice.get("finish_reason"):
                result.finish_reason = choice["finish_reason"]

        usage = event.get("usage")
        if usage:
            result.prompt_tokens = usage.get("prompt_tokens")
            result.output_tokens = usage.get("completion_tokens")
            details = usage.get("prompt_tokens_details") or {}
            result.cached_tokens = details.get("cached_tokens")

        timings = event.get("timings")
        if timings:
            result.prefill_ms = timings.get("prompt_ms")
            result.decode_ms = timings.get("predicted_ms")
            # Fallbacks in case usage was not sent.
            if result.cached_tokens is None and "cache_n" in timings:
                result.cached_tokens = timings["cache_n"]
            if result.prompt_tokens is None and "prompt_n" in timings:
                result.prompt_tokens = timings["prompt_n"] + timings.get("cache_n", 0)
            if result.output_tokens is None and "predicted_n" in timings:
                result.output_tokens = timings["predicted_n"]


def _error_message(payload: str | dict) -> str:
    """Pull the message out of llama-server's {"error": {"message": ...}} body."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return payload[:500]
    error = payload.get("error", payload) if isinstance(payload, dict) else payload
    if isinstance(error, dict):
        return str(error.get("message", error))
    return str(error)
