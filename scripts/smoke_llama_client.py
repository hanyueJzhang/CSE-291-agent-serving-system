"""Check LlamaClient against an already-running llama-server.

    python -m scripts.smoke_llama_client [--url http://localhost:8080]

Only sends three short requests; it never starts or stops the server.

Sends a plain request, then repeats it (should hit the prompt cache), then a
multi-turn request with a "tool" message, which is what the workload runner
sends after a tool step.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict

from workload.llama_client import LlamaClient


def show(label, completion):
    print(f"--- {label}")
    fields = asdict(completion)
    fields["ttft_s"] = completion.ttft_s
    fields["text"] = completion.text[:80]
    for key, value in fields.items():
        print(f"  {key:14} {value}")


async def main(base_url: str) -> int:
    async with LlamaClient(base_url) as client:
        await client.wait_ready()
        prompt = [{"role": "user", "content": "List three prime numbers. " + "token " * 200}]

        first = await client.complete(prompt, max_tokens=32)
        show("plain request", first)
        second = await client.complete(prompt, max_tokens=32)
        show("same request again (expect cached_tokens > 0)", second)

        with_tool = prompt + [
            {"role": "assistant", "content": first.text or "2, 3, 5"},
            {"role": "tool", "content": "Simulated tool result"},
            {"role": "user", "content": "Continue."},
        ]
        third = await client.complete(with_tool, max_tokens=32)
        show("multi-turn with a tool message", third)

        results = [first, second, third]
        failed = [c for c in results if c.status != "ok"]
        if failed:
            print("\nFAILED:", [c.error for c in failed])
            return 1
        print("\nOK")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    raise SystemExit(asyncio.run(main(parser.parse_args().url)))
