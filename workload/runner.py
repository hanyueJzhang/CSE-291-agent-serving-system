"""Execute one agent task step by step."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from scheduler import Request


async def run_task(
    task: dict[str, Any],
    scheduler: Any,
    model_client: Any,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> dict[str, Any]:
    """Run one task and return its responses and scheduler requests.

    LLM steps use scheduler admission. Tool steps run after admission has been
    released, so another agent can use the model while this task waits.
    """
    context: list[dict[str, str]] = []
    responses: list[str] = []
    requests: list[Request] = []

    for step_idx, step in enumerate(task["steps"]):
        kind = step["kind"]

        if kind == "tool":
            delay = step["delay_s"]
            if delay["dist"] != "fixed":
                raise ValueError(f"unsupported tool delay distribution: {delay['dist']!r}")
            await sleep(delay["value"])
            context.append(
                {
                    "role": "tool",
                    "content": step.get("result", "Simulated tool result"),
                }
            )
            continue

        if kind != "llm":
            raise ValueError(f"unsupported step kind: {kind!r}")

        if context and not step.get("carry_context", False):
            context = []

        context.append({"role": "user", "content": step["prompt"]})
        request = Request(
            task_id=task["id"],
            step_idx=step_idx,
            task_type=task["type"],
            prompt_tokens=step["prompt_tokens"],
            max_output_tokens=step["max_output_tokens"],
        )

        async with scheduler.admit(request):
            response = await model_client.generate(
                messages=list(context),
                max_tokens=step["max_output_tokens"],
            )

        context.append({"role": "assistant", "content": response})
        responses.append(response)
        requests.append(request)

    return {
        "task_id": task["id"],
        "status": "ok",
        "responses": responses,
        "context": context,
        "requests": requests,
    }
