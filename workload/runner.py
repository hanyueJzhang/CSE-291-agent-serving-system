"""Execute one agent task step by step."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from scheduler import Request

from .llama_client import Completion

# Completion.status -> task status (results/SCHEMA.md)
_TASK_STATUS = {"error": "failed", "timeout": "timeout"}


async def run_task(
    task: dict[str, Any],
    scheduler: Any,
    model_client: Any,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Run one task and return its responses, scheduler requests and timings.

    LLM steps use scheduler admission. Tool steps run after admission has been
    released, so another agent can use the model while this task waits.
    A failed or timed-out LLM request ends the task; later steps are skipped.
    """
    t_arrive = clock()
    context: list[dict[str, str]] = []
    responses: list[str] = []
    requests: list[Request] = []
    completions: list[Completion] = []
    total_tool_s = 0.0
    num_steps = 0
    status = "ok"

    for step_idx, step in enumerate(task["steps"]):
        kind = step["kind"]

        if kind == "tool":
            t_tool = clock()
            await sleep(step["delay_s"])
            total_tool_s += clock() - t_tool
            context.append(
                {
                    "role": "tool",
                    "content": step.get("result", "Simulated tool result"),
                }
            )
            num_steps += 1
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
            completion = await model_client.complete(
                messages=list(context),
                max_tokens=step["max_output_tokens"],
            )

        requests.append(request)
        completions.append(completion)
        if completion.status in _TASK_STATUS:
            status = _TASK_STATUS[completion.status]
            break

        context.append({"role": "assistant", "content": completion.text})
        responses.append(completion.text)
        num_steps += 1

    return {
        "task_id": task["id"],
        "task_type": task["type"],
        "status": status,
        "responses": responses,
        "context": context,
        "requests": requests,
        "completions": completions,
        "t_arrive": t_arrive,
        "t_finish": clock(),
        "num_steps": num_steps,
        "total_queue_s": sum(r.queue_s or 0.0 for r in requests),
        "total_tool_s": total_tool_s,
    }


async def run_workload(
    tasks: list[dict[str, Any]],
    scheduler: Any,
    model_client: Any,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> list[dict[str, Any]]:
    """Run concrete tasks at their generated arrival times."""

    async def run_at_arrival(task: dict[str, Any]) -> dict[str, Any]:
        if task["arrival_s"] > 0:
            await sleep(task["arrival_s"])
        return await run_task(task, scheduler, model_client, sleep=sleep, clock=clock)

    return await asyncio.gather(*(run_at_arrival(task) for task in tasks))
