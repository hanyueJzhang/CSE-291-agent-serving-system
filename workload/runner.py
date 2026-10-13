"""Execute one agent task step by step."""

from __future__ import annotations

from typing import Any

from scheduler import Request


async def run_task(task: dict[str, Any], scheduler: Any, model_client: Any) -> dict[str, Any]:
    """Run one task and return its responses and scheduler requests.

    This first milestone supports a task containing LLM steps. Tool steps and
    workload-file generation will be added in later milestones.
    """
    context: list[dict[str, str]] = []
    responses: list[str] = []
    requests: list[Request] = []

    for step_idx, step in enumerate(task["steps"]):
        if step["kind"] != "llm":
            raise ValueError(f"unsupported step kind: {step['kind']!r}")

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
