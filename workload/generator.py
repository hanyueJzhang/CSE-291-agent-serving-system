"""Load workload definitions and generate concrete task instances."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

# Rough chat-template overhead per message, for context-fit estimates only.
_TEMPLATE_TOKENS_PER_MESSAGE = 8
_TOOL_RESULT_TOKENS = 20


def load_workload(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def _make_prompt(task_id: str, step_idx: int, token_count: int) -> str:
    words = [f"{task_id}-step-{step_idx}"]
    words.extend("token" for _ in range(token_count - 1))
    return " ".join(words)


def concrete_steps(
    task_id: str,
    template_steps: list[dict[str, Any]],
    rng: random.Random,
    first_step_idx: int = 0,
) -> list[dict[str, Any]]:
    """Materialize template steps: prompts for LLM steps, delays for tool steps."""
    steps = []

    for step_idx, step in enumerate(template_steps, start=first_step_idx):
        if step["kind"] == "llm":
            prompt_tokens = (
                step["prompt_tokens"]
                if "prompt_tokens" in step
                else step["append_tokens"]
            )
            concrete_step = {
                "kind": "llm",
                "prompt": _make_prompt(task_id, step_idx, prompt_tokens),
                "prompt_tokens": prompt_tokens,
                "max_output_tokens": step["max_output_tokens"],
            }
            if "carry_context" in step:
                concrete_step["carry_context"] = step["carry_context"]
            steps.append(concrete_step)
            continue

        delay = step["delay_s"]
        if delay["dist"] == "fixed":
            delay_s = delay["value"]
        elif delay["dist"] == "uniform":
            delay_s = rng.uniform(delay["min"], delay["max"])
        else:
            raise ValueError(f"unsupported tool delay distribution: {delay['dist']!r}")

        steps.append(
            {
                "kind": "tool",
                "delay_s": delay_s,
                "result": step.get(
                    "result",
                    f"Simulated tool result for {task_id} step {step_idx}",
                ),
            }
        )

    return steps


def generate_tasks(workload: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate reproducible concrete tasks from a workload definition."""
    if "trace" in workload:
        from .traces.mooncake import generate_trace_tasks

        return generate_trace_tasks(workload)

    rng = random.Random(workload["seed"])
    arrival = workload["arrival"]
    task_types = workload["task_types"]
    weights = [task_type["weight"] for task_type in task_types]
    arrival_s = 0.0
    tasks = []

    for task_index in range(arrival["num_tasks"]):
        arrival_s += rng.expovariate(arrival["rate_per_s"])
        template = rng.choices(task_types, weights=weights, k=1)[0]
        task_id = f"task-{task_index:04d}"
        tasks.append(
            {
                "id": task_id,
                "type": template["type"],
                "arrival_s": arrival_s,
                "steps": concrete_steps(task_id, template["steps"], rng),
            }
        )

    return tasks


def estimate_peak_context(task: dict[str, Any]) -> int:
    """Approximate the largest context (prompt + output) any request of the task needs."""
    peak = context = 0
    for step in task["steps"]:
        if step["kind"] == "tool":
            context += _TOOL_RESULT_TOKENS + _TEMPLATE_TOKENS_PER_MESSAGE
            continue
        if not step.get("carry_context", False):
            context = 0
        context += step["prompt_tokens"] + _TEMPLATE_TOKENS_PER_MESSAGE
        peak = max(peak, context + step["max_output_tokens"])
        context += step["max_output_tokens"] + _TEMPLATE_TOKENS_PER_MESSAGE
    return peak
