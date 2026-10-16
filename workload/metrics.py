"""Turn run results into the CSV rows of results/SCHEMA.md."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

REQUEST_COLUMNS = [
    "run_id", "task_id", "step_idx", "slot_id",
    "t_arrive", "t_dispatch", "t_first_token", "t_finish",
    "prompt_tokens", "cached_tokens", "output_tokens",
    "prefill_ms", "decode_ms",
    "status", "error",
]

TASK_COLUMNS = [
    "run_id", "task_id", "task_type",
    "t_arrive", "t_finish",
    "num_steps", "total_queue_s", "total_tool_s",
    "status",
]

MEMORY_COLUMNS = [
    "t",
    "gpu_mem_used_mb", "gpu_util_pct", "server_rss_mb",
    "slot_tokens_total", "server_requests_deferred",
]


def request_rows(run_id: str, results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for request, completion in zip(result["requests"], result["completions"]):
            rows.append({
                "run_id": run_id,
                "task_id": request.task_id,
                "step_idx": request.step_idx,
                "slot_id": completion.slot_id,
                "t_arrive": request.t_arrive,
                "t_dispatch": request.t_dispatch,
                "t_first_token": completion.t_first_token,
                "t_finish": completion.t_finish if completion.t_finish is not None else request.t_finish,
                "prompt_tokens": completion.prompt_tokens,
                "cached_tokens": completion.cached_tokens,
                "output_tokens": completion.output_tokens,
                "prefill_ms": completion.prefill_ms,
                "decode_ms": completion.decode_ms,
                "status": completion.status,
                "error": completion.error,
            })
    return rows


def task_rows(run_id: str, results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "run_id": run_id,
            "task_id": result["task_id"],
            "task_type": result["task_type"],
            "t_arrive": result["t_arrive"],
            "t_finish": result["t_finish"],
            "num_steps": result["num_steps"],
            "total_queue_s": result["total_queue_s"],
            "total_tool_s": result["total_tool_s"],
            "status": result["status"],
        }
        for result in results
    ]


def write_csv(path: str | Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
