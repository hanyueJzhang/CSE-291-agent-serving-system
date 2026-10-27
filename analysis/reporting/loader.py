"""Load experiment artifacts into normalized report tables.

Measurements always come from ``summary.json``.  A legacy manifest may fill
semantic experiment dimensions that old runs did not record (for example,
``initial_prompt_tokens``); it must not replace measured values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunData:
    """One loaded run and its normalized report records."""

    path: Path
    summary: dict[str, Any]
    config: dict[str, Any]
    dimensions: dict[str, Any]
    record: dict[str, Any]
    step_records: tuple[dict[str, Any], ...]


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _get(obj: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = obj
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _relative_key(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_manifest(path: str | Path | None, *, root: Path = REPO_ROOT) -> dict[str, dict[str, Any]]:
    """Return legacy metadata keyed by normalized run path."""
    if path is None:
        return {}
    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = _read_json(manifest_path)
    if manifest.get("version") != 1:
        raise ValueError(f"{manifest_path}: unsupported or missing manifest version")
    runs = manifest.get("runs")
    if not isinstance(runs, dict):
        raise ValueError(f"{manifest_path}: 'runs' must be an object")
    normalized: dict[str, dict[str, Any]] = {}
    for raw_path, metadata in runs.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"{manifest_path}: metadata for {raw_path!r} must be an object")
        run_path = Path(raw_path)
        if not run_path.is_absolute():
            run_path = root / run_path
        normalized[_relative_key(run_path, root)] = dict(metadata.get("dimensions", metadata))
    return normalized


def _merge_dimensions(sources: Iterable[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    origins: dict[str, str] = {}
    for source_name, values in sources:
        if not isinstance(values, dict):
            raise ValueError(f"{source_name}: dimensions must be an object")
        for key, value in values.items():
            if key in merged and merged[key] != value:
                raise ValueError(
                    f"dimension {key!r} conflicts: {origins[key]} has {merged[key]!r}, "
                    f"but {source_name} has {value!r}"
                )
            merged[key] = value
            origins[key] = source_name
    return merged


RUN_FIELDS = {
    "run_id": "run_id",
    "workload": "workload",
    "scheduler": "scheduler",
    "concurrency": "concurrency",
    "rate_per_s": "rate_per_s",
    "seed": "seed",
    "server_slots": "server_slots",
    "ctx_per_slot": "ctx_per_slot",
    "cache_prompt": "cache_prompt",
    "makespan_s": "makespan_s",
    "tasks": "counts.tasks",
    "tasks_ok": "counts.tasks_ok",
    "tasks_failed": "counts.tasks_failed",
    "tasks_timeout": "counts.tasks_timeout",
    "requests": "counts.requests",
    "requests_ok": "counts.requests_ok",
    "requests_error": "counts.requests_error",
    "requests_timeout": "counts.requests_timeout",
    "tasks_per_s": "throughput.tasks_per_s",
    "requests_per_s": "throughput.requests_per_s",
    "output_tokens_per_s": "throughput.output_tokens_per_s",
    "uncached_prompt_tokens_per_s": "throughput.uncached_prompt_tokens_per_s",
    "queue_p50_s": "requests.queue_s.p50",
    "queue_p95_s": "requests.queue_s.p95",
    "queue_p99_s": "requests.queue_s.p99",
    "ttft_p50_s": "requests.ttft_s.p50",
    "ttft_p95_s": "requests.ttft_s.p95",
    "ttft_p99_s": "requests.ttft_s.p99",
    "server_ttft_p50_s": "requests.server_ttft_s.p50",
    "server_ttft_p95_s": "requests.server_ttft_s.p95",
    "request_e2e_p50_s": "requests.e2e_s.p50",
    "request_e2e_p95_s": "requests.e2e_s.p95",
    "task_latency_p50_s": "tasks.latency_s.p50",
    "task_latency_p95_s": "tasks.latency_s.p95",
    "task_latency_p99_s": "tasks.latency_s.p99",
    "task_latency_excl_tool_p50_s": "tasks.latency_excl_tool_s.p50",
    "task_latency_excl_tool_p95_s": "tasks.latency_excl_tool_s.p95",
    "task_queue_p50_s": "tasks.queue_s.p50",
    "task_queue_p95_s": "tasks.queue_s.p95",
    "task_queue_p99_s": "tasks.queue_s.p99",
    "decode_ms_per_token_p50": "requests.decode_ms_per_token.p50",
    "cached_ratio_all": "prompt_cache.cached_ratio_all",
    "cached_ratio_followup": "prompt_cache.cached_ratio_followup_steps",
    "prefill_followup_p50_ms": "prompt_cache.followup_steps.prefill_ms.p50",
    "prefill_followup_p95_ms": "prompt_cache.followup_steps.prefill_ms.p95",
    "uncached_followup_p50": "prompt_cache.followup_steps.uncached_tokens.p50",
    "gpu_util_mean_pct": "memory.gpu_util_mean_pct",
    "gpu_mem_peak_mb": "memory.gpu_mem_peak_mb",
    "server_rss_peak_mb": "memory.server_rss_peak_mb",
    "slot_tokens_peak": "memory.slot_tokens_peak",
    "active_slot_tokens_peak": "memory.active_slot_tokens_peak",
    "idle_slot_tokens_peak": "memory.idle_slot_tokens_peak",
    "idle_slot_tokens_mean": "memory.idle_slot_tokens_mean",
    "server_requests_deferred_max": "memory.server_requests_deferred_max",
}


def _run_record(run_path: Path, summary: dict[str, Any], dimensions: dict[str, Any], root: Path) -> dict[str, Any]:
    record = {name: _get(summary, path) for name, path in RUN_FIELDS.items()}
    record.update({
        "path": _relative_key(run_path, root),
        "group": run_path.parent.name,
    })
    for key, value in dimensions.items():
        if key in record and record[key] not in (None, value):
            raise ValueError(
                f"{run_path}: dimension {key!r}={value!r} conflicts with measured value {record[key]!r}"
            )
        record[key] = value
    record.setdefault("workload_family", record.get("workload"))

    slots, ctx = record.get("server_slots"), record.get("ctx_per_slot")
    capacity = slots * ctx if slots is not None and ctx is not None else None
    record["slot_capacity_tokens"] = capacity
    peak = record.get("slot_tokens_peak")
    record["slot_occupancy_ratio"] = peak / capacity if peak is not None and capacity else None
    tasks, ok = record.get("tasks"), record.get("tasks_ok")
    record["task_success_rate"] = ok / tasks if ok is not None and tasks else None
    return record


def _step_records(
    run_path: Path,
    summary: dict[str, Any],
    dimensions: dict[str, Any],
    base_record: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    rows = []
    for name, values in (summary.get("by_step") or {}).items():
        task_type, separator, raw_step = name.rpartition("/step")
        if not separator:
            raise ValueError(f"{run_path}: invalid by_step key {name!r}")
        row = {
            "path": base_record["path"],
            "group": base_record["group"],
            "run_id": base_record["run_id"],
            "workload": base_record["workload"],
            "task_type": task_type,
            "step_idx": int(raw_step),
            "prompt_tokens_p50": _get(values, "prompt_tokens.p50"),
            "prompt_tokens_p95": _get(values, "prompt_tokens.p95"),
            "uncached_tokens_p50": _get(values, "uncached_tokens.p50"),
            "uncached_tokens_p95": _get(values, "uncached_tokens.p95"),
            "prefill_p50_ms": _get(values, "prefill_ms.p50"),
            "prefill_p95_ms": _get(values, "prefill_ms.p95"),
            "server_ttft_p50_s": _get(values, "server_ttft_s.p50"),
            "server_ttft_p95_s": _get(values, "server_ttft_s.p95"),
            "decode_ms_per_token_p50": _get(values, "decode_ms_per_token.p50"),
        }
        row.update(dimensions)
        rows.append(row)
    return tuple(rows)


def load_run(
    path: str | Path,
    *,
    manifest: dict[str, dict[str, Any]] | None = None,
    root: Path = REPO_ROOT,
) -> RunData:
    """Load one run without modifying its stored artifacts."""
    run_path = Path(path)
    if not run_path.is_absolute():
        run_path = root / run_path
    summary = _read_json(run_path / "summary.json")
    config = _read_json(run_path / "run_config.json", required=False)
    legacy = (manifest or {}).get(_relative_key(run_path, root), {})
    config_dimensions = config.get("dimensions") or _get(config, "experiment.dimensions", {}) or {}
    summary_dimensions = summary.get("dimensions") or {}
    dimensions = _merge_dimensions([
        (f"{run_path}/run_config.json", config_dimensions),
        (f"{run_path}/summary.json", summary_dimensions),
        ("legacy manifest", legacy),
    ])
    record = _run_record(run_path, summary, dimensions, root)
    return RunData(
        path=run_path,
        summary=summary,
        config=config,
        dimensions=dimensions,
        record=record,
        step_records=_step_records(run_path, summary, dimensions, record),
    )


def load_runs(
    paths: Iterable[str | Path],
    *,
    manifest_path: str | Path | None = None,
    root: Path = REPO_ROOT,
) -> list[RunData]:
    manifest = load_manifest(manifest_path, root=root)
    return [load_run(path, manifest=manifest, root=root) for path in paths]


def run_frame(runs: Iterable[RunData]) -> pd.DataFrame:
    """Return one normalized row per experiment run."""
    rows = [run.record for run in runs]
    return pd.DataFrame(rows)


def step_frame(runs: Iterable[RunData]) -> pd.DataFrame:
    """Return one normalized row per run/task-type/LLM-step."""
    rows = [row for run in runs for row in run.step_records]
    return pd.DataFrame(rows)

