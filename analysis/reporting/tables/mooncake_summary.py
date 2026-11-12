"""Generate exact Mooncake load and host-cache summary tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..validation import validate_figure


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _load_label(value: float) -> str:
    return f"L{round(value * 100):g}"


def render(frame: pd.DataFrame, spec: dict[str, Any], output_base: Path) -> list[Path]:
    required = [
        "load_fraction",
        "trace_window",
        "cache_ram_mb",
        "tasks_per_s",
        "task_latency_p95_s",
        "cached_ratio_step0",
        "cached_ratio_followup",
        "prefill_followup_p50_ms",
        "gpu_util_mean_pct",
        "server_rss_peak_mb",
    ]
    validate_figure(frame, spec, required)
    capacity = float(spec["calibrated_capacity_tasks_per_s"])

    baseline = frame[frame["cache_ram_mb"] == 0].copy()
    load = baseline.groupby("load_fraction", sort=True).agg(
        windows=("trace_window", "nunique"),
        task_throughput_mean=("tasks_per_s", "mean"),
        task_latency_p95_mean=("task_latency_p95_s", "mean"),
        task_latency_p95_min=("task_latency_p95_s", "min"),
        task_latency_p95_max=("task_latency_p95_s", "max"),
        step0_cached_ratio_mean=("cached_ratio_step0", "mean"),
        followup_cached_ratio_mean=("cached_ratio_followup", "mean"),
        followup_prefill_p50_ms_mean=("prefill_followup_p50_ms", "mean"),
        gpu_util_mean_pct=("gpu_util_mean_pct", "mean"),
    ).reset_index()
    load.insert(1, "nominal_offered_tasks_per_s", load["load_fraction"] * capacity)

    host = frame[
        (frame["trace_window"] == spec.get("host_cache_window", "W1"))
        & (frame["load_fraction"].isin(spec.get("host_cache_loads", [0.75, 0.95])))
    ].copy()
    host = host.sort_values(["load_fraction", "cache_ram_mb"])[[
        "load_fraction",
        "cache_ram_mb",
        "task_latency_p95_s",
        "cached_ratio_step0",
        "cached_ratio_followup",
        "prefill_followup_p50_ms",
        "gpu_util_mean_pct",
        "server_rss_peak_mb",
    ]]
    host["server_rss_peak_gib"] = host.pop("server_rss_peak_mb") / 1024

    output_base.parent.mkdir(parents=True, exist_ok=True)
    load_csv = output_base.with_name(f"{output_base.name}_load.csv")
    host_csv = output_base.with_name(f"{output_base.name}_host_cache.csv")
    markdown = output_base.with_suffix(".md")
    load.to_csv(load_csv, index=False)
    host.to_csv(host_csv, index=False)

    lines = [
        "# Mooncake FIFO baseline summary",
        "",
        (
            f"Load levels are fractions of the calibrated capacity "
            f"X = {capacity:.3f} tasks/s. Baseline values are means across W1–W3; "
            "the bracketed latency values show the full window range."
        ),
        "",
        "| Load | Nominal offered rate | Completed throughput | Task latency P95 | Step-0 reuse | Follow-up reuse | Follow-up prefill P50 | Mean GPU util. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in load.itertuples(index=False):
        lines.append(
            "| "
            f"{_load_label(row.load_fraction)} | "
            f"{row.nominal_offered_tasks_per_s:.3f} tasks/s | "
            f"{row.task_throughput_mean:.3f} tasks/s | "
            f"{row.task_latency_p95_mean:.1f} s [{row.task_latency_p95_min:.1f}–{row.task_latency_p95_max:.1f}] | "
            f"{_pct(row.step0_cached_ratio_mean)} | "
            f"{_pct(row.followup_cached_ratio_mean)} | "
            f"{row.followup_prefill_p50_ms_mean:.0f} ms | "
            f"{row.gpu_util_mean_pct:.1f}% |"
        )

    lines.extend([
        "",
        f"## Host-cache comparison ({spec.get('host_cache_window', 'W1')} only)",
        "",
        "| Load | Host cache | Task latency P95 | Step-0 reuse | Follow-up reuse | Follow-up prefill P50 | Mean GPU util. | Peak server RSS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in host.itertuples(index=False):
        cache = "Off" if row.cache_ram_mb == 0 else f"{row.cache_ram_mb / 1024:g} GiB"
        lines.append(
            "| "
            f"{_load_label(row.load_fraction)} | {cache} | "
            f"{row.task_latency_p95_s:.1f} s | "
            f"{_pct(row.cached_ratio_step0)} | "
            f"{_pct(row.cached_ratio_followup)} | "
            f"{row.prefill_followup_p50_ms:.0f} ms | "
            f"{row.gpu_util_mean_pct:.1f}% | "
            f"{row.server_rss_peak_gib:.1f} GiB |"
        )
    lines.extend([
        "",
        "Step-0 reuse measures shared-prefix reuse between trace-derived tasks. Follow-up reuse measures whether a task's carried context survives its synthetic tool waits.",
        "",
    ])
    markdown.write_text("\n".join(lines), encoding="utf-8")
    return [markdown, load_csv, host_csv]
