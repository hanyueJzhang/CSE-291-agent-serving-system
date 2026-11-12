"""Mooncake host prompt-cache benefit and memory-cost figure."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..style import COLORS, finish_axes, save_figure
from ..validation import validate_figure


def _cache_label(cache_ram_mb: float) -> str:
    if cache_ram_mb == 0:
        return "Host cache off"
    return f"{cache_ram_mb / 1024:g} GiB host cache"


def render(
    frame: pd.DataFrame,
    spec: dict[str, Any],
    output_base: Path,
    formats: Iterable[str],
) -> list[Path]:
    required = [
        "load_fraction",
        "cache_ram_mb",
        "task_latency_p95_s",
        "cached_ratio_followup",
        "server_rss_peak_mb",
    ]
    validate_figure(frame, spec, required)
    data = frame.sort_values(["load_fraction", "cache_ram_mb"]).copy()
    loads = sorted(data["load_fraction"].unique())
    cache_sizes = sorted(data["cache_ram_mb"].unique())
    expected = {(load, cache) for load in loads for cache in cache_sizes}
    actual = set(zip(data["load_fraction"], data["cache_ram_mb"]))
    if actual != expected or len(data) != len(expected):
        raise ValueError("mooncake_host_cache: expected one run for every load/cache combination")

    x = np.arange(len(loads), dtype=float)
    width = 0.36
    metrics = [
        ("task_latency_p95_s", "Task completion latency P95 (s)", "Tail latency", 1.0),
        ("cached_ratio_followup", "Follow-up prompt tokens cached (%)", "Context reuse", 100.0),
        ("server_rss_peak_mb", "Peak server RSS (GiB)", "Host-memory cost", 1 / 1024),
    ]
    colors = [COLORS["gray"], COLORS["blue"]]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.1))
    handles = []
    for cache_index, cache_size in enumerate(cache_sizes):
        cache_data = data[data["cache_ram_mb"] == cache_size].set_index("load_fraction").reindex(loads)
        offset = (cache_index - (len(cache_sizes) - 1) / 2) * width
        for axis_index, (column, ylabel, title, scale) in enumerate(metrics):
            values = cache_data[column].astype(float).to_numpy() * scale
            bars = axes[axis_index].bar(
                x + offset,
                values,
                width,
                color=colors[cache_index % len(colors)],
                label=_cache_label(float(cache_size)),
            )
            axes[axis_index].bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
            if axis_index == 0:
                handles.append(bars)

    labels = [f"L{round(load * 100):g}" for load in loads]
    for axis, (_, ylabel, title, _) in zip(axes, metrics):
        axis.set_xticks(x, labels)
        axis.set_xlabel("Offered load level")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
    axes[1].set_ylim(0, 100)

    finish_axes(axes, title=spec.get("title"))
    fig.legend(
        handles,
        [_cache_label(float(value)) for value in cache_sizes],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(cache_sizes),
        frameon=False,
    )
    return save_figure(fig, output_base, formats)
