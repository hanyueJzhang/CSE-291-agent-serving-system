"""Tune FIFO request concurrency for a fixed-slot server."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd

from ..style import PALETTE, finish_axes, save_figure
from ..validation import validate_figure


def render(frame: pd.DataFrame, spec: dict[str, Any], output_base: Path,
           formats: Iterable[str]) -> list[Path]:
    required = [
        "concurrency", "rate_per_s", "server_slots", "tasks_per_s",
        "task_latency_p95_s", "server_requests_deferred_max",
    ]
    validate_figure(frame, spec, required)
    data = frame.sort_values(["rate_per_s", "concurrency"])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    throughput, latency = axes
    groups = list(data.groupby("rate_per_s", sort=True))
    deferred_labeled = False
    for index, (rate, group) in enumerate(groups):
        group = group.sort_values("concurrency")
        color = PALETTE[index % len(PALETTE)]
        label = f"offered {rate:g} tasks/s"
        throughput.plot(group["concurrency"], group["tasks_per_s"], marker="o",
                        color=color, label=label)
        latency.plot(group["concurrency"], group["task_latency_p95_s"], marker="o",
                     color=color, label=label)

        deferred = group[group["server_requests_deferred_max"].fillna(0) > 0]
        if not deferred.empty:
            deferred_label = "llama.cpp queued internally" if not deferred_labeled else None
            throughput.scatter(deferred["concurrency"], deferred["tasks_per_s"], marker="x",
                               color="black", s=70, linewidth=1.5,
                               label=deferred_label)
            latency.scatter(deferred["concurrency"], deferred["task_latency_p95_s"], marker="x",
                            color="black", s=70, linewidth=1.5, label=deferred_label)
            deferred_labeled = True

    throughput.set(xlabel="Scheduler request concurrency limit",
                   ylabel="Completed throughput (tasks/s)")
    latency.set(xlabel="Scheduler request concurrency limit",
                ylabel="Task latency P95 (s)")
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        legend_order = sorted(range(len(labels)), key=lambda i: labels[i].startswith("llama.cpp"))
        axis.legend([handles[i] for i in legend_order], [labels[i] for i in legend_order],
                    loc="center right")
    finish_axes(axes, title=spec.get("title"))
    return save_figure(fig, output_base, formats)

