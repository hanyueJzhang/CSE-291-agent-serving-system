"""Request concurrency versus server-slot comparison figure."""

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
        "task_latency_p50_s", "task_latency_p95_s", "server_requests_deferred_max",
    ]
    validate_figure(frame, spec, required)
    data = frame.sort_values(["rate_per_s", "server_slots", "concurrency"])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    throughput, latency = axes
    groups = list(data.groupby(["rate_per_s", "server_slots"], sort=True))
    deferred_labeled = False
    for index, ((rate, slots), group) in enumerate(groups):
        group = group.sort_values("concurrency")
        color = PALETTE[index % len(PALETTE)]
        label = f"rate {rate:g}/s, {int(slots)} slots"
        x = group["concurrency"].astype(float)
        throughput.plot(x, group["tasks_per_s"].astype(float), marker="o", color=color, label=label)
        p50 = group["task_latency_p50_s"].astype(float)
        p95 = group["task_latency_p95_s"].astype(float)
        latency.plot(x, p50, marker="o", color=color, label=f"{label} P50")
        if len(group) > 1:
            latency.fill_between(x.to_numpy(), p50.to_numpy(), p95.to_numpy(), color=color,
                                 alpha=0.14, label=f"{label} P50–P95")
        else:
            latency.scatter(x, p95, marker="^", color=color, s=45, label=f"{label} P95")

        deferred = group[group["server_requests_deferred_max"].fillna(0) > 0]
        if not deferred.empty:
            throughput.scatter(deferred["concurrency"], deferred["tasks_per_s"], marker="x",
                               color="black", s=70, linewidth=1.5,
                               label="server deferred" if not deferred_labeled else None)
            deferred_labeled = True

    throughput.set(xlabel="Scheduler request concurrency", ylabel="Completed throughput (tasks/s)")
    latency.set(xlabel="Scheduler request concurrency", ylabel="Task latency (s)")
    throughput.legend()
    latency.legend(ncol=2)
    finish_axes(axes, title=spec.get("title"))
    return save_figure(fig, output_base, formats)

