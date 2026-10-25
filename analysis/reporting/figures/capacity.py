"""FIFO offered-load, throughput, and latency figure."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd

from ..style import COLORS, finish_axes, save_figure
from ..validation import validate_figure


def render(frame: pd.DataFrame, spec: dict[str, Any], output_base: Path,
           formats: Iterable[str]) -> list[Path]:
    required = [
        "rate_per_s", "tasks_per_s", "task_latency_p50_s",
        "task_latency_p95_s", "queue_p95_s",
    ]
    validate_figure(frame, spec, required)
    data = frame.sort_values("rate_per_s")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    throughput, latency = axes

    x = data["rate_per_s"].astype(float)
    y = data["tasks_per_s"].astype(float)
    throughput.plot(x, y, marker="o", color=COLORS["blue"], label="achieved")
    limit = max(float(x.max()), float(y.max()))
    throughput.plot([0, limit], [0, limit], linestyle="--", color=COLORS["gray"], label="ideal y=x")
    throughput.set(xlabel="Offered task rate (tasks/s)", ylabel="Completed throughput (tasks/s)")
    throughput.legend()

    latency.plot(x, data["task_latency_p50_s"].astype(float), marker="o",
                 color=COLORS["blue"], label="task P50")
    latency.plot(x, data["task_latency_p95_s"].astype(float), marker="o",
                 color=COLORS["orange"], label="task P95")
    latency.plot(x, data["queue_p95_s"].astype(float), marker="s", linestyle="--",
                 color=COLORS["red"], label="request queue P95")
    latency.set(xlabel="Offered task rate (tasks/s)", ylabel="Latency (s)")
    latency.legend()

    finish_axes(axes, title=spec.get("title"))
    return save_figure(fig, output_base, formats)

