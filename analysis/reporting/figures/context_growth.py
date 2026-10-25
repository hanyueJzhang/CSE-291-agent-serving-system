"""Context size, occupied KV working set, and performance figure."""

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
        "initial_prompt_tokens", "slot_occupancy_ratio", "prefill_followup_p50_ms",
        "tasks_per_s", "task_latency_p50_s", "task_latency_p95_s",
    ]
    validate_figure(frame, spec, required)
    data = frame.sort_values("initial_prompt_tokens")
    x = data["initial_prompt_tokens"].astype(float)

    fig, axes_grid = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
    axes = axes_grid.ravel()

    axes[0].plot(x, data["slot_occupancy_ratio"].astype(float) * 100, marker="o",
                 color=COLORS["purple"])
    axes[0].set(ylabel="Peak occupied slot capacity (%)", title="KV working-set pressure")
    axes[0].set_ylim(bottom=0)

    axes[1].plot(x, data["prefill_followup_p50_ms"].astype(float), marker="o",
                 color=COLORS["orange"])
    axes[1].set(ylabel="Follow-up prefill P50 (ms)", title="Prompt recomputation cost")

    axes[2].plot(x, data["tasks_per_s"].astype(float), marker="o", color=COLORS["green"])
    axes[2].set(xlabel="Initial prompt size (tokens)", ylabel="Completed throughput (tasks/s)",
                title="Completed agent work")

    axes[3].plot(x, data["task_latency_p50_s"].astype(float), marker="o",
                 color=COLORS["blue"], label="P50")
    axes[3].plot(x, data["task_latency_p95_s"].astype(float), marker="o",
                 color=COLORS["red"], label="P95")
    axes[3].set(xlabel="Initial prompt size (tokens)", ylabel="Task latency (s)",
                title="End-to-end completion latency")
    axes[3].legend()

    finish_axes(axes, title=spec.get("title"))
    return save_figure(fig, output_base, formats)

