"""Tool-wait duration, cache survival, and follow-up cost figure."""

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
        "tool_delay_s", "cached_ratio_followup", "prefill_followup_p50_ms",
        "task_latency_excl_tool_p50_s", "task_latency_excl_tool_p95_s",
        "idle_slot_tokens_mean",
    ]
    validate_figure(frame, spec, required)
    data = frame.sort_values("tool_delay_s")
    x = data["tool_delay_s"].astype(float)

    fig, axes_grid = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
    axes = axes_grid.ravel()

    axes[0].plot(x, data["cached_ratio_followup"].astype(float) * 100, marker="o",
                 color=COLORS["green"])
    axes[0].set(ylabel="Follow-up prompt cached (%)", title="Context reuse")
    axes[0].set_ylim(bottom=0)

    axes[1].plot(x, data["prefill_followup_p50_ms"].astype(float), marker="o",
                 color=COLORS["orange"])
    axes[1].set(ylabel="Follow-up prefill P50 (ms)", title="Recomputation cost")

    axes[2].plot(x, data["task_latency_excl_tool_p50_s"].astype(float), marker="o",
                 color=COLORS["blue"], label="P50")
    axes[2].plot(x, data["task_latency_excl_tool_p95_s"].astype(float), marker="o",
                 color=COLORS["red"], label="P95")
    axes[2].set(xlabel="Fixed tool delay (s)", ylabel="Task latency excluding tools (s)",
                title="Serving cost after removing tool sleep")
    axes[2].legend()

    axes[3].plot(x, data["idle_slot_tokens_mean"].astype(float), marker="o",
                 color=COLORS["purple"])
    axes[3].set(xlabel="Fixed tool delay (s)", ylabel="Estimated mean idle slot tokens",
                title="Retained idle KV estimate")

    finish_axes(axes, title=spec.get("title"))
    return save_figure(fig, output_base, formats)

