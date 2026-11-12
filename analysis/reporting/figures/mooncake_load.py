"""Mooncake load-scaling latency and prompt-reuse figure."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd

from ..style import COLORS, finish_axes, save_figure
from ..validation import validate_figure


def _range(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame.groupby("load_fraction", sort=True)[column].agg(["mean", "min", "max"])


def _validate_window_grid(frame: pd.DataFrame, expected_windows: list[str]) -> None:
    expected = set(expected_windows)
    for load, group in frame.groupby("load_fraction"):
        actual = set(group["trace_window"])
        if actual != expected:
            raise ValueError(
                f"mooncake_load: load {load:g} has windows {sorted(actual)!r}; "
                f"expected {sorted(expected)!r}"
            )


def render(
    frame: pd.DataFrame,
    spec: dict[str, Any],
    output_base: Path,
    formats: Iterable[str],
) -> list[Path]:
    required = [
        "load_fraction",
        "trace_window",
        "task_latency_p95_s",
        "cached_ratio_step0",
        "cached_ratio_followup",
    ]
    validate_figure(frame, spec, required)
    data = frame.sort_values(["load_fraction", "trace_window"]).copy()
    expected_windows = spec.get("expected_windows", ["W1", "W2", "W3"])
    _validate_window_grid(data, expected_windows)

    latency = _range(data, "task_latency_p95_s")
    step0 = _range(data, "cached_ratio_step0") * 100
    followup = _range(data, "cached_ratio_followup") * 100
    x = latency.index.to_numpy(dtype=float) * 100

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.1))

    axes[0].fill_between(
        x,
        latency["min"].to_numpy(dtype=float),
        latency["max"].to_numpy(dtype=float),
        color=COLORS["blue"],
        alpha=0.16,
        label="W1–W3 range",
    )
    axes[0].plot(
        x,
        latency["mean"].to_numpy(dtype=float),
        color=COLORS["blue"],
        marker="o",
        linewidth=2,
        label="Mean",
    )
    axes[0].set(
        title="Tail latency",
        xlabel="Offered load (% of calibrated capacity)",
        ylabel="Task completion latency P95 (s)",
        xticks=x,
    )
    axes[0].legend(loc="upper left")

    for stats, color, label in [
        (step0, COLORS["blue"], "Step 0 (shared prefix)"),
        (followup, COLORS["orange"], "Follow-up (same task)"),
    ]:
        axes[1].fill_between(
            x,
            stats["min"].to_numpy(dtype=float),
            stats["max"].to_numpy(dtype=float),
            color=color,
            alpha=0.12,
        )
        axes[1].plot(
            x,
            stats["mean"].to_numpy(dtype=float),
            color=color,
            marker="o",
            linewidth=2,
            label=label,
        )
    axes[1].set(
        title="Prompt-token reuse",
        xlabel="Offered load (% of calibrated capacity)",
        ylabel="Prompt tokens served from cache (%)",
        xticks=x,
        ylim=(0, 100),
    )
    axes[1].legend(loc="upper right")

    finish_axes(axes, title=spec.get("title"))
    return save_figure(fig, output_base, formats)
