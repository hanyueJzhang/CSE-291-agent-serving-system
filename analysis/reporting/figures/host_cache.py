"""Host prompt-cache reuse and follow-up prefill figure."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd

from ..style import PALETTE, finish_axes, save_figure
from ..validation import validate_figure


def _cache_labels(values) -> list[str]:
    labels = []
    for value in values:
        if value == 0:
            labels.append("Off")
        elif value % 1024 == 0:
            labels.append(f"{value / 1024:g} GiB")
        else:
            labels.append(f"{value:g} MiB")
    return labels


def render(frame: pd.DataFrame, spec: dict[str, Any], output_base: Path,
           formats: Iterable[str]) -> list[Path]:
    required = [
        "cache_ram_mb", "rate_per_s", "cached_ratio_followup",
        "prefill_followup_p50_ms",
    ]
    validate_figure(frame, spec, required)
    data = frame.sort_values(["rate_per_s", "cache_ram_mb"])
    cache_sizes = sorted(data["cache_ram_mb"].unique())
    x_positions = list(range(len(cache_sizes)))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharex=True)
    for index, (rate, group) in enumerate(data.groupby("rate_per_s", sort=True)):
        group = group.set_index("cache_ram_mb").reindex(cache_sizes)
        color = PALETTE[index % len(PALETTE)]
        label = f"task arrival {rate:g}/s"
        axes[0].plot(x_positions, group["cached_ratio_followup"].astype(float) * 100,
                     marker="o", color=color, label=label)
        axes[1].plot(x_positions, group["prefill_followup_p50_ms"].astype(float),
                     marker="o", color=color, label=label)

    labels = _cache_labels(cache_sizes)
    for axis in axes:
        axis.set_xticks(x_positions, labels)
        axis.set_xlabel("Host prompt-cache cap")
        axis.legend(loc="center right")
    axes[0].set(ylabel="Follow-up prompt tokens cached (%)",
                title="Follow-up context reuse", ylim=(0, 100))
    axes[1].set(ylabel="Follow-up prefill P50 (ms)", title="Follow-up prefill time")

    finish_axes(axes, title=spec.get("title"))
    return save_figure(fig, output_base, formats)

