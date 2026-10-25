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
    ]
    validate_figure(frame, spec, required)
    data = frame.sort_values("tool_delay_s")
    x = data["tool_delay_s"].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharex=True)

    axes[0].plot(x, data["cached_ratio_followup"].astype(float) * 100, marker="o",
                 color=COLORS["blue"])
    axes[0].set(xlabel="Fixed tool delay per call (s)",
                ylabel="Follow-up prompt tokens cached (%)",
                title="Follow-up context reuse")
    axes[0].set_ylim(bottom=0)

    axes[1].plot(x, data["prefill_followup_p50_ms"].astype(float), marker="o",
                 color=COLORS["blue"])
    axes[1].set(xlabel="Fixed tool delay per call (s)",
                ylabel="Follow-up prefill P50 (ms)",
                title="Follow-up prefill time")

    finish_axes(axes, title=spec.get("title"))
    return save_figure(fig, output_base, formats)

