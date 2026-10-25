"""Shared report figure styling and output helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "blue": "#4C78A8",
    "orange": "#F58518",
    "red": "#E45756",
    "teal": "#72B7B2",
    "green": "#54A24B",
    "purple": "#B279A2",
    "gray": "#79706E",
}

PALETTE = [COLORS[key] for key in ("blue", "orange", "green", "red", "purple", "teal", "gray")]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 180,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def finish_axes(axes: Iterable, *, title: str | None = None) -> None:
    axes = list(axes)
    for axis in axes:
        axis.grid(True, alpha=0.25)
    if title and axes:
        axes[0].figure.suptitle(title, fontsize=13)


def save_figure(fig, output_base: Path, formats: Iterable[str]) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    written = []
    for extension in formats:
        path = output_base.with_suffix(f".{extension}")
        fig.savefig(path, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written

