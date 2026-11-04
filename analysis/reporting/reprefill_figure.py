"""Avoidable re-prefill across load, host cache, and output scaling (three seeds).

    python -m analysis.reporting.reprefill_figure

Writes report/figures/phase1/reprefill.{png,pdf}. Bars are means over seeds
42-44, whiskers span min to max, and dots are individual seeds. The W1 load
levels use the tuned FIFO concurrency of five (w1_c5_loads, w1_concurrency,
and cache_budget for the 8 GiB host cache). The output-scaled control was run
at concurrency four (mooncake_out4 and phase1_repeats). Re-prefill comes from
analysis.reprefill; task P95 from each run's summary.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

from ..reprefill import analyze as reprefill
from .style import COLORS, apply_style, save_figure

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
REPEATS = RESULTS / "phase1_repeats"
SEEDS = (43, 44)


def _seeds(seed42: str, others: str) -> list[str]:
    """seed42 folder plus <others>_s43 and _s44."""
    return [seed42] + [f"{others}_s{seed}" for seed in SEEDS]


# group label -> {cache: run folders relative to results/}
GROUPS = {
    "W1\n0.50X": {"off": _seeds("w1_c5_loads/w1_fifo_c5_l50", "w1_c5_loads/w1_fifo_c5_l50")},
    "W1\n0.75X": {"off": _seeds("w1_c5_loads/w1_fifo_c5_l75", "w1_c5_loads/w1_fifo_c5_l75")},
    "W1\n0.95X": {"off": _seeds("w1_concurrency/w1_fifo_c5_l95", "w1_concurrency/w1_fifo_c5_l95"),
                  "on": _seeds("cache_budget/w1_fifo_c5_l95_cram8g", "cache_budget/w1_fifo_c5_l95_cram8g")},
    "W1, outputs /4\n550 tasks (c=4)": {
        "off": _seeds("mooncake_out4/w1out4_fifo_c4_n550", "phase1_repeats/w1out4_fifo_c4_n550"),
        "on": _seeds("mooncake_out4/w1out4_fifo_c4_n550_cram8g", "phase1_repeats/w1out4_fifo_c4_n550_cram8g")},
}
SERIES = {"off": ("Host cache off", COLORS["gray"]), "on": ("Host cache 8 GiB", COLORS["blue"])}


def run_dirs(folders: list[str]) -> list[Path]:
    return [RESULTS / f for f in folders if (RESULTS / f / "requests.csv").is_file()]


def metrics(run_dir: Path) -> dict[str, float]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return {
        "reprefill": reprefill(run_dir)["share_of_gpu_busy_pct"],
        "p95": summary["tasks"]["latency_s"]["p95"],
    }


def collect() -> dict[str, dict[str, list[dict[str, float]]]]:
    return {label: {cache: [metrics(d) for d in run_dirs(folders)] for cache, folders in caches.items()}
            for label, caches in GROUPS.items()}


def _bars(axis, data, key: str, ylabel: str, title: str, label_fmt: str) -> None:
    width = 0.36
    for g, (label, caches) in enumerate(data.items()):
        for s, cache in enumerate(("off", "on")):
            runs = caches.get(cache)
            if not runs:
                continue
            values = [r[key] for r in runs]
            mean = statistics.mean(values)
            x = g + (s - 0.5) * width
            name, color = SERIES[cache]
            axis.bar(x, mean, width * 0.92, color=color, label=name if g == 2 else None, zorder=2)
            if len(values) > 1:
                axis.errorbar(x, mean, yerr=[[mean - min(values)], [max(values) - mean]],
                              fmt="none", ecolor="black", elinewidth=1, capsize=3, zorder=3)
            axis.scatter([x] * len(values), values, s=9, color="black", zorder=4)
            top = max(values)
            axis.annotate(label_fmt.format(mean) + ("" if len(values) > 1 else "\n(n=1)"),
                          (x, top), textcoords="offset points", xytext=(0, 3),
                          ha="center", va="bottom", fontsize=7)
    axis.set_xticks(range(len(data)), list(data))
    axis.set(ylabel=ylabel, title=title)
    axis.set_ylim(0, axis.get_ylim()[1] * 1.15)
    axis.grid(True, axis="y", alpha=0.25)
    axis.grid(False, axis="x")


def render(output_dir: Path, formats=("png", "pdf")) -> list[Path]:
    apply_style()
    data = collect()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    _bars(axes[0], data, "reprefill", "Avoidable re-prefill (% of GPU-busy time)",
          "Re-prefill of displaced context", "{:.1f}%")
    _bars(axes[1], data, "p95", "Task P95 (s)", "Task tail latency", "{:.1f}")
    axes[0].legend(loc="upper left", frameon=False)
    return save_figure(fig, output_dir / "reprefill", formats)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "report/figures/phase1")
    args = parser.parse_args(argv)
    for path in render(args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
