"""Avoidable re-prefill across load, host cache, and output scaling (three seeds).

    python -m analysis.reporting.reprefill_figure

Writes report/figures/phase1/reprefill.{png,pdf}. Bars are means over the
seeds available for each configuration, whiskers span min to max, and dots are
individual seeds. Seed 42 lives in mooncake_baseline / mooncake_out4; seeds 43
and 44 in phase1_repeats. W1 0.75X with the host cache was run with seed 42
only, so its bar has no whisker. Re-prefill comes from analysis.reprefill;
task P95 from each run's summary.json.
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

# group label -> {cache: (seed-42 folder, repeats run_id or None)}
GROUPS = {
    "W1\n0.50X": {"off": ("mooncake_baseline/w1_fifo_c4_l50", "w1_fifo_c4_l50")},
    "W1\n0.75X": {"off": ("mooncake_baseline/w1_fifo_c4_l75", "w1_fifo_c4_l75"),
                  "on": ("mooncake_baseline/w1_fifo_c4_l75_cram8g", None)},
    "W1\n0.95X": {"off": ("mooncake_baseline/w1_fifo_c4_l95", "w1_fifo_c4_l95"),
                  "on": ("mooncake_baseline/w1_fifo_c4_l95_cram8g", "w1_fifo_c4_l95_cram8g")},
    "W1, outputs /4\n550 tasks": {"off": ("mooncake_out4/w1out4_fifo_c4_n550", "w1out4_fifo_c4_n550"),
                                  "on": ("mooncake_out4/w1out4_fifo_c4_n550_cram8g", "w1out4_fifo_c4_n550_cram8g")},
}
SERIES = {"off": ("Host cache off", COLORS["gray"]), "on": ("Host cache 8 GiB", COLORS["blue"])}


def run_dirs(seed42: str, run_id: str | None) -> list[Path]:
    dirs = [RESULTS / seed42]
    if run_id:
        dirs += [REPEATS / f"{run_id}_s{seed}" for seed in (43, 44)]
    return [d for d in dirs if (d / "requests.csv").is_file()]


def metrics(run_dir: Path) -> dict[str, float]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return {
        "reprefill": reprefill(run_dir)["share_of_gpu_busy_pct"],
        "p95": summary["tasks"]["latency_s"]["p95"],
    }


def collect() -> dict[str, dict[str, list[dict[str, float]]]]:
    return {label: {cache: [metrics(d) for d in run_dirs(*src)] for cache, src in caches.items()}
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
