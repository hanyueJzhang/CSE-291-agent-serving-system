"""Generate a complete report figure set from a JSON specification.

Usage:
    python -m analysis.reporting.cli analysis/report_specs/phase1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .figures import RENDERERS
from .loader import REPO_ROOT, load_runs, run_frame, step_frame
from .style import apply_style


def _read_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or spec.get("version") != 1:
        raise ValueError(f"{path}: expected report specification version 1")
    if not isinstance(spec.get("figures"), list):
        raise ValueError(f"{path}: 'figures' must be a list")
    return spec


def generate(spec_path: str | Path, *, output_dir: str | Path | None = None) -> list[Path]:
    spec_path = Path(spec_path)
    if not spec_path.is_absolute():
        spec_path = REPO_ROOT / spec_path
    report = _read_spec(spec_path)
    manifest = report.get("manifest")
    formats = report.get("formats", ["png"])
    destination = Path(output_dir or report.get("output_dir", "report/figures"))
    if not destination.is_absolute():
        destination = REPO_ROOT / destination

    apply_style()
    written: list[Path] = []
    for figure in report["figures"]:
        figure_id = figure.get("id")
        figure_type = figure.get("type")
        if not figure_id or figure_type not in RENDERERS:
            raise ValueError(f"{spec_path}: invalid figure entry {figure!r}")
        paths = figure.get("runs")
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"{figure_id}: 'runs' must be a non-empty list")
        runs = load_runs(paths, manifest_path=manifest, root=REPO_ROOT)
        # Both normalized tables are produced here. Comparative Phase 1 plots use
        # run-level data today; future per-step renderers can consume steps.
        runs_df = run_frame(runs)
        _ = step_frame(runs)
        outputs = RENDERERS[figure_type](runs_df, figure, destination / figure_id, formats)
        written.extend(outputs)
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", help="JSON report specification")
    parser.add_argument("--output-dir", help="override the specification output directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for path in generate(args.spec, output_dir=args.output_dir):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

