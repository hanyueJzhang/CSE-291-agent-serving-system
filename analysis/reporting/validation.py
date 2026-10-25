"""Validation helpers for scientifically comparable report figures."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd


def require_columns(frame: pd.DataFrame, columns: Iterable[str], *, context: str = "data") -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{context}: missing required columns: {', '.join(missing)}")
    empty = [column for column in columns if frame[column].isna().all()]
    if empty:
        raise ValueError(f"{context}: required columns contain no values: {', '.join(empty)}")


def require_constant(frame: pd.DataFrame, columns: Iterable[str], *, context: str = "comparison") -> None:
    """Require each control column to have exactly one value, including null."""
    require_columns(frame, columns, context=context)
    mismatches: dict[str, list[Any]] = {}
    for column in columns:
        values = frame[column].drop_duplicates().tolist()
        if len(values) != 1:
            mismatches[column] = values
    if mismatches:
        detail = "; ".join(f"{key}={values!r}" for key, values in mismatches.items())
        raise ValueError(f"{context}: controlled variables differ: {detail}")


def select_runs(frame: pd.DataFrame, filters: Mapping[str, Any], *, context: str = "selection") -> pd.DataFrame:
    """Filter by equality or membership while preserving input order."""
    require_columns(frame, filters, context=context)
    selected = frame
    for column, expected in filters.items():
        if isinstance(expected, list):
            selected = selected[selected[column].isin(expected)]
        else:
            selected = selected[selected[column] == expected]
    if selected.empty:
        raise ValueError(f"{context}: no runs matched {dict(filters)!r}")
    return selected.copy()


def validate_figure(frame: pd.DataFrame, spec: Mapping[str, Any], required: Iterable[str]) -> None:
    figure_id = str(spec.get("id") or spec.get("type") or "figure")
    require_columns(frame, required, context=figure_id)
    require_constant(frame, spec.get("require_constant", []), context=figure_id)

