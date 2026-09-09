"""Availability and support-continuity diagnostics for two-period data."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Iterable

from .metrics import SupportSummary, summarize_support


@dataclass(frozen=True)
class CellPeriodCount:
    """Aggregated focal and effort counts for one cell-period combination."""

    period: str
    cell_id: str
    effort_count: float
    focal_count: float

    def __post_init__(self) -> None:
        if not str(self.cell_id).strip():
            raise ValueError("cell_id must not be empty")
        for name, value in (
            ("effort_count", self.effort_count),
            ("focal_count", self.focal_count),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative finite number")


@dataclass(frozen=True)
class DiagnosticResult:
    """Support summaries and estimator-availability status for one group."""

    group: str
    effort_support: SupportSummary
    species_support: SupportSummary
    stable_cell_count: int
    raw_estimable: bool
    corrected_estimable: bool
    early_relative_rate_sum: float
    late_relative_rate_sum: float
    availability_reason: str
    minimum_effort_per_cell_period: float
    minimum_stable_cells: int

    def to_dict(self) -> dict[str, str | int | float | bool]:
        result: dict[str, str | int | float | bool] = {
            "group": self.group,
            "stable_cell_count": self.stable_cell_count,
            "raw_estimable": self.raw_estimable,
            "corrected_estimable": self.corrected_estimable,
            "early_relative_rate_sum": self.early_relative_rate_sum,
            "late_relative_rate_sum": self.late_relative_rate_sum,
            "availability_reason": self.availability_reason,
            "minimum_effort_per_cell_period": self.minimum_effort_per_cell_period,
            "minimum_stable_cells": self.minimum_stable_cells,
        }
        result.update(
            {f"effort_{key}": value for key, value in asdict(self.effort_support).items()}
        )
        result.update(
            {
                f"species_{key}": value
                for key, value in asdict(self.species_support).items()
            }
        )
        return result


def diagnose_support(
    rows: Iterable[CellPeriodCount],
    *,
    group: str = "all",
    early_label: str = "early",
    late_label: str = "late",
    minimum_effort_per_cell_period: float = 10.0,
    minimum_stable_cells: int = 3,
) -> DiagnosticResult:
    """Diagnose support overlap and correction availability.

    Duplicate cell-period rows are summed. Effort-frame support contains cells
    with positive effort. Species support contains cells with positive focal
    counts. Stable cells meet the supplied effort minimum in both periods.
    No Jaccard cutoff is applied or recommended.
    """

    if early_label == late_label:
        raise ValueError("early_label and late_label must differ")
    if minimum_effort_per_cell_period <= 0 or not math.isfinite(
        minimum_effort_per_cell_period
    ):
        raise ValueError("minimum_effort_per_cell_period must be positive")
    if minimum_stable_cells < 1:
        raise ValueError("minimum_stable_cells must be at least 1")

    counts: dict[str, dict[str, list[float]]] = {
        early_label: defaultdict(lambda: [0.0, 0.0]),
        late_label: defaultdict(lambda: [0.0, 0.0]),
    }
    for row in rows:
        if row.period not in counts:
            raise ValueError(
                f"unexpected period {row.period!r}; expected "
                f"{early_label!r} or {late_label!r}"
            )
        aggregate = counts[row.period][str(row.cell_id)]
        aggregate[0] += float(row.effort_count)
        aggregate[1] += float(row.focal_count)

    early = counts[early_label]
    late = counts[late_label]
    effort_early = {cell for cell, values in early.items() if values[0] > 0}
    effort_late = {cell for cell, values in late.items() if values[0] > 0}
    species_early = {cell for cell, values in early.items() if values[1] > 0}
    species_late = {cell for cell, values in late.items() if values[1] > 0}

    all_cells = set(early) | set(late)
    stable_cells = {
        cell
        for cell in all_cells
        if early.get(cell, [0.0, 0.0])[0] >= minimum_effort_per_cell_period
        and late.get(cell, [0.0, 0.0])[0] >= minimum_effort_per_cell_period
    }

    early_focal_total = sum(values[1] for values in early.values())
    late_focal_total = sum(values[1] for values in late.values())
    raw_estimable = early_focal_total > 0 and late_focal_total > 0

    early_rate_sum = sum(early[cell][1] / early[cell][0] for cell in stable_cells)
    late_rate_sum = sum(late[cell][1] / late[cell][0] for cell in stable_cells)

    reasons: list[str] = []
    if early_focal_total <= 0:
        reasons.append("no_early_focal_records")
    if late_focal_total <= 0:
        reasons.append("no_late_focal_records")
    if len(stable_cells) < minimum_stable_cells:
        reasons.append("insufficient_stable_effort_cells")
    if stable_cells and early_rate_sum <= 0:
        reasons.append("no_early_focal_rate_on_stable_cells")
    if stable_cells and late_rate_sum <= 0:
        reasons.append("no_late_focal_rate_on_stable_cells")

    corrected_estimable = (
        raw_estimable
        and len(stable_cells) >= minimum_stable_cells
        and early_rate_sum > 0
        and late_rate_sum > 0
    )
    reason = "available" if corrected_estimable else ";".join(reasons)

    return DiagnosticResult(
        group=str(group),
        effort_support=summarize_support(effort_early, effort_late),
        species_support=summarize_support(species_early, species_late),
        stable_cell_count=len(stable_cells),
        raw_estimable=raw_estimable,
        corrected_estimable=corrected_estimable,
        early_relative_rate_sum=early_rate_sum,
        late_relative_rate_sum=late_rate_sum,
        availability_reason=reason,
        minimum_effort_per_cell_period=float(minimum_effort_per_cell_period),
        minimum_stable_cells=int(minimum_stable_cells),
    )
