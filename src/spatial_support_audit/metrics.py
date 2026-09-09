"""Set-based spatial-support metrics used by the manuscript workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Hashable, Iterable


@dataclass(frozen=True)
class SupportSummary:
    """Counts and overlap metrics for early and late support sets."""

    early_cells: int
    late_cells: int
    intersection_cells: int
    union_cells: int
    jaccard: float
    overlap_min_fraction: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _as_set(values: Iterable[Hashable]) -> set[Hashable]:
    return set(values)


def jaccard_similarity(
    early: Iterable[Hashable], late: Iterable[Hashable]
) -> float:
    """Return intersection over union, or NaN when both supports are empty."""

    early_set = _as_set(early)
    late_set = _as_set(late)
    union = early_set | late_set
    if not union:
        return math.nan
    return len(early_set & late_set) / len(union)


def overlap_min_fraction(
    early: Iterable[Hashable], late: Iterable[Hashable]
) -> float:
    """Return intersection divided by the smaller support size.

    The result is NaN when either support is empty, matching the manuscript
    pipeline's convention.
    """

    early_set = _as_set(early)
    late_set = _as_set(late)
    denominator = min(len(early_set), len(late_set))
    if denominator == 0:
        return math.nan
    return len(early_set & late_set) / denominator


def summarize_support(
    early: Iterable[Hashable], late: Iterable[Hashable]
) -> SupportSummary:
    """Summarize temporal overlap between two cell-support sets."""

    early_set = _as_set(early)
    late_set = _as_set(late)
    intersection = early_set & late_set
    union = early_set | late_set
    return SupportSummary(
        early_cells=len(early_set),
        late_cells=len(late_set),
        intersection_cells=len(intersection),
        union_cells=len(union),
        jaccard=jaccard_similarity(early_set, late_set),
        overlap_min_fraction=overlap_min_fraction(early_set, late_set),
    )


def grid_cell_id(latitude: float, longitude: float, resolution: float = 0.5) -> str:
    """Assign a coordinate to the manuscript's floor-indexed grid cell."""

    if not math.isfinite(latitude) or not math.isfinite(longitude):
        raise ValueError("latitude and longitude must be finite")
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be a positive finite number")
    lat_index = math.floor(latitude / resolution)
    lon_index = math.floor(longitude / resolution)
    return f"{lat_index}_{lon_index}"
