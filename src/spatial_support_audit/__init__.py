"""Spatial-support continuity diagnostics."""

from .diagnostics import (
    CellPeriodCount,
    DiagnosticResult,
    diagnose_support,
)
from .metrics import (
    SupportSummary,
    grid_cell_id,
    jaccard_similarity,
    overlap_min_fraction,
    summarize_support,
)

__all__ = [
    "CellPeriodCount",
    "DiagnosticResult",
    "SupportSummary",
    "diagnose_support",
    "grid_cell_id",
    "jaccard_similarity",
    "overlap_min_fraction",
    "summarize_support",
]

__version__ = "0.1.0"
