
# Cell 23D — Locate or reconstruct the PA same-window matched-plot count
#
# No scientific experiment is rerun.
#
# This cell:
# 1. scans notebook inputs for the exact PA FIA construction cell;
# 2. scans TAR/current files for PA-specific FIA plot-level tables;
# 3. extracts only highly relevant candidate files;
# 4. inspects schemas and, where justified, computes a candidate count of
#    physical plots represented in both early and late periods;
# 5. never guesses a count from unrelated narrative text.
#
# A count is marked "table-derived candidate" only when the source is
# explicitly PA-specific and contains plot identifiers plus a period/year
# field. It still must be checked against the exact notebook filtering code.

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import tarfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from IPython.display import display

WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")

AUDIT_DIR = WORKING / "submission_critical_repair_audit"
TAR_PATH = WORKING / "20260711q1test.tar"

OUT_DIR = AUDIT_DIR / "pa_exact_count_recovery"
EXTRACT_DIR = OUT_DIR / "extracted_pa_candidates"

OUT_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

EARLY_YEARS = set(range(2013, 2019))
LATE_YEARS = set(range(2020, 2026))

MAX_TEXT_SIZE = 50 * 1024 * 1024
MAX_DATA_EXTRACT_SIZE = 600 * 1024 * 1024
MAX_DATA_FILES_TO_INSPECT = 20

RUN_UTC = datetime.now(
    timezone.utc
).strftime("%Y-%m-%d %H:%M:%S UTC")

print("RUN_UTC:", RUN_UTC)
print("TAR_PATH:", TAR_PATH)
print("OUT_DIR:", OUT_DIR)
print("EXPERIMENT_RUN: False")
print("GPU: not used")


# ============================================================
# Search vocabulary
# ============================================================
CODE_TERMS = {
    "n_matched_physical_plots": 30,
    "matched live-tree physical plots": 30,
    "matched_physical_plots": 25,
    "physical_plot_id": 20,
    "STATECD": 7,
    "UNITCD": 7,
    "COUNTYCD": 7,
    "PLOT": 5,
    "same_window": 18,
    "same-window": 18,
    "live_tree": 12,
    "live-tree": 12,
    "fia_same_window": 20,
    "Pennsylvania": 12,
    "state_code = \"PA\"": 15,
    "state_code='PA'": 15,
    "\"PA\"": 2,
    "'PA'": 2,
}

DATA_PATH_TERMS = {
    "pa_species_external_validation": 30,
    "pennsylvania": 20,
    "/pa_": 15,
    "_pa_": 15,
    "same_window": 20,
    "same-window": 20,
    "physical_plot": 20,
    "plot_level": 18,
    "plot-level": 18,
    "fia_reference": 15,
    "fia_same_window": 25,
    "matched": 10,
    "live_tree": 10,
    "live-tree": 10,
}

TEXT_SUFFIXES = {
    ".ipynb",
    ".py",
    ".txt",
    ".md",
    ".json",
}

DATA_SUFFIXES = {
    ".parquet",
    ".csv",
}


# ============================================================
# Helpers
# ============================================================
def score_text(text, weights):
    lower = str(text).lower()
    return int(
        sum(
            weight
            for term, weight in weights.items()
            if term.lower() in lower
        )
    )


def output_to_text(output):
    if not isinstance(output, dict):
        return ""

    output_type = output.get("output_type", "")

    if output_type == "stream":
        value = output.get("text", "")
        if isinstance(value, list):
            return "".join(str(item) for item in value)
        return str(value)

    if output_type in {"execute_result", "display_data"}:
        data = output.get("data", {})
        parts = []

        for key in ["text/plain", "text/markdown", "text/html"]:
            value = data.get(key)

            if value is None:
                continue

            if isinstance(value, list):
                value = "".join(str(item) for item in value)

            parts.append(str(value))

        return "\n".join(parts)

    if output_type == "error":
        return (
            str(output.get("ename", ""))
            + ": "
            + str(output.get("evalue", ""))
        )

    return ""


def safe_name(original):
    path = Path(str(original))
    name = "__".join(path.parts[-5:])
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def normalize_columns(columns):
    return {
        str(column).strip().lower(): str(column)
        for column in columns
    }


def first_existing_column(columns, candidates):
    lookup = normalize_columns(columns)

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def build_plot_id(frame):
    direct = first_existing_column(
        frame.columns,
        [
            "physical_plot_id",
            "physicalplotid",
            "plot_id",
            "plotid",
        ],
    )

    if direct is not None:
        return frame[direct].astype(str)

    required = [
        "STATECD",
        "UNITCD",
        "COUNTYCD",
        "PLOT",
    ]

    lookup = normalize_columns(frame.columns)

    if all(column.lower() in lookup for column in required):
        actual = [
            lookup[column.lower()]
            for column in required
        ]

        return (
            frame[actual]
            .astype(str)
            .agg("_".join, axis=1)
        )

    return None


def derive_period(frame):
    period_column = first_existing_column(
        frame.columns,
        [
            "period",
            "time_period",
            "study_period",
            "window",
        ],
    )

    if period_column is not None:
        values = (
            frame[period_column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        mapped = values.map(
            {
                "early": "early",
                "late": "late",
                "2013-2018": "early",
                "2013–2018": "early",
                "2020-2025": "late",
                "2020–2025": "late",
            }
        )

        return mapped.where(
            mapped.notna(),
            values.where(
                values.isin(["early", "late"])
            ),
        )

    year_column = first_existing_column(
        frame.columns,
        [
            "year",
            "measurement_year",
            "measyear",
            "invyr",
        ],
    )

    if year_column is None:
        return None

    years = pd.to_numeric(
        frame[year_column],
        errors="coerce",
    )

    return pd.Series(
        np.where(
            years.isin(EARLY_YEARS),
            "early",
            np.where(
                years.isin(LATE_YEARS),
                "late",
                None,
            ),
        ),
        index=frame.index,
        dtype="object",
    )


def inspect_dataframe(
    frame,
    source,
):
    result = {
        "source": source,
        "rows": int(len(frame)),
        "columns": "|".join(
            str(column)
            for column in frame.columns
        ),
        "has_plot_id": False,
        "has_period": False,
        "early_unique_plots": np.nan,
        "late_unique_plots": np.nan,
        "matched_both_periods": np.nan,
        "count_status": "NO_COUNT",
    }

    plot_id = build_plot_id(frame)
    period = derive_period(frame)

    if plot_id is None:
        result["count_status"] = "NO_PLOT_IDENTIFIER"
        return result

    result["has_plot_id"] = True

    if period is None:
        result["count_status"] = "NO_PERIOD_OR_YEAR"
        return result

    result["has_period"] = True

    work = pd.DataFrame(
        {
            "plot_id": plot_id,
            "period": period,
        }
    ).dropna()

    work = work[
        work["period"].isin(
            [
                "early",
                "late",
            ]
        )
    ].copy()

    early = set(
        work.loc[
            work["period"] == "early",
            "plot_id",
        ]
    )

    late = set(
        work.loc[
            work["period"] == "late",
            "plot_id",
        ]
    )

    result[
        "early_unique_plots"
    ] = int(len(early))

    result[
        "late_unique_plots"
    ] = int(len(late))

    result[
        "matched_both_periods"
    ] = int(
        len(
            early
            & late
        )
    )

    result[
        "count_status"
    ] = (
        "TABLE_DERIVED_CANDIDATE"
        if len(
            early
            & late
        ) > 0
        else "ZERO_INTERSECTION"
    )

    return result


# ============================================================
# 1. Scan notebook input files and current notebooks
# ============================================================
notebook_files = []

for root in [
    INPUT / "notebooks",
    INPUT,
    WORKING,
]:
    if not root.exists():
        continue

    for path in root.rglob("*.ipynb"):
        if path.is_file():
            notebook_files.append(path)

notebook_files = sorted(
    set(notebook_files)
)

notebook_cell_rows = []
direct_count_rows = []

count_patterns = [
    re.compile(
        r"n[_ ]?matched[_ ]?physical[_ ]?plots"
        r"\s*[:=]\s*([0-9][0-9,]*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"matched live[- ]tree physical plots"
        r"\s*[:=]\s*([0-9][0-9,]*)",
        re.IGNORECASE,
    ),
]

for notebook_path in tqdm(
    notebook_files,
    desc="Scanning notebook inputs",
    unit="notebook",
):
    try:
        notebook = json.loads(
            notebook_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        )
    except Exception:
        continue

    path_score = score_text(
        notebook_path,
        DATA_PATH_TERMS,
    )

    for cell_index, cell in enumerate(
        notebook.get("cells", [])
    ):
        source = cell.get("source", "")

        if isinstance(source, list):
            source = "".join(
                str(item)
                for item in source
            )

        output_text = "\n".join(
            output_to_text(output)
            for output in cell.get(
                "outputs",
                [],
            )
        )

        combined = (
            str(notebook_path)
            + "\n"
            + source
            + "\n"
            + output_text
        )

        relevance = (
            path_score
            + score_text(
                combined,
                CODE_TERMS,
            )
        )

        if relevance <= 0:
            continue

        notebook_cell_rows.append(
            {
                "notebook": str(
                    notebook_path
                ),
                "cell_index": int(
                    cell_index
                ),
                "cell_type": cell.get(
                    "cell_type",
                    ""
                ),
                "relevance_score": int(
                    relevance
                ),
                "source": source,
                "saved_output": output_text,
            }
        )

        for pattern in count_patterns:
            for match in pattern.finditer(
                combined
            ):
                direct_count_rows.append(
                    {
                        "source": str(
                            notebook_path
                        ),
                        "cell_index": int(
                            cell_index
                        ),
                        "candidate_count": match.group(
                            1
                        ),
                        "context": re.sub(
                            r"\s+",
                            " ",
                            combined[
                                max(
                                    0,
                                    match.start()
                                    - 250,
                                ):
                                min(
                                    len(combined),
                                    match.end()
                                    + 400,
                                )
                            ],
                        ),
                    }
                )


# ============================================================
# 2. Rank current and TAR PA-specific data candidates
# ============================================================
filesystem_data_rows = []

for root in [
    INPUT,
    WORKING,
]:
    if not root.exists():
        continue

    for current_root, _, filenames in os.walk(
        root
    ):
        for filename in filenames:
            path = Path(current_root) / filename

            if path.suffix.lower() not in DATA_SUFFIXES:
                continue

            score = score_text(
                path,
                DATA_PATH_TERMS,
            )

            if score <= 0:
                continue

            try:
                size = path.stat().st_size
            except Exception:
                continue

            filesystem_data_rows.append(
                {
                    "location": "filesystem",
                    "source": str(path),
                    "size_bytes": int(size),
                    "score": int(score),
                }
            )

tar_data_rows = []

if not TAR_PATH.exists():
    raise FileNotFoundError(
        f"TAR not found: {TAR_PATH}"
    )

with tarfile.open(
    TAR_PATH,
    "r",
) as archive:
    members = archive.getmembers()

    for member in tqdm(
        members,
        desc="Ranking PA data members in TAR",
        unit="member",
    ):
        if not member.isfile():
            continue

        suffix = Path(
            member.name
        ).suffix.lower()

        if suffix not in DATA_SUFFIXES:
            continue

        score = score_text(
            member.name,
            DATA_PATH_TERMS,
        )

        if score <= 0:
            continue

        tar_data_rows.append(
            {
                "location": "tar",
                "source": member.name,
                "size_bytes": int(
                    member.size
                ),
                "score": int(score),
            }
        )

data_candidates = pd.DataFrame(
    filesystem_data_rows
    + tar_data_rows,
    columns=[
        "location",
        "source",
        "size_bytes",
        "score",
    ],
)

if not data_candidates.empty:
    data_candidates = (
        data_candidates.sort_values(
            [
                "score",
                "size_bytes",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .drop_duplicates(
            [
                "location",
                "source",
            ]
        )
        .reset_index(
            drop=True
        )
    )

data_candidates.to_csv(
    OUT_DIR
    / "ranked_pa_data_candidates.csv",
    index=False,
)

print("\nTOP PA DATA CANDIDATES")
display(
    data_candidates.head(50)
)


# ============================================================
# 3. Extract and inspect top data candidates
# ============================================================
inspection_rows = []
extracted_manifest_rows = []

selected_data = data_candidates[
    data_candidates[
        "size_bytes"
    ]
    <= MAX_DATA_EXTRACT_SIZE
].head(
    MAX_DATA_FILES_TO_INSPECT
)

tar_members_by_name = {}

with tarfile.open(
    TAR_PATH,
    "r",
) as archive:
    tar_members_by_name = {
        member.name: member
        for member in archive.getmembers()
    }

    for _, row in tqdm(
        selected_data.iterrows(),
        total=len(
            selected_data
        ),
        desc="Inspecting PA data candidates",
        unit="file",
    ):
        location = row[
            "location"
        ]

        source = str(
            row[
                "source"
            ]
        )

        suffix = Path(
            source
        ).suffix.lower()

        if location == "filesystem":
            local_path = Path(
                source
            )

        else:
            member = tar_members_by_name.get(
                source
            )

            if member is None:
                continue

            extracted = archive.extractfile(
                member
            )

            if extracted is None:
                continue

            raw = extracted.read()

            local_path = (
                EXTRACT_DIR
                / safe_name(
                    source
                )
            )

            local_path.write_bytes(
                raw
            )

            extracted_manifest_rows.append(
                {
                    "member": source,
                    "local_path": str(
                        local_path
                    ),
                    "size_bytes": len(
                        raw
                    ),
                    "sha256": hashlib.sha256(
                        raw
                    ).hexdigest(),
                }
            )

        try:
            if suffix == ".parquet":
                parquet_file = pq.ParquetFile(
                    local_path
                )

                columns = parquet_file.schema.names

                needed = []

                for candidate in [
                    "physical_plot_id",
                    "physicalplotid",
                    "plot_id",
                    "plotid",
                    "STATECD",
                    "UNITCD",
                    "COUNTYCD",
                    "PLOT",
                    "period",
                    "time_period",
                    "study_period",
                    "window",
                    "year",
                    "measurement_year",
                    "MEASYEAR",
                    "INVYR",
                ]:
                    actual = first_existing_column(
                        columns,
                        [
                            candidate
                        ],
                    )

                    if actual is not None:
                        needed.append(
                            actual
                        )

                needed = list(
                    dict.fromkeys(
                        needed
                    )
                )

                if not needed:
                    inspection_rows.append(
                        {
                            "source": source,
                            "rows": int(
                                parquet_file.metadata.num_rows
                            ),
                            "columns": "|".join(
                                columns
                            ),
                            "has_plot_id": False,
                            "has_period": False,
                            "early_unique_plots": np.nan,
                            "late_unique_plots": np.nan,
                            "matched_both_periods": np.nan,
                            "count_status": (
                                "NO_RELEVANT_COLUMNS"
                            ),
                        }
                    )
                    continue

                frame = pd.read_parquet(
                    local_path,
                    columns=needed,
                )

            else:
                frame = pd.read_csv(
                    local_path,
                    low_memory=False,
                )

            inspection_rows.append(
                inspect_dataframe(
                    frame,
                    source,
                )
            )

        except Exception as exc:
            inspection_rows.append(
                {
                    "source": source,
                    "rows": np.nan,
                    "columns": "",
                    "has_plot_id": False,
                    "has_period": False,
                    "early_unique_plots": np.nan,
                    "late_unique_plots": np.nan,
                    "matched_both_periods": np.nan,
                    "count_status": (
                        f"INSPECTION_ERROR:"
                        f"{type(exc).__name__}:"
                        f"{exc}"
                    ),
                }
            )

inspection_df = pd.DataFrame(
    inspection_rows
)

extracted_manifest_df = pd.DataFrame(
    extracted_manifest_rows
)

inspection_df.to_csv(
    OUT_DIR
    / "pa_data_schema_and_count_inspection.csv",
    index=False,
)

extracted_manifest_df.to_csv(
    OUT_DIR
    / "extracted_pa_data_manifest.csv",
    index=False,
)

print("\nPA DATA INSPECTION")
display(
    inspection_df
)


# ============================================================
# 4. Save notebook-cell report
# ============================================================
notebook_cell_df = pd.DataFrame(
    notebook_cell_rows
)

if not notebook_cell_df.empty:
    notebook_cell_df = (
        notebook_cell_df.sort_values(
            [
                "relevance_score",
                "notebook",
                "cell_index",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

notebook_cell_df.to_csv(
    OUT_DIR
    / "ranked_pa_notebook_cells.csv",
    index=False,
)

direct_count_df = pd.DataFrame(
    direct_count_rows
)

direct_count_df.to_csv(
    OUT_DIR
    / "direct_pa_count_candidates_from_notebooks.csv",
    index=False,
)

report_lines = []

for _, row in notebook_cell_df.head(
    30
).iterrows():
    report_lines.extend(
        [
            "=" * 100,
            (
                f"NOTEBOOK: {row['notebook']} | "
                f"CELL: {row['cell_index']} | "
                f"SCORE: {row['relevance_score']}"
            ),
            "SOURCE:",
            str(
                row[
                    "source"
                ]
            ),
            "SAVED OUTPUT:",
            str(
                row[
                    "saved_output"
                ]
            ),
            "",
        ]
    )

report_path = (
    OUT_DIR
    / "PA_TOP_NOTEBOOK_CELLS.txt"
)

report_path.write_text(
    "\n".join(
        report_lines
    ),
    encoding="utf-8",
)


# ============================================================
# 5. Decision
# ============================================================
valid_direct_counts = direct_count_df.copy()

table_count_candidates = inspection_df[
    inspection_df[
        "count_status"
    ]
    == "TABLE_DERIVED_CANDIDATE"
].copy()

if len(
    valid_direct_counts
) > 0:
    status = (
        "PA_DIRECT_NOTEBOOK_COUNT_FOUND_VERIFY_CONTEXT"
    )

    next_step = (
        "Send the direct notebook count rows and the matching source cell. "
        "Use the value only after confirming the locked same-window filters."
    )

elif len(
    table_count_candidates
) > 0:
    status = (
        "PA_PLOT_TABLE_COUNT_CANDIDATE_FOUND"
    )

    next_step = (
        "Send the table-derived candidate rows and the top notebook cells. "
        "Confirm that the table already applies live-tree and nearest-midpoint "
        "filters before using the count."
    )

elif len(
    notebook_cell_df
) > 0:
    status = (
        "PA_RECONSTRUCTION_CODE_LOCATED"
    )

    next_step = (
        "Send the top five notebook-cell rows. A minimal PA-only count cell "
        "can now be reconstructed from the exact original code."
    )

else:
    status = (
        "PA_EXACT_SOURCE_STILL_NOT_LOCATED"
    )

    next_step = (
        "Do not guess the count. Temporarily remove the matched-plot column "
        "from Table 1 and reacquire/rebuild PA FIA plot-level inputs."
    )

summary = {
    "run_utc": RUN_UTC,
    "status": status,
    "notebooks_scanned": len(
        notebook_files
    ),
    "relevant_notebook_cells": len(
        notebook_cell_df
    ),
    "direct_count_candidates": len(
        direct_count_df
    ),
    "data_candidates": len(
        data_candidates
    ),
    "data_files_inspected": len(
        inspection_df
    ),
    "table_count_candidates": len(
        table_count_candidates
    ),
    "next_step": next_step,
}

(
    OUT_DIR
    / "pa_exact_count_recovery_summary.json"
).write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print("\nPA EXACT COUNT RECOVERY SUMMARY")
print(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    )
)

if not direct_count_df.empty:
    print("\nDIRECT NOTEBOOK COUNT CANDIDATES")
    display(
        direct_count_df
    )

if not table_count_candidates.empty:
    print("\nTABLE-DERIVED COUNT CANDIDATES")
    display(
        table_count_candidates
    )

print("\nTOP NOTEBOOK CELLS")
display(
    notebook_cell_df.head(
        20
    )[
        [
            "notebook",
            "cell_index",
            "cell_type",
            "relevance_score",
            "source",
            "saved_output",
        ]
    ]
    if not notebook_cell_df.empty
    else pd.DataFrame()
)

print(
    "\n========== RUN SUMMARY =========="
)
print(
    f"CELL_STATUS: {status}"
)
print(
    f"OUT_DIR: {OUT_DIR}"
)
print(
    f"NOTEBOOKS_SCANNED: {len(notebook_files)}"
)
print(
    f"RELEVANT_NOTEBOOK_CELLS: {len(notebook_cell_df)}"
)
print(
    f"DIRECT_COUNT_CANDIDATES: {len(direct_count_df)}"
)
print(
    f"DATA_CANDIDATES: {len(data_candidates)}"
)
print(
    f"DATA_FILES_INSPECTED: {len(inspection_df)}"
)
print(
    f"TABLE_COUNT_CANDIDATES: {len(table_count_candidates)}"
)
print(
    f"NOTEBOOK_CELLS_CSV: "
    f"{OUT_DIR / 'ranked_pa_notebook_cells.csv'}"
)
print(
    f"DATA_INSPECTION_CSV: "
    f"{OUT_DIR / 'pa_data_schema_and_count_inspection.csv'}"
)
print(
    f"NOTEBOOK_REPORT: {report_path}"
)
print(
    f"NEXT_STEP: {next_step}"
)
print(
    "============================================"
)
