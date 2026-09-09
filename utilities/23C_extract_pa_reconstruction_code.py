
# Cell 23C — Extract the exact PA same-window FIA reconstruction code
#
# No experiment is run.
#
# This cell:
# 1. reads the audit outputs from Cell 23B;
# 2. ranks notebook/code members related to PA same-window FIA matching;
# 3. extracts only the top small code/notebook files from the TAR;
# 4. parses notebook code cells and saved outputs;
# 5. writes a readable report containing the exact candidate cells;
# 6. searches more flexibly for a previously printed PA matched-plot count.
#
# It does NOT extract the full TAR and does NOT modify the manuscript.

from pathlib import Path
from datetime import datetime, timezone
import json
import re
import tarfile
import hashlib

import pandas as pd
from tqdm.auto import tqdm
from IPython.display import display

WORKING = Path("/kaggle/working")
AUDIT_DIR = WORKING / "submission_critical_repair_audit"
TAR_PATH = WORKING / "20260711q1test.tar"

OUT_DIR = AUDIT_DIR / "pa_reconstruction_extraction"
EXTRACT_DIR = OUT_DIR / "extracted_candidates"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

CODE_MATCH_PATH = AUDIT_DIR / "pa_recovery_code_matches.csv"
TAR_CANDIDATE_PATH = AUDIT_DIR / "pa_recovery_tar_candidates.csv"

RUN_UTC = datetime.now(timezone.utc).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)

MAX_EXTRACT_FILES = 12
MAX_MEMBER_SIZE = 40 * 1024 * 1024

CORE_TERMS = [
    "n_matched_physical_plots",
    "matched_physical_plots",
    "physical_plot_id",
    "same_window",
    "same-window",
    "STATECD",
    "UNITCD",
    "COUNTYCD",
    "PLOT",
    "live-tree",
    "fia_same_window",
    "pa_species_external_validation",
    "Pennsylvania",
]

OUTPUT_COUNT_PATTERNS = [
    re.compile(
        r"n[_ ]?matched[_ ]?physical[_ ]?plots"
        r"\s*[:=]\s*([0-9][0-9,]*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:PA|Pennsylvania).{0,120}"
        r"(?:matched|same[-_ ]window).{0,120}"
        r"(?:plot|plots).{0,40}"
        r"[:=]?\s*([0-9][0-9,]*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:matched|same[-_ ]window).{0,100}"
        r"(?:physical).{0,60}"
        r"(?:plot|plots).{0,40}"
        r"[:=]?\s*([0-9][0-9,]*)",
        re.IGNORECASE,
    ),
]

print("RUN_UTC:", RUN_UTC)
print("TAR_PATH:", TAR_PATH)
print("OUT_DIR:", OUT_DIR)
print("EXPERIMENT_RUN: False")
print("GPU: not used")


# ============================================================
# Validation
# ============================================================
required_paths = [
    TAR_PATH,
    CODE_MATCH_PATH,
    TAR_CANDIDATE_PATH,
]

missing = [
    str(path)
    for path in required_paths
    if not path.exists()
]

if missing:
    raise FileNotFoundError(
        "Missing Cell 23B outputs or TAR: "
        + " | ".join(missing)
    )

code_matches = pd.read_csv(
    CODE_MATCH_PATH
)

tar_candidates = pd.read_csv(
    TAR_CANDIDATE_PATH
)


# ============================================================
# Rank candidate members
# ============================================================
def keyword_score(text):
    text_lower = str(text).lower()

    weights = {
        "n_matched_physical_plots": 20,
        "matched_physical_plots": 16,
        "physical_plot_id": 14,
        "fia_same_window": 14,
        "same_window": 12,
        "same-window": 12,
        "pa_species_external_validation": 12,
        "statecd": 6,
        "unitcd": 6,
        "countycd": 6,
        "live-tree": 5,
        "pennsylvania": 5,
        "/pa_": 4,
        "_pa_": 4,
    }

    return sum(
        weight
        for term, weight in weights.items()
        if term in text_lower
    )


match_aggregate = (
    code_matches.groupby(
        [
            "archive",
            "member",
        ],
        as_index=False,
    )
    .agg(
        matched_keywords=(
            "keyword",
            lambda values: "|".join(
                sorted(
                    set(
                        str(value)
                        for value in values
                    )
                )
            ),
        ),
        snippets=(
            "snippet",
            lambda values: " || ".join(
                str(value)
                for value in list(values)[:5]
            ),
        ),
        n_matches=(
            "keyword",
            "size",
        ),
    )
)

ranked = tar_candidates.copy()

ranked = ranked.merge(
    match_aggregate,
    how="left",
    left_on=[
        "archive",
        "member",
    ],
    right_on=[
        "archive",
        "member",
    ],
)

ranked[
    "n_matches"
] = ranked[
    "n_matches"
].fillna(0).astype(int)

ranked[
    "matched_keywords"
] = ranked[
    "matched_keywords"
].fillna("")

ranked[
    "snippets"
] = ranked[
    "snippets"
].fillna("")

ranked[
    "content_score"
] = ranked.apply(
    lambda row: keyword_score(
        str(row["member"])
        + " "
        + str(row["matched_keywords"])
        + " "
        + str(row["snippets"])
    ),
    axis=1,
)

ranked[
    "extension_bonus"
] = ranked[
    "member"
].map(
    lambda value: {
        ".ipynb": 8,
        ".py": 6,
        ".txt": 3,
        ".md": 2,
        ".json": 2,
    }.get(
        Path(str(value)).suffix.lower(),
        0,
    )
)

ranked[
    "total_score"
] = (
    ranked[
        "content_score"
    ]
    + ranked[
        "extension_bonus"
    ]
    + ranked[
        "name_score"
    ].fillna(0)
    + ranked[
        "n_matches"
    ]
)

ranked = ranked[
    ranked[
        "size_bytes"
    ]
    <= MAX_MEMBER_SIZE
].sort_values(
    [
        "total_score",
        "n_matches",
        "size_bytes",
    ],
    ascending=[
        False,
        False,
        True,
    ],
).reset_index(
    drop=True
)

ranked.to_csv(
    OUT_DIR / "ranked_pa_reconstruction_candidates.csv",
    index=False,
)

print("\nTOP RANKED CANDIDATES")
display(
    ranked.head(
        30
    )[
        [
            "member",
            "size_bytes",
            "matched_keywords",
            "n_matches",
            "total_score",
        ]
    ]
)


# ============================================================
# Select diverse top code/notebook files
# ============================================================
selected_rows = []
seen_members = set()

for _, row in ranked.iterrows():
    member = str(
        row[
            "member"
        ]
    )

    suffix = Path(
        member
    ).suffix.lower()

    if suffix not in {
        ".ipynb",
        ".py",
        ".txt",
        ".md",
        ".json",
    }:
        continue

    if member in seen_members:
        continue

    selected_rows.append(
        row
    )

    seen_members.add(
        member
    )

    if len(
        selected_rows
    ) >= MAX_EXTRACT_FILES:
        break

selected_df = pd.DataFrame(
    selected_rows
)

selected_df.to_csv(
    OUT_DIR / "selected_pa_code_candidates.csv",
    index=False,
)

if selected_df.empty:
    raise RuntimeError(
        "No extractable PA reconstruction code candidates found."
    )


# ============================================================
# Safe extraction and parsing
# ============================================================
def safe_output_path(member_name):
    member_path = Path(
        member_name
    )

    safe_name = "__".join(
        member_path.parts[-4:]
    )

    safe_name = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        safe_name,
    )

    return EXTRACT_DIR / safe_name


def output_to_text(output):
    if not isinstance(
        output,
        dict,
    ):
        return ""

    output_type = output.get(
        "output_type",
        ""
    )

    if output_type == "stream":
        text = output.get(
            "text",
            ""
        )

        if isinstance(
            text,
            list,
        ):
            return "".join(
                str(item)
                for item in text
            )

        return str(
            text
        )

    if output_type in {
        "execute_result",
        "display_data",
    }:
        data = output.get(
            "data",
            {}
        )

        parts = []

        for key in [
            "text/plain",
            "text/markdown",
            "text/html",
        ]:
            value = data.get(
                key
            )

            if value is None:
                continue

            if isinstance(
                value,
                list,
            ):
                value = "".join(
                    str(item)
                    for item in value
                )

            parts.append(
                str(
                    value
                )
            )

        return "\n".join(
            parts
        )

    if output_type == "error":
        return (
            str(
                output.get(
                    "ename",
                    ""
                )
            )
            + ": "
            + str(
                output.get(
                    "evalue",
                    ""
                )
            )
        )

    return ""


def cell_relevance(source, output_text):
    combined = (
        str(
            source
        )
        + "\n"
        + str(
            output_text
        )
    )

    return keyword_score(
        combined
    )


report_sections = []
cell_rows = []
count_candidate_rows = []
extracted_file_rows = []

with tarfile.open(
    TAR_PATH,
    "r",
) as archive:
    members_by_name = {
        member.name: member
        for member in archive.getmembers()
    }

    for _, row in tqdm(
        selected_df.iterrows(),
        total=len(
            selected_df
        ),
        desc="Extracting and parsing PA candidates",
        unit="file",
    ):
        member_name = str(
            row[
                "member"
            ]
        )

        member = members_by_name.get(
            member_name
        )

        if member is None:
            continue

        extracted = archive.extractfile(
            member
        )

        if extracted is None:
            continue

        raw_bytes = extracted.read()

        output_path = safe_output_path(
            member_name
        )

        output_path.write_bytes(
            raw_bytes
        )

        sha256 = hashlib.sha256(
            raw_bytes
        ).hexdigest()

        extracted_file_rows.append(
            {
                "member": member_name,
                "extracted_path": str(
                    output_path
                ),
                "size_bytes": len(
                    raw_bytes
                ),
                "sha256": sha256,
                "rank_score": float(
                    row[
                        "total_score"
                    ]
                ),
            }
        )

        text = raw_bytes.decode(
            "utf-8",
            errors="ignore",
        )

        suffix = Path(
            member_name
        ).suffix.lower()

        if suffix == ".ipynb":
            try:
                notebook = json.loads(
                    text
                )
            except Exception as exc:
                report_sections.append(
                    f"\nFILE: {member_name}\n"
                    f"NOTEBOOK_PARSE_ERROR: "
                    f"{type(exc).__name__}: {exc}\n"
                )
                continue

            relevant_cells = []

            cells = notebook.get(
                "cells",
                []
            )

            for cell_index, cell in enumerate(
                cells
            ):
                source = cell.get(
                    "source",
                    ""
                )

                if isinstance(
                    source,
                    list,
                ):
                    source = "".join(
                        str(item)
                        for item in source
                    )

                outputs = cell.get(
                    "outputs",
                    []
                )

                output_text = "\n".join(
                    output_to_text(
                        output
                    )
                    for output in outputs
                )

                relevance = cell_relevance(
                    source,
                    output_text,
                )

                if relevance <= 0:
                    continue

                relevant_cells.append(
                    (
                        relevance,
                        cell_index,
                        cell.get(
                            "cell_type",
                            ""
                        ),
                        source,
                        output_text,
                    )
                )

                cell_rows.append(
                    {
                        "member": member_name,
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
                        "source_preview": re.sub(
                            r"\s+",
                            " ",
                            source[:1000],
                        ),
                        "output_preview": re.sub(
                            r"\s+",
                            " ",
                            output_text[:1000],
                        ),
                    }
                )

                combined = (
                    source
                    + "\n"
                    + output_text
                )

                for pattern in OUTPUT_COUNT_PATTERNS:
                    for match in pattern.finditer(
                        combined
                    ):
                        count_candidate_rows.append(
                            {
                                "member": member_name,
                                "cell_index": int(
                                    cell_index
                                ),
                                "candidate_count": match.group(
                                    1
                                ),
                                "matched_text": re.sub(
                                    r"\s+",
                                    " ",
                                    combined[
                                        max(
                                            0,
                                            match.start()
                                            - 180,
                                        ):
                                        min(
                                            len(
                                                combined
                                            ),
                                            match.end()
                                            + 280,
                                        )
                                    ],
                                ),
                            }
                        )

            relevant_cells.sort(
                key=lambda item: (
                    -item[
                        0
                    ],
                    item[
                        1
                    ],
                )
            )

            section_lines = [
                "",
                "=" * 90,
                f"FILE: {member_name}",
                f"EXTRACTED_PATH: {output_path}",
                f"RANK_SCORE: {row['total_score']}",
                f"RELEVANT_CELLS: {len(relevant_cells)}",
                "=" * 90,
            ]

            for (
                relevance,
                cell_index,
                cell_type,
                source,
                output_text,
            ) in relevant_cells[:20]:
                section_lines.extend(
                    [
                        "",
                        "-" * 80,
                        (
                            f"CELL_INDEX: {cell_index} | "
                            f"TYPE: {cell_type} | "
                            f"RELEVANCE: {relevance}"
                        ),
                        "SOURCE:",
                        source,
                    ]
                )

                if output_text.strip():
                    section_lines.extend(
                        [
                            "SAVED_OUTPUT:",
                            output_text,
                        ]
                    )

            report_sections.append(
                "\n".join(
                    section_lines
                )
            )

        else:
            lower_text = text.lower()

            occurrence_positions = []

            for term in CORE_TERMS:
                start = 0

                while True:
                    index = lower_text.find(
                        term.lower(),
                        start,
                    )

                    if index < 0:
                        break

                    occurrence_positions.append(
                        (
                            index,
                            term,
                        )
                    )

                    start = (
                        index
                        + len(
                            term
                        )
                    )

            occurrence_positions = sorted(
                occurrence_positions
            )[:30]

            section_lines = [
                "",
                "=" * 90,
                f"FILE: {member_name}",
                f"EXTRACTED_PATH: {output_path}",
                f"RANK_SCORE: {row['total_score']}",
                f"MATCHES: {len(occurrence_positions)}",
                "=" * 90,
            ]

            for index, term in occurrence_positions:
                start = max(
                    0,
                    index - 700,
                )

                end = min(
                    len(
                        text
                    ),
                    index + 1800,
                )

                section_lines.extend(
                    [
                        "",
                        "-" * 80,
                        f"TERM: {term}",
                        text[
                            start:end
                        ],
                    ]
                )

            report_sections.append(
                "\n".join(
                    section_lines
                )
            )

            for pattern in OUTPUT_COUNT_PATTERNS:
                for match in pattern.finditer(
                    text
                ):
                    count_candidate_rows.append(
                        {
                            "member": member_name,
                            "cell_index": -1,
                            "candidate_count": match.group(
                                1
                            ),
                            "matched_text": re.sub(
                                r"\s+",
                                " ",
                                text[
                                    max(
                                        0,
                                        match.start()
                                        - 180,
                                    ):
                                    min(
                                        len(
                                            text
                                        ),
                                        match.end()
                                        + 280,
                                    )
                                ],
                            ),
                        }
                    )


# ============================================================
# Save reports
# ============================================================
extracted_file_df = pd.DataFrame(
    extracted_file_rows
)

cell_df = pd.DataFrame(
    cell_rows
)

count_df = pd.DataFrame(
    count_candidate_rows
)

extracted_file_df.to_csv(
    OUT_DIR / "extracted_candidate_manifest.csv",
    index=False,
)

cell_df.to_csv(
    OUT_DIR / "ranked_relevant_notebook_cells.csv",
    index=False,
)

count_df.to_csv(
    OUT_DIR / "pa_count_candidates_deep_scan.csv",
    index=False,
)

report_path = (
    OUT_DIR
    / "PA_RECONSTRUCTION_CANDIDATE_CELLS.txt"
)

report_path.write_text(
    "\n".join(
        report_sections
    ),
    encoding="utf-8",
)

print("\nEXTRACTED FILES")
display(
    extracted_file_df
)

print("\nTOP RELEVANT NOTEBOOK CELLS")
if not cell_df.empty:
    display(
        cell_df.sort_values(
            [
                "relevance_score",
                "member",
                "cell_index",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        ).head(
            50
        )
    )
else:
    print("No notebook cells parsed.")

print("\nDEEP COUNT CANDIDATES")
if not count_df.empty:
    display(
        count_df
    )
else:
    print("No directly printed PA count was found.")


# ============================================================
# Choose next step
# ============================================================
if not count_df.empty:
    status = (
        "PA_COUNT_CANDIDATE_FOUND_NEEDS_CODE_VERIFICATION"
    )

    next_step = (
        "Send the count-candidate rows and top relevant cell rows. "
        "Verify the exact filter before using the count."
    )

elif not cell_df.empty:
    status = (
        "PA_RECONSTRUCTION_CELLS_EXTRACTED"
    )

    next_step = (
        "Send the top relevant notebook-cell rows. "
        "A minimal PA-only reconstruction cell can now be written."
    )

else:
    status = (
        "PA_CODE_FILES_EXTRACTED_MANUAL_REVIEW_REQUIRED"
    )

    next_step = (
        "Open PA_RECONSTRUCTION_CANDIDATE_CELLS.txt and send the "
        "highest-scoring candidate section."
    )

summary = {
    "run_utc": RUN_UTC,
    "status": status,
    "selected_files": len(
        selected_df
    ),
    "extracted_files": len(
        extracted_file_df
    ),
    "relevant_notebook_cells": len(
        cell_df
    ),
    "count_candidates": len(
        count_df
    ),
    "report_path": str(
        report_path
    ),
    "next_step": next_step,
}

(
    OUT_DIR
    / "pa_reconstruction_extraction_summary.json"
).write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
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
    f"SELECTED_FILES: {len(selected_df)}"
)
print(
    f"EXTRACTED_FILES: {len(extracted_file_df)}"
)
print(
    f"RELEVANT_NOTEBOOK_CELLS: {len(cell_df)}"
)
print(
    f"COUNT_CANDIDATES: {len(count_df)}"
)
print(
    f"RANKED_CELLS_CSV: "
    f"{OUT_DIR / 'ranked_relevant_notebook_cells.csv'}"
)
print(
    f"COUNT_CANDIDATES_CSV: "
    f"{OUT_DIR / 'pa_count_candidates_deep_scan.csv'}"
)
print(
    f"CANDIDATE_REPORT: {report_path}"
)
print(
    f"NEXT_STEP: {next_step}"
)
print(
    "============================================"
)
