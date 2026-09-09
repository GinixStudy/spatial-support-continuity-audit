
# Cell 23B — Submission-critical repair audit
#
# No experiment is run.
#
# Tasks:
# 1. Capture exact software versions from the current environment that
#    successfully reproduced Cell 22F.
# 2. Search current Kaggle files and TAR archives for the original PA
#    same-window FIA matching code, outputs, or plot-level tables.
# 3. Produce a compact recovery report without extracting large raw files.
#
# The cell does not modify manuscript files.

from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
import platform
import re
import subprocess
import sys
import tarfile

import pandas as pd
from tqdm.auto import tqdm
from IPython.display import display

WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
OUT_DIR = WORKING / "submission_critical_repair_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_UTC = datetime.now(timezone.utc).strftime(
    "%Y-%m-%d %H:%M:%S UTC"
)

print("RUN_UTC:", RUN_UTC)
print("OUT_DIR:", OUT_DIR)
print("GPU: not used")
print("EXPERIMENT_RUN: False")


# ============================================================
# 1. Capture the successfully reproduced software environment
# ============================================================
packages = [
    "pandas",
    "numpy",
    "scipy",
    "pyarrow",
    "matplotlib",
    "tqdm",
    "statsmodels",
    "scikit-learn",
]

version_rows = [
    {
        "package": "python",
        "version": platform.python_version(),
    }
]

for package in packages:
    try:
        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        version = "NOT_INSTALLED"

    version_rows.append(
        {
            "package": package,
            "version": version,
        }
    )

version_df = pd.DataFrame(version_rows)

version_df.to_csv(
    OUT_DIR / "software_versions_core.csv",
    index=False,
)

core_requirements = "\n".join(
    f"{row['package']}=={row['version']}"
    for row in version_rows
    if row["package"] != "python"
    and row["version"] != "NOT_INSTALLED"
)

(OUT_DIR / "requirements_core.txt").write_text(
    core_requirements + "\n",
    encoding="utf-8",
)

environment_info = {
    "captured_utc": RUN_UTC,
    "python": platform.python_version(),
    "platform": platform.platform(),
    "executable": sys.executable,
    "kaggle_cpu_environment": True,
    "gpu_used": False,
    "note": (
        "Captured after successful exact rerun of the frozen "
        "600,000-replicate confirmation."
    ),
}

(OUT_DIR / "environment_info.json").write_text(
    json.dumps(
        environment_info,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

freeze_result = subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "freeze",
    ],
    capture_output=True,
    text=True,
    check=True,
)

(OUT_DIR / "requirements_full.txt").write_text(
    freeze_result.stdout,
    encoding="utf-8",
)

print("\nSOFTWARE VERSIONS")
display(version_df)


# ============================================================
# 2. Locate candidate PA/FIA recovery materials
# ============================================================
name_keywords = [
    "pa",
    "pennsylvania",
    "fia",
    "same_window",
    "same-window",
    "matched",
    "physical_plot",
    "physical-plot",
    "plot_reference",
    "species_external_validation",
]

content_keywords = [
    "n_matched_physical_plots",
    "matched_physical_plots",
    "same_window",
    "same-window",
    "STATECD",
    "UNITCD",
    "COUNTYCD",
    "PLOT",
    "live-tree",
    "physical_plot_id",
]

interesting_suffixes = {
    ".ipynb",
    ".py",
    ".txt",
    ".md",
    ".json",
    ".csv",
    ".parquet",
}

all_files = []

for root in [WORKING, INPUT]:
    if not root.exists():
        continue

    for current_root, _, filenames in os.walk(root):
        for filename in filenames:
            path = Path(current_root) / filename
            if path.suffix.lower() in interesting_suffixes:
                all_files.append(path)

candidate_rows = []

for path in tqdm(
    all_files,
    desc="Scanning current Kaggle files",
    unit="file",
):
    path_lower = str(path).lower()

    name_score = sum(
        keyword in path_lower
        for keyword in name_keywords
    )

    if name_score == 0:
        continue

    try:
        size_bytes = path.stat().st_size
    except Exception:
        continue

    candidate_rows.append(
        {
            "source": "filesystem",
            "path": str(path),
            "size_bytes": int(size_bytes),
            "name_score": int(name_score),
        }
    )


# ============================================================
# 3. Inspect TAR member names and small text/notebook members
# ============================================================
tar_paths = [
    path
    for path in WORKING.glob("*.tar")
    if path.is_file()
]

tar_member_rows = []
text_match_rows = []

for tar_path in tqdm(
    tar_paths,
    desc="Inspecting TAR archives",
    unit="archive",
):
    with tarfile.open(tar_path, "r") as archive:
        members = archive.getmembers()

        for member in tqdm(
            members,
            desc=f"Indexing {tar_path.name}",
            unit="member",
            leave=False,
        ):
            if not member.isfile():
                continue

            member_lower = member.name.lower()

            name_score = sum(
                keyword in member_lower
                for keyword in name_keywords
            )

            if name_score > 0:
                tar_member_rows.append(
                    {
                        "archive": str(tar_path),
                        "member": member.name,
                        "size_bytes": int(member.size),
                        "name_score": int(name_score),
                    }
                )

            suffix = Path(member.name).suffix.lower()

            if (
                suffix not in {".ipynb", ".py", ".txt", ".md", ".json"}
                or member.size > 30 * 1024 * 1024
            ):
                continue

            extracted = archive.extractfile(member)

            if extracted is None:
                continue

            try:
                text = extracted.read().decode(
                    "utf-8",
                    errors="ignore",
                )
            except Exception:
                continue

            matched_keywords = [
                keyword
                for keyword in content_keywords
                if keyword.lower() in text.lower()
            ]

            if not matched_keywords:
                continue

            lower_text = text.lower()

            for keyword in matched_keywords:
                index = lower_text.find(
                    keyword.lower()
                )

                start = max(
                    0,
                    index - 250,
                )

                end = min(
                    len(text),
                    index + 600,
                )

                snippet = re.sub(
                    r"\s+",
                    " ",
                    text[start:end],
                )

                text_match_rows.append(
                    {
                        "archive": str(tar_path),
                        "member": member.name,
                        "keyword": keyword,
                        "snippet": snippet,
                    }
                )


# ============================================================
# 4. Inspect current small text/notebook files
# ============================================================
for path in tqdm(
    all_files,
    desc="Searching current code and notebook content",
    unit="file",
):
    if (
        path.suffix.lower()
        not in {".ipynb", ".py", ".txt", ".md", ".json"}
    ):
        continue

    try:
        if path.stat().st_size > 30 * 1024 * 1024:
            continue

        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        continue

    lower_text = text.lower()

    matched_keywords = [
        keyword
        for keyword in content_keywords
        if keyword.lower() in lower_text
    ]

    for keyword in matched_keywords:
        index = lower_text.find(
            keyword.lower()
        )

        start = max(
            0,
            index - 250,
        )

        end = min(
            len(text),
            index + 600,
        )

        snippet = re.sub(
            r"\s+",
            " ",
            text[start:end],
        )

        text_match_rows.append(
            {
                "archive": "",
                "member": str(path),
                "keyword": keyword,
                "snippet": snippet,
            }
        )


candidate_df = pd.DataFrame(
    candidate_rows,
    columns=[
        "source",
        "path",
        "size_bytes",
        "name_score",
    ],
)

tar_member_df = pd.DataFrame(
    tar_member_rows,
    columns=[
        "archive",
        "member",
        "size_bytes",
        "name_score",
    ],
)

text_match_df = pd.DataFrame(
    text_match_rows,
    columns=[
        "archive",
        "member",
        "keyword",
        "snippet",
    ],
)

if not candidate_df.empty:
    candidate_df = candidate_df.sort_values(
        [
            "name_score",
            "size_bytes",
        ],
        ascending=[
            False,
            False,
        ],
    )

if not tar_member_df.empty:
    tar_member_df = tar_member_df.sort_values(
        [
            "name_score",
            "size_bytes",
        ],
        ascending=[
            False,
            False,
        ],
    )

candidate_df.to_csv(
    OUT_DIR / "pa_recovery_filesystem_candidates.csv",
    index=False,
)

tar_member_df.to_csv(
    OUT_DIR / "pa_recovery_tar_candidates.csv",
    index=False,
)

text_match_df.to_csv(
    OUT_DIR / "pa_recovery_code_matches.csv",
    index=False,
)


# ============================================================
# 5. Look for plausible printed PA matched-plot counts
# ============================================================
count_patterns = [
    re.compile(
        r"(?:PA|Pennsylvania).{0,100}?"
        r"(?:matched|physical).{0,100}?"
        r"(?:plot|plots).{0,30}?"
        r"[:=]\s*([0-9][0-9,]*)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"n_matched_physical_plots.{0,30}?"
        r"[:=]\s*([0-9][0-9,]*)",
        flags=re.IGNORECASE,
    ),
]

count_rows = []

for _, row in text_match_df.iterrows():
    snippet = str(
        row["snippet"]
    )

    for pattern in count_patterns:
        for match in pattern.finditer(
            snippet
        ):
            count_rows.append(
                {
                    "source": (
                        row["archive"]
                        or row["member"]
                    ),
                    "member": row["member"],
                    "candidate_count": match.group(1),
                    "snippet": snippet,
                }
            )

count_df = pd.DataFrame(
    count_rows,
    columns=[
        "source",
        "member",
        "candidate_count",
        "snippet",
    ],
)

count_df.to_csv(
    OUT_DIR / "pa_matched_plot_count_candidates.csv",
    index=False,
)


# ============================================================
# 6. Summary and decision
# ============================================================
has_count_candidate = len(
    count_df
) > 0

has_code_match = len(
    text_match_df
) > 0

has_data_candidate = (
    len(
        candidate_df
    )
    + len(
        tar_member_df
    )
) > 0

if has_count_candidate:
    recovery_status = (
        "PA_COUNT_CANDIDATE_FOUND_VERIFY_SOURCE"
    )

    next_step = (
        "Verify the candidate against the exact PA same-window "
        "filtering code; do not copy it blindly."
    )

elif has_code_match:
    recovery_status = (
        "PA_RECONSTRUCTION_CODE_FOUND"
    )

    next_step = (
        "Extract or open the highest-scoring matching notebook/code "
        "and rerun only the PA same-window physical-plot construction."
    )

elif has_data_candidate:
    recovery_status = (
        "PA_DATA_CANDIDATES_FOUND_CODE_NOT_IDENTIFIED"
    )

    next_step = (
        "Use the candidate manifest to identify raw PA FIA inputs, "
        "then rebuild only the plot-matching step."
    )

else:
    recovery_status = (
        "PA_RECOVERY_SOURCE_NOT_FOUND"
    )

    next_step = (
        "Remove the matched-FIA-plot column from Table 1 for the next "
        "draft, and reacquire PA FIA raw tables before submission."
    )

summary = {
    "run_utc": RUN_UTC,
    "software_versions_captured": True,
    "full_requirements_captured": True,
    "filesystem_candidates": len(
        candidate_df
    ),
    "tar_candidates": len(
        tar_member_df
    ),
    "code_matches": len(
        text_match_df
    ),
    "count_candidates": len(
        count_df
    ),
    "recovery_status": recovery_status,
    "next_step": next_step,
}

(OUT_DIR / "repair_audit_summary.json").write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print("\nPA RECOVERY SUMMARY")
print(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    )
)

print("\nTOP FILESYSTEM CANDIDATES")
display(
    candidate_df.head(
        40
    )
)

print("\nTOP TAR CANDIDATES")
display(
    tar_member_df.head(
        40
    )
)

print("\nTOP CODE MATCHES")
display(
    text_match_df.head(
        40
    )
)

if has_count_candidate:
    print("\nPA COUNT CANDIDATES — VERIFY BEFORE USE")
    display(
        count_df.head(
            20
        )
    )

print(
    "\n========== RUN SUMMARY =========="
)
print(
    "CELL_STATUS: SUBMISSION_CRITICAL_REPAIR_AUDIT_COMPLETE"
)
print(
    f"OUT_DIR: {OUT_DIR}"
)
print(
    f"SOFTWARE_VERSIONS_CORE: "
    f"{OUT_DIR / 'software_versions_core.csv'}"
)
print(
    f"REQUIREMENTS_CORE: "
    f"{OUT_DIR / 'requirements_core.txt'}"
)
print(
    f"REQUIREMENTS_FULL: "
    f"{OUT_DIR / 'requirements_full.txt'}"
)
print(
    f"PA_RECOVERY_STATUS: {recovery_status}"
)
print(
    f"FILESYSTEM_CANDIDATES: {len(candidate_df)}"
)
print(
    f"TAR_CANDIDATES: {len(tar_member_df)}"
)
print(
    f"CODE_MATCHES: {len(text_match_df)}"
)
print(
    f"COUNT_CANDIDATES: {len(count_df)}"
)
print(
    f"NEXT_STEP: {next_step}"
)
print(
    "============================================"
)
