# Spatial-support continuity audit workflow

Repository: https://github.com/GinixStudy/spatial-support-continuity-audit

This directory converts the notebook history into an explicit, auditable
pipeline. It does not change locked statistics and does not run anything by
default. It is the local release candidate for the public repository supporting
the manuscript *Computational evaluation of observation-drift correction in
opportunistic biodiversity data using spatial support continuity*.

The repository has two connected components:

1. `spatial-support-audit` v0.1.0, a lightweight installable Python toolkit for
   two-period support diagnostics;
2. the frozen research pipeline and audit materials used to reproduce the
   manuscript.

## Companion toolkit

The toolkit calculates:

- effort-frame Jaccard overlap;
- focal-species cell Jaccard overlap;
- intersection, union, and smaller-support overlap counts;
- the number of cells meeting a user-supplied effort minimum in both periods;
- raw and corrected-estimator availability under the supplied minimum number
  of stable cells.

It does not impose or recommend a universal Jaccard cutoff. The manuscript
shows that support diagnostics are method-specific, so users must choose effort
and cell-count requirements for their own estimator and sampling design.

### Install

From the repository root:

```bash
python -m pip install .
```

The package has no third-party runtime dependencies. Python 3.10 or newer is
required.

### One-command local test on Windows

From PowerShell in the repository directory, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_local_test.ps1
```

The script creates an isolated installation under `build/`, unpacks the local
wheel without downloading dependencies, runs all unit tests, and writes
`examples/local_test_output.csv` from the bundled example data. It does not run
the scientific experiments or modify the frozen result archive.

### Test with the paper's real three-state data

The final archive can be converted into the software's aggregated CSV format
and checked against all 43 locked state-species Jaccard values with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_real_data_test.ps1
```

The script reads the registered PA, VA, and NC GBIF Parquet files directly from
the final TAR, applies the locked deterministic filtering, 0.05-degree
thinning, and 0.5-degree grid rules, runs the companion tool, and writes the
results under `real_data/`. It does not rerun the paper's estimators,
simulations, bootstrap analyses, permutations, or inference. See
`real_data/README.md` for the files and provenance.

The included verified outputs pass 21 locked conversion-parity checks and
reproduce all 43 locked species Jaccard values.

### CSV interface

Input is an aggregated cell-period table with these columns:

- `period`: normally `early` or `late`;
- `cell_id`: a spatial cell identifier;
- `effort_count`: target-frame records in that cell and period;
- `focal_count`: focal-species records in that cell and period;
- an optional grouping column, such as `species`.

Run the included example:

```bash
ssc-audit examples/example_support_counts.csv \
  --output examples/example_diagnostics.csv \
  --group-column species \
  --minimum-effort 10 \
  --minimum-stable-cells 1
```

The output contains one diagnostic row per group. `corrected_estimable` reports
whether both periods contain focal records, enough stable effort cells exist,
and both relative-rate sums are positive on that common frame. The
`availability_reason` field records why an estimate is unavailable. This is a
diagnostic result, not a claim that the correction will improve accuracy.

The same interface can be called from R with `system2()` or another process
runner, and the CSV output can be read by modelling workflows built around
packages such as `sdm` or `biomod2`.

## Quick verification without rerunning experiments

From this directory:

```bash
python validate_release.py
python validate_release.py --archive /path/to/20260712_full_reproduction_output.tar
```

The first command runs the companion-tool unit tests, parses all 37 pipeline
scripts, and verifies the machine-readable report from the successful
138-check final audit. The second also checks the final frozen archive against
20 exact manuscript-result assertions. Neither command repeats the
600,000-replicate experiment.

## Full Kaggle run

Read `DATA_REQUIREMENTS.md`, attach the registered GBIF dataset, and then run:

```bash
python run_pipeline.py list
python run_pipeline.py run --dry-run
python run_pipeline.py run --confirm-heavy
```

The last command is intentionally guarded because it includes the frozen
600,000-replicate simulation. Each script runs in a fresh Python process, so
the result does not depend on hidden notebook variables. Logs are written to
`/kaggle/working/reproduction_logs`.

`repairable` stages are historical producers that may return an error after
writing the intermediate files consumed by the immediately following locked
repair cell. Other nonzero exits stop the pipeline.

## Contents

- `src/spatial_support_audit/`: installable companion-tool source code.
- `tests/`: unit and command-line tests for the companion tool.
- `examples/`: a small language-neutral CSV example and its output.
- `pipeline/`: exact exported code for the final scientific chain.
- `utilities/`: environment capture and PA recovery tools.
- `pipeline_order.tsv`: explicit execution order and error policy.
- `expected_results.json`: full-precision locked numerical assertions.
- `verify_frozen_results.py`: archive/output verifier.
- `verify_pa_repair.py`: verifies the 2,199-row PA summary and Parquet.
- `validation/final_reproduction_audit.py`: standalone export of the end-to-end
  audit used after the successful frozen run.
- `validation/reference_reports/`: machine-readable outputs showing 138 of 138
  checks passed in the successful run on 2026-07-12.
- `validation/verify_reference_audit.py`: lightweight integrity check for those
  reports.
- `validate_release.py`: one-command source and report validation.
- `build_reproduction_archive.py`: builds a new checked archive and refuses
  to overwrite `20260711q1test.tar`.
- `requirements_*.txt`, `software_versions_core.csv`,
  `environment_info.json`: captured successful confirmation environment.
- `SOURCE_MAP.md`: notebook provenance and supersession decisions.

The unpinned `ipython`, `requests` and `urllib3` lines are direct runtime
dependencies whose exact versions were not printed in the notebook output now
available locally. All numerical core packages are pinned to the versions
captured by Cell 23B; this limitation is stated rather than guessed.

## End-to-end audit

The end-to-end audit performs no new scientific inference. Run it only after
the locked pipeline and exact Pennsylvania matched-plot reconstruction have
produced their outputs. Its original Kaggle paths remain the defaults, and the
following environment variables can point it to equivalent mounted locations:

- `REPRO_BASE_DIR`
- `REPRO_AUDIT_DIR`
- `REPRO_PA_COUNT_DIR`
- `REPRO_FINAL_PACKAGE`

```bash
python validation/final_reproduction_audit.py
```

The reference reports are retained so that the published claim of 138 passed
automated validation checks can be inspected without rerunning the heavy
analysis.

## Containerized workflow

The repository includes a Dockerfile based on Python 3.12.13 and a dedicated
container validation command. GitHub Actions builds the image and verifies the
captured core environment, the release manifest, the installed toolkit, the
11 unit and command-line tests, all 37 frozen pipeline scripts, the record of
138 passed scientific audit checks, and the 43-group real-data diagnostic.

```bash
docker build --tag spatial-support-audit:0.1.0 .
docker run --rm spatial-support-audit:0.1.0
```

The default command does not rerun the 600,000-replicate experiment. The image
contains the complete frozen pipeline, and a full rerun is enabled by mounting
the registered GBIF download and writable Kaggle-compatible working
directories. See `container/README.md` and `DATA_REQUIREMENTS.md` for the exact
mount contract.

## Build the new submission archive

After the PA summary and matched-plot Parquet have been restored, run:

```bash
python build_reproduction_archive.py \
  --project-root /kaggle/working/fia_temporal_observation_drift \
  --repair-root /kaggle/working/submission_critical_repair_audit \
  --output /kaggle/working/manuscript_reproduction_v1.tar.gz
```

Add `--include-large-outputs` to include the full replicate-level Parquets.

## License

The software and original source code in this repository are released under
the MIT License. Third-party biodiversity data remain subject to the terms of
their respective providers.
