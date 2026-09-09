# Real three-state software test data

This directory contains a deterministic, aggregated test input derived from
the final frozen reproduction archive. It covers the 43 state-species
observations used in the manuscript (PA: 15, VA: 11, NC: 17).

## Files

- `three_state_support_counts.csv`: direct input to `spatial-support-audit`;
- `three_state_support_diagnostics.csv`: software output for all 43 groups;
- `locked_reference_cell_jaccard.csv`: the corresponding locked manuscript
  values used only for validation;
- `conversion_summary.csv`: state-level row and support counts;
- `locked_conversion_parity.csv`: 21 checks against the frozen state-processing
  and 0.5-degree grid summaries;
- `validation_report.json`: comparison of recalculated and locked Jaccard
  values;
- `provenance.json`: source members, rules, versions, hashes, and the explicit
  statement that no scientific experiment was rerun.

The input columns used by the software are `analysis_group`, `period`,
`cell_id`, `effort_count`, and `focal_count`. The state, FIA species code, and
scientific name are retained as provenance columns.

## Locked conversion rules

- early period: 2013--2018;
- late period: 2020--2025;
- maximum coordinate uncertainty: 10,000 m;
- within-day spatial thinning: 0.05 degrees;
- analysis grid: 0.5 degrees;
- stable effort cell: at least 10 target-frame records in each period.

These are copied from the final formal three-state parity script. The
conversion performs filtering, thinning, gridding, and counting only. It does
not rerun distribution-shift estimates, simulations, bootstrap analyses,
permutations, or inference.

The verified local build produced 5,665 input rows. All 21 state-processing and
grid-count parity checks passed, and all 43 species Jaccard values matched the
locked manuscript table (maximum absolute difference
`5.551115123125783e-17`).

## Rebuild and test

From PowerShell in the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_real_data_test.ps1
```

To recalculate and verify the SHA-256 of the approximately 2 GB source archive
as part of the run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_real_data_test.ps1 -VerifyArchiveHash
```
