# Containerized workflow

The root `Dockerfile` recreates the captured Python 3.12.13 environment,
installs the companion toolkit, and contains the complete frozen pipeline.
The default container command performs lightweight validation without
rerunning any scientific experiment.

## Build and validate

```bash
docker build --tag spatial-support-audit:0.1.0 .
docker run --rm spatial-support-audit:0.1.0
```

The validation command checks:

- the captured Python and core dependency versions;
- the release SHA-256 manifest;
- installed-package dependency consistency;
- 11 unit and command-line tests;
- syntax and presence of the 37 frozen pipeline scripts;
- the machine-readable record of 138 passed scientific audit checks;
- all 43 real state-species support diagnostics against the locked Jaccard
  reference table.

## Run the complete scientific pipeline

The exported research code retains its original Kaggle filesystem contract.
A complete rerun therefore requires the inputs described in
`DATA_REQUIREMENTS.md` and writable mounts at the original locations. From a
Linux or macOS shell, the general form is:

```bash
docker run --rm \
  -v /local/registered-gbif:/kaggle/input/datasets/nanjide/20260711gbif:ro \
  -v /local/fia-work:/kaggle/working/fia_temporal_observation_drift \
  -v /local/repair:/kaggle/working/submission_critical_repair_audit \
  -v /local/recovered-q1:/kaggle/working/recovered_q1:ro \
  spatial-support-audit:0.1.0 \
  python run_pipeline.py run --confirm-heavy
```

On Windows PowerShell, replace the four `/local/...` placeholders with
absolute Windows directories. Docker Desktop translates those bind mounts
into the Linux container paths shown above.

The full command includes downloads and the 600,000-replicate confirmation.
It is intentionally never triggered by the default container command or by
continuous integration.
