# Source map

The scientific pipeline was mechanically exported from the four notebooks
below. Scientific
constants, random seeds, thresholds and output names were not edited.

- `20260710test.ipynb`: raw FIA/GBIF construction, controlled benchmark,
  PA discovery, VA/NC construction and the first three-region synthesis.
- `q1.ipynb`: registered-GBIF parity, final inference, threshold-free and
  clustered validation, and the frozen two-axis experiment.
- `最新.ipynb`: same scientific chain plus manuscript integration.
- `20260712.ipynb`: submission repair audit and exact PA matched-plot recovery.

The standalone end-to-end audit in
`validation/final_reproduction_audit.py` was exported from the uniquely marked
`FINAL_REPRODUCTION_AUDIT_COMPLETE` code cell in
`20260712biodiversityinformatics.ipynb`. Only path configuration and a
plain-text display fallback were added; the 138 scientific validation checks
and their locked expected values were not changed.

Conflict rule: a later explicitly labelled repair supersedes the corresponding
earlier failed cell without changing its locked design. The default order
therefore retains three `repairable` producers followed by their repair cells:
06A/06B, 12A/12B and final 17A/17B.

Excluded from the default run:

- mount/schema diagnostics and failed download alternatives (01A-02A);
- eBird-BBS feasibility (20A), because it is not part of the current paper;
- the confounded phase designs (22A-22B), superseded by 22C-22F;
- recovery search attempts (23B-23D2), retained under `utilities/`;
- exact PA reconstruction (23E), retained under `utilities/` because it uses
  the restored legacy TAR rather than the normal analysis working tree.
