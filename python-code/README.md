# Python analysis code

This directory preserves every Python script and notebook found in the supplied figures-and-code archive, plus the newer repository versions that were not byte-identical to any archived copy.

| Directory | Purpose |
| --- | --- |
| `current-analysis/` | Latest repository versions of the confirmatory, robustness, and paper-sensitivity analyses |
| `frozen_reproducible_pipeline/` | Frozen public preparation, metadata audit, confirmatory, robustness, test, and verification pipeline; its internal layout is intentionally preserved |
| `main_commented_analyses/` | Commented analysis versions from the supplied archive |
| `prospective_replication_executed/` | Executed prospective-replication analysis and tests |
| `prospective_replication_github_upload/` | Publication-ready replication code and tests |
| `exploratory_joint_mechanism/` | Exploratory joint-geometry and spin-sensitivity analyses |
| `figure_generation_and_validation/` | Figure creation, correction, rendering, and validation tools |
| `site-tools/` | Local launcher plus the verified distribution-archive builder |

Several archived scripts use their location to find sibling files or project directories. Their subdirectory structure has therefore been retained instead of flattening files with duplicate names.

Current inventory: **55 Python files** and **12 Jupyter notebooks**. The Windows launcher is retained as an additional `.bat` script.
