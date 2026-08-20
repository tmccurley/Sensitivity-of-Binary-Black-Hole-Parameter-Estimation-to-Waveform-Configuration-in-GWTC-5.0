"""Static guardrail confirming that the eligibility audit has no outcome code."""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE = Path(__file__).with_name("GWTC5_replication_metadata_audit.py")
TEXT = SOURCE.read_text(encoding="utf-8")
TREE = ast.parse(TEXT)

FORBIDDEN_IMPORTS = {"scipy", "matplotlib", "seaborn"}
FORBIDDEN_CALL_TOKENS = {
    "wasserstein_distance",
    "wilcoxon",
    "spearmanr",
    "quantile",
    "median",
    "histogram",
}

imports: set[str] = set()
calls: set[str] = set()
for node in ast.walk(TREE):
    if isinstance(node, ast.Import):
        imports.update(alias.name.split(".")[0] for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module:
        imports.add(node.module.split(".")[0])
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            calls.add(node.func.attr)

bad_imports = imports & FORBIDDEN_IMPORTS
bad_calls = calls & FORBIDDEN_CALL_TOKENS
if bad_imports or bad_calls:
    raise SystemExit(
        f"Metadata-audit seal failed: imports={sorted(bad_imports)}, "
        f"calls={sorted(bad_calls)}"
    )

if "posterior.fields(" in TEXT or "posterior[" in TEXT:
    raise SystemExit("Metadata-audit seal failed: posterior array read detected.")

print("PASS: metadata audit contains no replication-outcome calculation path.")

