"""Static guard for the narrow prospective replication analysis surface."""

from __future__ import annotations

import ast
from pathlib import Path


SCRIPT = Path(__file__).with_name("GWTC5_prospective_h3_replication_analysis.py")
ALLOWED_PARAMETERS = {"chi_eff", "luminosity_distance"}
FORBIDDEN_OUTCOME_TERMS = {
    "chi_p",
    "chirp_mass",
    "mass_ratio",
    "jensen_shannon",
    "ks_2samp",
    "spearmanr",
    "pearsonr",
}


def main() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SCRIPT))

    parameter_assignment = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "PARAMETERS":
                    parameter_assignment = ast.literal_eval(node.value)
    if set(parameter_assignment or ()) != ALLOWED_PARAMETERS:
        raise RuntimeError("The script's parameter set differs from the lock.")

    lowered = source.lower()
    found = sorted(term for term in FORBIDDEN_OUTCOME_TERMS if term in lowered)
    if found:
        raise RuntimeError(f"Forbidden exploratory outcome terms found: {found}")

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    wilcoxon_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "wilcoxon"
    ]
    if len(wilcoxon_calls) != 1:
        raise RuntimeError("Expected exactly one Wilcoxon call in the analysis script.")
    keywords = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in wilcoxon_calls[0].keywords
        if keyword.arg is not None
    }
    expected = {
        "alternative": "greater",
        "zero_method": "wilcox",
        "method": "auto",
    }
    if keywords != expected:
        raise RuntimeError(f"The Wilcoxon call differs from the lock: {keywords}")

    print("Replication analysis seal check passed.")


if __name__ == "__main__":
    main()
