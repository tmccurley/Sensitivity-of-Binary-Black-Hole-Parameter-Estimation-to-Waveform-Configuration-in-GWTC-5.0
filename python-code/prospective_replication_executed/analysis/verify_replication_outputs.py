"""Independently verify the sealed replication outputs without reopening HDF5."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "replication_analysis"


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    integrity = read_csv("replication_input_integrity.csv")
    preflight = read_csv("replication_preflight_samples.csv")
    metrics = read_csv("replication_parameter_metrics.csv")
    events = read_csv("replication_event_deltas.csv")
    hypothesis_rows = read_csv("replication_hypothesis_result.csv")
    manifest = json.loads(
        (RESULTS / "replication_analysis_manifest.json").read_text(encoding="utf-8")
    )

    assert len(integrity) == 9
    assert all(row["checksum_match"].lower() == "true" for row in integrity)
    assert all(row["expected_md5"] == row["actual_md5"] for row in integrity)

    assert len(preflight) == 36
    assert all(row["sampling_method"] == "all_samples" for row in preflight)
    assert all(
        row["original_sample_count"]
        == row["stored_sample_count_before_finite_filter"]
        == row["finite_sample_count"]
        for row in preflight
    )

    assert len(metrics) == 18
    assert {row["parameter"] for row in metrics} == {
        "chi_eff",
        "luminosity_distance",
    }

    assert len(events) == 9
    deltas = [
        float(row["delta_chi_eff_minus_luminosity_distance"]) for row in events
    ]
    assert len(set(deltas)) == 9
    assert all(delta > 0 for delta in deltas)

    assert len(hypothesis_rows) == 1
    hypothesis = hypothesis_rows[0]
    assert int(hypothesis["positive_delta_count"]) == 9
    assert int(hypothesis["zero_delta_count"]) == 0
    assert int(hypothesis["negative_delta_count"]) == 0
    assert float(hypothesis["wilcoxon_statistic"]) == 45.0
    assert math.isclose(float(hypothesis["median_delta"]), statistics.median(deltas))
    # With nine nonzero positive differences, the exact one-sided tail is 1/2^9.
    assert math.isclose(float(hypothesis["raw_p_value"]), 1.0 / (2**9))
    assert hypothesis["reject_at_alpha_0_05"].lower() == "true"

    assert manifest["scope"]["exploratory_outputs_created"] is False
    for name, expected_hash in manifest["result_hashes"].items():
        assert sha256_file(RESULTS / name) == expected_hash
    assert (
        sha256_file(Path(manifest["analysis_code"]["path"]))
        == manifest["analysis_code"]["sha256"]
    )

    print("Replication output verification passed.")
    print(
        f"n=9; positive=9; median_delta={statistics.median(deltas):.12f}; "
        f"W=45; one_sided_p={1.0 / (2**9):.9f}"
    )


if __name__ == "__main__":
    main()
