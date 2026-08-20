"""Audit the first prespecified replacement without reading posterior values."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py

from GWTC5_replication_metadata_audit import audit_event, file_md5, write_csv


EVENT = "GW240716_034900"
REPLACES_EVENT = "GW240915_105151"
EXPECTED_MD5 = "ba62bbb2da44a6abfd0219176d513495"
ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
SOURCE = (
    REPOSITORY_ROOT
    / "data"
    / "raw"
    / "events"
    / "IGWN-GWTC5p0-29ebe06b7_25-GW240716_034900-combined_PEDataRelease.hdf5"
)
OUTPUT_DIR = ROOT / "results" / "metadata_audit_replacement_round1"


def main() -> int:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    actual_md5 = file_md5(SOURCE)
    if actual_md5 != EXPECTED_MD5:
        raise RuntimeError("Replacement checksum mismatch; audit stopped.")
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Replacement audit outputs already exist: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary, comparisons = audit_event(EVENT, SOURCE)
    summary["replaces_event"] = REPLACES_EVENT

    write_csv(
        OUTPUT_DIR / "replacement_metadata_event_summary.csv",
        [summary],
        [
            "event",
            "replaces_event",
            "source_path",
            "file_opened",
            "xphm_present",
            "xpnr_present",
            "xphm_sample_count",
            "xpnr_sample_count",
            "xphm_parameters_present",
            "xpnr_parameters_present",
            "required_parameters_present",
            "review_required_count",
            "provisional_status",
            "error",
        ],
    )
    write_csv(
        OUTPUT_DIR / "replacement_pair_metadata_comparison.csv",
        comparisons,
        [
            "event",
            "section",
            "setting",
            "comparison_category",
            "xphm_value_json",
            "xpnr_value_json",
        ],
    )
    manifest = {
        "purpose": "sealed metadata-only replacement eligibility audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "event": EVENT,
        "replaces_event": REPLACES_EVENT,
        "source_md5": actual_md5,
        "python": sys.version,
        "h5py": h5py.__version__,
        "provisional_status": summary["provisional_status"],
        "sealed_guarantees": {
            "posterior_arrays_loaded": False,
            "posterior_values_output": False,
            "distribution_summaries_calculated": False,
            "distance_metrics_calculated": False,
            "hypothesis_tests_calculated": False,
            "figures_created": False,
        },
    }
    (OUTPUT_DIR / "replacement_metadata_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
