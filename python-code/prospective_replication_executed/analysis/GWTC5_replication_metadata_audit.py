"""Run the sealed metadata-only eligibility audit for the nine-event replication.

This script deliberately cannot calculate the replication outcome. It never
loads a posterior column: it inspects only posterior dataset shape and field
names, then reads configuration, metadata, sampler records, and small analytic
prior records needed for the frozen Tier A comparison.

Outputs contain no posterior values, distribution summaries, distances,
figures, hypothesis tests, or event rankings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py


XPHM = "C00:IMRPhenomXPHM-SpinTaylor"
XPNR = "C00:IMRPhenomXPNR"
MODELS = [XPHM, XPNR]
REQUIRED_PARAMETERS = ["chi_eff", "luminosity_distance"]

CONFIG_KEYS = [
    "waveform-approximant",
    "waveform-arguments-dict",
    "sampler",
    "sampler-kwargs",
    "detectors",
    "minimum-frequency",
    "maximum-frequency",
    "reference-frequency",
    "duration",
    "sampling-frequency",
    "likelihood-type",
    "calibration-marginalization",
    "calibration-model",
    "distance-marginalization",
    "phase-marginalization",
    "time-marginalization",
    "time-reference",
    "reference-frame",
    "cosmology",
    "prior-dict",
    "psd-dict",
    "channel-dict",
    "trigger-time",
    "generation-seed",
    "sampling-seed",
]

METADATA_KEYS = [
    "IFOs",
    "approximant",
    "f_low",
    "f_start",
    "f_ref",
    "f_final",
    "duration",
    "sampling_frequency",
    "delta_f",
    "start_time",
    "calibration_marginalization",
    "distance_marginalization",
    "phase_marginalization",
    "time_marginalization",
    "time_reference",
    "reference_frame",
    "cosmology",
    "frequency_domain_source_model",
    "parameter_conversion",
]

SAMPLER_KEYS = [
    "pe_algorithm",
    "nsamples",
    "ln_evidence",
    "ln_evidence_error",
    "ln_noise_evidence",
    "ln_bayes_factor",
]

# These are the only model-dependent settings allowed by the frozen protocol.
PERMITTED_WAVEFORM_DIFFERENCES = {
    ("top_level", "approximant"),
    ("top_level", "description"),
    ("config", "waveform-approximant"),
    ("config", "waveform-arguments-dict"),
    ("metadata", "approximant"),
}

# These fields are results of separate sampling runs, not sampler settings.
PERMITTED_RUN_OUTPUT_DIFFERENCES = {
    ("sampler", "nsamples"),
    ("sampler", "ln_evidence"),
    ("sampler", "ln_evidence_error"),
    ("sampler", "ln_bayes_factor"),
}

NUMBER_TOKEN = re.compile(
    r"(?<![A-Za-z_])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Metadata-only audit; no replication outcomes are computed."
    )
    parser.add_argument(
        "--verification",
        type=Path,
        default=root / "results" / "download_verification.csv",
    )
    parser.add_argument(
        "--event-lock",
        type=Path,
        default=root / "design" / "replication_event_lock.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "metadata_audit",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing metadata-audit output directory.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item") and not isinstance(value, (str, list, tuple, dict)):
        try:
            return convert_value(value.item())
        except ValueError:
            pass
    if hasattr(value, "tolist"):
        return convert_value(value.tolist())
    if isinstance(value, tuple):
        return [convert_value(item) for item in value]
    if isinstance(value, list):
        return [convert_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): convert_value(item) for key, item in value.items()}
    return value


def read_dataset(source: h5py.File, path: str) -> Any:
    if path not in source or not isinstance(source[path], h5py.Dataset):
        return None
    return convert_value(source[path][()])


def read_named_datasets(
    source: h5py.File, base_path: str, names: list[str]
) -> dict[str, Any]:
    return {
        name: value
        for name in names
        if (value := read_dataset(source, f"{base_path}/{name}")) is not None
    }


def read_all_small_datasets(source: h5py.File, group_path: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if group_path not in source or not isinstance(source[group_path], h5py.Group):
        return output

    def visitor(relative_name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and obj.size <= 1000:
            output[relative_name] = convert_value(obj[()])

    source[group_path].visititems(visitor)
    return output


def numeric_text_equivalent(left: str, right: str) -> bool:
    """Treat whitespace and negligible float serialization as non-substantive."""

    a = re.sub(r"\s+", "", left)
    b = re.sub(r"\s+", "", right)
    a_parts = NUMBER_TOKEN.split(a)
    b_parts = NUMBER_TOKEN.split(b)
    if a_parts != b_parts:
        return False
    a_numbers = NUMBER_TOKEN.findall(a)
    b_numbers = NUMBER_TOKEN.findall(b)
    if len(a_numbers) != len(b_numbers):
        return False
    return all(
        math.isclose(float(x), float(y), rel_tol=1e-12, abs_tol=1e-15)
        for x, y in zip(a_numbers, b_numbers, strict=True)
    )


def equivalent(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-15)
    if isinstance(left, str) and isinstance(right, str):
        return numeric_text_equivalent(left, right)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            equivalent(x, y) for x, y in zip(left, right, strict=True)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            equivalent(left[key], right[key]) for key in left
        )
    return left == right


def difference_category(section: str, setting: str, same: bool) -> str:
    key = (section, setting)
    if same:
        return "identical_or_numeric_serialization"
    if key in PERMITTED_WAVEFORM_DIFFERENCES:
        return "permitted_waveform_difference"
    if key in PERMITTED_RUN_OUTPUT_DIFFERENCES:
        return "permitted_run_output_difference"
    return "review_required"


def model_metadata(source: h5py.File, label: str) -> dict[str, dict[str, Any]]:
    base = f"/{label}"
    return {
        "top_level": {
            "approximant": read_dataset(source, f"{base}/approximant"),
            "description": read_dataset(source, f"{base}/description"),
        },
        "config": read_named_datasets(
            source, f"{base}/config_file/config", CONFIG_KEYS
        ),
        "metadata": read_named_datasets(
            source, f"{base}/meta_data/meta_data", METADATA_KEYS
        ),
        "sampler": read_named_datasets(
            source, f"{base}/meta_data/sampler", SAMPLER_KEYS
        ),
        "analytic_priors": read_all_small_datasets(
            source, f"{base}/priors/analytic"
        ),
    }


def audit_event(event: str, path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary: dict[str, Any] = {
        "event": event,
        "source_path": str(path),
        "file_opened": False,
        "xphm_present": False,
        "xpnr_present": False,
        "xphm_sample_count": "",
        "xpnr_sample_count": "",
        "xphm_parameters_present": False,
        "xpnr_parameters_present": False,
        "required_parameters_present": False,
        "review_required_count": "",
        "provisional_status": "audit_error",
        "error": "",
    }
    rows: list[dict[str, Any]] = []

    try:
        with h5py.File(path, "r") as source:
            summary["file_opened"] = True
            metadata: dict[str, dict[str, dict[str, Any]]] = {}

            for label in MODELS:
                present = label in source and isinstance(source[label], h5py.Group)
                summary["xphm_present" if label == XPHM else "xpnr_present"] = present
                if not present or "posterior_samples" not in source[label]:
                    continue

                posterior = source[label]["posterior_samples"]
                if not isinstance(posterior, h5py.Dataset):
                    continue

                # Sealed boundary: inspect schema and row count only. Do not index
                # or read the posterior dataset here.
                names = set(posterior.dtype.names or [])
                count_key = (
                    "xphm_sample_count" if label == XPHM else "xpnr_sample_count"
                )
                summary[count_key] = int(posterior.shape[0])
                metadata[label] = model_metadata(source, label)
                summary[f"{count_key.split('_')[0]}_parameters_present"] = all(
                    parameter in names for parameter in REQUIRED_PARAMETERS
                )

            summary["required_parameters_present"] = bool(
                summary.get("xphm_parameters_present")
                and summary.get("xpnr_parameters_present")
            )

            if set(metadata) != set(MODELS):
                summary["provisional_status"] = "missing_target_model_or_posterior"
                return summary, rows

            for section in sorted({key for model in MODELS for key in metadata[model]}):
                settings = sorted(
                    {
                        key
                        for model in MODELS
                        for key in metadata[model].get(section, {})
                    }
                )
                for setting in settings:
                    left = metadata[XPHM].get(section, {}).get(setting)
                    right = metadata[XPNR].get(section, {}).get(setting)
                    same = equivalent(left, right)
                    category = difference_category(section, setting, same)
                    rows.append(
                        {
                            "event": event,
                            "section": section,
                            "setting": setting,
                            "comparison_category": category,
                            "xphm_value_json": json.dumps(
                                left, ensure_ascii=False, sort_keys=True, default=str
                            ),
                            "xpnr_value_json": json.dumps(
                                right, ensure_ascii=False, sort_keys=True, default=str
                            ),
                        }
                    )

            review_count = sum(
                row["comparison_category"] == "review_required" for row in rows
            )
            summary["review_required_count"] = review_count
            if not summary["required_parameters_present"]:
                summary["provisional_status"] = "missing_required_parameter"
            elif review_count:
                summary["provisional_status"] = "manual_tier_review_required"
            else:
                summary["provisional_status"] = "provisional_tier_A"

    except Exception as error:  # Preserve failure without trying another event.
        summary["error"] = f"{type(error).__name__}: {error}"

    return summary, rows


def main() -> int:
    args = parse_args()
    lock_rows = read_csv(args.event_lock)
    verification_rows = read_csv(args.verification)
    verified = {row["event"]: row for row in verification_rows}

    if len(lock_rows) != 9 or len({row["cell_id"] for row in lock_rows}) != 9:
        raise RuntimeError("The frozen event lock must contain nine unique cells.")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} already contains files; pass --overwrite intentionally."
            )
        for path in args.output_dir.iterdir():
            if path.is_file():
                path.unlink()
            else:
                raise RuntimeError(f"Refusing to remove nested output directory: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    integrity_rows: list[dict[str, Any]] = []
    event_summaries: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for lock in lock_rows:
        event = lock["event"]
        if event not in verified:
            raise RuntimeError(f"No download-verification row for {event}.")
        record = verified[event]
        path = Path(record["actual_path"])
        if not path.is_file():
            raise FileNotFoundError(path)

        expected_md5 = record["expected_md5"].lower()
        actual_md5 = file_md5(path)
        matches = actual_md5 == expected_md5
        integrity_rows.append(
            {
                "event": event,
                "source_path": str(path),
                "size_bytes": path.stat().st_size,
                "expected_md5": expected_md5,
                "actual_md5": actual_md5,
                "checksum_match": matches,
            }
        )
        if not matches:
            raise RuntimeError(f"Checksum mismatch for {event}; audit stopped.")

        summary, rows = audit_event(event, path)
        event_summaries.append(summary)
        comparison_rows.extend(rows)

    write_csv(
        args.output_dir / "replication_input_integrity.csv",
        integrity_rows,
        [
            "event",
            "source_path",
            "size_bytes",
            "expected_md5",
            "actual_md5",
            "checksum_match",
        ],
    )
    write_csv(
        args.output_dir / "replication_metadata_event_summary.csv",
        event_summaries,
        [
            "event",
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
        args.output_dir / "replication_pair_metadata_comparison.csv",
        comparison_rows,
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
        "purpose": "sealed metadata-only replication eligibility audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "h5py": h5py.__version__,
        "events_requested": len(lock_rows),
        "events_opened": sum(bool(row["file_opened"]) for row in event_summaries),
        "provisional_tier_A": sum(
            row["provisional_status"] == "provisional_tier_A"
            for row in event_summaries
        ),
        "manual_tier_review_required": sum(
            row["provisional_status"] == "manual_tier_review_required"
            for row in event_summaries
        ),
        "sealed_guarantees": {
            "posterior_arrays_loaded": False,
            "posterior_values_output": False,
            "distribution_summaries_calculated": False,
            "distance_metrics_calculated": False,
            "hypothesis_tests_calculated": False,
            "figures_created": False,
        },
    }
    (args.output_dir / "replication_metadata_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
