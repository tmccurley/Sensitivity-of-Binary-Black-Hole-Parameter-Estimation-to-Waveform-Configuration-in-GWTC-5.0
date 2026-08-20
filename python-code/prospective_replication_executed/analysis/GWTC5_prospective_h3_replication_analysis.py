"""Run the single locked H3 replication after the final sample is sealed.

This script intentionally has a narrow scope. It reads only the two posterior
parameters named in the prospective protocol, calculates the parent's exact
normalized Wasserstein-1 (NW1) metric, and runs the parent's exact one-sided
Wilcoxon signed-rank call. It does not calculate alternative metrics,
parameters, normalizations, or exploratory tests.

The script refuses to start unless the final nine-event CSV still matches its
pre-outcome lock and every raw input still matches its recorded checksum.
Existing result files are also protected from accidental replacement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy
from scipy.stats import wasserstein_distance, wilcoxon


XPHM = "C00:IMRPhenomXPHM-SpinTaylor"
XPNR = "C00:IMRPhenomXPNR"
MODELS = (XPHM, XPNR)
PARAMETERS = ("chi_eff", "luminosity_distance")

MAX_STORED_SAMPLES_PER_ANALYSIS = 100_000
EXTRACTION_SEED = 20260802
ALPHA = 0.05

LOCK_STATUS = "FINAL_9_EVENT_TIER_A_SAMPLE_LOCKED_BEFORE_OUTCOME_INSPECTION"
EXPECTED_OUTPUT_FILES = {
    "replication_input_integrity.csv",
    "replication_preflight_samples.csv",
    "replication_parameter_metrics.csv",
    "replication_event_deltas.csv",
    "replication_hypothesis_result.csv",
    "replication_analysis_manifest.json",
}


def parse_args() -> argparse.Namespace:
    """Define paths relative to the replication root by default."""

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run the frozen nine-event prospective H3 replication."
    )
    parser.add_argument(
        "--final-sample",
        type=Path,
        default=root / "results" / "replication_final_strict_9_event_sample.csv",
    )
    parser.add_argument(
        "--sample-lock",
        type=Path,
        default=root / "results" / "replication_final_sample_lock.json",
    )
    parser.add_argument(
        "--verification",
        type=Path,
        default=root / "results" / "download_verification.csv",
    )
    parser.add_argument(
        "--replacement-verification",
        type=Path,
        default=root
        / "results"
        / "download_verification_replacement_round1.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "replication_analysis",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the known result files from a documented full rerun.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV into records without changing stored text values."""

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write records using an explicit column order for stable outputs."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a SHA-256 digest without loading a complete file into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the catalog's published checksum format for a raw HDF5 file."""

    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_clean_output(output_dir: Path, overwrite: bool) -> None:
    """Prevent an old result set from being silently mixed with a new run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_files = [path for path in output_dir.iterdir() if path.is_file()]
    unexpected = [path for path in existing_files if path.name not in EXPECTED_OUTPUT_FILES]
    if unexpected:
        raise FileExistsError(
            "Output directory contains unexpected files; refusing to alter it:\n"
            + "\n".join(str(path) for path in unexpected)
        )
    if existing_files and not overwrite:
        raise FileExistsError(
            f"Output directory already contains results: {output_dir}\n"
            "Use --overwrite only for a documented full rerun."
        )
    if overwrite:
        for path in existing_files:
            path.unlink()


def load_and_verify_final_sample(
    final_sample_path: Path,
    lock_path: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Verify that the exact pre-outcome sample and order remain unchanged."""

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != LOCK_STATUS:
        raise RuntimeError("The final sample lock does not have the expected status.")
    if int(lock.get("strict_event_count", -1)) != 9:
        raise RuntimeError("The final sample lock does not specify nine events.")

    expected_hash = lock.get("source_hashes", {}).get(
        "results/replication_final_strict_9_event_sample.csv"
    )
    actual_hash = sha256_file(final_sample_path)
    if not expected_hash or actual_hash.lower() != str(expected_hash).lower():
        raise RuntimeError("The final sample CSV no longer matches its sealed hash.")

    rows = read_csv(final_sample_path)
    events = [row["event"] for row in rows]
    if len(rows) != 9 or len(set(events)) != 9:
        raise RuntimeError("The final sample must contain nine unique events.")
    if events != list(lock.get("ordered_events", [])):
        raise RuntimeError("The final sample order differs from the sealed order.")
    if len({row["cell_id"] for row in rows}) != 9:
        raise RuntimeError("The final sample does not contain one event per cell.")
    if any(row["classification"] != "Tier A" for row in rows):
        raise RuntimeError("At least one final-sample event is not locked as Tier A.")
    if any(
        row["posterior_metrics_inspected_before_lock"].strip().lower() != "false"
        for row in rows
    ):
        raise RuntimeError("The final sample does not preserve the sealed outcome state.")

    sealed_state = lock.get("sealed_outcome_state", {})
    if any(bool(value) for value in sealed_state.values()):
        raise RuntimeError("The lock indicates that an outcome was opened before sealing.")
    return rows, lock


def load_verified_inputs(
    sample_rows: list[dict[str, str]],
    verification_paths: tuple[Path, Path],
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    """Recheck every raw file against the checksum recorded before analysis."""

    verification_rows: list[dict[str, str]] = []
    for path in verification_paths:
        verification_rows.extend(read_csv(path))
    by_event = {row["event"]: row for row in verification_rows}

    event_paths: dict[str, Path] = {}
    integrity_rows: list[dict[str, Any]] = []
    for sample_row in sample_rows:
        event = sample_row["event"]
        if event not in by_event:
            raise FileNotFoundError(f"No download-verification record for {event}.")
        record = by_event[event]
        path = Path(record["actual_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Verified raw input is missing for {event}: {path}")
        if record.get("checksum_match", "").strip().lower() != "true":
            raise RuntimeError(f"The prior checksum verification failed for {event}.")

        expected_md5 = record["expected_md5"].strip().lower()
        actual_md5 = md5_file(path).lower()
        size_bytes = path.stat().st_size
        recorded_size = int(record["size_bytes"])
        matches = actual_md5 == expected_md5 and size_bytes == recorded_size
        if not matches:
            raise RuntimeError(f"Raw input integrity check failed for {event}: {path}")

        event_paths[event] = path
        integrity_rows.append(
            {
                "event": event,
                "source_path": str(path),
                "size_bytes": size_bytes,
                "expected_md5": expected_md5,
                "actual_md5": actual_md5,
                "checksum_match": matches,
            }
        )
    return event_paths, integrity_rows


def interval_summary(values: np.ndarray) -> dict[str, float]:
    """Return the same 5th/50th/95th-percentile summary as the parent code."""

    lower, median, upper = np.quantile(values, [0.05, 0.50, 0.95])
    return {
        "lower_90": float(lower),
        "median": float(median),
        "upper_90": float(upper),
        "width_90": float(upper - lower),
    }


def pair_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Calculate the parent's exact NW1 definition for one posterior pair."""

    sx = interval_summary(x)
    sy = interval_summary(y)
    average_width = 0.5 * (sx["width_90"] + sy["width_90"])
    raw_w1 = float(wasserstein_distance(x, y))
    normalized_w1 = raw_w1 / average_width if average_width > 0 else math.nan
    return {
        "xphm_lower_90": sx["lower_90"],
        "xphm_median": sx["median"],
        "xphm_upper_90": sx["upper_90"],
        "xphm_width_90": sx["width_90"],
        "xpnr_lower_90": sy["lower_90"],
        "xpnr_median": sy["median"],
        "xpnr_upper_90": sy["upper_90"],
        "xpnr_width_90": sy["width_90"],
        "average_width_90": average_width,
        "wasserstein_1": raw_w1,
        "normalized_wasserstein_1": normalized_w1,
    }


def load_locked_samples(
    sample_rows: list[dict[str, str]],
    event_paths: dict[str, Path],
) -> tuple[dict[tuple[str, str, str], np.ndarray], list[dict[str, Any]]]:
    """Load only the two frozen fields, applying the frozen sample cap if needed."""

    samples: dict[tuple[str, str, str], np.ndarray] = {}
    preflight_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(EXTRACTION_SEED)

    for sample_row in sample_rows:
        event = sample_row["event"]
        source_path = event_paths[event]
        with h5py.File(source_path, "r") as source:
            for model in MODELS:
                if model not in source or "posterior_samples" not in source[model]:
                    raise KeyError(f"Missing posterior for {event} / {model}.")
                posterior = source[model]["posterior_samples"]
                if not isinstance(posterior, h5py.Dataset):
                    raise TypeError(f"Posterior is not a dataset for {event} / {model}.")

                names = set(posterior.dtype.names or [])
                missing = [parameter for parameter in PARAMETERS if parameter not in names]
                if missing:
                    raise KeyError(f"Missing {event} / {model} fields: {missing}")
                weight_names = {
                    "weight",
                    "weights",
                    "posterior_weight",
                    "posterior_weights",
                    "log_weight",
                    "log_weights",
                }
                found_weights = sorted(names.intersection(weight_names))
                if found_weights:
                    raise RuntimeError(
                        f"Explicit posterior weights found for {event} / {model}: "
                        f"{found_weights}. The locked protocol requires analysis to pause."
                    )

                original_count = int(posterior.shape[0])
                stored_count = min(original_count, MAX_STORED_SAMPLES_PER_ANALYSIS)
                if stored_count < original_count:
                    indices = np.sort(
                        rng.choice(original_count, size=stored_count, replace=False)
                    )
                    sampling_method = "random_without_replacement"
                else:
                    indices = np.arange(original_count)
                    sampling_method = "all_samples"

                for parameter in PARAMETERS:
                    values = np.asarray(
                        posterior.fields(parameter)[indices], dtype=np.float64
                    ).reshape(-1)
                    values = values[np.isfinite(values)]
                    if values.size < 4:
                        raise RuntimeError(
                            f"Too few finite samples for {event} / {model} / {parameter}: "
                            f"{values.size}"
                        )
                    samples[(event, model, parameter)] = values
                    preflight_rows.append(
                        {
                            "event": event,
                            "cell_id": sample_row["cell_id"],
                            "model": model,
                            "parameter": parameter,
                            "source_file": source_path.name,
                            "original_sample_count": original_count,
                            "stored_sample_count_before_finite_filter": stored_count,
                            "finite_sample_count": int(values.size),
                            "sampling_method": sampling_method,
                            "extraction_seed": EXTRACTION_SEED,
                        }
                    )
    return samples, preflight_rows


def calculate_locked_result(
    sample_rows: list[dict[str, str]],
    samples: dict[tuple[str, str, str], np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Calculate the two NW1 values per event and the sole locked hypothesis test."""

    parameter_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for sample_row in sample_rows:
        event = sample_row["event"]
        event_nw1: dict[str, float] = {}
        for parameter in PARAMETERS:
            x = samples[(event, XPHM, parameter)]
            y = samples[(event, XPNR, parameter)]
            metrics = pair_metrics(x, y)
            event_nw1[parameter] = metrics["normalized_wasserstein_1"]
            parameter_rows.append(
                {
                    "event": event,
                    "cell_id": sample_row["cell_id"],
                    "parameter": parameter,
                    "xphm_finite_sample_count": int(x.size),
                    "xpnr_finite_sample_count": int(y.size),
                    **metrics,
                }
            )

        delta = event_nw1["chi_eff"] - event_nw1["luminosity_distance"]
        event_rows.append(
            {
                "event": event,
                "cell_id": sample_row["cell_id"],
                "score_stratum": sample_row["score_stratum"],
                "snr_stratum": sample_row["snr_stratum"],
                "chi_eff_normalized_wasserstein_1": event_nw1["chi_eff"],
                "luminosity_distance_normalized_wasserstein_1": event_nw1[
                    "luminosity_distance"
                ],
                "delta_chi_eff_minus_luminosity_distance": delta,
                "delta_positive": delta > 0,
            }
        )

    chi_eff = np.asarray(
        [row["chi_eff_normalized_wasserstein_1"] for row in event_rows],
        dtype=float,
    )
    luminosity_distance = np.asarray(
        [row["luminosity_distance_normalized_wasserstein_1"] for row in event_rows],
        dtype=float,
    )
    deltas = chi_eff - luminosity_distance

    # This call is copied exactly from the parent H3 analysis.
    result = wilcoxon(
        chi_eff,
        luminosity_distance,
        alternative="greater",
        zero_method="wilcox",
        method="auto",
    )
    hypothesis = {
        "hypothesis": "prospective_H3_replication",
        "test": "one-sided Wilcoxon signed-rank",
        "alternative": "chi_eff NW1 > luminosity_distance NW1",
        "zero_method": "wilcox",
        "method_requested": "auto",
        "alpha": ALPHA,
        "event_count": len(event_rows),
        "positive_delta_count": int(np.sum(deltas > 0)),
        "zero_delta_count": int(np.sum(deltas == 0)),
        "negative_delta_count": int(np.sum(deltas < 0)),
        "median_delta": float(np.median(deltas)),
        "wilcoxon_statistic": float(result.statistic),
        "raw_p_value": float(result.pvalue),
        "reject_at_alpha_0_05": bool(result.pvalue < ALPHA),
        "multiplicity_adjustment": "none; one locked hypothesis",
    }
    return parameter_rows, event_rows, hypothesis


def write_results(
    output_dir: Path,
    integrity_rows: list[dict[str, Any]],
    preflight_rows: list[dict[str, Any]],
    parameter_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    hypothesis: dict[str, Any],
) -> None:
    """Write the prespecified audit trail and result tables."""

    write_csv(
        output_dir / "replication_input_integrity.csv",
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
        output_dir / "replication_preflight_samples.csv",
        preflight_rows,
        [
            "event",
            "cell_id",
            "model",
            "parameter",
            "source_file",
            "original_sample_count",
            "stored_sample_count_before_finite_filter",
            "finite_sample_count",
            "sampling_method",
            "extraction_seed",
        ],
    )
    write_csv(
        output_dir / "replication_parameter_metrics.csv",
        parameter_rows,
        [
            "event",
            "cell_id",
            "parameter",
            "xphm_finite_sample_count",
            "xpnr_finite_sample_count",
            "xphm_lower_90",
            "xphm_median",
            "xphm_upper_90",
            "xphm_width_90",
            "xpnr_lower_90",
            "xpnr_median",
            "xpnr_upper_90",
            "xpnr_width_90",
            "average_width_90",
            "wasserstein_1",
            "normalized_wasserstein_1",
        ],
    )
    write_csv(
        output_dir / "replication_event_deltas.csv",
        event_rows,
        [
            "event",
            "cell_id",
            "score_stratum",
            "snr_stratum",
            "chi_eff_normalized_wasserstein_1",
            "luminosity_distance_normalized_wasserstein_1",
            "delta_chi_eff_minus_luminosity_distance",
            "delta_positive",
        ],
    )
    write_csv(
        output_dir / "replication_hypothesis_result.csv",
        [hypothesis],
        list(hypothesis),
    )


def build_manifest(
    output_dir: Path,
    input_paths: list[Path],
    integrity_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Record code, environment, inputs, and result hashes after the run."""

    result_paths = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "replication_analysis_manifest.json"
    )
    return {
        "analysis": "prospective within-GWTC-5 H3 replication",
        "scope": {
            "models": list(MODELS),
            "parameters": list(PARAMETERS),
            "metric": "NW1 = W1 / mean(XPHM width90, XPNR width90)",
            "test": {
                "function": "scipy.stats.wilcoxon",
                "alternative": "greater",
                "zero_method": "wilcox",
                "method": "auto",
                "alpha": ALPHA,
            },
            "exploratory_outputs_created": False,
        },
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "h5py": h5py.__version__,
            "hdf5": h5py.version.hdf5_version,
        },
        "analysis_code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "locked_input_hashes": {
            str(path.resolve()): sha256_file(path) for path in input_paths
        },
        "raw_input_md5": {
            row["event"]: row["actual_md5"] for row in integrity_rows
        },
        "result_hashes": {
            path.name: sha256_file(path) for path in result_paths
        },
    }


def main() -> None:
    """Verify the lock, open the outcomes once, and write the frozen result."""

    args = parse_args()
    ensure_clean_output(args.output_dir, args.overwrite)

    sample_rows, _ = load_and_verify_final_sample(args.final_sample, args.sample_lock)
    event_paths, integrity_rows = load_verified_inputs(
        sample_rows,
        (args.verification, args.replacement_verification),
    )
    samples, preflight_rows = load_locked_samples(sample_rows, event_paths)
    parameter_rows, event_rows, hypothesis = calculate_locked_result(
        sample_rows, samples
    )
    write_results(
        args.output_dir,
        integrity_rows,
        preflight_rows,
        parameter_rows,
        event_rows,
        hypothesis,
    )

    manifest = build_manifest(
        args.output_dir,
        [
            args.final_sample,
            args.sample_lock,
            args.verification,
            args.replacement_verification,
        ],
        integrity_rows,
    )
    (args.output_dir / "replication_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(hypothesis, indent=2))


if __name__ == "__main__":
    main()
