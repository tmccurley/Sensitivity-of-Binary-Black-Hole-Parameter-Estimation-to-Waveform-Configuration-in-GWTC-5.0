"""Run exploratory robustness checks for the GWTC-5 confirmatory results.

This script consumes the locked sample, compact posteriors, and outputs of the
frozen confirmatory analysis. It evaluates leave-one-event-out stability,
alternative endpoints and normalizations, agreement among disagreement
metrics, H3 resampling checks, JSD bin sensitivity, and descriptive results for
events excluded during sample assembly.

The robustness outputs are written separately under
``results/robustness_analysis``. Hash and count checks ensure that these checks
are tied to the same inputs used by the confirmatory run; they do not modify or
replace the confirmatory results.
"""

#Robustness analysis

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, spearmanr, wasserstein_distance, wilcoxon


# Model labels and parameter sets mirror the confirmatory analysis so all
# robustness comparisons address the same posterior quantities.
XPHM = "C00:IMRPhenomXPHM-SpinTaylor"
XPNR = "C00:IMRPhenomXPNR"
MODELS = [XPHM, XPNR]

PRIMARY_PARAMETERS = [
    "chirp_mass",
    "mass_ratio",
    "chi_eff",
    "luminosity_distance",
]
SCREENING_PARAMETERS = [
    "chirp_mass_source",
    "chi_eff",
    "luminosity_distance",
]
ALL_PARAMETERS = [
    "chirp_mass",
    "mass_ratio",
    "chi_eff",
    "luminosity_distance",
    "chirp_mass_source",
]

# Dedicated robustness seeds keep exploratory random streams independent of
# the frozen confirmatory seeds while making every rerun deterministic.
ROBUSTNESS_PERMUTATION_SEED = 20260804
ROBUSTNESS_BOOTSTRAP_SEED = 20260805
ROBUSTNESS_SPLIT_SEED = 20260806
N_PERMUTATIONS = 100_000
N_BOOTSTRAPS = 100_000
N_SPLITS_EXCLUDED = 100
JSD_BIN_COUNTS = [30, 60, 120]
ALPHA = 0.05


@dataclass(frozen=True)
class RobustnessPaths:
    """Collect confirmatory inputs and the separate robustness output paths."""

    project_root: Path
    processed_dir: Path
    design_dir: Path
    final_assembly_dir: Path
    confirmatory_dir: Path
    lock_file: Path
    queue_file: Path
    replacement_history_file: Path
    confirmatory_manifest_file: Path
    parameter_metrics_file: Path
    event_endpoints_file: Path
    input_file_manifest_file: Path
    output_dir: Path


def find_project_root(start: Path | None = None) -> Path:
    """Locate a project containing both processed data and confirmatory results."""

    env_root = os.environ.get("GWTC5_PROJECT_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if (root / "results" / "confirmatory_analysis").exists():
            return root
        raise FileNotFoundError(
            "GWTC5_PROJECT_ROOT does not contain results/confirmatory_analysis: "
            f"{root}"
        )

    start = Path(start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (
            (candidate / "results" / "confirmatory_analysis").exists()
            and (candidate / "data" / "processed").exists()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate the GWTC5 project root. Set GWTC5_PROJECT_ROOT "
        "or run inside the project."
    )


def resolve_paths(project_root: Path) -> RobustnessPaths:
    """Build and validate every path required for the robustness workflow."""

    processed_dir = project_root / "data" / "processed"
    design_dir = processed_dir / "confirmatory_design"
    final_assembly_dir = project_root / "results" / "confirmatory_final_assembly"
    confirmatory_dir = project_root / "results" / "confirmatory_analysis"
    output_dir = project_root / "results" / "robustness_analysis"

    paths = RobustnessPaths(
        project_root=project_root,
        processed_dir=processed_dir,
        design_dir=design_dir,
        final_assembly_dir=final_assembly_dir,
        confirmatory_dir=confirmatory_dir,
        lock_file=final_assembly_dir / "confirmatory_final_sample_lock.json",
        queue_file=design_dir / "confirmatory_randomization_queue.csv",
        replacement_history_file=(
            final_assembly_dir / "confirmatory_complete_replacement_history.csv"
        ),
        confirmatory_manifest_file=(
            confirmatory_dir / "confirmatory_analysis_manifest.json"
        ),
        parameter_metrics_file=(
            confirmatory_dir / "confirmatory_parameter_metrics.csv"
        ),
        event_endpoints_file=(
            confirmatory_dir / "confirmatory_event_endpoints.csv"
        ),
        input_file_manifest_file=(
            confirmatory_dir / "confirmatory_input_file_manifest.csv"
        ),
        output_dir=output_dir,
    )

    required = [
        paths.processed_dir,
        paths.design_dir,
        paths.lock_file,
        paths.queue_file,
        paths.replacement_history_file,
        paths.confirmatory_manifest_file,
        paths.parameter_metrics_file,
        paths.event_endpoints_file,
        paths.input_file_manifest_file,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required robustness-analysis inputs:\n"
            + "\n".join(str(path) for path in missing)
        )

    return paths


def ensure_clean_output(output_dir: Path, overwrite: bool) -> None:
    """Prevent new robustness files from being mixed with an earlier run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [path for path in output_dir.iterdir() if path.is_file()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}\n"
            "Use --overwrite only for a documented full robustness rerun."
        )
    if overwrite:
        for path in existing:
            path.unlink()


def write_log(log_file: Path, message: str = "") -> None:
    """Write a progress message to both standard output and the run log."""

    print(message)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for provenance verification."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_and_verify_confirmatory_inputs(
    paths: RobustnessPaths,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load confirmatory products and verify counts, event sets, and hashes.

    These checks bind the exploratory analysis to the locked 18-event run
    and stop execution if the lock, queue, metrics, or endpoints are
    inconsistent with the confirmatory manifest.
    """

    manifest = json.loads(
        paths.confirmatory_manifest_file.read_text(encoding="utf-8")
    )
    lock = json.loads(paths.lock_file.read_text(encoding="utf-8"))
    metrics = pd.read_csv(paths.parameter_metrics_file)
    endpoints = pd.read_csv(paths.event_endpoints_file)
    input_manifest = pd.read_csv(paths.input_file_manifest_file)
    queue = pd.read_csv(paths.queue_file)

    if not lock.get("confirmatory_sample_locked", False):
        raise RuntimeError("The confirmatory sample lock is not active.")
    if int(lock.get("strict_event_count", -1)) != 18:
        raise RuntimeError("The sample lock does not specify 18 events.")
    if int(manifest.get("strict_event_count", -1)) != 18:
        raise RuntimeError("The confirmatory manifest does not specify 18 events.")
    if int(manifest.get("counts", {}).get("full_posterior_metric_rows", -1)) != 90:
        raise RuntimeError("The confirmatory manifest does not specify 90 metric rows.")
    if len(metrics) != 90:
        raise RuntimeError(f"Expected 90 metric rows, found {len(metrics)}.")
    if len(endpoints) != 18 or endpoints["event"].nunique() != 18:
        raise RuntimeError("Expected 18 unique event endpoints.")

    locked_events = [row["event"] for row in lock.get("events", [])]
    if set(locked_events) != set(endpoints["event"]):
        raise RuntimeError("Event endpoints do not match the locked event set.")
    if set(metrics["event"]) != set(locked_events):
        raise RuntimeError("Parameter metrics do not match the locked event set.")

    # Hash comparisons detect changed design inputs even when filenames and
    # table dimensions still look correct.
    expected_lock_hash = manifest.get("sample_lock_sha256")
    actual_lock_hash = sha256_file(paths.lock_file)
    if expected_lock_hash and expected_lock_hash != actual_lock_hash:
        raise RuntimeError(
            "The current sample lock hash differs from the confirmatory run."
        )

    expected_queue_hash = manifest.get("queue_sha256")
    actual_queue_hash = sha256_file(paths.queue_file)
    if expected_queue_hash and expected_queue_hash != actual_queue_hash:
        raise RuntimeError(
            "The current randomization queue hash differs from the confirmatory run."
        )

    return manifest, metrics, endpoints, input_manifest, queue


def locate_confirmatory_posterior_files(
    processed_dir: Path,
    input_manifest: pd.DataFrame,
) -> list[Path]:
    """Find and hash-check the exact compact files listed by confirmation."""

    resolved: list[Path] = []

    for row in input_manifest.itertuples(index=False):
        filename = str(row.filename)
        matches = sorted(processed_dir.rglob(filename))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one compact posterior file named {filename}; "
                f"found {len(matches)} under {processed_dir}."
            )
        path = matches[0]
        expected_hash = str(row.sha256)
        actual_hash = sha256_file(path)
        if expected_hash != actual_hash:
            raise RuntimeError(
                f"Hash mismatch for {path}:\n"
                f"expected {expected_hash}\nactual   {actual_hash}"
            )
        resolved.append(path)

    return resolved


def index_events_in_files(files: Iterable[Path]) -> dict[str, Path]:
    """Map each stored event to exactly one confirmed posterior file."""

    event_to_file: dict[str, Path] = {}
    for path in files:
        with h5py.File(path, "r") as source:
            for event in source.keys():
                if event in event_to_file:
                    raise RuntimeError(
                        f"Event {event} appears in multiple confirmatory files."
                    )
                event_to_file[event] = path
    return event_to_file


def find_model_group(event_group: h5py.Group, model_label: str) -> h5py.Group:
    """Resolve a waveform-model group from its original or sanitized label."""

    for key in event_group.keys():
        group = event_group[key]
        original_label = group.attrs.get("original_label", "")
        if isinstance(original_label, bytes):
            original_label = original_label.decode("utf-8", errors="replace")
        if str(original_label) == model_label:
            return group

    safe_name = (
        model_label.replace("\\", "_")
        .replace("/", "_")
        .replace(":", "__")
        .replace(" ", "_")
    )
    if safe_name in event_group:
        return event_group[safe_name]
    raise KeyError(f"Missing model group {model_label}")


def load_samples(
    event_to_file: dict[str, Path],
    events: list[str],
    parameters: list[str] = ALL_PARAMETERS,
) -> dict[tuple[str, str, str], np.ndarray]:
    """Load finite posterior arrays keyed by event, model, and parameter."""

    samples: dict[tuple[str, str, str], np.ndarray] = {}
    handles: dict[Path, h5py.File] = {}
    try:
        for event in events:
            if event not in event_to_file:
                raise FileNotFoundError(
                    f"Event {event} is absent from the confirmed compact files."
                )
            path = event_to_file[event]
            source = handles.setdefault(path, h5py.File(path, "r"))
            event_group = source[event]
            for model in MODELS:
                model_group = find_model_group(event_group, model)
                for parameter in parameters:
                    if parameter not in model_group:
                        raise KeyError(f"Missing {event}/{model}/{parameter}")
                    values = np.asarray(
                        model_group[parameter][:], dtype=np.float64
                    ).reshape(-1)
                    values = values[np.isfinite(values)]
                    if values.size < 4:
                        raise RuntimeError(
                            f"Too few finite samples for {event}/{model}/{parameter}"
                        )
                    samples[(event, model, parameter)] = values
    finally:
        for handle in handles.values():
            handle.close()
    return samples


def interval_summary(values: np.ndarray) -> dict[str, float]:
    """Return equal-tailed 90% bounds, width, median, and sample variance."""

    lower, median, upper = np.quantile(values, [0.05, 0.50, 0.95])
    return {
        "lower_90": float(lower),
        "median": float(median),
        "upper_90": float(upper),
        "width_90": float(upper - lower),
        "variance": float(np.var(values, ddof=1)),
    }


def js_divergence_bits(x: np.ndarray, y: np.ndarray, bins: int) -> float:
    """Estimate base-2 Jensen-Shannon divergence on a shared pooled grid."""

    pooled_min = float(min(np.min(x), np.min(y)))
    pooled_max = float(max(np.max(x), np.max(y)))
    if pooled_min == pooled_max:
        return 0.0
    edges = np.linspace(pooled_min, pooled_max, bins + 1)
    px = np.histogram(x, bins=edges)[0].astype(np.float64)
    py = np.histogram(y, bins=edges)[0].astype(np.float64)
    px /= px.sum()
    py /= py.sum()
    midpoint = 0.5 * (px + py)

    def kl_bits(p: np.ndarray, q: np.ndarray) -> float:
        """Compute the nonzero-bin contribution to KL divergence in bits."""

        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

    return 0.5 * kl_bits(px, midpoint) + 0.5 * kl_bits(py, midpoint)


def pair_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Compute primary and alternative disagreement metrics for a model pair.

    Alongside the frozen arithmetic-width normalization, this exploratory
    version reports geometric- and maximum-width normalizations for
    sensitivity comparisons.
    """

    sx = interval_summary(x)
    sy = interval_summary(y)
    raw_w1 = float(wasserstein_distance(x, y))
    average_width = 0.5 * (sx["width_90"] + sy["width_90"])
    geometric_width = math.sqrt(sx["width_90"] * sy["width_90"])
    maximum_width = max(sx["width_90"], sy["width_90"])
    median_denominator = math.sqrt(sx["variance"] + sy["variance"])

    intersection = max(
        0.0,
        min(sx["upper_90"], sy["upper_90"])
        - max(sx["lower_90"], sy["lower_90"]),
    )
    union = (
        max(sx["upper_90"], sy["upper_90"])
        - min(sx["lower_90"], sy["lower_90"])
    )

    return {
        "xphm_width_90": sx["width_90"],
        "xpnr_width_90": sy["width_90"],
        "wasserstein_1": raw_w1,
        "normalized_wasserstein_1": (
            raw_w1 / average_width if average_width > 0 else math.nan
        ),
        "normalized_w1_geometric_width": (
            raw_w1 / geometric_width if geometric_width > 0 else math.nan
        ),
        "normalized_w1_max_width": (
            raw_w1 / maximum_width if maximum_width > 0 else math.nan
        ),
        "js_divergence_bits": js_divergence_bits(x, y, bins=60),
        "standardized_median_displacement": (
            abs(sx["median"] - sy["median"]) / median_denominator
            if median_denominator > 0
            else math.nan
        ),
        "interval_overlap_90": intersection / union if union > 0 else 1.0,
        "log_width_ratio": (
            math.log(sx["width_90"] / sy["width_90"])
            if sx["width_90"] > 0 and sy["width_90"] > 0
            else math.nan
        ),
    }


def equal_size_w1(x: np.ndarray, y: np.ndarray) -> float:
    """Compute empirical one-dimensional W1 for equal-length arrays."""

    if x.size != y.size:
        raise ValueError("equal_size_w1 requires equal sample sizes")
    return float(np.mean(np.abs(np.sort(x) - np.sort(y))))


def split_normalized_w1(first: np.ndarray, second: np.ndarray) -> float:
    """Compute normalized W1 between two equal posterior halves."""

    s1 = interval_summary(first)
    s2 = interval_summary(second)
    average_width = 0.5 * (s1["width_90"] + s2["width_90"])
    return equal_size_w1(first, second) / average_width


def compute_excluded_noise_thresholds(
    samples: dict[tuple[str, str, str], np.ndarray],
    events: list[str],
) -> pd.DataFrame:
    """Recreate split-sample noise thresholds for excluded events.

    The same fixed-seed, 100-split construction is used descriptively so
    excluded and locked events can be summarized on a comparable scale.
    """

    rng = np.random.default_rng(ROBUSTNESS_SPLIT_SEED)
    rows: list[dict] = []
    for event in events:
        for parameter in ALL_PARAMETERS:
            model_p95: dict[str, float] = {}
            for model in MODELS:
                values = samples[(event, model, parameter)]
                half_size = values.size // 2
                replicates: list[float] = []
                for _ in range(N_SPLITS_EXCLUDED):
                    order = rng.permutation(values.size)
                    first = values[order[:half_size]]
                    second = values[order[half_size : 2 * half_size]]
                    replicates.append(split_normalized_w1(first, second))
                model_p95[model] = float(np.quantile(replicates, 0.95))
            rows.append(
                {
                    "event": event,
                    "parameter": parameter,
                    "xphm_split_NW1_p95": model_p95[XPHM],
                    "xpnr_split_NW1_p95": model_p95[XPNR],
                    "pair_noise_threshold_NW1": max(model_p95.values()),
                }
            )
    return pd.DataFrame(rows)


def spearman_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_permutations: int = N_PERMUTATIONS,
    chunk_size: int = 5000,
) -> tuple[float, float]:
    """Run a chunked one-sided rank-correlation permutation test.

    The plus-one numerator and denominator correction yields a valid
    finite Monte Carlo p-value even when no permutation is as extreme as
    the observed statistic.
    """

    if x.size != y.size:
        raise ValueError("x and y must have equal length")
    xr = rankdata(x).astype(np.float64)
    yr = rankdata(y).astype(np.float64)
    xc = xr - xr.mean()
    yc = yr - yr.mean()
    denominator = math.sqrt(float(np.dot(xc, xc) * np.dot(yc, yc)))
    if denominator == 0:
        raise RuntimeError("Spearman statistic is undefined for constant input")
    observed = float(np.dot(xc, yc) / denominator)

    exceedances = 0
    completed = 0
    n = x.size
    while completed < n_permutations:
        current = min(chunk_size, n_permutations - completed)
        order = np.argsort(rng.random((current, n)), axis=1)
        statistics = yc[order] @ xc / denominator
        exceedances += int(np.count_nonzero(statistics >= observed))
        completed += current

    return observed, float((exceedances + 1) / (n_permutations + 1))


def leave_one_out_analysis(
    endpoints: pd.DataFrame,
    metrics: pd.DataFrame,
    rng_seed_sequence: np.random.SeedSequence,
) -> pd.DataFrame:
    """Recompute H1--H3 after omitting each locked event in turn.

    Child seed sequences give every H1 and H2 rerun a deterministic,
    independent random stream. The results reveal whether conclusions are
    driven by any single event.
    """

    event_order = endpoints["event"].tolist()
    h3_wide = metrics[
        metrics["parameter"].isin(["chi_eff", "luminosity_distance"])
    ].pivot(
        index="event",
        columns="parameter",
        values="normalized_wasserstein_1",
    ).reindex(event_order)

    children = rng_seed_sequence.spawn(len(event_order) * 2)
    rows: list[dict] = []

    for index, omitted in enumerate(event_order):
        reduced = endpoints[endpoints["event"] != omitted]
        h1_rng = np.random.default_rng(children[2 * index])
        h2_rng = np.random.default_rng(children[2 * index + 1])

        h1_rho, h1_p = spearman_permutation_test(
            reduced["network_matched_filter_snr"].to_numpy(float),
            reduced["event_median_NW1"].to_numpy(float),
            h1_rng,
        )
        h2_rho, h2_p = spearman_permutation_test(
            reduced["maximum_screening_shift"].to_numpy(float),
            reduced["screening_max_NW1"].to_numpy(float),
            h2_rng,
        )

        reduced_h3 = h3_wide.drop(index=omitted)
        chi = reduced_h3["chi_eff"].to_numpy(float)
        distance = reduced_h3["luminosity_distance"].to_numpy(float)
        h3 = wilcoxon(
            chi,
            distance,
            alternative="greater",
            zero_method="wilcox",
            method="auto",
        )

        rows.extend(
            [
                {
                    "omitted_event": omitted,
                    "hypothesis": "H1",
                    "effect_statistic": h1_rho,
                    "raw_p_value": h1_p,
                    "reject_0_05": bool(h1_p < ALPHA),
                    "effect_description": "Spearman rho",
                },
                {
                    "omitted_event": omitted,
                    "hypothesis": "H2",
                    "effect_statistic": h2_rho,
                    "raw_p_value": h2_p,
                    "reject_0_05": bool(h2_p < ALPHA),
                    "effect_description": "Spearman rho",
                },
                {
                    "omitted_event": omitted,
                    "hypothesis": "H3",
                    "effect_statistic": float(h3.statistic),
                    "raw_p_value": float(h3.pvalue),
                    "reject_0_05": bool(h3.pvalue < ALPHA),
                    "effect_description": "Wilcoxon W",
                    "paired_median_difference": float(np.median(chi - distance)),
                },
            ]
        )

    return pd.DataFrame(rows)


def summarize_leave_one_out(loo: pd.DataFrame) -> pd.DataFrame:
    """Condense leave-one-out effects, p-values, and rejection counts by hypothesis."""

    rows: list[dict] = []
    for hypothesis, group in loo.groupby("hypothesis", sort=False):
        rows.append(
            {
                "hypothesis": hypothesis,
                "leave_one_out_runs": len(group),
                "effect_min": float(group["effect_statistic"].min()),
                "effect_median": float(group["effect_statistic"].median()),
                "effect_max": float(group["effect_statistic"].max()),
                "p_value_min": float(group["raw_p_value"].min()),
                "p_value_median": float(group["raw_p_value"].median()),
                "p_value_max": float(group["raw_p_value"].max()),
                "significant_runs": int(group["reject_0_05"].sum()),
                "significant_fraction": float(group["reject_0_05"].mean()),
            }
        )
    return pd.DataFrame(rows)


def alternative_endpoint_analysis(
    endpoints: pd.DataFrame,
    metrics: pd.DataFrame,
    seed_sequence: np.random.SeedSequence,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test H1 and H2 against alternative event-level summaries.

    These exploratory endpoints vary aggregation and W1 normalization to
    show how strongly the rank association depends on the frozen endpoint
    definition.
    """

    # Recalculate alternative normalizations from the preserved raw W1 and
    # interval widths rather than altering the frozen confirmatory column.
    primary = metrics[metrics["parameter"].isin(PRIMARY_PARAMETERS)].copy()
    primary["normalized_w1_geometric_width"] = (
        primary["wasserstein_1"]
        / np.sqrt(primary["xphm_width_90"] * primary["xpnr_width_90"])
    )
    primary["normalized_w1_max_width"] = (
        primary["wasserstein_1"]
        / primary[["xphm_width_90", "xpnr_width_90"]].max(axis=1)
    )

    grouped = primary.groupby("event", sort=False)
    summaries = pd.DataFrame(
        {
            "event": grouped.size().index,
            "median_NW1": grouped["normalized_wasserstein_1"].median(),
            "mean_NW1": grouped["normalized_wasserstein_1"].mean(),
            "q75_NW1": grouped["normalized_wasserstein_1"].quantile(0.75),
            "max_NW1": grouped["normalized_wasserstein_1"].max(),
            "median_NW1_geometric_width": grouped[
                "normalized_w1_geometric_width"
            ].median(),
            "median_NW1_max_width": grouped[
                "normalized_w1_max_width"
            ].median(),
        }
    ).reset_index(drop=True)

    merged = endpoints[["event", "network_matched_filter_snr"]].merge(
        summaries, on="event", how="left", validate="one_to_one"
    )

    endpoint_names = [
        "median_NW1",
        "mean_NW1",
        "q75_NW1",
        "max_NW1",
        "median_NW1_geometric_width",
        "median_NW1_max_width",
    ]
    children = seed_sequence.spawn(len(endpoint_names))
    rows: list[dict] = []
    for name, child in zip(endpoint_names, children):
        rho, p_value = spearman_permutation_test(
            merged["network_matched_filter_snr"].to_numpy(float),
            merged[name].to_numpy(float),
            np.random.default_rng(child),
        )
        rows.append(
            {
                "analysis_family": "H1_alternative_event_endpoint",
                "endpoint": name,
                "spearman_rho": rho,
                "one_sided_permutation_p": p_value,
                "reject_unadjusted_0_05": bool(p_value < ALPHA),
                "permutations": N_PERMUTATIONS,
            }
        )

    screening = metrics[
        metrics["parameter"].isin(SCREENING_PARAMETERS)
    ].groupby("event", sort=False)["normalized_wasserstein_1"].agg(
        screening_min_NW1="min",
        screening_median_NW1="median",
        screening_mean_NW1="mean",
        screening_max_NW1="max",
    ).reset_index()

    screening_merged = endpoints[
        ["event", "maximum_screening_shift"]
    ].merge(screening, on="event", how="left", validate="one_to_one")

    screening_names = [
        "screening_min_NW1",
        "screening_median_NW1",
        "screening_mean_NW1",
        "screening_max_NW1",
    ]
    screen_children = seed_sequence.spawn(len(screening_names))
    screening_rows: list[dict] = []
    for name, child in zip(screening_names, screen_children):
        rho, p_value = spearman_permutation_test(
            screening_merged["maximum_screening_shift"].to_numpy(float),
            screening_merged[name].to_numpy(float),
            np.random.default_rng(child),
        )
        screening_rows.append(
            {
                "analysis_family": "H2_alternative_screening_endpoint",
                "endpoint": name,
                "spearman_rho": rho,
                "one_sided_permutation_p": p_value,
                "reject_unadjusted_0_05": bool(p_value < ALPHA),
                "permutations": N_PERMUTATIONS,
            }
        )

    return pd.DataFrame(rows + screening_rows), merged


def cross_metric_correlations(metrics: pd.DataFrame) -> pd.DataFrame:
    """Measure rank agreement among complementary disagreement metrics.

    Correlations are reported overall, for the four primary parameters,
    and separately by parameter; two-sided asymptotic p-values are
    descriptive rather than replacements for confirmatory tests.
    """

    # Orient all metrics so larger values consistently mean more disagreement.
    transformed = metrics.copy()
    transformed["one_minus_interval_overlap_90"] = (
        1.0 - transformed["interval_overlap_90"]
    )
    transformed["absolute_log_width_ratio"] = transformed[
        "log_width_ratio"
    ].abs()

    metric_columns = [
        "normalized_wasserstein_1",
        "js_divergence_bits",
        "standardized_median_displacement",
        "one_minus_interval_overlap_90",
        "absolute_log_width_ratio",
    ]

    scopes: list[tuple[str, pd.DataFrame]] = [
        ("all_five_parameters", transformed),
        (
            "primary_four_parameters",
            transformed[transformed["parameter"].isin(PRIMARY_PARAMETERS)],
        ),
    ]
    scopes.extend(
        (f"parameter:{parameter}", transformed[transformed["parameter"] == parameter])
        for parameter in ALL_PARAMETERS
    )

    rows: list[dict] = []
    for scope_name, frame in scopes:
        for metric_a in metric_columns:
            for metric_b in metric_columns:
                result = spearmanr(
                    frame[metric_a].to_numpy(float),
                    frame[metric_b].to_numpy(float),
                )
                rows.append(
                    {
                        "scope": scope_name,
                        "metric_a": metric_a,
                        "metric_b": metric_b,
                        "spearman_rho": float(result.statistic),
                        "two_sided_asymptotic_p": float(result.pvalue),
                        "n_pairs": len(frame),
                    }
                )
    return pd.DataFrame(rows)


def parameter_and_noise_summaries(
    metrics: pd.DataFrame,
    endpoints: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build parameter summaries, event rankings, and a noise-ratio matrix."""

    frame = metrics.copy()
    frame["noise_ratio"] = (
        frame["normalized_wasserstein_1"]
        / frame["pair_noise_threshold_NW1"]
    )

    parameter_summary = (
        frame.groupby("parameter", sort=False)
        .agg(
            event_count=("event", "nunique"),
            median_NW1=("normalized_wasserstein_1", "median"),
            q25_NW1=("normalized_wasserstein_1", lambda x: x.quantile(0.25)),
            q75_NW1=("normalized_wasserstein_1", lambda x: x.quantile(0.75)),
            maximum_NW1=("normalized_wasserstein_1", "max"),
            median_noise_ratio=("noise_ratio", "median"),
            minimum_noise_ratio=("noise_ratio", "min"),
            events_above_noise=("NW1_above_noise", "sum"),
            fraction_above_noise=("NW1_above_noise", "mean"),
        )
        .reset_index()
    )

    primary = frame[frame["parameter"].isin(PRIMARY_PARAMETERS)]
    event_summary = (
        primary.groupby("event", sort=False)
        .agg(
            event_median_NW1=("normalized_wasserstein_1", "median"),
            event_mean_NW1=("normalized_wasserstein_1", "mean"),
            event_max_NW1=("normalized_wasserstein_1", "max"),
            event_median_noise_ratio=("noise_ratio", "median"),
            event_minimum_noise_ratio=("noise_ratio", "min"),
            primary_parameters_above_noise=("NW1_above_noise", "sum"),
        )
        .reset_index()
    )

    dominant = primary.loc[
        primary.groupby("event")["normalized_wasserstein_1"].idxmax(),
        ["event", "parameter", "normalized_wasserstein_1"],
    ].rename(
        columns={
            "parameter": "dominant_primary_parameter",
            "normalized_wasserstein_1": "dominant_parameter_NW1",
        }
    )

    event_summary = (
        event_summary.merge(dominant, on="event", validate="one_to_one")
        .merge(
            endpoints[
                [
                    "event",
                    "batch",
                    "network_matched_filter_snr",
                    "maximum_screening_shift",
                    "total_mass_source",
                ]
            ],
            on="event",
            validate="one_to_one",
        )
        .sort_values("event_median_NW1", ascending=False)
        .reset_index(drop=True)
    )
    event_summary.insert(0, "event_median_rank", np.arange(1, len(event_summary) + 1))

    noise_matrix = primary.pivot(
        index="event",
        columns="parameter",
        values="noise_ratio",
    ).reindex(endpoints["event"])[PRIMARY_PARAMETERS]
    noise_matrix = noise_matrix.reset_index()

    return parameter_summary, event_summary, noise_matrix


def h3_additional_robustness(
    metrics: pd.DataFrame,
    bootstrap_rng: np.random.Generator,
) -> pd.DataFrame:
    """Add sign-test and bootstrap summaries for the paired H3 contrast.

    Resampling is performed at the event level, preserving the paired
    ``chi_eff`` versus distance comparison within every bootstrap draw.
    """

    wide = metrics[
        metrics["parameter"].isin(["chi_eff", "luminosity_distance"])
    ].pivot(
        index="event",
        columns="parameter",
        values="normalized_wasserstein_1",
    )
    chi = wide["chi_eff"].to_numpy(float)
    distance = wide["luminosity_distance"].to_numpy(float)
    difference = chi - distance
    ratio = chi / distance

    n = difference.size
    # Sample event indices with replacement; both parameter values follow the
    # same indices, preserving the paired H3 structure.
    indices = bootstrap_rng.integers(0, n, size=(N_BOOTSTRAPS, n))
    boot_median_difference = np.median(difference[indices], axis=1)
    boot_median_ratio = np.median(ratio[indices], axis=1)

    sign = binomtest(
        int(np.count_nonzero(difference > 0)),
        n=n,
        p=0.5,
        alternative="greater",
    )

    return pd.DataFrame(
        [
            {
                "analysis": "paired_sign_test",
                "event_count": n,
                "positive_differences": int(np.count_nonzero(difference > 0)),
                "zero_differences": int(np.count_nonzero(difference == 0)),
                "negative_differences": int(np.count_nonzero(difference < 0)),
                "effect_value": float(np.median(difference)),
                "effect_label": "median chi_eff minus distance NW1",
                "lower_95": math.nan,
                "upper_95": math.nan,
                "p_value": float(sign.pvalue),
            },
            {
                "analysis": "bootstrap_median_difference",
                "event_count": n,
                "positive_differences": int(np.count_nonzero(difference > 0)),
                "zero_differences": int(np.count_nonzero(difference == 0)),
                "negative_differences": int(np.count_nonzero(difference < 0)),
                "effect_value": float(np.median(difference)),
                "effect_label": "median chi_eff minus distance NW1",
                "lower_95": float(np.quantile(boot_median_difference, 0.025)),
                "upper_95": float(np.quantile(boot_median_difference, 0.975)),
                "p_value": math.nan,
            },
            {
                "analysis": "bootstrap_median_ratio",
                "event_count": n,
                "positive_differences": int(np.count_nonzero(difference > 0)),
                "zero_differences": int(np.count_nonzero(difference == 0)),
                "negative_differences": int(np.count_nonzero(difference < 0)),
                "effect_value": float(np.median(ratio)),
                "effect_label": "median chi_eff divided by distance NW1",
                "lower_95": float(np.quantile(boot_median_ratio, 0.025)),
                "upper_95": float(np.quantile(boot_median_ratio, 0.975)),
                "p_value": math.nan,
            },
        ]
    )


def jsd_bin_sensitivity(
    samples: dict[tuple[str, str, str], np.ndarray],
    events: list[str],
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute JSD at 30, 60, and 120 bins and compare rank orderings."""

    rows: list[dict] = []
    for event in events:
        for parameter in ALL_PARAMETERS:
            x = samples[(event, XPHM, parameter)]
            y = samples[(event, XPNR, parameter)]
            row = {"event": event, "parameter": parameter}
            for bins in JSD_BIN_COUNTS:
                row[f"jsd_bits_{bins}_bins"] = js_divergence_bits(x, y, bins)
            rows.append(row)
    sensitivity = pd.DataFrame(rows).merge(
        metrics[["event", "parameter", "normalized_wasserstein_1"]],
        on=["event", "parameter"],
        validate="one_to_one",
    )

    variables = [
        "jsd_bits_30_bins",
        "jsd_bits_60_bins",
        "jsd_bits_120_bins",
        "normalized_wasserstein_1",
    ]
    correlation_rows: list[dict] = []
    for variable_a in variables:
        for variable_b in variables:
            result = spearmanr(
                sensitivity[variable_a].to_numpy(float),
                sensitivity[variable_b].to_numpy(float),
            )
            correlation_rows.append(
                {
                    "variable_a": variable_a,
                    "variable_b": variable_b,
                    "spearman_rho": float(result.statistic),
                    "two_sided_asymptotic_p": float(result.pvalue),
                    "n_pairs": len(sensitivity),
                }
            )

    return sensitivity, pd.DataFrame(correlation_rows)


def excluded_event_analysis(
    replacement_history: pd.DataFrame,
    event_to_file: dict[str, Path],
    queue: pd.DataFrame,
    strict_event_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe originally selected events that failed metadata eligibility.

    This is an exploratory extension: excluded events are not returned to
    the confirmatory sample. Their endpoint percentiles are measured only
    relative to the locked sample for context.
    """

    excluded_events = replacement_history["original_event"].astype(str).tolist()
    samples = load_samples(event_to_file, excluded_events)
    noise_thresholds = compute_excluded_noise_thresholds(samples, excluded_events)

    metric_rows: list[dict] = []
    for event in excluded_events:
        for parameter in ALL_PARAMETERS:
            row = {
                "event": event,
                "parameter": parameter,
                **pair_metrics(
                    samples[(event, XPHM, parameter)],
                    samples[(event, XPNR, parameter)],
                ),
            }
            metric_rows.append(row)
    excluded_metrics = pd.DataFrame(metric_rows).merge(
        noise_thresholds,
        on=["event", "parameter"],
        validate="one_to_one",
    )
    excluded_metrics["NW1_above_noise"] = (
        excluded_metrics["normalized_wasserstein_1"]
        > excluded_metrics["pair_noise_threshold_NW1"]
    )

    history_columns = [
        "original_event",
        "original_batch",
        "original_classification",
        "replacement_event",
        "reason",
    ]
    history = replacement_history[history_columns].rename(
        columns={"original_event": "event"}
    )
    excluded_metrics = excluded_metrics.merge(
        history, on="event", validate="many_to_one"
    )

    primary = excluded_metrics[
        excluded_metrics["parameter"].isin(PRIMARY_PARAMETERS)
    ]
    endpoint = (
        primary.groupby("event", sort=False)
        .agg(
            event_median_NW1=("normalized_wasserstein_1", "median"),
            event_mean_NW1=("normalized_wasserstein_1", "mean"),
            event_max_NW1=("normalized_wasserstein_1", "max"),
            primary_parameters_above_noise=("NW1_above_noise", "sum"),
        )
        .reset_index()
    )
    endpoint = endpoint.merge(history, on="event", validate="one_to_one")
    endpoint = endpoint.merge(
        queue[
            [
                "event",
                "score_stratum",
                "snr_stratum",
                "maximum_screening_shift",
                "network_matched_filter_snr",
                "total_mass_source",
            ]
        ].drop_duplicates("event"),
        on="event",
        validate="one_to_one",
    )

    # Percentiles provide descriptive context only; excluded events remain
    # outside all confirmatory hypothesis tests.
    strict_values = strict_event_summary["event_median_NW1"].to_numpy(float)
    endpoint["strict_sample_percentile_event_median_NW1"] = endpoint[
        "event_median_NW1"
    ].map(lambda value: 100.0 * np.mean(strict_values <= value))

    endpoint = endpoint.sort_values("event_median_NW1", ascending=False)
    return excluded_metrics, endpoint


def create_figures(
    output_dir: Path,
    loo: pd.DataFrame,
    alternative: pd.DataFrame,
    correlations: pd.DataFrame,
    parameter_summary: pd.DataFrame,
    event_summary: pd.DataFrame,
    noise_matrix: pd.DataFrame,
    excluded_endpoints: pd.DataFrame,
    jsd_correlations: pd.DataFrame,
) -> list[str]:
    """Create robustness figures and return the generated filenames."""

    figure_files: list[str] = []

    for hypothesis in ["H1", "H2"]:
        frame = loo[loo["hypothesis"] == hypothesis].copy()
        frame = frame.sort_values("effect_statistic")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(frame["effect_statistic"], np.arange(len(frame)))
        ax.axvline(0.0, linewidth=1)
        ax.set_yticks(np.arange(len(frame)), frame["omitted_event"])
        ax.set_xlabel("Leave-one-event-out Spearman rho")
        ax.set_ylabel("Omitted event")
        ax.set_title(f"{hypothesis} leave-one-event-out effect stability")
        fig.tight_layout()
        path = output_dir / f"figure_LOO_{hypothesis}_rho.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        figure_files.append(path.name)

    h3 = loo[loo["hypothesis"] == "H3"].copy()
    h3 = h3.sort_values("paired_median_difference")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(h3["paired_median_difference"], np.arange(len(h3)))
    ax.axvline(0.0, linewidth=1)
    ax.set_yticks(np.arange(len(h3)), h3["omitted_event"])
    ax.set_xlabel("Median chi_eff minus distance NW1")
    ax.set_ylabel("Omitted event")
    ax.set_title("H3 leave-one-event-out paired-effect stability")
    fig.tight_layout()
    path = output_dir / "figure_LOO_H3_median_difference.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_files.append(path.name)

    h1_alternatives = alternative[
        alternative["analysis_family"] == "H1_alternative_event_endpoint"
    ].sort_values("spearman_rho")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(
        h1_alternatives["spearman_rho"],
        np.arange(len(h1_alternatives)),
    )
    ax.axvline(0.0, linewidth=1)
    ax.set_yticks(np.arange(len(h1_alternatives)), h1_alternatives["endpoint"])
    ax.set_xlabel("Spearman rho with network SNR")
    ax.set_ylabel("Exploratory event endpoint")
    ax.set_title("H1 sensitivity to event-level summary choice")
    fig.tight_layout()
    path = output_dir / "figure_H1_alternative_endpoints.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_files.append(path.name)

    corr_scope = correlations[
        correlations["scope"] == "all_five_parameters"
    ]
    corr_matrix = corr_scope.pivot(
        index="metric_a", columns="metric_b", values="spearman_rho"
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(corr_matrix.to_numpy(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_matrix.index)), corr_matrix.index)
    ax.set_title("Cross-metric Spearman agreement")
    fig.colorbar(image, ax=ax, label="Spearman rho")
    fig.tight_layout()
    path = output_dir / "figure_cross_metric_correlations.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_files.append(path.name)

    ordered_parameters = PRIMARY_PARAMETERS + ["chirp_mass_source"]

    # Parameter-level NW1 distribution from summary quantiles is shown as points.
    ordered = parameter_summary.set_index("parameter").reindex(ordered_parameters).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        np.arange(len(ordered)),
        ordered["median_NW1"],
        yerr=np.vstack(
            [
                ordered["median_NW1"] - ordered["q25_NW1"],
                ordered["q75_NW1"] - ordered["median_NW1"],
            ]
        ),
        fmt="o",
        capsize=4,
    )
    ax.set_xticks(np.arange(len(ordered)), ordered["parameter"], rotation=30, ha="right")
    ax.set_ylabel("Normalized W1")
    ax.set_title("Parameter-level median and interquartile range")
    fig.tight_layout()
    path = output_dir / "figure_parameter_NW1_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_files.append(path.name)

    matrix = noise_matrix.set_index("event")[PRIMARY_PARAMETERS]
    fig, ax = plt.subplots(figsize=(8, 8))
    image = ax.imshow(matrix.to_numpy(), aspect="auto")
    ax.set_xticks(range(len(PRIMARY_PARAMETERS)), PRIMARY_PARAMETERS)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title("Full-posterior NW1 divided by split-noise threshold")
    fig.colorbar(image, ax=ax, label="NW1 / noise threshold")
    fig.tight_layout()
    path = output_dir / "figure_noise_ratio_matrix.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_files.append(path.name)

    strict_plot = event_summary[["event", "event_median_NW1"]].copy()
    strict_plot["sample"] = "Strict Tier A"
    excluded_plot = excluded_endpoints[["event", "event_median_NW1"]].copy()
    excluded_plot["sample"] = "Excluded Tier B/C"
    combined = pd.concat([strict_plot, excluded_plot], ignore_index=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for x_position, sample_name in enumerate(["Strict Tier A", "Excluded Tier B/C"]):
        sample_values = combined.loc[
            combined["sample"] == sample_name, "event_median_NW1"
        ].to_numpy(float)
        jitter = np.linspace(-0.08, 0.08, len(sample_values)) if len(sample_values) > 1 else np.array([0.0])
        ax.scatter(np.full(len(sample_values), x_position) + jitter, sample_values)
    ax.set_xticks([0, 1], ["Strict Tier A", "Excluded Tier B/C"])
    ax.set_ylabel("Event median normalized W1")
    ax.set_title("Descriptive comparison of strict and excluded events")
    fig.tight_layout()
    path = output_dir / "figure_excluded_event_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_files.append(path.name)

    jsd_matrix = jsd_correlations.pivot(
        index="variable_a", columns="variable_b", values="spearman_rho"
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(jsd_matrix.to_numpy(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(jsd_matrix.columns)), jsd_matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(jsd_matrix.index)), jsd_matrix.index)
    ax.set_title("JSD bin-count sensitivity")
    fig.colorbar(image, ax=ax, label="Spearman rho")
    fig.tight_layout()
    path = output_dir / "figure_JSD_bin_sensitivity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_files.append(path.name)

    return figure_files


def run_analysis(project_root: Path, overwrite: bool = False) -> Path:
    """Execute the full, separately labeled robustness workflow."""

    paths = resolve_paths(project_root)
    ensure_clean_output(paths.output_dir, overwrite)
    log_file = paths.output_dir / "robustness_analysis_execution_log.txt"
    if log_file.exists():
        log_file.unlink()

    started = time.perf_counter()
    write_log(log_file, "GWTC-5 robustness and interpretation analysis")
    write_log(log_file, f"UTC start: {datetime.now(timezone.utc).isoformat()}")
    write_log(log_file, f"Project root: {paths.project_root}")
    write_log(log_file, f"Python: {sys.version}")
    write_log(log_file, f"NumPy: {np.__version__}")
    write_log(log_file, f"pandas: {pd.__version__}")
    write_log(log_file, f"h5py: {h5py.__version__}")

    # Load and authenticate the frozen products before any exploratory work.
    (
        confirmatory_manifest,
        metrics,
        endpoints,
        input_manifest,
        queue,
    ) = load_and_verify_confirmatory_inputs(paths)

    posterior_files = locate_confirmatory_posterior_files(
        paths.processed_dir, input_manifest
    )
    event_to_file = index_events_in_files(posterior_files)
    locked_events = endpoints["event"].astype(str).tolist()
    samples = load_samples(event_to_file, locked_events)

    replacement_history = pd.read_csv(paths.replacement_history_file)

    # Each robustness family writes its own tables; none overwrites a
    # confirmatory artifact.
    write_log(log_file, "Running leave-one-event-out analyses...")
    # Child sequences make the leave-one-out and alternative-endpoint Monte
    # Carlo results reproducible without coupling their random draws.
    root_seed = np.random.SeedSequence(ROBUSTNESS_PERMUTATION_SEED)
    loo_seed, alternative_seed = root_seed.spawn(2)
    loo = leave_one_out_analysis(endpoints, metrics, loo_seed)
    loo_summary = summarize_leave_one_out(loo)
    loo.to_csv(paths.output_dir / "robustness_leave_one_out.csv", index=False)
    loo_summary.to_csv(
        paths.output_dir / "robustness_leave_one_out_summary.csv", index=False
    )

    write_log(log_file, "Testing alternative event-level summaries...")
    alternative, alternative_event_values = alternative_endpoint_analysis(
        endpoints, metrics, alternative_seed
    )
    alternative.to_csv(
        paths.output_dir / "robustness_alternative_endpoints.csv", index=False
    )
    alternative_event_values.to_csv(
        paths.output_dir / "robustness_alternative_event_values.csv", index=False
    )

    write_log(log_file, "Computing cross-metric agreement...")
    correlations = cross_metric_correlations(metrics)
    correlations.to_csv(
        paths.output_dir / "robustness_cross_metric_correlations.csv", index=False
    )

    parameter_summary, event_summary, noise_matrix = parameter_and_noise_summaries(
        metrics, endpoints
    )
    parameter_summary.to_csv(
        paths.output_dir / "robustness_parameter_summary.csv", index=False
    )
    event_summary.to_csv(
        paths.output_dir / "robustness_event_ranking.csv", index=False
    )
    noise_matrix.to_csv(
        paths.output_dir / "robustness_noise_ratio_matrix.csv", index=False
    )

    write_log(log_file, "Running H3 sign-test and bootstrap checks...")
    h3_robustness = h3_additional_robustness(
        metrics, np.random.default_rng(ROBUSTNESS_BOOTSTRAP_SEED)
    )
    h3_robustness.to_csv(
        paths.output_dir / "robustness_H3_additional_checks.csv", index=False
    )

    write_log(log_file, "Checking JSD bin-count sensitivity...")
    jsd_sensitivity, jsd_correlations = jsd_bin_sensitivity(
        samples, locked_events, metrics
    )
    jsd_sensitivity.to_csv(
        paths.output_dir / "robustness_JSD_bin_sensitivity.csv", index=False
    )
    jsd_correlations.to_csv(
        paths.output_dir / "robustness_JSD_bin_correlations.csv", index=False
    )

    write_log(log_file, "Computing descriptive excluded-event extensions...")
    excluded_metrics, excluded_endpoints = excluded_event_analysis(
        replacement_history, event_to_file, queue, event_summary
    )
    excluded_metrics.to_csv(
        paths.output_dir / "robustness_excluded_event_metrics.csv", index=False
    )
    excluded_endpoints.to_csv(
        paths.output_dir / "robustness_excluded_event_endpoints.csv", index=False
    )

    figure_files = create_figures(
        paths.output_dir,
        loo,
        alternative,
        correlations,
        parameter_summary,
        event_summary,
        noise_matrix,
        excluded_endpoints,
        jsd_correlations,
    )

    # Preserve source hashes, seeds, output counts, and headline ranges in a
    # machine-readable manifest for this exploratory run.
    summary = {
        "analysis_label": "exploratory robustness and interpretation",
        "confirmatory_results_unchanged": True,
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(paths.project_root),
        "confirmatory_manifest_sha256": sha256_file(
            paths.confirmatory_manifest_file
        ),
        "confirmatory_parameter_metrics_sha256": sha256_file(
            paths.parameter_metrics_file
        ),
        "confirmatory_event_endpoints_sha256": sha256_file(
            paths.event_endpoints_file
        ),
        "seeds": {
            "permutation_root": ROBUSTNESS_PERMUTATION_SEED,
            "bootstrap": ROBUSTNESS_BOOTSTRAP_SEED,
            "excluded_split_noise": ROBUSTNESS_SPLIT_SEED,
        },
        "counts": {
            "strict_events": len(locked_events),
            "leave_one_out_rows": len(loo),
            "alternative_endpoint_rows": len(alternative),
            "cross_metric_rows": len(correlations),
            "excluded_events": len(excluded_endpoints),
            "excluded_metric_rows": len(excluded_metrics),
            "jsd_sensitivity_rows": len(jsd_sensitivity),
        },
        "headline_robustness": {
            "H1_leave_one_out_rho_min": float(
                loo.loc[loo["hypothesis"] == "H1", "effect_statistic"].min()
            ),
            "H1_leave_one_out_rho_max": float(
                loo.loc[loo["hypothesis"] == "H1", "effect_statistic"].max()
            ),
            "H1_significant_leave_one_out_runs": int(
                loo.loc[loo["hypothesis"] == "H1", "reject_0_05"].sum()
            ),
            "H2_leave_one_out_rho_min": float(
                loo.loc[loo["hypothesis"] == "H2", "effect_statistic"].min()
            ),
            "H2_leave_one_out_rho_max": float(
                loo.loc[loo["hypothesis"] == "H2", "effect_statistic"].max()
            ),
            "H2_significant_leave_one_out_runs": int(
                loo.loc[loo["hypothesis"] == "H2", "reject_0_05"].sum()
            ),
            "H3_positive_pairs": int(
                h3_robustness.loc[
                    h3_robustness["analysis"] == "paired_sign_test",
                    "positive_differences",
                ].iloc[0]
            ),
            "H3_sign_test_p": float(
                h3_robustness.loc[
                    h3_robustness["analysis"] == "paired_sign_test",
                    "p_value",
                ].iloc[0]
            ),
        },
        "figures": figure_files,
        "confirmatory_hypotheses": confirmatory_manifest.get("hypotheses", []),
    }
    (paths.output_dir / "robustness_analysis_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    elapsed = time.perf_counter() - started
    write_log(log_file, "")
    write_log(log_file, "ROBUSTNESS ANALYSIS COMPLETE")
    write_log(log_file, f"Elapsed seconds: {elapsed:.3f}")
    for row in loo_summary.itertuples(index=False):
        write_log(
            log_file,
            f"{row.hypothesis} LOO effect range "
            f"[{row.effect_min:.6g}, {row.effect_max:.6g}], "
            f"significant {row.significant_runs}/{row.leave_one_out_runs}",
        )
    sign_row = h3_robustness[h3_robustness["analysis"] == "paired_sign_test"].iloc[0]
    write_log(
        log_file,
        f"H3 sign test: {int(sign_row.positive_differences)}/"
        f"{int(sign_row.event_count)} positive, p={sign_row.p_value:.6g}",
    )
    write_log(log_file, f"Outputs: {paths.output_dir}")

    return paths.output_dir


def parse_args() -> argparse.Namespace:
    """Parse the project-root and deliberate-overwrite command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the separately labeled GWTC-5 robustness and interpretation "
            "analysis without altering confirmatory outputs."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Path to the GWTC5 project root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing files in results/robustness_analysis.",
    )
    return parser.parse_args()


def main() -> None:
    """Resolve the project root and launch the robustness analysis."""

    args = parse_args()
    root = (
        args.project_root.expanduser().resolve()
        if args.project_root is not None
        else find_project_root()
    )
    run_analysis(root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
