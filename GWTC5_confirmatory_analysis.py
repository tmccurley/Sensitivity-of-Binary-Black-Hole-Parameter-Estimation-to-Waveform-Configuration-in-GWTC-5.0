#Confirmatory analysis

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
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
from scipy.stats import rankdata, spearmanr, wasserstein_distance, wilcoxon


# Frozen waveform labels and parameter sets. These strings must match the
# original-label metadata stored in the compact HDF5 files.
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

# Prespecified numerical settings. Separate seeds isolate the half-split
# reference from the hypothesis-test permutations for reproducibility.
N_BINS_JSD = 60
N_SPLITS = 100
N_PERMUTATIONS = 100_000
SPLIT_SEED = 20260802
PERMUTATION_SEED = 20260803
ALPHA = 0.05


@dataclass(frozen=True)
class AnalysisPaths:
    """Collect the project locations used by the confirmatory pipeline."""

    project_root: Path
    processed_dir: Path
    design_dir: Path
    final_assembly_dir: Path
    lock_file: Path
    queue_file: Path
    output_dir: Path


def find_project_root(start: Path | None = None) -> Path:
    """Locate the project root from the environment or the current path.

    A candidate root must contain both the processed data and results
    directories, which prevents an analysis from silently using an
    unrelated working directory.
    """

    env_root = os.environ.get("GWTC5_PROJECT_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if (root / "data" / "processed").exists():
            return root
        raise FileNotFoundError(
            "GWTC5_PROJECT_ROOT does not contain data/processed: "
            f"{root}"
        )

    start = Path(start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (
            (candidate / "data" / "processed").exists()
            and (candidate / "results").exists()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate the GWTC5 project root. Set the "
        "GWTC5_PROJECT_ROOT environment variable or run inside the project."
    )


def resolve_paths(project_root: Path) -> AnalysisPaths:
    """Build and validate all required confirmatory input and output paths."""

    processed_dir = project_root / "data" / "processed"
    design_dir = processed_dir / "confirmatory_design"
    final_assembly_dir = (
        project_root / "results" / "confirmatory_final_assembly"
    )
    lock_file = final_assembly_dir / "confirmatory_final_sample_lock.json"
    queue_file = design_dir / "confirmatory_randomization_queue.csv"
    output_dir = project_root / "results" / "confirmatory_analysis"

    required = [processed_dir, design_dir, lock_file, queue_file]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required confirmatory inputs:\n"
            + "\n".join(str(path) for path in missing)
        )

    return AnalysisPaths(
        project_root=project_root,
        processed_dir=processed_dir,
        design_dir=design_dir,
        final_assembly_dir=final_assembly_dir,
        lock_file=lock_file,
        queue_file=queue_file,
        output_dir=output_dir,
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for an input or provenance record."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def ensure_clean_output(output_dir: Path, overwrite: bool) -> None:
    """Create the output directory and guard against accidental mixed reruns.

    With ``overwrite=False``, any existing output file stops the run. With
    ``overwrite=True``, existing files in this one output directory are
    removed before the new analysis begins.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [path for path in output_dir.iterdir() if path.is_file()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}\n"
            "Use --overwrite only for a documented full rerun."
        )
    if overwrite:
        for path in existing:
            path.unlink()


def write_log(log_file: Path, message: str = "") -> None:
    """Write the same progress message to standard output and the run log."""

    print(message)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def load_lock(lock_file: Path) -> tuple[dict, pd.DataFrame]:
    """Load the frozen sample lock and enforce its 18-event invariants."""

    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    if not lock.get("confirmatory_sample_locked", False):
        raise RuntimeError("The final confirmatory sample is not marked locked.")
    if int(lock.get("strict_event_count", -1)) != 18:
        raise RuntimeError("The lock file does not specify 18 strict events.")

    events = pd.DataFrame(lock.get("events", []))
    if len(events) != 18 or events["event"].nunique() != 18:
        raise RuntimeError("The lock file does not contain 18 unique events.")
    if not events["strict_primary"].astype(bool).all():
        raise RuntimeError("At least one locked event is not strict primary.")

    return lock, events


def discover_posterior_files(processed_dir: Path) -> list[Path]:
    """Find compact posterior HDF5 files below the processed-data directory."""

    files = sorted(
        path
        for path in processed_dir.rglob("*posteriors.hdf5")
        if path.is_file()
    )
    if not files:
        raise FileNotFoundError(
            f"No compact posterior HDF5 files found under {processed_dir}"
        )
    return files


def index_events_in_files(
    posterior_files: Iterable[Path],
) -> tuple[dict[str, Path], list[dict]]:
    """Map each event to one compact file and record file-level provenance.

    Duplicate event storage is rejected because otherwise the selected
    posterior source would depend on file-discovery order.
    """

    event_to_file: dict[str, Path] = {}
    manifest_rows: list[dict] = []

    for path in posterior_files:
        with h5py.File(path, "r") as source:
            events = sorted(source.keys())
        manifest_rows.append(
            {
                "path": str(path),
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "event_count": len(events),
                "events": ";".join(events),
            }
        )
        for event in events:
            if event in event_to_file:
                raise RuntimeError(
                    f"Event {event} appears in more than one compact file:\n"
                    f"{event_to_file[event]}\n{path}"
                )
            event_to_file[event] = path

    return event_to_file, manifest_rows


def find_model_group(event_group: h5py.Group, model_label: str) -> h5py.Group:
    """Resolve a waveform-model group by its original label.

    The sanitized-name fallback supports compact files whose HDF5 group
    names could not retain punctuation from the original model label.
    """

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
    locked_events: list[str],
) -> tuple[dict[tuple[str, str, str], np.ndarray], list[dict]]:
    """Load finite one-dimensional posterior arrays for every locked comparison.

    The returned mapping is keyed by ``(event, model, parameter)``. A
    preflight table records the source and usable sample count for every
    array, while shared HDF5 handles avoid repeatedly opening the same file.
    """

    samples: dict[tuple[str, str, str], np.ndarray] = {}
    preflight_rows: list[dict] = []

    handles: dict[Path, h5py.File] = {}
    try:
        for event in locked_events:
            if event not in event_to_file:
                raise FileNotFoundError(
                    f"Locked event {event} is absent from compact posterior files."
                )
            path = event_to_file[event]
            source = handles.setdefault(path, h5py.File(path, "r"))
            event_group = source[event]

            for model in MODELS:
                model_group = find_model_group(event_group, model)
                for parameter in ALL_PARAMETERS:
                    if parameter not in model_group:
                        raise KeyError(
                            f"Missing {event} / {model} / {parameter}"
                        )
                    values = np.asarray(
                        model_group[parameter][:], dtype=np.float64
                    ).reshape(-1)
                    values = values[np.isfinite(values)]
                    if values.size < 4:
                        raise RuntimeError(
                            f"Too few finite samples for {event} / {model} / "
                            f"{parameter}: {values.size}"
                        )
                    samples[(event, model, parameter)] = values
                    preflight_rows.append(
                        {
                            "event": event,
                            "model": model,
                            "parameter": parameter,
                            "source_file": path.name,
                            "finite_sample_count": int(values.size),
                        }
                    )
    finally:
        for source in handles.values():
            source.close()

    return samples, preflight_rows


def interval_summary(values: np.ndarray) -> dict[str, float]:
    """Summarize a posterior by its equal-tailed 90% interval and variance."""

    lower, median, upper = np.quantile(values, [0.05, 0.50, 0.95])
    return {
        "lower_90": float(lower),
        "median": float(median),
        "upper_90": float(upper),
        "width_90": float(upper - lower),
        "variance": float(np.var(values, ddof=1)),
    }


def js_divergence_bits(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = N_BINS_JSD,
) -> float:
    """Estimate Jensen-Shannon divergence with pooled equal-width bins.

    Both posteriors use the same pooled range and the base-2 logarithm, so
    the result is expressed in bits. Identical point-mass ranges return
    zero rather than constructing degenerate histogram edges.
    """

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
        """Compute the finite contribution to KL divergence in base-2 units."""

        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

    return 0.5 * kl_bits(px, midpoint) + 0.5 * kl_bits(py, midpoint)


def pair_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float | bool]:
    """Compute the prespecified disagreement summaries for one posterior pair.

    The primary normalized Wasserstein-1 value divides raw W1 by the mean
    of the two 90% interval widths. The additional fields describe median
    displacement, interval overlap, width change, and distributional
    divergence without changing that primary definition.
    """

    sx = interval_summary(x)
    sy = interval_summary(y)

    average_width = 0.5 * (sx["width_90"] + sy["width_90"])
    raw_w1 = float(wasserstein_distance(x, y))
    normalized_w1 = (
        raw_w1 / average_width if average_width > 0 else math.nan
    )

    median_denominator = math.sqrt(sx["variance"] + sy["variance"])
    standardized_median = (
        abs(sx["median"] - sy["median"]) / median_denominator
        if median_denominator > 0
        else math.nan
    )

    intersection = max(
        0.0,
        min(sx["upper_90"], sy["upper_90"])
        - max(sx["lower_90"], sy["lower_90"]),
    )
    union = (
        max(sx["upper_90"], sy["upper_90"])
        - min(sx["lower_90"], sy["lower_90"])
    )
    overlap = intersection / union if union > 0 else 1.0

    log_width_ratio = (
        math.log(sx["width_90"] / sy["width_90"])
        if sx["width_90"] > 0 and sy["width_90"] > 0
        else math.nan
    )

    return {
        "xphm_sample_count": int(x.size),
        "xpnr_sample_count": int(y.size),
        "xphm_median": sx["median"],
        "xpnr_median": sy["median"],
        "xphm_lower_90": sx["lower_90"],
        "xphm_upper_90": sx["upper_90"],
        "xpnr_lower_90": sy["lower_90"],
        "xpnr_upper_90": sy["upper_90"],
        "xphm_width_90": sx["width_90"],
        "xpnr_width_90": sy["width_90"],
        "average_width_90": average_width,
        "wasserstein_1": raw_w1,
        "normalized_wasserstein_1": normalized_w1,
        "js_divergence_bits": js_divergence_bits(x, y),
        "standardized_median_displacement": standardized_median,
        "interval_overlap_90": overlap,
        "log_width_ratio": log_width_ratio,
    }


def equal_size_w1(x: np.ndarray, y: np.ndarray) -> float:
    """Compute empirical one-dimensional W1 for two equally sized samples."""

    if x.size != y.size:
        raise ValueError("equal_size_w1 requires equal sample sizes")
    return float(np.mean(np.abs(np.sort(x) - np.sort(y))))


def split_normalized_w1(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Normalize half-sample W1 by their mean 90% interval width."""

    s1 = interval_summary(first)
    s2 = interval_summary(second)
    average_width = 0.5 * (s1["width_90"] + s2["width_90"])
    if average_width <= 0:
        return math.nan
    return equal_size_w1(first, second) / average_width


def compute_split_noise(
    samples: dict[tuple[str, str, str], np.ndarray],
    event_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate a within-model sampling-noise reference for every pair.

    Each posterior is randomly divided into equal halves 100 times. The
    pair threshold is the larger of the two model-specific 95th
    percentiles, making the comparison conservative with respect to the
    noisier posterior representation.
    """

    rng = np.random.default_rng(SPLIT_SEED)
    replicate_rows: list[dict] = []
    threshold_rows: list[dict] = []

    for event in event_order:
        for parameter in ALL_PARAMETERS:
            model_values: dict[str, list[float]] = {}
            for model in MODELS:
                values = samples[(event, model, parameter)]
                half_size = values.size // 2
                if half_size < 2:
                    raise RuntimeError(
                        f"Insufficient split size for {event}/{model}/{parameter}"
                    )

                split_values: list[float] = []
                for replicate in range(1, N_SPLITS + 1):
                    permutation = rng.permutation(values.size)
                    first = values[permutation[:half_size]]
                    second = values[
                        permutation[half_size : 2 * half_size]
                    ]
                    nwd = split_normalized_w1(first, second)
                    split_values.append(nwd)
                    replicate_rows.append(
                        {
                            "event": event,
                            "model": model,
                            "parameter": parameter,
                            "replicate": replicate,
                            "half_sample_size": half_size,
                            "split_normalized_wasserstein_1": nwd,
                        }
                    )
                model_values[model] = split_values

            xphm_p95 = float(np.nanquantile(model_values[XPHM], 0.95))
            xpnr_p95 = float(np.nanquantile(model_values[XPNR], 0.95))
            threshold_rows.append(
                {
                    "event": event,
                    "parameter": parameter,
                    "xphm_split_NW1_p95": xphm_p95,
                    "xpnr_split_NW1_p95": xpnr_p95,
                    "pair_noise_threshold_NW1": max(xphm_p95, xpnr_p95),
                }
            )

    return pd.DataFrame(replicate_rows), pd.DataFrame(threshold_rows)


def spearman_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_permutations: int = N_PERMUTATIONS,
    alternative: str = "greater",
    chunk_size: int = 5000,
) -> tuple[float, float]:
    """Run the frozen one-sided Spearman permutation test.

    Ranking first reproduces Spearman correlation. Permutations are
    generated in chunks to limit memory use, and the plus-one correction
    prevents a Monte Carlo p-value of exactly zero.
    """

    if x.size != y.size:
        raise ValueError("x and y must have equal length")
    if alternative != "greater":
        raise NotImplementedError("Only the frozen greater alternative is used")

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
        permuted_y = yc[order]
        statistics = permuted_y @ xc / denominator
        exceedances += int(np.count_nonzero(statistics >= observed))
        completed += current

    p_value = (exceedances + 1) / (n_permutations + 1)
    return observed, float(p_value)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Apply the step-down Holm adjustment while preserving monotonicity."""

    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for index, (name, p_value) in enumerate(ordered):
        candidate = min(1.0, (m - index) * p_value)
        running_max = max(running_max, candidate)
        adjusted[name] = running_max
    return adjusted


def create_figures(
    event_endpoints: pd.DataFrame,
    parameter_metrics: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    """Create the four prespecified diagnostic figures and return their names."""

    figure_files: list[str] = []

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        event_endpoints["network_matched_filter_snr"],
        event_endpoints["event_median_NW1"],
    )
    ax.set_xlabel("Catalog network matched-filter SNR")
    ax.set_ylabel("Event median normalized W1")
    ax.set_title("H1: SNR and event-level model sensitivity")
    fig.tight_layout()
    path = output_dir / "figure_H1_snr_vs_event_median_NW1.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    figure_files.append(path.name)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        event_endpoints["maximum_screening_shift"],
        event_endpoints["screening_max_NW1"],
    )
    ax.set_xlabel("Catalog maximum screening shift")
    ax.set_ylabel("Maximum full-posterior screening NW1")
    ax.set_title("H2: Screening score validation")
    fig.tight_layout()
    path = output_dir / "figure_H2_screening_validation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_files.append(path.name)

    h3 = parameter_metrics[
        parameter_metrics["parameter"].isin(
            ["chi_eff", "luminosity_distance"]
        )
    ].pivot(
        index="event",
        columns="parameter",
        values="normalized_wasserstein_1",
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in h3.iterrows():
        ax.plot(
            [0, 1],
            [row["luminosity_distance"], row["chi_eff"]],
            marker="o",
        )
    ax.set_xticks([0, 1], ["Luminosity distance", "chi_eff"])
    ax.set_ylabel("Normalized W1")
    ax.set_title("H3: Paired parameter comparison")
    fig.tight_layout()
    path = output_dir / "figure_H3_chi_eff_vs_distance_NW1.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_files.append(path.name)

    primary = parameter_metrics[
        parameter_metrics["parameter"].isin(PRIMARY_PARAMETERS)
    ].pivot(
        index="event",
        columns="parameter",
        values="normalized_wasserstein_1",
    )
    primary = primary.reindex(event_endpoints["event"])
    primary = primary[PRIMARY_PARAMETERS]
    fig, ax = plt.subplots(figsize=(8, 8))
    image = ax.imshow(primary.to_numpy(), aspect="auto")
    ax.set_xticks(range(len(PRIMARY_PARAMETERS)), PRIMARY_PARAMETERS)
    ax.set_yticks(range(len(primary.index)), primary.index)
    ax.set_title("Normalized Wasserstein-1 across locked events")
    fig.colorbar(image, ax=ax, label="Normalized W1")
    fig.tight_layout()
    path = output_dir / "figure_primary_NW1_matrix.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_files.append(path.name)

    return figure_files


def run_analysis(project_root: Path, overwrite: bool = False) -> Path:
    """Execute and document the complete confirmatory workflow."""

    paths = resolve_paths(project_root)
    ensure_clean_output(paths.output_dir, overwrite)
    log_file = paths.output_dir / "confirmatory_analysis_execution_log.txt"
    if log_file.exists():
        log_file.unlink()

    started = time.perf_counter()
    write_log(log_file, "GWTC-5 confirmatory analysis")
    write_log(log_file, f"UTC start: {datetime.now(timezone.utc).isoformat()}")
    write_log(log_file, f"Project root: {paths.project_root}")
    write_log(log_file, f"Python: {sys.version}")
    write_log(log_file, f"NumPy: {np.__version__}")
    write_log(log_file, f"pandas: {pd.__version__}")
    write_log(log_file, f"h5py: {h5py.__version__}")

    # Establish the frozen event order before any posterior metric is computed.
    lock, locked = load_lock(paths.lock_file)
    event_order = locked["event"].astype(str).tolist()
    queue = pd.read_csv(paths.queue_file)

    posterior_files = discover_posterior_files(paths.processed_dir)
    event_to_file, file_manifest_rows = index_events_in_files(posterior_files)

    # Retain hashes only for files that actually supply a locked event.
    relevant_files = sorted({event_to_file[event] for event in event_order})
    file_manifest = pd.DataFrame(
        row
        for row in file_manifest_rows
        if Path(row["path"]) in relevant_files
    )
    file_manifest.to_csv(
        paths.output_dir / "confirmatory_input_file_manifest.csv",
        index=False,
    )

    write_log(log_file, f"Locked events: {len(event_order)}")
    write_log(log_file, f"Relevant compact posterior files: {len(relevant_files)}")

    samples, preflight_rows = load_samples(event_to_file, event_order)
    preflight = pd.DataFrame(preflight_rows)
    preflight.to_csv(
        paths.output_dir / "confirmatory_preflight_samples.csv",
        index=False,
    )
    write_log(log_file, f"Verified posterior arrays: {len(preflight)}")

    # Compute model-pair metrics from the full stored posteriors.
    metric_rows: list[dict] = []
    for event in event_order:
        for parameter in ALL_PARAMETERS:
            x = samples[(event, XPHM, parameter)]
            y = samples[(event, XPNR, parameter)]
            metric_rows.append(
                {
                    "event": event,
                    "parameter": parameter,
                    **pair_metrics(x, y),
                }
            )
    parameter_metrics = pd.DataFrame(metric_rows)
    write_log(log_file, f"Full-posterior metric rows: {len(parameter_metrics)}")

    # The split reference estimates disagreement caused by finite posterior
    # sampling within a model, rather than disagreement between models.
    write_log(log_file, "Computing 100 half-split noise replicates...")
    split_replicates, noise_thresholds = compute_split_noise(
        samples, event_order
    )
    split_replicates.to_csv(
        paths.output_dir / "confirmatory_split_noise_replicates.csv",
        index=False,
    )
    noise_thresholds.to_csv(
        paths.output_dir / "confirmatory_split_noise_thresholds.csv",
        index=False,
    )

    parameter_metrics = parameter_metrics.merge(
        noise_thresholds,
        on=["event", "parameter"],
        how="left",
        validate="one_to_one",
    )
    parameter_metrics["NW1_above_noise"] = (
        parameter_metrics["normalized_wasserstein_1"]
        > parameter_metrics["pair_noise_threshold_NW1"]
    )
    parameter_metrics.to_csv(
        paths.output_dir / "confirmatory_parameter_metrics.csv",
        index=False,
    )

    primary = parameter_metrics[
        parameter_metrics["parameter"].isin(PRIMARY_PARAMETERS)
    ]
    # Collapse parameter-level results to the prespecified event endpoints.
    event_primary = (
        primary.groupby("event", sort=False)
        .agg(
            event_median_NW1=("normalized_wasserstein_1", "median"),
            event_max_NW1=("normalized_wasserstein_1", "max"),
            primary_parameters_above_noise=("NW1_above_noise", "sum"),
            primary_fraction_above_noise=("NW1_above_noise", "mean"),
        )
        .reset_index()
    )

    screening = parameter_metrics[
        parameter_metrics["parameter"].isin(SCREENING_PARAMETERS)
    ]
    screening_event = (
        screening.groupby("event", sort=False)["normalized_wasserstein_1"]
        .max()
        .rename("screening_max_NW1")
        .reset_index()
    )

    catalog_columns = [
        "event",
        "maximum_screening_shift",
        "median_screening_shift",
        "network_matched_filter_snr",
        "total_mass_source",
    ]
    catalog = queue[catalog_columns].drop_duplicates("event")

    event_endpoints = (
        locked[
            [
                "batch",
                "slot",
                "event",
                "original_selected_event",
                "replacement_status",
                "score_stratum",
                "snr_stratum",
            ]
        ]
        .merge(event_primary, on="event", how="left", validate="one_to_one")
        .merge(screening_event, on="event", how="left", validate="one_to_one")
        .merge(catalog, on="event", how="left", validate="one_to_one")
    )
    if event_endpoints[
        [
            "event_median_NW1",
            "screening_max_NW1",
            "maximum_screening_shift",
            "network_matched_filter_snr",
        ]
    ].isna().any().any():
        raise RuntimeError("Missing event endpoint or catalog metadata")

    event_endpoints.to_csv(
        paths.output_dir / "confirmatory_event_endpoints.csv",
        index=False,
    )
    # Browser-ready data keeps the visualization self-contained even when the
    # HTML file is opened directly rather than through a web server.
    dashboard_metrics = parameter_metrics.replace(
        {np.nan: None, np.inf: None, -np.inf: None}
    ).to_dict(orient="records")
    dashboard_events = event_endpoints.replace(
        {np.nan: None, np.inf: None, -np.inf: None}
    ).to_dict(orient="records")
    (paths.output_dir / "waveform-comparison-data.js").write_text(
        "window.GWTC5_WAVEFORM_DATA = "
        + json.dumps(dashboard_metrics, ensure_ascii=False)
        + ";\nwindow.GWTC5_EVENT_DATA = "
        + json.dumps(dashboard_events, ensure_ascii=False)
        + ";\n",
        encoding="utf-8",
    )

    # Spawn deterministic child generators so H1 and H2 do not share a
    # mutable random stream.
    seed_sequence = np.random.SeedSequence(PERMUTATION_SEED)
    h1_rng, h2_rng = [
        np.random.default_rng(child)
        for child in seed_sequence.spawn(2)
    ]

    # H1 and H2 test positive monotonic associations at the event level.
    h1_rho, h1_p = spearman_permutation_test(
        event_endpoints["network_matched_filter_snr"].to_numpy(float),
        event_endpoints["event_median_NW1"].to_numpy(float),
        h1_rng,
    )
    h2_rho, h2_p = spearman_permutation_test(
        event_endpoints["maximum_screening_shift"].to_numpy(float),
        event_endpoints["screening_max_NW1"].to_numpy(float),
        h2_rng,
    )

    # H3 preserves event pairing when comparing chi_eff with distance NW1.
    h3_wide = parameter_metrics[
        parameter_metrics["parameter"].isin(
            ["chi_eff", "luminosity_distance"]
        )
    ].pivot(
        index="event",
        columns="parameter",
        values="normalized_wasserstein_1",
    ).reindex(event_order)
    h3_result = wilcoxon(
        h3_wide["chi_eff"].to_numpy(float),
        h3_wide["luminosity_distance"].to_numpy(float),
        alternative="greater",
        zero_method="wilcox",
        method="auto",
    )
    h3_statistic = float(h3_result.statistic)
    h3_p = float(h3_result.pvalue)

    # H2 and H3 form the prespecified secondary family; H1 remains the
    # separate primary test and therefore keeps its raw p-value.
    holm = holm_adjust({"H2": h2_p, "H3": h3_p})

    hypothesis_rows = [
        {
            "hypothesis": "H1",
            "family": "primary",
            "test": "One-sided Spearman permutation test",
            "alternative": "rho > 0",
            "effect_statistic": h1_rho,
            "raw_p_value": h1_p,
            "adjusted_p_value": h1_p,
            "alpha": ALPHA,
            "reject": bool(h1_p < ALPHA),
            "permutations": N_PERMUTATIONS,
        },
        {
            "hypothesis": "H2",
            "family": "secondary_Holm",
            "test": "One-sided Spearman permutation test",
            "alternative": "rho > 0",
            "effect_statistic": h2_rho,
            "raw_p_value": h2_p,
            "adjusted_p_value": holm["H2"],
            "alpha": ALPHA,
            "reject": bool(holm["H2"] < ALPHA),
            "permutations": N_PERMUTATIONS,
        },
        {
            "hypothesis": "H3",
            "family": "secondary_Holm",
            "test": "One-sided Wilcoxon signed-rank test",
            "alternative": "chi_eff NW1 > luminosity_distance NW1",
            "effect_statistic": h3_statistic,
            "raw_p_value": h3_p,
            "adjusted_p_value": holm["H3"],
            "alpha": ALPHA,
            "reject": bool(holm["H3"] < ALPHA),
            "permutations": 0,
        },
    ]
    hypothesis_results = pd.DataFrame(hypothesis_rows)
    hypothesis_results.to_csv(
        paths.output_dir / "confirmatory_hypothesis_results.csv",
        index=False,
    )

    figure_files = create_figures(
        event_endpoints, parameter_metrics, paths.output_dir
    )

    # Record definitions, seeds, counts, input hashes, and results needed to
    # audit or reproduce this exact run.
    result_manifest = {
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(paths.project_root),
        "sample_lock_sha256": sha256_file(paths.lock_file),
        "queue_sha256": sha256_file(paths.queue_file),
        "strict_event_count": len(event_order),
        "models": MODELS,
        "primary_parameters": PRIMARY_PARAMETERS,
        "screening_parameters": SCREENING_PARAMETERS,
        "definitions": {
            "normalized_wasserstein_1": (
                "W1 divided by the mean of the two 90% interval widths"
            ),
            "js_divergence_bits": (
                "Base-2 JSD from 60 equal-width bins over pooled range"
            ),
            "standardized_median_displacement": (
                "Absolute median difference divided by "
                "sqrt(var_XPHM + var_XPNR)"
            ),
            "interval_overlap_90": (
                "Intersection length divided by union length"
            ),
            "log_width_ratio": "log(width_XPHM / width_XPNR)",
            "noise_threshold": (
                "Larger model-specific 95th percentile from 100 "
                "random half-split normalized-W1 replicates"
            ),
        },
        "seeds": {
            "split_noise": SPLIT_SEED,
            "permutation_root": PERMUTATION_SEED,
        },
        "counts": {
            "full_posterior_metric_rows": len(parameter_metrics),
            "split_noise_rows": len(split_replicates),
            "event_endpoint_rows": len(event_endpoints),
        },
        "hypotheses": hypothesis_rows,
        "figures": figure_files,
        "input_files": file_manifest.to_dict(orient="records"),
    }
    (paths.output_dir / "confirmatory_analysis_manifest.json").write_text(
        json.dumps(result_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - started
    write_log(log_file, "")
    write_log(log_file, "CONFIRMATORY ANALYSIS COMPLETE")
    write_log(log_file, f"Elapsed seconds: {elapsed:.3f}")
    write_log(log_file, f"H1 rho={h1_rho:.6g}, p={h1_p:.6g}")
    write_log(
        log_file,
        f"H2 rho={h2_rho:.6g}, raw p={h2_p:.6g}, "
        f"Holm p={holm['H2']:.6g}",
    )
    write_log(
        log_file,
        f"H3 W={h3_statistic:.6g}, raw p={h3_p:.6g}, "
        f"Holm p={holm['H3']:.6g}",
    )
    write_log(log_file, f"Outputs: {paths.output_dir}")

    return paths.output_dir


def parse_args() -> argparse.Namespace:
    """Parse the project-root and deliberate-overwrite command-line options."""

    parser = argparse.ArgumentParser(
        description="Run the frozen GWTC-5 confirmatory analysis."
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
        help="Delete existing files in the confirmatory output directory.",
    )
    return parser.parse_args()


def main() -> None:
    """Resolve the requested project root and launch the analysis."""

    args = parse_args()
    root = (
        args.project_root.expanduser().resolve()
        if args.project_root is not None
        else find_project_root()
    )
    run_analysis(root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
