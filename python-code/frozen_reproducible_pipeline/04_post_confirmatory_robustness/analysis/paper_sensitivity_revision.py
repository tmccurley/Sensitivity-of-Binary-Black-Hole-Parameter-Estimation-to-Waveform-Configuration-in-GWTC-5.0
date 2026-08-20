#!/usr/bin/env python3
"""Post-confirmatory sensitivity checks added during manuscript revision.

These checks do not replace or alter the frozen confirmatory analysis.
They produce exploratory design-aware H1/H2 checks, an H1 bootstrap
interval, H3 parameterization/normalization checks, and an array-level
posterior-sample audit.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import glob
import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wasserstein_distance, ks_2samp, wilcoxon

XPHM = "C00:IMRPhenomXPHM-SpinTaylor"
XPNR = "C00:IMRPhenomXPNR"
PRIMARY = ["chirp_mass", "mass_ratio", "chi_eff", "luminosity_distance"]
ALL_PARAMS = PRIMARY + ["chirp_mass_source"]


def corr(a: np.ndarray, b: np.ndarray) -> float:
    ac = a - np.mean(a)
    bc = b - np.mean(b)
    den = np.sqrt(np.dot(ac, ac) * np.dot(bc, bc))
    return float(np.dot(ac, bc) / den)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return corr(rankdata(a), rankdata(b))


def within_stratum_permutation(
    predictor: np.ndarray,
    response: np.ndarray,
    strata: np.ndarray,
    seed: int,
    permutations: int = 100_000,
) -> tuple[float, int, float]:
    rx = rankdata(predictor)
    ry = rankdata(response)
    observed = corr(rx, ry)
    rng = np.random.default_rng(seed)
    unique = np.unique(strata)
    count = 0
    chunk = 10_000
    ac = rx - rx.mean()
    ac_ss = np.dot(ac, ac)
    for start in range(0, permutations, chunk):
        n = min(chunk, permutations - start)
        arr = np.tile(ry, (n, 1))
        for stratum in unique:
            idx = np.where(strata == stratum)[0]
            order = np.argsort(rng.random((n, len(idx))), axis=1)
            arr[:, idx] = ry[idx][order]
        bc = arr - arr.mean(axis=1, keepdims=True)
        rho = (bc @ ac) / np.sqrt((bc * bc).sum(axis=1) * ac_ss)
        count += int(np.count_nonzero(rho >= observed - 1e-15))
    p_value = (count + 1) / (permutations + 1)
    return observed, count, p_value


def bootstrap_spearman(
    predictor: np.ndarray,
    response: np.ndarray,
    seed: int,
    resamples: int = 100_000,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=float)
    valid = 0
    n = len(predictor)
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        a = rankdata(predictor[idx])
        b = rankdata(response[idx])
        ac = a - a.mean()
        bc = b - b.mean()
        den = np.sqrt(np.dot(ac, ac) * np.dot(bc, bc))
        if den > 0:
            values[valid] = np.dot(ac, bc) / den
            valid += 1
    values = values[:valid]
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(np.median(values)), float(lower), float(upper)


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    rxy = spearman(x, y)
    rxz = spearman(x, z)
    ryz = spearman(y, z)
    den = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    return float((rxy - rxz * ryz) / den)


def width90(x: np.ndarray) -> float:
    q05, q95 = np.quantile(x, [0.05, 0.95])
    return float(q95 - q05)


def decode_attr(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def locate_posteriors(processed_dir: Path, events: set[str]):
    mapping: dict[str, tuple[Path, dict[str, str]]] = {}
    for path in processed_dir.rglob("*posteriors.hdf5"):
        with h5py.File(path, "r") as handle:
            for event in handle.keys():
                if event not in events:
                    continue
                labels: dict[str, str] = {}
                for group_name in handle[event].keys():
                    group = handle[event][group_name]
                    if not isinstance(group, h5py.Group):
                        continue
                    label = decode_attr(group.attrs.get("original_label", ""))
                    if label in {XPHM, XPNR}:
                        labels[label] = group_name
                if len(labels) == 2:
                    if event in mapping and mapping[event][0] != path:
                        raise RuntimeError(
                            f"Event {event} appears in multiple compact files: "
                            f"{mapping[event][0]} and {path}"
                        )
                    mapping[event] = (path, labels)
    missing = events.difference(mapping)
    if missing:
        raise FileNotFoundError(f"Missing posterior files for: {sorted(missing)}")
    return mapping


def load(mapping, event: str, label: str, parameter: str) -> np.ndarray:
    path, labels = mapping[event]
    with h5py.File(path, "r") as handle:
        values = np.asarray(
            handle[event][labels[label]][parameter], dtype=float
        ).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 4:
        raise RuntimeError(
            f"Too few finite samples for {event}/{label}/{parameter}: "
            f"{values.size}"
        )
    return values


def group_audit(mapping, event: str, label: str) -> dict[str, object]:
    path, labels = mapping[event]
    with h5py.File(path, "r") as handle:
        group = handle[event][labels[label]]
        return {
            "source_file": path.name,
            "sampling_method": decode_attr(group.attrs.get("sampling_method", "")),
            "original_sample_count": int(group.attrs.get("original_sample_count", -1)),
            "stored_sample_count": int(group.attrs.get("stored_sample_count", -1)),
        }


def run(project_root: Path, output_dir: Path) -> None:
    confirm_dir = project_root / "results" / "confirmatory_analysis"
    endpoints = pd.read_csv(confirm_dir / "confirmatory_event_endpoints.csv")
    metrics = pd.read_csv(confirm_dir / "confirmatory_parameter_metrics.csv")
    events = set(endpoints["event"])

    h1_rho, h1_exceed, h1_strat_p = within_stratum_permutation(
        endpoints["network_matched_filter_snr"].to_numpy(),
        endpoints["event_median_NW1"].to_numpy(),
        endpoints["score_stratum"].to_numpy(),
        seed=20260807,
    )
    h2_rho, h2_exceed, h2_strat_p = within_stratum_permutation(
        endpoints["maximum_screening_shift"].to_numpy(),
        endpoints["screening_max_NW1"].to_numpy(),
        endpoints["snr_stratum"].to_numpy(),
        seed=20260808,
    )
    boot_median, boot_lower, boot_upper = bootstrap_spearman(
        endpoints["network_matched_filter_snr"].to_numpy(),
        endpoints["event_median_NW1"].to_numpy(),
        seed=20260809,
    )
    partial = partial_spearman(
        endpoints["network_matched_filter_snr"].to_numpy(),
        endpoints["event_median_NW1"].to_numpy(),
        endpoints["maximum_screening_shift"].to_numpy(),
    )

    design_rows = [
        {
            "analysis": "H1 within-screening-stratum permutation",
            "effect": h1_rho,
            "exceedances": h1_exceed,
            "iterations": 100_000,
            "p_value": h1_strat_p,
            "seed": 20260807,
        },
        {
            "analysis": "H2 within-SNR-stratum permutation",
            "effect": h2_rho,
            "exceedances": h2_exceed,
            "iterations": 100_000,
            "p_value": h2_strat_p,
            "seed": 20260808,
        },
        {
            "analysis": "H1 partial Spearman controlling screening score",
            "effect": partial,
            "exceedances": np.nan,
            "iterations": 0,
            "p_value": np.nan,
            "seed": np.nan,
        },
        {
            "analysis": "H1 bootstrap Spearman median",
            "effect": boot_median,
            "exceedances": np.nan,
            "iterations": 100_000,
            "p_value": np.nan,
            "seed": 20260809,
            "lower_95": boot_lower,
            "upper_95": boot_upper,
        },
    ]

    mapping = locate_posteriors(project_root / "data" / "processed", events)
    h3_event_rows = []
    audit_rows = []
    for event in endpoints["event"]:
        parameter_values = {}
        for parameter in ["chi_eff", "luminosity_distance"]:
            x = load(mapping, event, XPHM, parameter)
            y = load(mapping, event, XPNR, parameter)
            raw_w1 = wasserstein_distance(x, y)
            wx, wy = width90(x), width90(y)
            values = {
                "arithmetic_width_NW1": raw_w1 / ((wx + wy) / 2),
                "geometric_width_NW1": raw_w1 / np.sqrt(wx * wy),
                "maximum_width_NW1": raw_w1 / max(wx, wy),
                "KS": ks_2samp(x, y, method="asymp").statistic,
            }
            if parameter == "luminosity_distance":
                if np.any(x <= 0) or np.any(y <= 0):
                    raise ValueError(
                        f"Nonpositive luminosity distance for {event}"
                    )
                lx, ly = np.log(x), np.log(y)
                values["log_distance_NW1"] = wasserstein_distance(lx, ly) / (
                    (width90(lx) + width90(ly)) / 2
                )
            parameter_values[parameter] = values

        metric_subset = metrics[
            (metrics["event"] == event)
            & metrics["parameter"].isin(["chi_eff", "luminosity_distance"])
        ].copy()
        expected_parameters = {"chi_eff", "luminosity_distance"}
        if len(metric_subset) != 2 or set(metric_subset["parameter"]) != expected_parameters:
            raise RuntimeError(
                f"Expected exactly one H3 metric row per parameter for {event}; "
                f"found {metric_subset[["parameter"]].to_dict(orient="records")}"
            )
        if metric_subset["parameter"].duplicated().any():
            raise RuntimeError(f"Duplicate H3 metric rows for {event}")
        metric_subset["noise_ratio"] = (
            metric_subset["normalized_wasserstein_1"]
            / metric_subset["pair_noise_threshold_NW1"]
        )
        noise = metric_subset.set_index("parameter")["noise_ratio"]

        h3_event_rows.append(
            {
                "event": event,
                "chi_eff_arithmetic_NW1": parameter_values["chi_eff"]["arithmetic_width_NW1"],
                "distance_arithmetic_NW1": parameter_values["luminosity_distance"]["arithmetic_width_NW1"],
                "chi_eff_geometric_NW1": parameter_values["chi_eff"]["geometric_width_NW1"],
                "distance_geometric_NW1": parameter_values["luminosity_distance"]["geometric_width_NW1"],
                "chi_eff_maximum_width_NW1": parameter_values["chi_eff"]["maximum_width_NW1"],
                "distance_maximum_width_NW1": parameter_values["luminosity_distance"]["maximum_width_NW1"],
                "log_distance_NW1": parameter_values["luminosity_distance"]["log_distance_NW1"],
                "chi_eff_KS": parameter_values["chi_eff"]["KS"],
                "distance_KS": parameter_values["luminosity_distance"]["KS"],
                "chi_eff_split_reference_ratio": noise["chi_eff"],
                "distance_split_reference_ratio": noise["luminosity_distance"],
            }
        )

        for label in [XPHM, XPNR]:
            group_info = group_audit(mapping, event, label)
            for parameter in ALL_PARAMS:
                x = load(mapping, event, label, parameter)
                duplicate_fraction = 1 - np.unique(x).size / x.size
                lag1 = (
                    float(np.corrcoef(x[:-1], x[1:])[0, 1])
                    if x.size >= 2
                    else np.nan
                )
                audit_rows.append(
                    {
                        "event": event,
                        "model": label,
                        "parameter": parameter,
                        "sample_count": x.size,
                        "duplicate_fraction": duplicate_fraction,
                        "lag1_correlation": lag1,
                        **group_info,
                        "stored_equals_original": (
                            group_info["stored_sample_count"]
                            == group_info["original_sample_count"]
                            == x.size
                        ),
                    }
                )

    h3_events = pd.DataFrame(h3_event_rows)
    h3_checks = []

    def paired_check(name: str, a: str, b: str) -> None:
        first = h3_events[a].to_numpy()
        second = h3_events[b].to_numpy()
        result = wilcoxon(first, second, alternative="greater", method="auto")
        h3_checks.append(
            {
                "analysis": name,
                "positive_pairs": int(np.sum(first > second)),
                "median_first": float(np.median(first)),
                "median_second": float(np.median(second)),
                "median_difference": float(np.median(first - second)),
                "wilcoxon_W": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )

    paired_check(
        "Frozen NW1: arithmetic mean width",
        "chi_eff_arithmetic_NW1",
        "distance_arithmetic_NW1",
    )
    paired_check(
        "NW1: geometric mean width",
        "chi_eff_geometric_NW1",
        "distance_geometric_NW1",
    )
    paired_check(
        "NW1: larger width",
        "chi_eff_maximum_width_NW1",
        "distance_maximum_width_NW1",
    )
    paired_check(
        "chi_eff NW1 vs log-distance NW1",
        "chi_eff_arithmetic_NW1",
        "log_distance_NW1",
    )
    paired_check(
        "Two-sample KS distance",
        "chi_eff_KS",
        "distance_KS",
    )
    paired_check(
        "Split-reference ratio",
        "chi_eff_split_reference_ratio",
        "distance_split_reference_ratio",
    )

    # Practical screening summaries within the stratified sample.
    screen_rank = rankdata(-endpoints["maximum_screening_shift"], method="average")
    full_rank = rankdata(-endpoints["screening_max_NW1"], method="average")
    rank_error = np.abs(screen_rank - full_rank)
    practical = []
    for k in [3, 4, 5, 6]:
        top_screen = set(endpoints.iloc[np.argsort(screen_rank)[:k]]["event"])
        top_full = set(endpoints.iloc[np.argsort(full_rank)[:k]]["event"])
        practical.append(
            {
                "top_k": k,
                "recovered": len(top_screen & top_full),
                "recall": len(top_screen & top_full) / k,
                "mean_absolute_rank_error": float(np.mean(rank_error)),
                "maximum_rank_error": float(np.max(rank_error)),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(design_rows).to_csv(
        output_dir / "design_aware_sensitivity.csv", index=False
    )
    h3_events.to_csv(output_dir / "h3_parameterization_events.csv", index=False)
    pd.DataFrame(h3_checks).to_csv(
        output_dir / "h3_parameterization_summary.csv", index=False
    )
    pd.DataFrame(audit_rows).to_csv(
        output_dir / "posterior_array_audit.csv", index=False
    )
    pd.DataFrame(practical).to_csv(
        output_dir / "screening_practical_performance.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project_root.resolve(), args.output_dir.resolve())
