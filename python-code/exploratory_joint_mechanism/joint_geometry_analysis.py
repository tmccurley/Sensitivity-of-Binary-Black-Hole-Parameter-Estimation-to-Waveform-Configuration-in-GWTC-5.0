from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


ROOT = Path(r"C:\Users\tmark\Documents\Codex\2026-08-15\referenced-chatgpt-conversation-this-is-an")
ARCHIVE = ROOT / "work" / "gwtc5_h1_rerun" / "GWTC5_reproducible_archive_v1_0_0"
REPLICATION = ROOT / "work" / "gwtc5_h1_rerun" / "GWTC5_prospective_h3_replication"
OUTPUT = ROOT / "work" / "gwtc5_joint_mechanism_exploratory"
DOWNLOADS = Path(r"C:\Users\tmark\Downloads")

XPHM = "C00:IMRPhenomXPHM-SpinTaylor"
XPNR = "C00:IMRPhenomXPNR"
MODELS = (XPHM, XPNR)
PARAMETERS = ("mass_ratio", "chi_eff", "chi_p")
CONFIRM_SAFE = {
    XPHM: "C00__IMRPhenomXPHM-SpinTaylor",
    XPNR: "C00__IMRPhenomXPNR",
}
DIRECTION_COUNT = 360
REPLICATION_CAP = 20_000
REPLICATION_SEED = 20260802


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def central_width(values: np.ndarray) -> float:
    lo, hi = np.quantile(values, [0.05, 0.95])
    return float(hi - lo)


def wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Exact empirical W1 for one-dimensional arrays with unequal counts."""
    u = np.sort(np.asarray(x, dtype=np.float64))
    v = np.sort(np.asarray(y, dtype=np.float64))
    all_values = np.concatenate((u, v))
    all_values.sort()
    if all_values.size < 2:
        return 0.0
    deltas = np.diff(all_values)
    points = all_values[:-1]
    u_cdf = np.searchsorted(u, points, side="right") / u.size
    v_cdf = np.searchsorted(v, points, side="right") / v.size
    return float(np.sum(np.abs(u_cdf - v_cdf) * deltas))


def nw1(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    scale = 0.5 * (central_width(x) + central_width(y))
    raw = wasserstein_1d(x, y)
    return raw, scale, raw / scale if scale > 0 else math.nan


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(np.asarray(x, dtype=np.float64))
    ry = rankdata(np.asarray(y, dtype=np.float64))
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float(np.dot(rx, rx) * np.dot(ry, ry)))
    return float(np.dot(rx, ry) / denom) if denom > 0 else math.nan


def sliced_wasserstein(
    x_q: np.ndarray,
    x_chi: np.ndarray,
    y_q: np.ndarray,
    y_chi: np.ndarray,
    q_scale: float,
    chi_scale: float,
) -> tuple[float, float, float]:
    pooled_q_center = float(np.median(np.concatenate((x_q, y_q))))
    pooled_chi_center = float(np.median(np.concatenate((x_chi, y_chi))))
    x = np.column_stack(((x_q - pooled_q_center) / q_scale, (x_chi - pooled_chi_center) / chi_scale))
    y = np.column_stack(((y_q - pooled_q_center) / q_scale, (y_chi - pooled_chi_center) / chi_scale))
    angles = np.arange(DIRECTION_COUNT, dtype=np.float64) * math.pi / DIRECTION_COUNT
    distances = np.empty(DIRECTION_COUNT, dtype=np.float64)
    batch = 24
    for start in range(0, DIRECTION_COUNT, batch):
        stop = min(start + batch, DIRECTION_COUNT)
        directions = np.column_stack((np.cos(angles[start:stop]), np.sin(angles[start:stop])))
        x_proj = x @ directions.T
        y_proj = y @ directions.T
        for offset in range(stop - start):
            distances[start + offset] = wasserstein_1d(x_proj[:, offset], y_proj[:, offset])
    return float(distances.mean()), float(distances.min()), float(distances.max())


def principal_alignment(
    x_q: np.ndarray,
    x_chi: np.ndarray,
    y_q: np.ndarray,
    y_chi: np.ndarray,
    q_scale: float,
    chi_scale: float,
    shift: np.ndarray,
) -> tuple[float, float, float]:
    def standardized_centered(q: np.ndarray, chi: np.ndarray) -> np.ndarray:
        return np.column_stack(
            ((q - np.median(q)) / q_scale, (chi - np.median(chi)) / chi_scale)
        )

    x = standardized_centered(x_q, x_chi)
    y = standardized_centered(y_q, y_chi)
    covariance = 0.5 * (np.cov(x, rowvar=False) + np.cov(y, rowvar=False))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    shift_norm = float(np.linalg.norm(shift))
    alignment = abs(float(np.dot(shift / shift_norm, axis))) if shift_norm > 0 else math.nan
    axis_angle = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    if axis_angle < 0:
        axis_angle += 180.0
    variance_fraction = float(np.max(eigenvalues) / np.sum(eigenvalues))
    return alignment, axis_angle, variance_fraction


def confirmation_paths(events: Iterable[str]) -> dict[str, Path]:
    wanted = set(events)
    found: dict[str, Path] = {}
    processed = ARCHIVE / "data" / "processed"
    for path in sorted(processed.rglob("*_posteriors.hdf5")):
        with h5py.File(path, "r") as source:
            for event in source.keys():
                if event in wanted:
                    if event in found:
                        raise RuntimeError(f"Duplicate compact event {event}")
                    found[event] = path
    missing = wanted - set(found)
    if missing:
        raise FileNotFoundError(f"Missing compact events: {sorted(missing)}")
    return found


def load_confirmatory() -> tuple[list[str], dict[tuple[str, str, str], np.ndarray], dict[str, str]]:
    rows = read_csv(
        ARCHIVE / "reference_results" / "confirmatory_analysis" / "confirmatory_event_endpoints.csv"
    )
    events = [row["event"] for row in rows]
    paths = confirmation_paths(events)
    samples: dict[tuple[str, str, str], np.ndarray] = {}
    source_names: dict[str, str] = {}
    handles: dict[Path, h5py.File] = {}
    try:
        for event in events:
            path = paths[event]
            source_names[event] = path.name
            source = handles.setdefault(path, h5py.File(path, "r"))
            for model in MODELS:
                group = source[event][CONFIRM_SAFE[model]]
                for parameter in PARAMETERS:
                    values = np.asarray(group[parameter][:], dtype=np.float64)
                    values = values[np.isfinite(values)]
                    samples[(event, model, parameter)] = values
    finally:
        for handle in handles.values():
            handle.close()
    return events, samples, source_names


def load_replication() -> tuple[list[str], dict[tuple[str, str, str], np.ndarray], dict[str, str]]:
    rows = read_csv(REPLICATION / "results" / "replication_final_strict_9_event_sample.csv")
    events = [row["event"] for row in rows]
    samples: dict[tuple[str, str, str], np.ndarray] = {}
    source_names: dict[str, str] = {}
    rng = np.random.default_rng(REPLICATION_SEED)
    for row in rows:
        event = row["event"]
        if event == "GW240705_053215":
            path = DOWNLOADS / "Unconfirmed 891143.crdownload"
        else:
            path = DOWNLOADS / row["expected_filename"]
        if not path.exists():
            raise FileNotFoundError(path)
        source_names[event] = path.name
        with h5py.File(path, "r") as source:
            for model in MODELS:
                posterior = source[model]["posterior_samples"]
                names = set(posterior.dtype.names or ())
                missing = set(PARAMETERS) - names
                if missing:
                    raise KeyError(f"{event}/{model} missing {sorted(missing)}")
                original_count = int(posterior.shape[0])
                stored_count = min(original_count, REPLICATION_CAP)
                if stored_count < original_count:
                    indices = np.sort(rng.choice(original_count, stored_count, replace=False))
                else:
                    indices = np.arange(original_count)
                for parameter in PARAMETERS:
                    values = np.asarray(posterior.fields(parameter)[indices], dtype=np.float64)
                    values = values[np.isfinite(values)]
                    samples[(event, model, parameter)] = values
    return events, samples, source_names


def analyze_stage(
    stage: str,
    events: list[str],
    samples: dict[tuple[str, str, str], np.ndarray],
    source_names: dict[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, event in enumerate(events, start=1):
        print(f"[{stage}] {index}/{len(events)} {event}", flush=True)
        xq = samples[(event, XPHM, "mass_ratio")]
        xc = samples[(event, XPHM, "chi_eff")]
        xp = samples[(event, XPHM, "chi_p")]
        yq = samples[(event, XPNR, "mass_ratio")]
        yc = samples[(event, XPNR, "chi_eff")]
        yp = samples[(event, XPNR, "chi_p")]

        q_w1, q_scale, q_nw1 = nw1(xq, yq)
        chi_w1, chi_scale, chi_nw1 = nw1(xc, yc)
        chip_w1, chip_scale, chip_nw1 = nw1(xp, yp)
        sw_mean, sw_min, sw_max = sliced_wasserstein(
            xq, xc, yq, yc, q_scale, chi_scale
        )

        delta_q = (float(np.median(yq)) - float(np.median(xq))) / q_scale
        delta_chi = (float(np.median(yc)) - float(np.median(xc))) / chi_scale
        shift = np.asarray([delta_q, delta_chi], dtype=np.float64)
        shift_norm = float(np.linalg.norm(shift))
        shift_l1 = abs(delta_q) + abs(delta_chi)
        chi_share = abs(delta_chi) / shift_l1 if shift_l1 > 0 else math.nan
        alignment, axis_angle, axis_fraction = principal_alignment(
            xq, xc, yq, yc, q_scale, chi_scale, shift
        )

        rows.append(
            {
                "stage": stage,
                "event": event,
                "source_file": source_names[event],
                "xphm_sample_count": len(xq),
                "xpnr_sample_count": len(yq),
                "mass_ratio_nw1": q_nw1,
                "chi_eff_nw1": chi_nw1,
                "chi_p_nw1": chip_nw1,
                "joint_q_chi_eff_sliced_w1": sw_mean,
                "joint_projection_min_w1": sw_min,
                "joint_projection_max_w1": sw_max,
                "standardized_median_shift_q": delta_q,
                "standardized_median_shift_chi_eff": delta_chi,
                "standardized_median_shift_norm": shift_norm,
                "chi_eff_share_of_l1_shift": chi_share,
                "xphm_spearman_q_chi_eff": spearman(xq, xc),
                "xpnr_spearman_q_chi_eff": spearman(yq, yc),
                "absolute_shift_pc1_alignment": alignment,
                "pc1_angle_degrees": axis_angle,
                "pc1_variance_fraction": axis_fraction,
                "chi_eff_nw1_exceeds_chi_p_nw1": chi_nw1 > chip_nw1,
                "mass_ratio_w1": q_w1,
                "chi_eff_w1": chi_w1,
                "chi_p_w1": chip_w1,
                "mass_ratio_average_width_90": q_scale,
                "chi_eff_average_width_90": chi_scale,
                "chi_p_average_width_90": chip_scale,
            }
        )
    return rows


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {
        "protocol_date": "2026-08-19",
        "direction_count": DIRECTION_COUNT,
        "event_count": len(rows),
        "stages": {},
    }
    for stage in ("confirmatory", "replication", "combined"):
        subset = rows if stage == "combined" else [row for row in rows if row["stage"] == stage]
        def arr(key: str) -> np.ndarray:
            return np.asarray([float(row[key]) for row in subset], dtype=np.float64)
        stage_summary = {
            "event_count": len(subset),
            "joint_sliced_w1_median": float(np.median(arr("joint_q_chi_eff_sliced_w1"))),
            "joint_sliced_w1_min": float(np.min(arr("joint_q_chi_eff_sliced_w1"))),
            "joint_sliced_w1_max": float(np.max(arr("joint_q_chi_eff_sliced_w1"))),
            "chi_eff_nw1_median": float(np.median(arr("chi_eff_nw1"))),
            "mass_ratio_nw1_median": float(np.median(arr("mass_ratio_nw1"))),
            "chi_p_nw1_median": float(np.median(arr("chi_p_nw1"))),
            "chi_eff_exceeds_chi_p_count": int(sum(bool(row["chi_eff_nw1_exceeds_chi_p_nw1"]) for row in subset)),
            "median_chi_eff_shift_share": float(np.median(arr("chi_eff_share_of_l1_shift"))),
            "median_absolute_shift_pc1_alignment": float(np.nanmedian(arr("absolute_shift_pc1_alignment"))),
            "median_pc1_variance_fraction": float(np.median(arr("pc1_variance_fraction"))),
            "rho_joint_vs_chi_eff_nw1": spearman(arr("joint_q_chi_eff_sliced_w1"), arr("chi_eff_nw1")),
            "rho_joint_vs_mass_ratio_nw1": spearman(arr("joint_q_chi_eff_sliced_w1"), arr("mass_ratio_nw1")),
            "rho_joint_vs_chi_p_nw1": spearman(arr("joint_q_chi_eff_sliced_w1"), arr("chi_p_nw1")),
            "median_xphm_q_chi_eff_spearman": float(np.median(arr("xphm_spearman_q_chi_eff"))),
            "median_xpnr_q_chi_eff_spearman": float(np.median(arr("xpnr_spearman_q_chi_eff"))),
            "median_absolute_change_q_chi_eff_spearman": float(np.median(np.abs(arr("xpnr_spearman_q_chi_eff") - arr("xphm_spearman_q_chi_eff")))),
        }
        summary["stages"][stage] = stage_summary
    return summary


def make_markdown(summary: dict[str, object], rows: list[dict[str, object]]) -> str:
    lines = [
        "# Exploratory joint-posterior mechanism results",
        "",
        "This analysis was frozen before its joint-posterior and chi_p outcomes were calculated. It is post-confirmatory and exploratory.",
        "",
        "## Stage summaries",
        "",
        "| Stage | n | Median joint SW1 | Median q NW1 | Median chi_eff NW1 | Median chi_p NW1 | chi_eff > chi_p | Median alignment |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in ("confirmatory", "replication", "combined"):
        s = summary["stages"][stage]
        lines.append(
            f"| {stage} | {s['event_count']} | {s['joint_sliced_w1_median']:.4f} | "
            f"{s['mass_ratio_nw1_median']:.4f} | {s['chi_eff_nw1_median']:.4f} | "
            f"{s['chi_p_nw1_median']:.4f} | {s['chi_eff_exceeds_chi_p_count']}/{s['event_count']} | "
            f"{s['median_absolute_shift_pc1_alignment']:.3f} |"
        )
    c = summary["stages"]["combined"]
    lines.extend(
        [
            "",
            "## Combined descriptive relationships",
            "",
            f"- Spearman rho between joint sliced W1 and chi_eff NW1: {c['rho_joint_vs_chi_eff_nw1']:.3f}",
            f"- Spearman rho between joint sliced W1 and mass-ratio NW1: {c['rho_joint_vs_mass_ratio_nw1']:.3f}",
            f"- Spearman rho between joint sliced W1 and chi_p NW1: {c['rho_joint_vs_chi_p_nw1']:.3f}",
            f"- Median share of the standardized median-shift L1 magnitude contributed by chi_eff: {c['median_chi_eff_shift_share']:.3f}",
            f"- Median absolute alignment of the configuration shift with the within-posterior first principal axis: {c['median_absolute_shift_pc1_alignment']:.3f}",
            f"- Median fraction of standardized within-posterior variance along that axis: {c['median_pc1_variance_fraction']:.3f}",
            f"- Median absolute change in within-posterior q--chi_eff Spearman correlation: {c['median_absolute_change_q_chi_eff_spearman']:.3f}",
            "",
            "## Boundaries",
            "",
            "These summaries can show whether the observed chi_eff marginal response accompanies joint q--chi_eff movement or reshaping. They do not isolate a waveform component, prove causation, establish catalog prevalence, or identify the more accurate configuration.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    confirm_events, confirm_samples, confirm_sources = load_confirmatory()
    replication_events, replication_samples, replication_sources = load_replication()
    rows = analyze_stage("confirmatory", confirm_events, confirm_samples, confirm_sources)
    rows.extend(analyze_stage("replication", replication_events, replication_samples, replication_sources))
    summary = summarize(rows)

    event_path = OUTPUT / "joint_geometry_event_metrics.csv"
    summary_path = OUTPUT / "joint_geometry_summary.json"
    markdown_path = OUTPUT / "joint_geometry_summary.md"
    write_csv(event_path, rows)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(make_markdown(summary, rows), encoding="utf-8")

    manifest = {
        "protocol": "EXPLORATORY_Q_CHIEFF_PROTOCOL_2026-08-19.md",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "event_metrics_sha256": hashlib.sha256(event_path.read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "parameters": list(PARAMETERS),
        "directions": DIRECTION_COUNT,
        "replication_seed": REPLICATION_SEED,
    }
    (OUTPUT / "joint_geometry_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
