from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from joint_geometry_analysis import (
    OUTPUT,
    XPHM,
    XPNR,
    central_width,
    load_confirmatory,
    load_replication,
    wasserstein_1d,
)

PARAMETERS = ("chi_eff", "chi_p")
METRICS = (
    "arithmetic_width_nw1",
    "geometric_width_nw1",
    "larger_width_nw1",
    "standardized_median_displacement",
    "ks_distance",
)


def ks_distance(x: np.ndarray, y: np.ndarray) -> float:
    u = np.sort(np.asarray(x, dtype=np.float64))
    v = np.sort(np.asarray(y, dtype=np.float64))
    points = np.sort(np.concatenate((u, v)))
    u_cdf = np.searchsorted(u, points, side="right") / u.size
    v_cdf = np.searchsorted(v, points, side="right") / v.size
    return float(np.max(np.abs(u_cdf - v_cdf)))


def metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    width_x = central_width(x)
    width_y = central_width(y)
    arithmetic = 0.5 * (width_x + width_y)
    geometric = math.sqrt(width_x * width_y)
    larger = max(width_x, width_y)
    w1 = wasserstein_1d(x, y)
    median_shift = abs(float(np.median(x)) - float(np.median(y)))
    return {
        "arithmetic_width_nw1": w1 / arithmetic,
        "geometric_width_nw1": w1 / geometric,
        "larger_width_nw1": w1 / larger,
        "standardized_median_displacement": median_shift / arithmetic,
        "ks_distance": ks_distance(x, y),
    }


def analyze_stage(stage: str, events, samples) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in events:
        values = {}
        for parameter in PARAMETERS:
            values[parameter] = metrics(
                samples[(event, XPHM, parameter)],
                samples[(event, XPNR, parameter)],
            )
        row: dict[str, object] = {"stage": stage, "event": event}
        for parameter in PARAMETERS:
            for metric in METRICS:
                row[f"{parameter}_{metric}"] = values[parameter][metric]
        for metric in METRICS:
            row[f"chi_eff_exceeds_chi_p_{metric}"] = (
                values["chi_eff"][metric] > values["chi_p"][metric]
            )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {"protocol_date": "2026-08-19", "stages": {}}
    for stage in ("confirmatory", "replication"):
        subset = [row for row in rows if row["stage"] == stage]
        stage_result: dict[str, object] = {"event_count": len(subset), "metrics": {}}
        for metric in METRICS:
            chi = np.asarray([float(row[f"chi_eff_{metric}"]) for row in subset])
            chip = np.asarray([float(row[f"chi_p_{metric}"]) for row in subset])
            stage_result["metrics"][metric] = {
                "chi_eff_median": float(np.median(chi)),
                "chi_p_median": float(np.median(chip)),
                "chi_eff_exceeds_chi_p_count": int(np.sum(chi > chip)),
                "minimum_chi_eff_minus_chi_p": float(np.min(chi - chip)),
                "median_chi_eff_minus_chi_p": float(np.median(chi - chip)),
            }
        result["stages"][stage] = stage_result
    return result


def markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Exploratory chi_eff versus chi_p sensitivity results",
        "",
        "This analysis was frozen after the initial NW1 ordering was observed and before the alternate-metric outcomes were calculated. It is post-confirmatory and exploratory.",
        "",
        "| Stage | Metric | chi_eff median | chi_p median | chi_eff > chi_p |",
        "|---|---|---:|---:|---:|",
    ]
    labels = {
        "arithmetic_width_nw1": "Arithmetic-width NW1",
        "geometric_width_nw1": "Geometric-width NW1",
        "larger_width_nw1": "Larger-width NW1",
        "standardized_median_displacement": "Standardized median displacement",
        "ks_distance": "KS distance",
    }
    for stage in ("confirmatory", "replication"):
        data = summary["stages"][stage]
        for metric in METRICS:
            item = data["metrics"][metric]
            lines.append(
                f"| {stage} | {labels[metric]} | {item['chi_eff_median']:.4f} | "
                f"{item['chi_p_median']:.4f} | {item['chi_eff_exceeds_chi_p_count']}/{data['event_count']} |"
            )
    lines.extend(
        [
            "",
            "The ordering is descriptive for these selected samples. The screening design included chi_eff disagreement but not chi_p disagreement, so these counts cannot establish selection-independent catalog prevalence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    confirm_events, confirm_samples, _ = load_confirmatory()
    replication_events, replication_samples, _ = load_replication()
    rows = analyze_stage("confirmatory", confirm_events, confirm_samples)
    rows.extend(analyze_stage("replication", replication_events, replication_samples))
    summary = summarize(rows)
    csv_path = OUTPUT / "chi_eff_vs_chi_p_sensitivity_events.csv"
    json_path = OUTPUT / "chi_eff_vs_chi_p_sensitivity_summary.json"
    md_path = OUTPUT / "chi_eff_vs_chi_p_sensitivity_summary.md"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown(summary), encoding="utf-8")
    manifest = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "events_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256(json_path.read_bytes()).hexdigest(),
        "metrics": list(METRICS),
    }
    (OUTPUT / "chi_eff_vs_chi_p_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
