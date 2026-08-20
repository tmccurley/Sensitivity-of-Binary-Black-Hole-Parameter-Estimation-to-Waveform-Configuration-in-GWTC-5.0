"""
Catalog-wide XPHM-XPNR summary screening for GWTC-5.0.

This script uses the CSV created by GWTC5_catalog_inspection.py:

data/processed/catalog_inspection/catalog_raw_summary.csv

It creates:

data/processed/catalog_screening/
├── catalog_run_manifest_corrected.csv
├── catalog_xphm_xpnr_screening_metrics.csv
├── catalog_xphm_xpnr_event_ranking.csv
├── diagnostic_batch_01.csv
├── diagnostic_batch_01_files.txt
├── catalog_screening_summary.json
└── catalog_screening_report.md

Important
---------
The summary table's lower and upper columns are one-sided error widths,
not interval endpoints. The reconstructed interval is:

    [median - lower, median + upper]

The screening score is only a prioritization proxy. It is not a
replacement for posterior-sample Wasserstein distance or a full
metadata audit.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


PILOT_EVENTS = {
    "GW240612_081540",
    "GW240908_082628",
    "GW241011_233834",
    "GW241230_233618",
    "GW250114_082203",
}

PARAMETERS = {
    "chirp_mass_source": {
        "median": "chirp_mass_source_median",
        "lower": "chirp_mass_source_lower",
        "upper": "chirp_mass_source_upper",
    },
    "chi_eff": {
        "median": "chi_eff_median",
        "lower": "chi_eff_lower",
        "upper": "chi_eff_upper",
    },
    "luminosity_distance": {
        "median": "luminosity_distance_median",
        "lower": "luminosity_distance_lower",
        "upper": "luminosity_distance_upper",
    },
}

HIGH_COUNT = 6
CONTROL_COUNT = 6


def find_project_root(start=None):
    start = Path(start or Path.cwd()).resolve()

    for candidate in [start, *start.parents]:
        if (
            candidate
            / "data"
            / "processed"
            / "catalog_inspection"
            / "catalog_raw_summary.csv"
        ).exists():
            return candidate

    raise FileNotFoundError(
        "Could not find catalog_raw_summary.csv beneath "
        "data/processed/catalog_inspection."
    )


PROJECT_ROOT = find_project_root()

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "catalog_inspection"
    / "catalog_raw_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "catalog_screening"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(INPUT_FILE)


def model_family(label):
    label = str(label)

    if "IMRPhenomXPHM" in label:
        return "XPHM"
    if "IMRPhenomXPNR" in label:
        return "XPNR"
    if "SEOBNRv5PHM" in label:
        return "SEOB"
    if "NRSur7dq4" in label:
        return "NRSUR"
    if "Mixed" in label:
        return "MIXED"

    return "OTHER"


runs = raw.copy()
runs["event"] = runs["gw_name"].astype(str)
runs["analysis_label"] = runs["result_samples_key"].astype(str)
runs["model_family"] = runs["analysis_label"].map(model_family)
runs["release_tag"] = (
    runs["analysis_label"].str.split(":").str[0]
)
runs["event_filename"] = runs["result_file_name"].astype(str)
runs["already_downloaded_pilot"] = (
    runs["event"].isin(PILOT_EVENTS)
)

runs.to_csv(
    OUTPUT_DIR / "catalog_run_manifest_corrected.csv",
    index=False,
)


def preferred_rows(frame, family):
    subset = frame[
        frame["model_family"] == family
    ].copy()

    subset["release_priority"] = np.where(
        subset["release_tag"] == "C00",
        0,
        1,
    )

    subset = subset.sort_values(
        [
            "event",
            "release_priority",
            "analysis_label",
        ]
    )

    return (
        subset.drop_duplicates("event", keep="first")
        .drop(columns="release_priority")
        .set_index("event")
    )


xphm = preferred_rows(runs, "XPHM")
xpnr = preferred_rows(runs, "XPNR")

paired_events = sorted(
    set(xphm.index) & set(xpnr.index)
)


def numeric(row, column):
    return pd.to_numeric(
        pd.Series([row[column]]),
        errors="coerce",
    ).iloc[0]


def interval_from_errors(row, mapping):
    median = numeric(row, mapping["median"])
    lower_error = numeric(row, mapping["lower"])
    upper_error = numeric(row, mapping["upper"])

    if not all(
        np.isfinite(value)
        for value in [
            median,
            lower_error,
            upper_error,
        ]
    ):
        return np.nan, np.nan, np.nan

    return (
        float(median),
        float(median - lower_error),
        float(median + upper_error),
    )


def overlap_fraction(
    lower_a,
    upper_a,
    lower_b,
    upper_b,
):
    intersection = max(
        0.0,
        min(upper_a, upper_b)
        - max(lower_a, lower_b),
    )

    union = (
        max(upper_a, upper_b)
        - min(lower_a, lower_b)
    )

    if union <= 0:
        return np.nan

    return float(intersection / union)


screening_rows = []

for event in paired_events:
    xphm_row = xphm.loc[event]
    xpnr_row = xpnr.loc[event]

    for parameter, mapping in PARAMETERS.items():
        median_a, lower_a, upper_a = (
            interval_from_errors(
                xphm_row,
                mapping,
            )
        )
        median_b, lower_b, upper_b = (
            interval_from_errors(
                xpnr_row,
                mapping,
            )
        )

        values = [
            median_a,
            lower_a,
            upper_a,
            median_b,
            lower_b,
            upper_b,
        ]

        if not all(
            np.isfinite(value)
            for value in values
        ):
            continue

        width_a = upper_a - lower_a
        width_b = upper_b - lower_b
        mean_width = 0.5 * (width_a + width_b)

        if mean_width <= 0:
            continue

        screening_rows.append(
            {
                "event": event,
                "event_filename": str(
                    xphm_row["event_filename"]
                ),
                "parameter": parameter,
                "xphm_analysis_label": str(
                    xphm_row["analysis_label"]
                ),
                "xpnr_analysis_label": str(
                    xpnr_row["analysis_label"]
                ),
                "xphm_median": median_a,
                "xpnr_median": median_b,
                "xphm_lower_90": lower_a,
                "xphm_upper_90": upper_a,
                "xpnr_lower_90": lower_b,
                "xpnr_upper_90": upper_b,
                "summary_normalized_median_shift": (
                    abs(median_a - median_b)
                    / mean_width
                ),
                "summary_interval_overlap_90": (
                    overlap_fraction(
                        lower_a,
                        upper_a,
                        lower_b,
                        upper_b,
                    )
                ),
                "already_downloaded_pilot": (
                    event in PILOT_EVENTS
                ),
            }
        )

screening = pd.DataFrame(screening_rows)

screening.to_csv(
    OUTPUT_DIR
    / "catalog_xphm_xpnr_screening_metrics.csv",
    index=False,
)


event_ranking = (
    screening.groupby(
        ["event", "event_filename"],
        as_index=False,
    )
    .agg(
        parameters_screened=(
            "parameter",
            "nunique",
        ),
        median_screening_shift=(
            "summary_normalized_median_shift",
            "median",
        ),
        maximum_screening_shift=(
            "summary_normalized_median_shift",
            "max",
        ),
        minimum_interval_overlap_90=(
            "summary_interval_overlap_90",
            "min",
        ),
    )
)


def average_feature(event, column):
    values = []

    for frame in [xphm, xpnr]:
        if event not in frame.index:
            continue

        value = numeric(
            frame.loc[event],
            column,
        )

        if np.isfinite(value):
            values.append(float(value))

    return (
        float(np.mean(values))
        if values
        else np.nan
    )


feature_columns = {
    "network_matched_filter_snr": (
        "network_matched_filter_snr_median"
    ),
    "total_mass_source": (
        "total_mass_source_median"
    ),
    "chirp_mass_source": (
        "chirp_mass_source_median"
    ),
    "chi_eff": "chi_eff_median",
    "luminosity_distance": (
        "luminosity_distance_median"
    ),
}

for output_name, source_column in (
    feature_columns.items()
):
    event_ranking[output_name] = [
        average_feature(
            event,
            source_column,
        )
        for event in event_ranking["event"]
    ]

event_ranking[
    "already_downloaded_pilot"
] = event_ranking["event"].isin(
    PILOT_EVENTS
)

event_ranking = event_ranking.sort_values(
    [
        "maximum_screening_shift",
        "median_screening_shift",
    ],
    ascending=False,
)

event_ranking.insert(
    0,
    "screening_rank",
    np.arange(
        1,
        len(event_ranking) + 1,
    ),
)

event_ranking.to_csv(
    OUTPUT_DIR
    / "catalog_xphm_xpnr_event_ranking.csv",
    index=False,
)


# Select six high-score events.
candidates = event_ranking[
    ~event_ranking["already_downloaded_pilot"]
].copy()

high = (
    candidates.sort_values(
        "maximum_screening_shift",
        ascending=False,
    )
    .head(HIGH_COUNT)
    .copy()
)

high["selection_group"] = (
    "high_summary_disagreement"
)
high["matched_to_event"] = ""
high["feature_distance"] = np.nan

remaining = candidates[
    ~candidates["event"].isin(
        high["event"]
    )
].copy()

matching_features = [
    "network_matched_filter_snr",
    "total_mass_source",
    "chirp_mass_source",
]

centers = {
    column: candidates[column].median()
    for column in matching_features
}

scales = {
    column: candidates[column].std(ddof=0)
    for column in matching_features
}

for column in matching_features:
    if (
        not np.isfinite(scales[column])
        or scales[column] == 0
    ):
        scales[column] = 1.0


def feature_vector(row):
    return np.array(
        [
            (
                row[column]
                - centers[column]
            )
            / scales[column]
            for column in matching_features
        ],
        dtype=float,
    )


control_rows = []
unused = set(remaining.index)

for _, high_row in high.iterrows():
    if len(control_rows) >= CONTROL_COUNT:
        break

    high_vector = feature_vector(high_row)
    best_index = None
    best_distance = np.inf

    for control_index in list(unused):
        control_row = remaining.loc[
            control_index
        ]
        control_vector = feature_vector(
            control_row
        )

        valid = (
            np.isfinite(high_vector)
            & np.isfinite(control_vector)
        )

        if valid.sum() == 0:
            continue

        distance = float(
            np.sqrt(
                np.mean(
                    (
                        high_vector[valid]
                        - control_vector[valid]
                    ) ** 2
                )
            )
        )

        # Favor lower disagreement among otherwise similar controls.
        distance += (
            0.20
            * float(
                control_row[
                    "maximum_screening_shift"
                ]
            )
        )

        if distance < best_distance:
            best_distance = distance
            best_index = control_index

    if best_index is not None:
        selected = remaining.loc[
            best_index
        ].copy()

        selected["selection_group"] = (
            "matched_lower_disagreement_control"
        )
        selected["matched_to_event"] = (
            high_row["event"]
        )
        selected["feature_distance"] = (
            best_distance
        )

        control_rows.append(selected)
        unused.remove(best_index)

controls = pd.DataFrame(control_rows)

batch = pd.concat(
    [high, controls],
    ignore_index=True,
    sort=False,
)

batch.insert(
    0,
    "download_order",
    np.arange(1, len(batch) + 1),
)

batch.to_csv(
    OUTPUT_DIR / "diagnostic_batch_01.csv",
    index=False,
)

with (
    OUTPUT_DIR
    / "diagnostic_batch_01_files.txt"
).open("w", encoding="utf-8") as file:
    for filename in batch["event_filename"]:
        file.write(str(filename) + "\n")


summary = {
    "summary_rows": int(len(raw)),
    "distinct_events": int(
        raw["gw_name"].nunique()
    ),
    "xphm_xpnr_candidates": int(
        len(paired_events)
    ),
    "screened_parameters": list(
        PARAMETERS
    ),
    "diagnostic_batch_size": int(
        len(batch)
    ),
    "high_disagreement_count": int(
        (
            batch["selection_group"]
            == "high_summary_disagreement"
        ).sum()
    ),
    "control_count": int(
        (
            batch["selection_group"]
            == (
                "matched_lower_"
                "disagreement_control"
            )
        ).sum()
    ),
}

with (
    OUTPUT_DIR
    / "catalog_screening_summary.json"
).open("w", encoding="utf-8") as file:
    json.dump(
        summary,
        file,
        indent=2,
    )

print("Catalog screening complete.")
print("Outputs:", OUTPUT_DIR)
print()
print(batch[
    [
        "download_order",
        "event",
        "selection_group",
        "matched_to_event",
        "maximum_screening_shift",
        "network_matched_filter_snr",
        "total_mass_source",
    ]
].to_string(index=False))
