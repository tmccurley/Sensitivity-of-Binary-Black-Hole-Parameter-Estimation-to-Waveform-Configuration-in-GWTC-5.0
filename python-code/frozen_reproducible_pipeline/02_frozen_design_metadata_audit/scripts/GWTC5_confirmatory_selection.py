"""
Reproduce the frozen GWTC-5.0 confirmatory event selection.

Input
-----
data/processed/catalog_screening/
    catalog_xphm_xpnr_event_ranking.csv

Outputs
-------
data/processed/confirmatory_design/
    confirmatory_selection_manifest.csv
    confirmatory_randomization_queue.csv
    confirmatory_first_reserves.csv
    confirmatory_batch_A.csv
    confirmatory_batch_B.csv
    confirmatory_batch_C.csv
    confirmatory_batch_A_files.txt
    confirmatory_batch_B_files.txt
    confirmatory_batch_C_files.txt
    confirmatory_all_primary_files.txt

The sample was frozen on 2026-08-01.
"""

from pathlib import Path
import numpy as np
import pandas as pd


DEVELOPMENT_EVENTS = {
    "GW240612_081540",
    "GW240908_082628",
    "GW241011_233834",
    "GW241230_233618",
    "GW250114_082203",
    "GW240920_124024",
    "GW241127_061008",
    "GW241129_021832",
    "GW240513_183302",
    "GW240615_113620",
    "GW241116_151753",
    "GW240621_195059",
    "GW241225_082815",
    "GW241130_034908",
    "GW240920_073424",
    "GW240919_061559",
    "GW241229_155844",
}

SELECTION_SEED = 20260801


def find_project_root(start=None):
    start = Path(start or Path.cwd()).resolve()

    for candidate in [start, *start.parents]:
        ranking_file = (
            candidate
            / "data"
            / "processed"
            / "catalog_screening"
            / "catalog_xphm_xpnr_event_ranking.csv"
        )

        if ranking_file.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate catalog_xphm_xpnr_event_ranking.csv."
    )


PROJECT_ROOT = find_project_root()

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "catalog_screening"
    / "catalog_xphm_xpnr_event_ranking.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "confirmatory_design"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ranking = pd.read_csv(INPUT_FILE)

untouched = (
    ranking[
        ~ranking["event"].isin(DEVELOPMENT_EVENTS)
    ]
    .copy()
    .sort_values("event")
    .reset_index(drop=True)
)

if len(untouched) != 87:
    raise RuntimeError(
        f"Expected 87 untouched events; found {len(untouched)}."
    )

untouched["score_stratum"] = pd.qcut(
    untouched["maximum_screening_shift"],
    q=3,
    labels=["low", "medium", "high"],
)

untouched["snr_stratum"] = pd.qcut(
    untouched["network_matched_filter_snr"],
    q=3,
    labels=["low", "medium", "high"],
)

rng = np.random.default_rng(SELECTION_SEED)
queue_parts = []

for (score_stratum, snr_stratum), group in untouched.groupby(
    ["score_stratum", "snr_stratum"],
    observed=True,
    sort=True,
):
    group = group.sort_values("event").copy()
    group = group.iloc[
        rng.permutation(len(group))
    ].copy()

    group["selection_order_within_stratum"] = np.arange(
        1,
        len(group) + 1,
    )

    group["selection_role"] = np.where(
        group["selection_order_within_stratum"] <= 2,
        "primary",
        "reserve",
    )

    group["score_stratum"] = str(score_stratum)
    group["snr_stratum"] = str(snr_stratum)
    queue_parts.append(group)

queue = pd.concat(queue_parts, ignore_index=True)

primary = queue[
    queue["selection_role"] == "primary"
].copy()

excluded_cells = {
    "A": {
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
    },
    "B": {
        ("low", "medium"),
        ("medium", "high"),
        ("high", "low"),
    },
    "C": {
        ("low", "high"),
        ("medium", "low"),
        ("high", "medium"),
    },
}

batch_rows = []

for (score_stratum, snr_stratum), group in primary.groupby(
    ["score_stratum", "snr_stratum"],
    sort=True,
):
    batches = sorted(
        batch
        for batch, excluded in excluded_cells.items()
        if (score_stratum, snr_stratum) not in excluded
    )

    group = group.sort_values(
        "selection_order_within_stratum"
    )

    for (_, row), batch in zip(group.iterrows(), batches):
        row = row.copy()
        row["download_batch"] = batch
        batch_rows.append(row)

selection = pd.DataFrame(batch_rows)

selection_columns = [
    "download_batch",
    "event",
    "event_filename",
    "score_stratum",
    "snr_stratum",
    "selection_order_within_stratum",
    "maximum_screening_shift",
    "median_screening_shift",
    "network_matched_filter_snr",
    "total_mass_source",
    "chirp_mass_source",
    "chi_eff",
    "luminosity_distance",
]

selection = (
    selection[selection_columns]
    .sort_values(
        [
            "download_batch",
            "score_stratum",
            "snr_stratum",
            "event",
        ]
    )
    .reset_index(drop=True)
)

selection.insert(
    0,
    "confirmatory_sample_order",
    np.arange(1, len(selection) + 1),
)

selection.to_csv(
    OUTPUT_DIR / "confirmatory_selection_manifest.csv",
    index=False,
)

queue_columns = [
    "event",
    "event_filename",
    "score_stratum",
    "snr_stratum",
    "selection_order_within_stratum",
    "selection_role",
    "maximum_screening_shift",
    "median_screening_shift",
    "network_matched_filter_snr",
    "total_mass_source",
    "chirp_mass_source",
    "chi_eff",
    "luminosity_distance",
]

queue_output = queue[queue_columns].sort_values(
    [
        "score_stratum",
        "snr_stratum",
        "selection_order_within_stratum",
    ]
)

queue_output.to_csv(
    OUTPUT_DIR / "confirmatory_randomization_queue.csv",
    index=False,
)

queue_output[
    queue_output["selection_order_within_stratum"] == 3
].to_csv(
    OUTPUT_DIR / "confirmatory_first_reserves.csv",
    index=False,
)

for batch_name in ["A", "B", "C"]:
    batch = selection[
        selection["download_batch"] == batch_name
    ]

    batch.to_csv(
        OUTPUT_DIR / f"confirmatory_batch_{batch_name}.csv",
        index=False,
    )

    with (
        OUTPUT_DIR / f"confirmatory_batch_{batch_name}_files.txt"
    ).open("w", encoding="utf-8") as file:
        for filename in batch["event_filename"]:
            file.write(str(filename) + "\n")

with (
    OUTPUT_DIR / "confirmatory_all_primary_files.txt"
).open("w", encoding="utf-8") as file:
    for filename in selection["event_filename"]:
        file.write(str(filename) + "\n")

print(selection.to_string(index=False))
