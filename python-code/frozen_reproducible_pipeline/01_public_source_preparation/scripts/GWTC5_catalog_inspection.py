
from pathlib import Path
import json
import re

import h5py
import numpy as np
import pandas as pd


# ============================================================
# 1. LOCATE THE PROJECT AND SUMMARY TABLE
# ============================================================

def find_project_root(start=None):
    start = Path(start or Path.cwd()).resolve()

    for candidate in [start, *start.parents]:
        if (candidate / "data" / "raw" / "catalog").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find a project root containing data/raw/catalog."
    )


PROJECT_ROOT = find_project_root()
CATALOG_DIR = PROJECT_ROOT / "data" / "raw" / "catalog"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "catalog_inspection"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

matches = sorted(CATALOG_DIR.glob("*PESummaryTable*.hdf5"))

if len(matches) != 1:
    raise RuntimeError(
        "Expected exactly one PESummaryTable HDF5 file in:\n"
        f"{CATALOG_DIR}\n"
        f"Found: {[path.name for path in matches]}"
    )

summary_file = matches[0]

print("Project root:", PROJECT_ROOT)
print("Summary file:", summary_file)


# ============================================================
# 2. FIND STRUCTURED DATASETS
# ============================================================

def decode_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, np.generic):
        return decode_value(value.item())

    return value


def dataset_to_dataframe(dataset):
    array = dataset[()]
    columns = {}

    for field in dataset.dtype.names:
        values = array[field]

        if values.ndim == 1:
            columns[field] = [
                decode_value(value)
                for value in values
            ]
        else:
            columns[field] = [
                json.dumps(
                    np.asarray(value).tolist(),
                    default=str,
                )
                for value in values
            ]

    return pd.DataFrame(columns)


dataset_inventory = []

with h5py.File(summary_file, "r") as source:

    def visitor(name, obj):
        if (
            isinstance(obj, h5py.Dataset)
            and obj.dtype.names is not None
            and obj.ndim >= 1
        ):
            dataset_inventory.append(
                {
                    "path": name,
                    "rows": int(obj.shape[0]),
                    "fields": len(obj.dtype.names),
                }
            )

    source.visititems(visitor)

    if not dataset_inventory:
        raise RuntimeError(
            "No structured dataset was found in the summary file."
        )

    selected = max(
        dataset_inventory,
        key=lambda row: row["rows"] * row["fields"],
    )

    table = dataset_to_dataframe(source[selected["path"]])

pd.DataFrame(dataset_inventory).to_csv(
    OUTPUT_DIR / "catalog_hdf5_dataset_inventory.csv",
    index=False,
)

table.to_csv(
    OUTPUT_DIR / "catalog_raw_summary.csv",
    index=False,
)

print("Selected dataset:", selected)
print("Summary rows:", len(table))
print("Columns:", len(table.columns))


# ============================================================
# 3. BUILD A COLUMN INVENTORY
# ============================================================

column_rows = []

for column in table.columns:
    numeric = pd.to_numeric(
        table[column],
        errors="coerce",
    )

    examples = (
        table[column]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .head(6)
        .tolist()
    )

    column_rows.append(
        {
            "column": column,
            "dtype": str(table[column].dtype),
            "nonmissing_count": int(table[column].notna().sum()),
            "numeric_count": int(numeric.notna().sum()),
            "unique_count": int(table[column].nunique(dropna=True)),
            "example_values": " | ".join(examples),
        }
    )

column_inventory = pd.DataFrame(column_rows)

column_inventory.to_csv(
    OUTPUT_DIR / "catalog_column_inventory.csv",
    index=False,
)


# ============================================================
# 4. IDENTIFY EVENT, ANALYSIS, AND FILENAME COLUMNS
# ============================================================

event_pattern = re.compile(r"^GW\d{6}_\d{6}$")

def choose_column(score_function, minimum_score):
    scores = []

    for column in table.columns:
        values = table[column].dropna().astype(str)

        if values.empty:
            continue

        scores.append(
            (
                float(score_function(values)),
                column,
            )
        )

    if not scores:
        return None

    score, column = max(scores)

    return column if score >= minimum_score else None


event_column = choose_column(
    lambda values: values.str.match(event_pattern).mean(),
    0.5,
)

model_tokens = [
    "IMRPhenomXPHM",
    "IMRPhenomXPNR",
    "SEOBNRv5PHM",
    "NRSur7dq4",
]

analysis_column = choose_column(
    lambda values: values.apply(
        lambda value: any(
            token in value
            for token in model_tokens
        )
    ).mean(),
    0.25,
)

filename_column = choose_column(
    lambda values: values.str.contains(
        "combined_PEDataRelease.hdf5",
        regex=False,
    ).mean(),
    0.25,
)

if event_column is None or analysis_column is None:
    raise RuntimeError(
        "Could not identify the event or analysis-label column. "
        "Review catalog_column_inventory.csv."
    )

print("Event column:", event_column)
print("Analysis column:", analysis_column)
print("Filename column:", filename_column)


# ============================================================
# 5. CREATE RUN AND EVENT MANIFESTS
# ============================================================

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

    return "OTHER"


run_manifest = pd.DataFrame(
    {
        "event": table[event_column].astype(str),
        "analysis_label": table[analysis_column].astype(str),
    }
)

run_manifest["model_family"] = (
    run_manifest["analysis_label"].map(model_family)
)

if filename_column is not None:
    run_manifest["event_filename"] = (
        table[filename_column].astype(str)
    )
else:
    run_manifest["event_filename"] = ""

run_manifest.to_csv(
    OUTPUT_DIR / "catalog_run_manifest.csv",
    index=False,
)

event_rows = []

for event, rows in run_manifest.groupby("event"):
    families = set(rows["model_family"])

    filenames = [
        value
        for value in rows["event_filename"].astype(str)
        if value and value.lower() != "nan"
    ]

    event_rows.append(
        {
            "event": event,
            "event_filename": filenames[0] if filenames else "",
            "run_count": int(len(rows)),
            "has_xphm": "XPHM" in families,
            "has_xpnr": "XPNR" in families,
            "has_seob": "SEOB" in families,
            "has_nrsur": "NRSUR" in families,
            "xphm_xpnr_candidate": (
                "XPHM" in families
                and "XPNR" in families
            ),
            "analysis_labels": " | ".join(
                sorted(rows["analysis_label"].unique())
            ),
        }
    )

event_manifest = pd.DataFrame(event_rows).sort_values("event")

event_manifest.to_csv(
    OUTPUT_DIR / "catalog_event_manifest.csv",
    index=False,
)


# ============================================================
# 6. SAVE INSPECTION SUMMARY
# ============================================================

inspection = {
    "summary_file": summary_file.name,
    "selected_dataset": selected,
    "summary_row_count": int(len(table)),
    "column_count": int(len(table.columns)),
    "event_column": event_column,
    "analysis_column": analysis_column,
    "filename_column": filename_column,
    "distinct_event_count": int(event_manifest["event"].nunique()),
    "xphm_xpnr_candidate_count": int(
        event_manifest["xphm_xpnr_candidate"].sum()
    ),
    "columns": list(table.columns),
}

with (
    OUTPUT_DIR / "catalog_inspection_summary.json"
).open("w", encoding="utf-8") as file:
    json.dump(inspection, file, indent=2)

print()
print("Inspection complete.")
print("Distinct events:", inspection["distinct_event_count"])
print(
    "XPHM-XPNR candidates:",
    inspection["xphm_xpnr_candidate_count"],
)
print("Outputs:", OUTPUT_DIR)
