"""
Extract the two prespecified Batch B replacement events.

Target events
-------------
GW240630_101703
GW240910_103535

Replacement mapping
-------------------
GW240630_101703 replaces GW250109_010541
GW240910_103535 replaces GW240413_022019

Expected raw files
------------------
IGWN-GWTC5p0-29ebe06b7_25-GW240630_101703-combined_PEDataRelease.hdf5
IGWN-GWTC5p0-29ebe06b7_25-GW240910_103535-combined_PEDataRelease.hdf5

Place the raw files in:
GWTC5/data/raw/events/

Outputs
-------
GWTC5/data/processed/confirmatory_batch_B_replacements_01/
"""

from __future__ import annotations

from pathlib import Path
import csv
import json
import sys
import traceback
from typing import Any

import h5py
import numpy as np


TARGET_EVENTS = ['GW240630_101703', 'GW240910_103535']

REPLACEMENT_FOR = {'GW240630_101703': 'GW250109_010541', 'GW240910_103535': 'GW240413_022019'}

PRIMARY_PARAMETERS = [
    "chirp_mass",
    "mass_ratio",
    "chi_eff",
    "luminosity_distance",
]

SECONDARY_PARAMETERS = [
    "chirp_mass_source",
    "chi_p",
    "total_mass",
    "total_mass_source",
    "mass_1",
    "mass_2",
    "mass_1_source",
    "mass_2_source",
    "redshift",
    "network_optimal_snr",
    "network_matched_filter_snr",
    "network_precessing_snr",
    "log_likelihood",
    "log_prior",
]

MAX_STORED_SAMPLES_PER_ANALYSIS = 100_000
RANDOM_SEED = 20260802
OVERWRITE = False

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


def find_project_root(start=None):
    start = Path(start or Path.cwd()).resolve()

    for candidate in [start, *start.parents]:
        if (
            candidate
            / "data"
            / "raw"
            / "events"
        ).exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate the GWTC5 project root."
    )


PROJECT_ROOT = find_project_root()

RAW_EVENT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "events"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "confirmatory_batch_B_replacements_01"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREFIX = "confirmatory_batch_B_replacements_01"

POSTERIOR_OUTPUT = (
    OUTPUT_DIR
    / f"{PREFIX}_posteriors.hdf5"
)
SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / f"{PREFIX}_parameter_summary.csv"
)
METADATA_OUTPUT = (
    OUTPUT_DIR
    / f"{PREFIX}_metadata_audit.json"
)
COMPARISON_OUTPUT = (
    OUTPUT_DIR
    / f"{PREFIX}_metadata_comparison.csv"
)
MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / f"{PREFIX}_manifest.json"
)
LOG_OUTPUT = (
    OUTPUT_DIR
    / f"{PREFIX}_processing_log.txt"
)


def log(message=""):
    print(message)

    with LOG_OUTPUT.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(str(message) + "\n")


def find_event_file(event_name):
    matches = sorted(
        RAW_EVENT_DIR.glob(
            f"*{event_name}*combined_PEDataRelease.hdf5"
        )
    )

    if not matches:
        return None

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple files found for {event_name}:\n"
            + "\n".join(
                str(path)
                for path in matches
            )
        )

    return matches[0]


def convert_value(value: Any):
    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    if isinstance(value, np.generic):
        return convert_value(value.item())

    if isinstance(value, np.ndarray):
        if value.size == 1:
            return convert_value(
                value.reshape(-1)[0]
            )

        return [
            convert_value(item)
            for item in value.tolist()
        ]

    if isinstance(value, tuple):
        return [
            convert_value(item)
            for item in value
        ]

    if isinstance(value, list):
        return [
            convert_value(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            str(key): convert_value(item)
            for key, item in value.items()
        }

    return value


def read_dataset(source, path):
    if path not in source:
        return None

    obj = source[path]

    if not isinstance(obj, h5py.Dataset):
        return None

    return convert_value(obj[()])


def read_named_datasets(
    source,
    base_path,
    names,
):
    output = {}

    for name in names:
        value = read_dataset(
            source,
            f"{base_path}/{name}",
        )

        if value is not None:
            output[name] = value

    return output


def read_all_small_datasets(
    source,
    group_path,
):
    output = {}

    if group_path not in source:
        return output

    group = source[group_path]

    if not isinstance(group, h5py.Group):
        return output

    def visitor(relative_name, obj):
        if (
            isinstance(obj, h5py.Dataset)
            and obj.size <= 1000
        ):
            output[relative_name] = (
                convert_value(obj[()])
            )

    group.visititems(visitor)
    return output


def posterior_analysis_labels(source):
    return [
        key
        for key in source.keys()
        if (
            isinstance(
                source[key],
                h5py.Group,
            )
            and "posterior_samples"
            in source[key]
            and isinstance(
                source[key][
                    "posterior_samples"
                ],
                h5py.Dataset,
            )
        )
    ]


def read_parameter(
    posterior,
    parameter,
):
    if posterior.dtype.names is None:
        raise TypeError(
            "posterior_samples is not a structured dataset."
        )

    if parameter not in posterior.dtype.names:
        raise KeyError(parameter)

    return np.asarray(
        posterior.fields(parameter)[:],
        dtype=np.float64,
    ).reshape(-1)


def safe_group_name(text):
    return (
        str(text)
        .replace("\\", "_")
        .replace("/", "_")
        .replace(":", "__")
        .replace(" ", "_")
    )


def canonical_text(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


if LOG_OUTPUT.exists():
    LOG_OUTPUT.unlink()

if (
    POSTERIOR_OUTPUT.exists()
    and not OVERWRITE
):
    raise FileExistsError(
        f"{POSTERIOR_OUTPUT} already exists."
    )

for path in [
    POSTERIOR_OUTPUT,
    SUMMARY_OUTPUT,
    METADATA_OUTPUT,
    COMPARISON_OUTPUT,
    MANIFEST_OUTPUT,
]:
    if path.exists() and OVERWRITE:
        path.unlink()

log(f"Project root: {PROJECT_ROOT}")
log(f"Target events: {TARGET_EVENTS}")
log(f"Raw directory: {RAW_EVENT_DIR}")
log(f"Output directory: {OUTPUT_DIR}")
log(f"Python: {sys.version}")
log(f"h5py: {h5py.__version__}")
log(f"NumPy: {np.__version__}")

rng = np.random.default_rng(
    RANDOM_SEED
)

manifest = {
    "replacement_for": REPLACEMENT_FOR,
    "target_events": TARGET_EVENTS,
    "primary_parameters": PRIMARY_PARAMETERS,
    "secondary_parameters": SECONDARY_PARAMETERS,
    "random_seed": RANDOM_SEED,
    "events": {},
}

metadata_audit = {
    "events": {},
}

summary_rows = []

with h5py.File(
    POSTERIOR_OUTPUT,
    "w",
) as reduced_output:

    reduced_output.attrs[
        "purpose"
    ] = (
        "Compact posterior samples for "
        "Batch B prespecified replacements"
    )

    for event_name in TARGET_EVENTS:
        raw_file = find_event_file(
            event_name
        )

        if raw_file is None:
            log(
                f"MISSING: {event_name}"
            )
            manifest["events"][
                event_name
            ] = {
                "status": "missing",
            }
            continue

        log(
            f"\nPROCESSING {event_name}"
        )
        log(
            f"Source: {raw_file.name}"
        )

        event_manifest = {
            "status": "processing",
            "source_file": raw_file.name,
            "source_file_size_bytes": (
                raw_file.stat().st_size
            ),
            "analyses": {},
        }

        event_metadata = {
            "source_file": raw_file.name,
            "analyses": {},
        }

        try:
            with h5py.File(
                raw_file,
                "r",
            ) as source:

                labels = (
                    posterior_analysis_labels(
                        source
                    )
                )

                if not labels:
                    raise RuntimeError(
                        "No posterior analysis groups were found."
                    )

                event_group = (
                    reduced_output.create_group(
                        event_name
                    )
                )

                for label in labels:
                    log(f"  {label}")

                    posterior = source[
                        label
                    ][
                        "posterior_samples"
                    ]

                    available_parameters = (
                        list(
                            posterior.dtype.names
                            or []
                        )
                    )
                    available_set = set(
                        available_parameters
                    )

                    original_count = int(
                        posterior.shape[0]
                    )

                    stored_count = min(
                        original_count,
                        MAX_STORED_SAMPLES_PER_ANALYSIS,
                    )

                    if (
                        stored_count
                        < original_count
                    ):
                        indices = np.sort(
                            rng.choice(
                                original_count,
                                size=stored_count,
                                replace=False,
                            )
                        )
                        sampling_method = (
                            "random_without_replacement"
                        )
                    else:
                        indices = np.arange(
                            original_count
                        )
                        sampling_method = (
                            "all_samples"
                        )

                    output_group = (
                        event_group.create_group(
                            safe_group_name(
                                label
                            )
                        )
                    )

                    output_group.attrs[
                        "original_label"
                    ] = label
                    output_group.attrs[
                        "original_sample_count"
                    ] = original_count
                    output_group.attrs[
                        "stored_sample_count"
                    ] = stored_count
                    output_group.attrs[
                        "sampling_method"
                    ] = sampling_method

                    extracted = []
                    missing = []
                    skipped = {}

                    for parameter in (
                        PRIMARY_PARAMETERS
                        + SECONDARY_PARAMETERS
                    ):
                        if (
                            parameter
                            not in available_set
                        ):
                            missing.append(
                                parameter
                            )
                            continue

                        try:
                            full_values = (
                                read_parameter(
                                    posterior,
                                    parameter,
                                )
                            )
                        except Exception as error:
                            skipped[
                                parameter
                            ] = (
                                f"{type(error).__name__}: "
                                f"{error}"
                            )
                            continue

                        finite_values = (
                            full_values[
                                np.isfinite(
                                    full_values
                                )
                            ]
                        )

                        if (
                            finite_values.size
                            == 0
                        ):
                            skipped[
                                parameter
                            ] = (
                                "No finite values."
                            )
                            continue

                        reduced_values = (
                            full_values[
                                indices
                            ]
                        )

                        output_group.create_dataset(
                            parameter,
                            data=reduced_values,
                            compression="gzip",
                            compression_opts=6,
                            shuffle=True,
                        )

                        extracted.append(
                            parameter
                        )

                        if (
                            parameter
                            in PRIMARY_PARAMETERS
                        ):
                            (
                                lower,
                                median,
                                upper,
                            ) = np.quantile(
                                finite_values,
                                [
                                    0.05,
                                    0.50,
                                    0.95,
                                ],
                            )

                            summary_rows.append(
                                {
                                    "event": event_name,
                                    "analysis_label": label,
                                    "parameter": parameter,
                                    "original_sample_count": original_count,
                                    "finite_sample_count": int(
                                        finite_values.size
                                    ),
                                    "stored_sample_count": int(
                                        reduced_values.size
                                    ),
                                    "mean": float(
                                        np.mean(
                                            finite_values
                                        )
                                    ),
                                    "standard_deviation": float(
                                        np.std(
                                            finite_values,
                                            ddof=1,
                                        )
                                    ),
                                    "median": float(
                                        median
                                    ),
                                    "lower_90": float(
                                        lower
                                    ),
                                    "upper_90": float(
                                        upper
                                    ),
                                    "credible_interval_width_90": float(
                                        upper - lower
                                    ),
                                }
                            )

                    event_manifest[
                        "analyses"
                    ][label] = {
                        "original_sample_count": original_count,
                        "stored_sample_count": stored_count,
                        "sampling_method": sampling_method,
                        "available_parameter_count": len(
                            available_parameters
                        ),
                        "extracted_parameters": extracted,
                        "missing_parameters": missing,
                        "skipped_parameters": skipped,
                    }

                    base = f"/{label}"

                    event_metadata[
                        "analyses"
                    ][label] = {
                        "top_level": {
                            "approximant": read_dataset(
                                source,
                                f"{base}/approximant",
                            ),
                            "description": read_dataset(
                                source,
                                f"{base}/description",
                            ),
                        },
                        "config": read_named_datasets(
                            source,
                            f"{base}/config_file/config",
                            CONFIG_KEYS,
                        ),
                        "metadata": read_named_datasets(
                            source,
                            f"{base}/meta_data/meta_data",
                            METADATA_KEYS,
                        ),
                        "sampler": read_named_datasets(
                            source,
                            f"{base}/meta_data/sampler",
                            SAMPLER_KEYS,
                        ),
                        "analytic_priors": read_all_small_datasets(
                            source,
                            f"{base}/priors/analytic",
                        ),
                    }

            event_manifest[
                "status"
            ] = "complete"

            manifest["events"][
                event_name
            ] = event_manifest

            metadata_audit["events"][
                event_name
            ] = event_metadata

            log(
                f"COMPLETE: {event_name}"
            )

        except Exception as error:
            event_manifest[
                "status"
            ] = "failed"
            event_manifest[
                "error"
            ] = (
                f"{type(error).__name__}: "
                f"{error}"
            )
            event_manifest[
                "traceback"
            ] = traceback.format_exc()

            manifest["events"][
                event_name
            ] = event_manifest

            log(
                f"FAILED: {event_name}: "
                f"{type(error).__name__}: "
                f"{error}"
            )

summary_columns = [
    "event",
    "analysis_label",
    "parameter",
    "original_sample_count",
    "finite_sample_count",
    "stored_sample_count",
    "mean",
    "standard_deviation",
    "median",
    "lower_90",
    "upper_90",
    "credible_interval_width_90",
]

with SUMMARY_OUTPUT.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=summary_columns,
    )
    writer.writeheader()
    writer.writerows(
        summary_rows
    )

with METADATA_OUTPUT.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metadata_audit,
        file,
        indent=2,
        ensure_ascii=False,
    )

with MANIFEST_OUTPUT.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        manifest,
        file,
        indent=2,
        ensure_ascii=False,
    )

comparison_rows = []

for event_name, event_data in (
    metadata_audit["events"].items()
):
    analyses = (
        event_data["analyses"]
    )
    labels = list(analyses)

    for section in [
        "top_level",
        "config",
        "metadata",
        "sampler",
        "analytic_priors",
    ]:
        keys = sorted(
            {
                key
                for label in labels
                for key in (
                    analyses[label][
                        section
                    ]
                )
            }
        )

        for key in keys:
            values = {
                label: canonical_text(
                    analyses[label][
                        section
                    ].get(key)
                )
                for label in labels
            }

            comparison_rows.append(
                {
                    "event": event_name,
                    "section": section,
                    "setting": key,
                    "identical_across_all_analyses": (
                        len(
                            set(
                                values.values()
                            )
                        )
                        == 1
                    ),
                    "values_json": json.dumps(
                        values,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )

with COMPARISON_OUTPUT.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "event",
            "section",
            "setting",
            "identical_across_all_analyses",
            "values_json",
        ],
    )
    writer.writeheader()
    writer.writerows(
        comparison_rows
    )

complete = [
    event
    for event, details
    in manifest["events"].items()
    if details.get("status")
    == "complete"
]

missing = [
    event
    for event, details
    in manifest["events"].items()
    if details.get("status")
    == "missing"
]

failed = [
    event
    for event, details
    in manifest["events"].items()
    if details.get("status")
    == "failed"
]

log("\nEXTRACTION FINISHED")
log(f"Complete events: {complete}")
log(f"Missing events: {missing}")
log(f"Failed events: {failed}")
log(
    f"Summary rows: {len(summary_rows)}"
)

if missing or failed:
    raise RuntimeError(
        "Replacement extraction did not complete successfully."
    )

print(
    "\nUpload the six compact outputs "
    f"from:\n{OUTPUT_DIR}"
)
