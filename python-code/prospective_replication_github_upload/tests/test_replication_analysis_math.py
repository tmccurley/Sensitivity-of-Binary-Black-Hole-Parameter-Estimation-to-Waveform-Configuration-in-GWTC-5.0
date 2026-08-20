"""Synthetic tests for the frozen replication calculations.

No GWTC-5 posterior is opened by this test module.
"""

from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "GWTC5_prospective_h3_replication_analysis.py"
)
SPEC = importlib.util.spec_from_file_location("replication_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_interval_width_uses_fifth_and_ninety_fifth_percentiles() -> None:
    values = np.arange(101, dtype=float)
    summary = MODULE.interval_summary(values)
    assert summary["lower_90"] == 5.0
    assert summary["median"] == 50.0
    assert summary["upper_90"] == 95.0
    assert summary["width_90"] == 90.0


def test_nw1_matches_simple_equal_width_shift() -> None:
    x = np.arange(101, dtype=float)
    y = x + 9.0
    metrics = MODULE.pair_metrics(x, y)
    assert np.isclose(metrics["wasserstein_1"], 9.0)
    assert np.isclose(metrics["average_width_90"], 90.0)
    assert np.isclose(metrics["normalized_wasserstein_1"], 0.1)


def test_nw1_is_invariant_to_a_common_positive_scale() -> None:
    x = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.asarray([0.5, 1.5, 2.5, 3.5, 4.5])
    original = MODULE.pair_metrics(x, y)["normalized_wasserstein_1"]
    scaled = MODULE.pair_metrics(17.0 * x, 17.0 * y)[
        "normalized_wasserstein_1"
    ]
    assert np.isclose(original, scaled)


def test_parent_wilcoxon_call_for_nine_positive_differences() -> None:
    lower = np.arange(1.0, 10.0)
    higher = lower + np.arange(1.0, 10.0)
    result = MODULE.wilcoxon(
        higher,
        lower,
        alternative="greater",
        zero_method="wilcox",
        method="auto",
    )
    assert result.statistic == 45.0
    assert np.isclose(result.pvalue, 1.0 / 512.0)


def test_loader_reads_only_locked_fields_from_raw_compound_dataset() -> None:
    dtype = np.dtype(
        [
            ("chi_eff", "f8"),
            ("luminosity_distance", "f8"),
            ("unrelated_parameter", "f8"),
        ]
    )
    posterior = np.zeros(12, dtype=dtype)
    posterior["chi_eff"] = np.linspace(-0.2, 0.3, posterior.size)
    posterior["luminosity_distance"] = np.linspace(100.0, 900.0, posterior.size)
    posterior["unrelated_parameter"] = np.nan

    # Keep the synthetic file inside the writable replication workspace.
    path = Path(__file__).resolve().parent / f"synthetic_{uuid.uuid4().hex}.hdf5"
    try:
        with MODULE.h5py.File(path, "w") as target:
            for model in MODULE.MODELS:
                target.create_group(model).create_dataset(
                    "posterior_samples", data=posterior
                )

        rows = [{"event": "synthetic", "cell_id": "synthetic_cell"}]
        samples, preflight = MODULE.load_locked_samples(
            rows, {"synthetic": path}
        )
    finally:
        path.unlink(missing_ok=True)

    assert len(samples) == 4
    assert len(preflight) == 4
    assert all(row["sampling_method"] == "all_samples" for row in preflight)
    assert all(values.size == 12 for values in samples.values())
