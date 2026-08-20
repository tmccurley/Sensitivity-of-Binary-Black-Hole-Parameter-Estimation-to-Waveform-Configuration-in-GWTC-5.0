"""Regenerate the corrected Figure 4 panels and Figure 5 from archived data."""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


ROOT = Path(
    r"work\gwtc5_h1_rerun\GWTC5_reproducible_archive_v1_0_0"
)
PROCESSED = ROOT / "data" / "processed"
REFERENCE = ROOT / "reference_results"
OUTPUT = Path(r"outputs\updated_figures")

MODELS = (
    "C00:IMRPhenomXPHM-SpinTaylor",
    "C00:IMRPhenomXPNR",
)


def index_events() -> dict[str, Path]:
    """Map each event to the compact HDF5 file containing its posterior arrays."""
    event_files: dict[str, Path] = {}
    for path in sorted(PROCESSED.rglob("*_posteriors.hdf5")):
        with h5py.File(path, "r") as source:
            for event in source.keys():
                if event in event_files:
                    raise RuntimeError(f"Duplicate compact event {event}")
                event_files[event] = path
    return event_files


def find_model_group(event_group: h5py.Group, model_label: str) -> h5py.Group:
    """Find a waveform group by the original model label stored in its attributes."""
    for group in event_group.values():
        original_label = group.attrs.get("original_label", "")
        if isinstance(original_label, bytes):
            original_label = original_label.decode("utf-8", errors="replace")
        if str(original_label) == model_label:
            return group
    raise KeyError(f"Missing model group {model_label}")


def load_pair(event_files: dict[str, Path], event: str, parameter: str) -> tuple[np.ndarray, np.ndarray]:
    """Load finite XPHM-ST and XPNR samples for one event and parameter."""
    with h5py.File(event_files[event], "r") as source:
        event_group = source[event]
        arrays = []
        for model in MODELS:
            values = np.asarray(
                find_model_group(event_group, model)[parameter][:], dtype=float
            ).reshape(-1)
            arrays.append(values[np.isfinite(values)])
    return arrays[0], arrays[1]


def normalized_w1(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate the manuscript's normalized Wasserstein-1 distance."""
    width_x = np.quantile(x, 0.95) - np.quantile(x, 0.05)
    width_y = np.quantile(y, 0.95) - np.quantile(y, 0.05)
    return float(wasserstein_distance(x, y) / (0.5 * (width_x + width_y)))


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save matching vector and high-resolution raster versions."""
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_cdf_panel(
    event_files: dict[str, Path],
    event: str,
    parameter: str,
    xlabel: str,
    stem: str,
    expected_nw1: float,
) -> None:
    """Create one corrected empirical-CDF panel for Figure 4."""
    xphm, xpnr = load_pair(event_files, event, parameter)
    value = normalized_w1(xphm, xpnr)
    if not np.isclose(value, expected_nw1, rtol=0, atol=5e-12):
        raise RuntimeError(
            f"Unexpected NW1 for {event}/{parameter}: {value} != {expected_nw1}"
        )

    fig, ax = plt.subplots(figsize=(4.1, 3.2))
    for samples, label, style in (
        (xphm, "XPHM-ST", {"color": "#2b6ea6", "linestyle": "-"}),
        (xpnr, "XPNR", {"color": "#f26445", "linestyle": "--"}),
    ):
        ordered = np.sort(samples)
        cumulative = np.arange(1, ordered.size + 1, dtype=float) / ordered.size
        ax.plot(ordered, cumulative, linewidth=2.0, label=label, **style)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative probability")
    # Use the literal event identifier; no TeX escape is needed inside Matplotlib text.
    ax.set_title(event, color="#425466", fontsize=10)
    ax.text(
        0.02,
        0.08,
        f"NW1 = {value:.4f}",
        transform=ax.transAxes,
        fontsize=8,
        color="#425466",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "pad": 2.0},
    )
    ax.set_ylim(0, 1.03)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(True, color="#d9e2ec", linewidth=0.8, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, stem)
    print(f"{stem}: NW1={value:.10f}")


def make_parameter_summary() -> None:
    """Create Figure 5 with the corrected final-event x-axis wording."""
    summary = pd.read_csv(
        REFERENCE / "robustness_analysis" / "robustness_parameter_summary.csv"
    ).set_index("parameter")
    order = [
        "chi_eff",
        "chirp_mass",
        "chirp_mass_source",
        "luminosity_distance",
        "mass_ratio",
    ]
    labels = [
        r"Effective spin, $\chi_{\mathrm{eff}}$",
        "Detector-frame chirp mass",
        "Source-frame chirp mass",
        "Luminosity distance",
        "Mass ratio",
    ]
    ordered = summary.loc[order]
    medians = ordered["median_NW1"].to_numpy(float)
    lower = medians - ordered["q25_NW1"].to_numpy(float)
    upper = ordered["q75_NW1"].to_numpy(float) - medians
    positions = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.errorbar(
        medians,
        positions,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="#2b6ea6",
        ecolor="#2b6ea6",
        linewidth=2.0,
        markersize=7,
        capsize=6,
    )
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("NW1 across the 18 final events")
    ax.set_xlim(left=0)
    ax.grid(axis="x", color="#d9e2ec", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "fig_parameter_summary_large")
    print("fig_parameter_summary_large: generated from robustness_parameter_summary.csv")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    event_files = index_events()
    make_cdf_panel(
        event_files,
        event="GW240630_101703",
        parameter="chi_eff",
        xlabel=r"$\chi_{\mathrm{eff}}$",
        stem="fig_cdf_chieff_threepanel",
        expected_nw1=0.1433495241632629,
    )
    make_cdf_panel(
        event_files,
        event="GW240910_103535",
        parameter="chirp_mass",
        xlabel=r"Detector-frame chirp mass ($M_\odot$)",
        stem="fig_cdf_chirpmass_high",
        expected_nw1=0.15119771828335038,
    )
    make_cdf_panel(
        event_files,
        event="GW241101_220523",
        parameter="chirp_mass",
        xlabel=r"Detector-frame chirp mass ($M_\odot$)",
        stem="fig_cdf_chirpmass_low",
        expected_nw1=0.007741618479245755,
    )
    make_parameter_summary()


if __name__ == "__main__":
    main()
