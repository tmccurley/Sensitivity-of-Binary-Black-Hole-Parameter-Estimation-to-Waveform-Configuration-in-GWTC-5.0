"""Create a publication-ready paired plot from the sealed replication CSV."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "replication_analysis"
SOURCE = RESULTS / "replication_event_deltas.csv"
OUTPUT = RESULTS / "figure_prospective_h3_replication.png"
MANIFEST = RESULTS / "figure_prospective_h3_replication_manifest.json"

WIDTH = 2400
HEIGHT = 1600
DPI = 300


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    windows_font = Path("C:/Windows/Fonts") / name
    if windows_font.is_file():
        return ImageFont.truetype(str(windows_font), size=size)
    return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size=size)


def main() -> None:
    if OUTPUT.exists() or MANIFEST.exists():
        raise FileExistsError("Figure output already exists; refusing to overwrite it.")

    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 9:
        raise RuntimeError("Expected exactly nine sealed event rows.")

    for row in rows:
        row["distance"] = float(row["luminosity_distance_normalized_wasserstein_1"])
        row["chi"] = float(row["chi_eff_normalized_wasserstein_1"])
        row["delta"] = float(row["delta_chi_eff_minus_luminosity_distance"])
    if not all(row["delta"] > 0 for row in rows):
        raise RuntimeError("The sealed CSV no longer has nine positive differences.")

    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)

    title_font = font(54, bold=True)
    subtitle_font = font(34)
    axis_font = font(34, bold=True)
    tick_font = font(28)
    legend_font = font(28)
    note_font = font(27)

    foreground = "#252a31"
    muted = "#5e6874"
    grid = "#d8dde3"
    strata_colors = {
        "high": "#2c7fb8",
        "medium": "#31a354",
        "low": "#e08214",
    }

    draw.text(
        (WIDTH / 2, 58),
        "Prospective H3 replication across nine locked GWTC-5 events",
        fill=foreground,
        font=title_font,
        anchor="ma",
    )
    draw.text(
        (WIDTH / 2, 132),
        "9/9 positive paired differences; median Δ = 0.05597; "
        "one-sided Wilcoxon W = 45, p = 0.001953",
        fill=muted,
        font=subtitle_font,
        anchor="ma",
    )

    legend_y = 218
    legend_x = 650
    draw.text(
        (legend_x, legend_y),
        "Screening-score stratum:",
        fill=foreground,
        font=legend_font,
        anchor="lm",
    )
    legend_x += 365
    for label in ("high", "medium", "low"):
        draw.line(
            (legend_x, legend_y, legend_x + 55, legend_y),
            fill=strata_colors[label],
            width=8,
        )
        legend_x += 72
        draw.text(
            (legend_x, legend_y),
            label.capitalize(),
            fill=foreground,
            font=legend_font,
            anchor="lm",
        )
        legend_x += 145

    plot_left = 330
    plot_right = 2260
    plot_top = 300
    plot_bottom = 1320
    x_distance = 860
    x_chi = 1730

    maximum = max(max(row["distance"], row["chi"]) for row in rows)
    tick_step = 0.025
    y_max = math.ceil(maximum / tick_step) * tick_step
    ticks = [index * tick_step for index in range(int(round(y_max / tick_step)) + 1)]

    def y_position(value: float) -> float:
        return plot_bottom - (value / y_max) * (plot_bottom - plot_top)

    for tick in ticks:
        y = y_position(tick)
        draw.line((plot_left, y, plot_right, y), fill=grid, width=3)
        draw.text(
            (plot_left - 24, y),
            f"{tick:.3f}",
            fill=muted,
            font=tick_font,
            anchor="rm",
        )

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=foreground, width=4)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=foreground, width=4)

    for row in rows:
        color = strata_colors[row["score_stratum"]]
        y_distance = y_position(row["distance"])
        y_chi = y_position(row["chi"])
        draw.line((x_distance, y_distance, x_chi, y_chi), fill=color, width=7)
        radius = 14
        draw.ellipse(
            (
                x_distance - radius,
                y_distance - radius,
                x_distance + radius,
                y_distance + radius,
            ),
            fill="white",
            outline=color,
            width=7,
        )
        draw.rectangle(
            (
                x_chi - radius,
                y_chi - radius,
                x_chi + radius,
                y_chi + radius,
            ),
            fill=color,
        )

    draw.text(
        (x_distance, plot_bottom + 55),
        "Luminosity distance",
        fill=foreground,
        font=axis_font,
        anchor="ma",
    )
    draw.text(
        (x_chi, plot_bottom + 55),
        "chi_eff",
        fill=foreground,
        font=axis_font,
        anchor="ma",
    )

    y_label = Image.new("RGBA", (1100, 80), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_label)
    y_draw.text(
        (550, 40),
        "Normalized Wasserstein-1 distance (NW1)",
        fill=foreground,
        font=axis_font,
        anchor="mm",
    )
    y_label = y_label.rotate(90, expand=True)
    image.paste(
        y_label,
        (62, int((plot_top + plot_bottom - y_label.height) / 2)),
        y_label,
    )

    draw.text(
        (WIDTH / 2, 1495),
        "Each line is one prospectively selected reserve event; open circles mark "
        "luminosity distance and filled squares mark chi_eff.",
        fill=muted,
        font=note_font,
        anchor="ma",
    )

    image.save(OUTPUT, format="PNG", dpi=(DPI, DPI), optimize=True)
    manifest = {
        "figure": OUTPUT.name,
        "source": SOURCE.name,
        "source_sha256": sha256_file(SOURCE),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "figure_sha256": sha256_file(OUTPUT),
        "dimensions_pixels": [WIDTH, HEIGHT],
        "dpi": DPI,
        "derived_statistics_only": True,
        "new_hypothesis_tests": False,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
