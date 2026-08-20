"""Regenerate the manuscript H1 figure from the frozen confirmatory results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


BLUE = "#2b6ea6"
GRID = "#d9e2ec"
AXIS = "#20252b"
BORDER = "#cbd5e1"
FONT_REGULAR = Path(r"C:\Windows\Fonts\arial.ttf")
FONT_ITALIC = Path(r"C:\Windows\Fonts\ariali.ttf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoints", type=Path, required=True)
    parser.add_argument("--hypotheses", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_and_verify(
    endpoints_path: Path, hypotheses_path: Path
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Load the frozen endpoints and verify the locked H1 statistic."""
    endpoints = pd.read_csv(endpoints_path)
    hypothesis = pd.read_csv(hypotheses_path).set_index("hypothesis").loc["H1"]
    x = endpoints["network_matched_filter_snr"].to_numpy(float)
    y = endpoints["event_median_NW1"].to_numpy(float)
    if len(x) != 18 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("Expected 18 finite H1 endpoint pairs")

    # Spearman correlation is Pearson correlation of the rank vectors.
    x_rank = pd.Series(x).rank(method="average").to_numpy(float)
    y_rank = pd.Series(y).rank(method="average").to_numpy(float)
    rho = float(np.corrcoef(x_rank, y_rank)[0, 1])
    locked_rho = float(hypothesis["effect_statistic"])
    locked_p = float(hypothesis["raw_p_value"])
    if not np.isclose(rho, locked_rho, rtol=0, atol=1e-14):
        raise RuntimeError(f"H1 rho mismatch: {rho} != {locked_rho}")
    return x, y, locked_rho, locked_p


def domains(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Return padded plot domains that keep every marker inside the frame."""
    x_pad = 0.05 * float(np.ptp(x))
    y_pad = 0.09 * float(np.ptp(y))
    return (
        float(x.min() - x_pad),
        float(x.max() + x_pad),
        max(0.0, float(y.min() - y_pad)),
        float(y.max() + y_pad),
    )


def render_png(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    rho: float,
    p_value: float,
) -> None:
    """Render a high-resolution PNG with margins reserved for full labels."""
    width, height = 2160, 1620
    left, right, top, bottom = 300, 90, 100, 240
    plot_left, plot_right = left, width - right
    plot_top, plot_bottom = top, height - bottom
    x_min, x_max, y_min, y_max = domains(x, y)

    regular = ImageFont.truetype(str(FONT_REGULAR), 49)
    tick_font = ImageFont.truetype(str(FONT_REGULAR), 43)
    annotation_font = ImageFont.truetype(str(FONT_ITALIC), 43)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    def sx(value: float) -> float:
        return plot_left + (value - x_min) / (x_max - x_min) * (plot_right - plot_left)

    def sy(value: float) -> float:
        return plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    x_ticks = [10.0, 12.5, 15.0, 17.5]
    y_ticks = [0.025, 0.050, 0.075, 0.100]
    for value in x_ticks:
        px = sx(value)
        draw.line((px, plot_top, px, plot_bottom), fill=GRID, width=3)
        draw.line((px, plot_bottom, px, plot_bottom + 12), fill=AXIS, width=4)
        text = f"{value:.1f}"
        box = draw.textbbox((0, 0), text, font=tick_font)
        draw.text((px - (box[2] - box[0]) / 2, plot_bottom + 24), text, font=tick_font, fill=AXIS)
    for value in y_ticks:
        py = sy(value)
        draw.line((plot_left, py, plot_right, py), fill=GRID, width=3)
        draw.line((plot_left - 12, py, plot_left, py), fill=AXIS, width=4)
        text = f"{value:.3f}"
        box = draw.textbbox((0, 0), text, font=tick_font)
        draw.text(
            (plot_left - 24 - (box[2] - box[0]), py - (box[3] - box[1]) / 2 - 4),
            text,
            font=tick_font,
            fill=AXIS,
        )

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=AXIS, width=5)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=AXIS, width=5)
    for x_value, y_value in zip(x, y, strict=True):
        px, py, radius = sx(float(x_value)), sy(float(y_value)), 18
        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius),
            fill=BLUE,
            outline="white",
            width=3,
        )

    x_label = "Catalog network matched-filter SNR"
    x_box = draw.textbbox((0, 0), x_label, font=regular)
    draw.text(
        ((width - (x_box[2] - x_box[0])) / 2, height - 96),
        x_label,
        font=regular,
        fill=AXIS,
    )

    y_label = "Median NW1 across four primary parameters"
    y_box = draw.textbbox((0, 0), y_label, font=regular)
    y_image = Image.new(
        "RGBA",
        (y_box[2] - y_box[0] + 20, y_box[3] - y_box[1] + 20),
        (255, 255, 255, 0),
    )
    ImageDraw.Draw(y_image).text((10, 4), y_label, font=regular, fill=AXIS)
    y_image = y_image.rotate(90, expand=True)
    image.paste(y_image, (52, int((height - y_image.height) / 2 - 5)), y_image)

    annotation = f"ρ = {rho:.4f}, p = {p_value:.4f}"
    a_box = draw.textbbox((0, 0), annotation, font=annotation_font)
    a_width, a_height = a_box[2] - a_box[0], a_box[3] - a_box[1]
    # The upper-right corner is empty, so the annotation does not obscure the
    # high-NW1 event near SNR 10.
    a_x, a_y = plot_right - a_width - 28, plot_top + 24
    draw.rounded_rectangle(
        (a_x - 14, a_y - 10, a_x + a_width + 14, a_y + a_height + 16),
        radius=7,
        fill="white",
        outline=BORDER,
        width=3,
    )
    draw.text((a_x, a_y), annotation, font=annotation_font, fill=AXIS)

    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Title", "H1: network SNR versus event-median NW1")
    metadata.add_text("Source", "Frozen GWTC-5 confirmatory endpoint table")
    image.save(path, dpi=(300, 300), pnginfo=metadata, optimize=True)


def render_pdf(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    rho: float,
    p_value: float,
) -> None:
    """Render a genuine vector PDF rather than a PNG with a PDF extension."""
    page_width, page_height = 518.4, 388.8
    left, right, top, bottom = 72.0, 17.0, 24.0, 58.0
    plot_left, plot_right = left, page_width - right
    plot_bottom, plot_top = bottom, page_height - top
    x_min, x_max, y_min, y_max = domains(x, y)

    pdfmetrics.registerFont(TTFont("FigureArial", str(FONT_REGULAR)))
    pdfmetrics.registerFont(TTFont("FigureArialItalic", str(FONT_ITALIC)))
    c = canvas.Canvas(str(path), pagesize=(page_width, page_height), pageCompression=1)
    c.setTitle("H1: network SNR versus event-median NW1")
    c.setAuthor("Thomas McCurley")
    c.setSubject("Generated from the frozen GWTC-5 confirmatory endpoint table")

    def sx(value: float) -> float:
        return plot_left + (value - x_min) / (x_max - x_min) * (plot_right - plot_left)

    def sy(value: float) -> float:
        return plot_bottom + (value - y_min) / (y_max - y_min) * (plot_top - plot_bottom)

    c.setLineWidth(0.7)
    c.setStrokeColor(GRID)
    for value in [10.0, 12.5, 15.0, 17.5]:
        px = sx(value)
        c.line(px, plot_bottom, px, plot_top)
    for value in [0.025, 0.050, 0.075, 0.100]:
        py = sy(value)
        c.line(plot_left, py, plot_right, py)

    c.setStrokeColor(AXIS)
    c.setLineWidth(1.2)
    c.line(plot_left, plot_bottom, plot_right, plot_bottom)
    c.line(plot_left, plot_bottom, plot_left, plot_top)
    c.setFont("FigureArial", 10.5)
    for value in [10.0, 12.5, 15.0, 17.5]:
        px = sx(value)
        c.line(px, plot_bottom, px, plot_bottom - 3)
        c.drawCentredString(px, plot_bottom - 16, f"{value:.1f}")
    for value in [0.025, 0.050, 0.075, 0.100]:
        py = sy(value)
        c.line(plot_left - 3, py, plot_left, py)
        c.drawRightString(plot_left - 7, py - 3.5, f"{value:.3f}")

    c.setFillColor(BLUE)
    c.setStrokeColorRGB(1, 1, 1)
    c.setLineWidth(0.6)
    for x_value, y_value in zip(x, y, strict=True):
        c.circle(sx(float(x_value)), sy(float(y_value)), 4.5, stroke=1, fill=1)

    c.setFillColor(AXIS)
    c.setFont("FigureArial", 12.5)
    c.drawCentredString(page_width / 2, 14, "Catalog network matched-filter SNR")
    c.saveState()
    c.translate(17, page_height / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "Median NW1 across four primary parameters")
    c.restoreState()

    annotation = f"ρ = {rho:.4f}, p = {p_value:.4f}"
    c.setFont("FigureArialItalic", 10.5)
    text_width = pdfmetrics.stringWidth(annotation, "FigureArialItalic", 10.5)
    a_x, a_y = plot_right - text_width - 7, plot_top - 19
    c.setFillColorRGB(1, 1, 1)
    c.setStrokeColor(BORDER)
    c.roundRect(a_x - 4, a_y - 3, text_width + 8, 16, 2, stroke=1, fill=1)
    c.setFillColor(AXIS)
    c.drawString(a_x, a_y, annotation)
    c.showPage()
    c.save()


def main() -> None:
    args = parse_args()
    x, y, rho, p_value = load_and_verify(args.endpoints, args.hypotheses)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "fig_h1_snr_large.png"
    pdf_path = args.output_dir / "fig_h1_snr_large.pdf"
    render_png(png_path, x, y, rho, p_value)
    render_pdf(pdf_path, x, y, rho, p_value)
    print(f"rows={len(x)}")
    print(f"rho={rho:.16g}")
    print(f"p={p_value:.16g}")
    print(png_path.resolve())
    print(pdf_path.resolve())


if __name__ == "__main__":
    main()
