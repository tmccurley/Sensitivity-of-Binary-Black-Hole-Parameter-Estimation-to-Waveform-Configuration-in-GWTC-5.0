"""Correct label-only defects in archived vector figures without changing plot data."""

from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas


SOURCE = Path(r"work\GWTC_FINAL_source_20260815\figures")
OUTPUT = Path(r"output\pdf\updated_figures")


def overlay_text(
    source: Path,
    destination: Path,
    *,
    cover: tuple[float, float, float, float],
    text: str,
    x: float,
    y: float,
    font_size: float,
    color: tuple[float, float, float],
) -> None:
    """Cover one label and replace it while retaining the original vector plot."""
    reader = PdfReader(source)
    page = reader.pages[0]
    width = float(page.mediabox.width)
    height = float(page.mediabox.height)

    packet = BytesIO()
    drawing = canvas.Canvas(packet, pagesize=(width, height))
    drawing.setFillColorRGB(1, 1, 1)
    drawing.rect(*cover, stroke=0, fill=1)
    drawing.setFillColorRGB(*color)
    drawing.setFont("Helvetica", font_size)
    drawing.drawCentredString(x, y, text)
    drawing.save()
    packet.seek(0)

    page.merge_page(PdfReader(packet).pages[0])
    writer = PdfWriter()
    writer.add_page(page)
    with destination.open("wb") as stream:
        writer.write(stream)


def render_png(pdf_path: Path, png_path: Path) -> None:
    """Render a high-resolution PNG companion for easy preview and insertion."""
    document = pdfium.PdfDocument(pdf_path)
    image = document[0].render(scale=4.1667).to_pil()
    image.save(png_path, dpi=(300, 300))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    title_color = (0.165, 0.216, 0.267)

    panels = {
        "fig_cdf_chieff_threepanel": "GW240630_101703",
        "fig_cdf_chirpmass_high": "GW240910_103535",
        "fig_cdf_chirpmass_low": "GW241101_220523",
    }
    for stem, title in panels.items():
        source = SOURCE / f"{stem}.pdf"
        page = PdfReader(source).pages[0]
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        destination = OUTPUT / f"{stem}.pdf"
        overlay_text(
            source,
            destination,
            cover=(0, height - 12.8, width, 12.8),
            text=title,
            x=width / 2,
            y=height - 10.2,
            font_size=10.4,
            color=title_color,
        )
        render_png(destination, OUTPUT / f"{stem}.png")

    source = SOURCE / "fig_parameter_summary_large.pdf"
    page = PdfReader(source).pages[0]
    width = float(page.mediabox.width)
    destination = OUTPUT / "fig_parameter_summary_large.pdf"
    overlay_text(
        source,
        destination,
        cover=(82, 0, width - 82, 14.5),
        text="NW1 across the 18 final events",
        x=203.5,
        y=2.4,
        font_size=10.5,
        color=(0, 0, 0),
    )
    render_png(destination, OUTPUT / "fig_parameter_summary_large.png")

    print(f"Corrected figures written to {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
