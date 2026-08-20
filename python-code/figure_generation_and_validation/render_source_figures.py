from pathlib import Path

import pypdfium2 as pdfium

source = Path(r"work\GWTC_FINAL_source_20260815\figures")
destination = Path(r"tmp\pdfs\source_figures")
destination.mkdir(parents=True, exist_ok=True)

for name in [
    "fig_cdf_chieff_threepanel",
    "fig_cdf_chirpmass_high",
    "fig_cdf_chirpmass_low",
    "fig_parameter_summary_large",
]:
    document = pdfium.PdfDocument(source / f"{name}.pdf")
    image = document[0].render(scale=4).to_pil()
    image.save(destination / f"{name}.png")
    print(name, image.size)
