# GWTC-5.0 waveform-sensitivity study

This repository contains the interactive website, the complete Python analysis collection, generated results, and the figures used by the project.

## Repository layout

| Path | Contents |
| --- | --- |
| `website/` | Canonical HTML, CSS, JavaScript, and browser-ready analysis data |
| `python-code/` | All Python scripts, notebooks, tests, analysis variants, and local site tools |
| `figures/` | Complete figure collection from the figures-and-code archive |
| `results/` | Generated confirmatory-analysis outputs used to refresh the site |
| `preview-assets/` | Supporting tables, figures, and audit data retained for reference |
| `waveform-comparison/` | Compatibility redirect for the former dashboard URL |

The small root `index.html` and `about.html` files are GitHub Pages compatibility entry points. They redirect to the canonical files in `website/`, allowing the repository to remain publishable from the branch root without mixing the main website source with the analysis code.

## View the website locally

Run:

```powershell
python python-code/site-tools/launch_dashboard.py
```

The launcher uses cached confirmatory outputs when available, refreshes `website/assets/waveform-comparison-data.js`, starts a local server, and opens the visualization. Add `--force` to rerun the confirmatory analysis before launching.

Windows users can also double-click `python-code/site-tools/launch_dashboard.bat`.

## Rebuild the complete ZIP

Run:

```powershell
python python-code/site-tools/build_distribution.py
```

This creates `downloads/GWTC5_all_figures_and_code_only.zip`, writes a SHA-256 file manifest into the archive, and verifies every repository `.py`, `.ipynb`, and `.bat` script against its archived copy.

## GitHub Pages

Publish from the repository branch root. The root entry point redirects visitors to `website/`, while old `waveform-comparison/` links continue to work.

## Python collection

See `python-code/README.md` for the purpose of each analysis directory. The downloadable figures-and-code ZIP is rebuilt from the organized folders and checked against the repository so every Python script is included.
