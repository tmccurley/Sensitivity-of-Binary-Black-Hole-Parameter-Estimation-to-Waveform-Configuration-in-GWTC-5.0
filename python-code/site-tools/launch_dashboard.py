"""Build confirmatory results, refresh website data, and serve the site."""

from __future__ import annotations

import argparse
import csv
import http.server
import json
import shutil
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    """Serve current visualization code instead of a browser-cached copy."""

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun the confirmatory analysis even when cached results exist.",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open a browser window."
    )
    return parser.parse_args()


def find_repository_root() -> Path:
    """Find the organized repository without depending on the current directory."""
    script_path = Path(__file__).resolve()
    for candidate in script_path.parents:
        if (candidate / "website" / "index.html").is_file():
            return candidate
    raise RuntimeError("Could not locate website/index.html above the launcher.")


def build_javascript_from_csv(
    metrics_csv: Path, events_csv: Path, destination: Path
) -> None:
    """Create the browser data file from the two confirmatory CSV tables."""
    with metrics_csv.open(encoding="utf-8", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    with events_csv.open(encoding="utf-8", newline="") as handle:
        events = list(csv.DictReader(handle))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "window.GWTC5_WAVEFORM_DATA = "
        + json.dumps(metrics)
        + ";\nwindow.GWTC5_EVENT_DATA = "
        + json.dumps(events)
        + ";\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    root = find_repository_root()
    website_root = root / "website"
    results_dir = root / "results" / "confirmatory_analysis"
    dashboard_data = results_dir / "waveform-comparison-data.js"
    metrics_csv = results_dir / "confirmatory_parameter_metrics.csv"
    events_csv = results_dir / "confirmatory_event_endpoints.csv"
    analysis_script = (
        root
        / "python-code"
        / "current-analysis"
        / "GWTC5_confirmatory_analysis.py"
    )

    if not dashboard_data.exists() and metrics_csv.exists() and events_csv.exists():
        build_javascript_from_csv(metrics_csv, events_csv, dashboard_data)

    if args.force or not dashboard_data.exists():
        command = [
            sys.executable,
            str(analysis_script),
            "--project-root",
            str(root),
        ]
        if args.force:
            command.append("--overwrite")
        print("Preparing confirmatory results. This can take several minutes...")
        subprocess.run(command, cwd=root, check=True)
    else:
        print(f"Using cached confirmatory results: {dashboard_data}")

    if not dashboard_data.is_file():
        raise FileNotFoundError(f"Analysis data was not created: {dashboard_data}")

    website_data = website_root / "assets" / "waveform-comparison-data.js"
    website_data.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dashboard_data, website_data)
    print(f"Website data refreshed: {website_data}")

    handler = lambda *a, **kw: NoCacheHandler(  # noqa: E731
        *a, directory=str(website_root), **kw
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Dashboard: {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
