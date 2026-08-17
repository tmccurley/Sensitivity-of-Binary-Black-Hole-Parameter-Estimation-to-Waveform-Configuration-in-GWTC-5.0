"""Build missing confirmatory results, serve the dashboard, and open it."""

from __future__ import annotations

import argparse
import csv
import http.server
import json
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
        help="Rerun the confirmatory analysis even when dashboard data exists.",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open a browser window."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    dashboard_data = (
        root
        / "results"
        / "confirmatory_analysis"
        / "waveform-comparison-data.js"
    )
    metrics_csv = dashboard_data.with_name("confirmatory_parameter_metrics.csv")
    events_csv = dashboard_data.with_name("confirmatory_event_endpoints.csv")

    if not dashboard_data.exists() and metrics_csv.exists() and events_csv.exists():
        with metrics_csv.open(encoding="utf-8", newline="") as handle:
            metrics = list(csv.DictReader(handle))
        with events_csv.open(encoding="utf-8", newline="") as handle:
            events = list(csv.DictReader(handle))
        dashboard_data.write_text(
            "window.GWTC5_WAVEFORM_DATA = "
            + json.dumps(metrics)
            + ";\nwindow.GWTC5_EVENT_DATA = "
            + json.dumps(events)
            + ";\n",
            encoding="utf-8",
        )

    if args.force or not dashboard_data.exists():
        command = [
            sys.executable,
            str(root / "GWTC5_confirmatory_analysis.py"),
            "--project-root",
            str(root),
        ]
        if args.force:
            command.append("--overwrite")
        print("Preparing confirmatory results. This can take several minutes...")
        subprocess.run(command, cwd=root, check=True)
    else:
        print(f"Using cached confirmatory results: {dashboard_data}")

    handler = lambda *a, **kw: NoCacheHandler(  # noqa: E731
        *a, directory=str(root), **kw
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/waveform-comparison/"
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
