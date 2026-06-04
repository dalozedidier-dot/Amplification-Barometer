#!/usr/bin/env python3
"""Public entry point for running an audit on a user CSV dataset.

This wrapper keeps the README command stable while delegating to tools/run_audit.py,
which is the canonical implementation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from run_audit import _run_one


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the public Amplification Barometer audit on a CSV dataset.")
    parser.add_argument("--dataset", required=True, help="CSV dataset with a date/timestamp column.")
    parser.add_argument("--name", default="", help="Optional dataset name used for output filenames.")
    parser.add_argument("--out-dir", default="_ci_out/public_tests", help="Output directory.")
    parser.add_argument("--window", type=int, default=5, help="Smoothing window for delta_d(t).")
    parser.add_argument("--plot", action="store_true", help="Write PNG plots.")
    parser.add_argument("--plotly", action="store_true", help="Write interactive Plotly HTML charts.")
    parser.add_argument("--no-html-report", action="store_true", help="Disable the consolidated HTML report.")
    args = parser.parse_args()

    csv_path = Path(args.dataset)
    if not csv_path.exists():
        raise SystemExit(f"Missing dataset: {csv_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name.strip() or csv_path.stem
    _run_one(
        csv_path,
        name=name,
        out_dir=out_dir,
        window=int(args.window),
        do_png=bool(args.plot),
        do_plotly=bool(args.plotly),
        do_html_report=not bool(args.no_html_report),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
