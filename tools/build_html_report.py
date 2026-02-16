#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from amplification_barometer.html_report import HtmlReportOptions, build_reports_index, build_self_contained_html_report


def _read_csv(path: Path) -> pd.DataFrame:
    # Default contract in this repo: a 'date' column
    df = pd.read_csv(path, parse_dates=["date"])
    if "date" in df.columns:
        df = df.set_index("date")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a self-contained HTML report from audit JSON + dataset CSV.")
    ap.add_argument("--audit-json", type=str, required=True, help="Path to audit_report_*.json")
    ap.add_argument("--dataset", type=str, required=True, help="Path to the dataset CSV (must include 'date' column).")
    ap.add_argument("--out-html", type=str, default="", help="Output HTML path. Default next to audit JSON.")
    ap.add_argument("--window", type=int, default=5, help="Δd(t) smoothing window.")
    ap.add_argument("--author", type=str, default="GPT-5.2 Thinking", help="Author string in footer.")
    args = ap.parse_args()

    audit_path = Path(args.audit_json)
    report_dict = json.loads(audit_path.read_text(encoding="utf-8"))
    df = _read_csv(Path(args.dataset))

    out_html = Path(args.out_html) if args.out_html else audit_path.with_suffix(".html")
    build_self_contained_html_report(
        df,
        report_dict,
        out_html=out_html,
        options=HtmlReportOptions(author=args.author, window=args.window),
    )
    print(f"Wrote: {out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
