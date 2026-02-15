#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from amplification_barometer.audit_report import build_audit_report, write_audit_report


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a small sector suite on sector datasets (2026, 2027+, etc.).")
    ap.add_argument("--sector-dir", type=str, default="data/sector_2026")
    ap.add_argument("--out-dir", type=str, default="_ci_out")
    ap.add_argument("--datasets", nargs="*", default=None, help="Override datasets to run")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.datasets:
        paths = [Path(p) for p in args.datasets]
    else:
        sdir = Path(args.sector_dir)
        paths = sorted(sdir.glob("*.csv"))

    rows = []
    for p in paths:
        if not p.exists():
            raise SystemExit(f"Missing dataset: {p}")
        name = p.stem
        df = _read_csv(p)
        rep = build_audit_report(df, dataset_name=name)
        write_audit_report(rep, out / f"audit_report_{name}.json")
        rows.append(
            {
                "dataset": name,
                "rule_execution_gap_mean": rep.targets.get("rule_execution_gap_mean"),
                "gap_meets_target": rep.targets.get("rule_execution_gap_meets_target"),
                "prevented_exceedance_rel": rep.targets.get("prevented_exceedance_rel"),
                "prevented_meets_target": rep.targets.get("prevented_meets_target"),
                "prevented_topk_excess_rel": rep.targets.get("prevented_topk_excess_rel"),
                "proactive_topk_frac": rep.targets.get("proactive_topk_frac"),
                "maturity_label": rep.maturity.get("label"),
                "l_verdict": rep.l_performance.get("verdict"),
                "l_verdict_proactive": rep.l_performance_proactive.get("verdict"),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "sector_suite_summary.csv", index=False)

    md_lines = [
        "# Sector suite summary",
        "",
        "Targets (demo):",
        "- rule_execution_gap_mean < 0.05",
        "- prevented_topk_excess_rel > 0.10 (proactive L variant)",
        "",
        "Results:",
        "",
    ]
    # Avoid optional dependency (tabulate). Write a minimal table.
    cols = list(summary.columns)
    md_lines.append("| " + " | ".join(cols) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in summary.iterrows():
        md_lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    (out / "sector_suite_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
