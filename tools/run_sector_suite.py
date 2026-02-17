#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from amplification_barometer.audit_report import build_audit_report, write_audit_report
from amplification_barometer.calibration import Thresholds, derive_thresholds


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        return df.dropna(subset=["date"]).set_index("date").sort_index()
    # fallback for other time columns
    for col in ("datetime", "timestamp", "time", "open_time", "close_time"):
        if col in df.columns:
            s = df[col]
            if pd.api.types.is_numeric_dtype(s):
                unit = "ms" if float(s.dropna().iloc[0]) > 1e11 else "s"
                dt = pd.to_datetime(s, unit=unit, utc=True)
            else:
                dt = pd.to_datetime(s, utc=True, errors="coerce")
            df = df.drop(columns=[col])
            df.insert(0, "date", dt)
            return df.dropna(subset=["date"]).set_index("date").sort_index()
    raise SystemExit(f"Cannot find a date/timestamp column in {path}")


def _load_thresholds(*, window: int) -> Optional[Thresholds]:
    repo_root = Path(__file__).resolve().parents[1]
    stable_path = repo_root / "data" / "synthetic" / "stable_regime.csv"
    if not stable_path.exists():
        return None
    stable_df = _read_csv(stable_path)
    return derive_thresholds(stable_df, window=window)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a small sector suite on sector datasets (2026, 2027+, etc.).")
    ap.add_argument("--sector-dir", type=str, default="data/sector_2026")
    ap.add_argument("--out-dir", type=str, default="_ci_out")
    ap.add_argument("--datasets", nargs="*", default=None, help="Override datasets to run")
    ap.add_argument("--window", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    thresholds = _load_thresholds(window=int(args.window))

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
        rep = build_audit_report(df, dataset_name=name, delta_d_window=int(args.window), thresholds=thresholds)
        write_audit_report(rep, out / f"audit_report_{name}.json")

        rows.append(
            {
                "dataset": name,
                "baseline_used": bool((rep.summary or {}).get("baseline_used", False)),
                "risk_thr": float((rep.summary or {}).get("risk_threshold", float("nan"))),
                "rule_execution_gap_mean": rep.targets.get("rule_execution_gap_mean"),
                "gap_meets_target": rep.targets.get("rule_execution_gap_meets_target"),
                "prevented_exceedance_rel": rep.targets.get("prevented_exceedance_rel"),
                "prevented_meets_target": rep.targets.get("prevented_meets_target"),
                "prevented_topk_excess_rel": rep.targets.get("prevented_topk_excess_rel"),
                "proactive_topk_excess_rel": rep.targets.get("proactive_topk_excess_rel"),
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
    cols = list(summary.columns)
    md_lines.append("| " + " | ".join(cols) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in summary.iterrows():
        md_lines.append("| " + " | ".join([str(row.get(c, "")) for c in cols]) + " |")

    (out / "sector_suite_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
