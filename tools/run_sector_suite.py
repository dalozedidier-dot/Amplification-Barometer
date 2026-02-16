#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from amplification_barometer.audit_report import build_audit_report, write_audit_report
from amplification_barometer.calibration import Thresholds, derive_thresholds


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df = df.set_index("date")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.set_index("timestamp")
    else:
        # Fallback: accept CSVs already indexed by a first column
        idx_col = df.columns[0]
        try:
            df[idx_col] = pd.to_datetime(df[idx_col], errors="coerce", utc=True)
            df = df.set_index(idx_col)
        except Exception as exc:
            raise ValueError(f"Dataset missing a date-like column: {path}") from exc

    df = df.sort_index()
    return df


def _thresholds_to_dict(th: Thresholds) -> Dict[str, float]:
    d = dataclasses.asdict(th)
    return {k: float(v) for k, v in d.items()}


def _call_build_audit_report(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    thresholds: Thresholds,
) -> Any:
    sig = inspect.signature(build_audit_report)
    params = sig.parameters

    # Prefer passing the Thresholds object if supported.
    if "thresholds" in params:
        return build_audit_report(df, dataset_name=dataset_name, thresholds=thresholds)

    # Some versions may accept a dict instead.
    if "baseline_thresholds" in params:
        return build_audit_report(df, dataset_name=dataset_name, baseline_thresholds=_thresholds_to_dict(thresholds))

    # If no threshold parameter exists, refuse silently generating incomparable reports.
    raise SystemExit(
        "build_audit_report does not accept baseline thresholds. "
        "Baseline comparability is required. "
        "Update audit_report.build_audit_report to accept thresholds."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a sector suite with a stable baseline applied to every dataset.")
    ap.add_argument("--sector-dir", type=str, default="data/sector_2026")
    ap.add_argument("--out-dir", type=str, default="_ci_out")
    ap.add_argument("--datasets", nargs="*", default=None, help="Override datasets to run")
    ap.add_argument("--baseline-csv", type=str, default="data/synthetic/stable_regime.csv")
    ap.add_argument("--baseline-window", type=int, default=5)
    ap.add_argument("--baseline-required", action="store_true", help="Fail if baseline is not applied")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    stable_path = Path(args.baseline_csv)
    if not stable_path.exists():
        raise SystemExit(f"Missing baseline CSV: {stable_path}")

    stable_df = _read_csv(stable_path)
    thresholds = derive_thresholds(stable_df, window=int(args.baseline_window))
    (out / "sector_suite_baseline_thresholds.json").write_text(
        json.dumps({"thresholds": _thresholds_to_dict(thresholds)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

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

        rep = _call_build_audit_report(df, dataset_name=name, thresholds=thresholds)

        baseline_used = bool(rep.verdict.get("notes", {}).get("baseline_used")) if hasattr(rep, "verdict") else False
        if args.baseline_required and not baseline_used:
            raise SystemExit(f"Baseline not applied for dataset '{name}'. Refusing to emit incomparable reports.")

        write_audit_report(rep, out / f"audit_report_{name}.json")

        rows.append(
            {
                "dataset": name,
                "baseline_used": baseline_used,
                "risk_threshold": rep.summary.get("risk_threshold"),
                "rule_execution_gap_mean": rep.targets.get("rule_execution_gap_mean"),
                "gap_meets_target": rep.targets.get("rule_execution_gap_meets_target"),
                "prevented_exceedance_rel": rep.targets.get("prevented_exceedance_rel"),
                "prevented_meets_target": rep.targets.get("prevented_meets_target"),
                "prevented_topk_excess_rel": rep.targets.get("prevented_topk_excess_rel"),
                "proactive_topk_excess_rel": rep.targets.get("prevented_topk_excess_rel_proactive"),
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
        "Baseline:",
        f"- baseline_csv: {stable_path.as_posix()}",
        f"- baseline_window: {int(args.baseline_window)}",
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
    for _, r in summary.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if pd.isna(v):
                vals.append("None")
            else:
                vals.append(str(v))
        md_lines.append("| " + " | ".join(vals) + " |")
    (out / "sector_suite_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
