#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
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


def _looks_like_placeholder_governance(df: pd.DataFrame) -> bool:
    """Detect the classic placeholder pattern in sector_2026 datasets.

    Placeholder pattern seen in old generated CSVs:
    - rule_execution_gap around 1.0
    - control_turnover around 1.0
    - sanction_delay around 1.0 (should be in double digits)
    """
    cols = {"rule_execution_gap", "control_turnover", "sanction_delay"}
    if not cols.issubset(set(df.columns)):
        return False
    gap_m = float(pd.to_numeric(df["rule_execution_gap"], errors="coerce").mean())
    turn_m = float(pd.to_numeric(df["control_turnover"], errors="coerce").mean())
    sanc_m = float(pd.to_numeric(df["sanction_delay"], errors="coerce").mean())
    return bool((gap_m > 0.5) and (turn_m > 0.5) and (sanc_m < 5.0))


def _maybe_regen_sector_2026(*, sector_dir: Path, n: int, seed: int, start_date: str, auto_regen: bool) -> None:
    """Regenerate sector_2026 datasets if they look like placeholders."""
    if not auto_regen:
        return
    repo_root = Path(__file__).resolve().parents[1]
    gen = repo_root / "tools" / "generate_sector_2026.py"
    if not gen.exists():
        return

    # If directory missing, always generate.
    if not sector_dir.exists():
        subprocess.run(
            [sys.executable, str(gen), "--out-dir", str(sector_dir), "--n", str(int(n)), "--start-date", str(start_date), "--seed", str(int(seed))],
            check=True,
        )
        return

    # If any *_2026_synth.csv looks placeholder-like, regenerate the whole folder.
    csvs = sorted(sector_dir.glob("*.csv"))
    for p in csvs:
        if not p.name.endswith("_2026_synth.csv"):
            continue
        try:
            df = _read_csv(p)
        except Exception:
            continue
        if _looks_like_placeholder_governance(df):
            subprocess.run(
                [sys.executable, str(gen), "--out-dir", str(sector_dir), "--n", str(int(n)), "--start-date", str(start_date), "--seed", str(int(seed))],
                check=True,
            )
            return


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a small sector suite on sector datasets (2026, 2027+, etc.).")
    ap.add_argument("--sector-dir", type=str, default="data/sector_2026")
    ap.add_argument("--out-dir", type=str, default="_ci_out")
    ap.add_argument("--datasets", nargs="*", default=None, help="Override datasets to run")
    ap.add_argument("--window", type=int, default=5)

    # Convenience: avoid stale sector_2026 datasets that still contain placeholder governance proxies.
    ap.add_argument("--no-auto-regen-sector-2026", action="store_true", help="Disable automatic regeneration of sector_2026 when governance proxies look like placeholders.")
    ap.add_argument("--regen-n", type=int, default=365, help="n for auto regeneration (sector_2026 only).")
    ap.add_argument("--regen-seed", type=int, default=7, help="seed for auto regeneration (sector_2026 only).")
    ap.add_argument("--regen-start-date", type=str, default="2026-01-01", help="start date for auto regeneration (sector_2026 only).")

    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    thresholds = _load_thresholds(window=int(args.window))

    sector_dir = Path(args.sector_dir)

    if not args.datasets:
        _maybe_regen_sector_2026(
            sector_dir=sector_dir,
            n=int(args.regen_n),
            seed=int(args.regen_seed),
            start_date=str(args.regen_start_date),
            auto_regen=not bool(args.no_auto_regen_sector_2026),
        )

    if args.datasets:
        paths = [Path(p) for p in args.datasets]
    else:
        paths = sorted(sector_dir.glob("*.csv"))

    rows = []
    for p in paths:
        if not p.exists():
            raise SystemExit(f"Missing dataset: {p}")
        name = p.stem
        df = _read_csv(p)

        # Hard fail with an explicit message when governance looks placeholder-like.
        if name.endswith("_2026_synth") and _looks_like_placeholder_governance(df):
            raise SystemExit(
                f"{name}: governance proxies look like placeholders. "
                f"Delete {sector_dir} and regenerate: python tools/generate_sector_2026.py --out-dir {sector_dir}"
            )

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
                "gov_placeholder_like": bool((rep.targets.get("governance_proxy_quality") or {}).get("flag_placeholder_like", False)),
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
