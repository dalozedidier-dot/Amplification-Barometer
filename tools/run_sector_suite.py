
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from amplification_barometer.audit_report import build_audit_report, write_audit_report
from amplification_barometer.calibration import derive_thresholds


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.set_index("date").sort_index()
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector-dir", default="data/sector", help="Directory with sector datasets")
    ap.add_argument("--baseline-csv", default="data/synthetic/stable_regime.csv", help="Stable baseline CSV")
    ap.add_argument("--out-dir", default="_ci_out/sector_suite", help="Output dir")
    ap.add_argument("--window", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = None
    base_path = Path(args.baseline_csv)
    if base_path.exists():
        thresholds = derive_thresholds(_read_csv(base_path), window=args.window)
        (out_dir / "baseline_thresholds.json").write_text(json.dumps(thresholds.__dict__, indent=2), encoding="utf-8")

    sector_dir = Path(args.sector_dir)
    csvs: List[Path] = sorted(sector_dir.glob("*.csv")) if sector_dir.exists() else []
    results = []
    for p in csvs:
        df = _read_csv(p)
        rep = build_audit_report(df, dataset_name=p.stem, window=args.window, thresholds=thresholds)
        write_audit_report(rep, out_dir / f"audit_report_{p.stem}.json")
        results.append({"name": p.stem, "maturity": rep.maturity.get("label"), "risk_mean": rep.summary.get("RISK", {}).get("mean")})

    (out_dir / "sector_suite_summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
