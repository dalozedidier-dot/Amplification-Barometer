#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from amplification_barometer.audit_report import build_audit_report, write_audit_report
from amplification_barometer.calibration import Thresholds, derive_thresholds
from amplification_barometer.html_report import render_audit_html
from amplification_barometer.real_data_adapters import ensure_datetime_index, finance_ohlcv_to_proxies, has_required_proxies


def _load_thresholds(*, window: int) -> Optional[Thresholds]:
    repo_root = Path(__file__).resolve().parents[1]
    stable_path = repo_root / "data" / "synthetic" / "stable_regime.csv"
    if not stable_path.exists():
        return None
    stable_df = pd.read_csv(stable_path)
    stable_df["date"] = pd.to_datetime(stable_df["date"], utc=True, errors="coerce")
    stable_df = stable_df.dropna(subset=["date"]).set_index("date").sort_index()
    return derive_thresholds(stable_df, window=window)


def _discover_inputs(repo_root: Path) -> List[Path]:
    candidates: List[Path] = []
    for sub in ("data/real", "data/real_fixtures"):
        p = repo_root / sub
        if p.exists():
            candidates += list(p.glob("*.csv"))
            candidates += list(p.glob("*.parquet"))
    return sorted(set(candidates))


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df


def _to_proxies(df: pd.DataFrame) -> pd.DataFrame:
    # Already proxies?
    if has_required_proxies(df):
        return ensure_datetime_index(df)

    # OHLCV style?
    cols = {c.lower() for c in df.columns}
    if {"open", "high", "low", "close"}.issubset(cols):
        # normalize column names
        ren = {c: c.lower() for c in df.columns}
        df2 = df.rename(columns=ren)
        df2 = ensure_datetime_index(df2)
        vol_col = "volume" if "volume" in df2.columns else None
        proxies = finance_ohlcv_to_proxies(df2, price_col="close", volume_col=vol_col)
        return proxies

    raise ValueError("Unsupported real-data format. Provide proxies columns or OHLCV columns.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Real data smoke: convert + audit + HTML report.")
    ap.add_argument("--out-dir", type=str, default="_ci_out/real_data")
    ap.add_argument("--window", type=int, default=5)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    thresholds = _load_thresholds(window=int(args.window))

    inputs = _discover_inputs(repo_root)
    if not inputs:
        note = {"note": "No real data files found under data/real or data/real_fixtures.", "how_to": "Drop CSV/Parquet with proxies or OHLCV columns."}
        (out / "real_data_smoke_note.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
        return 0

    for p in inputs:
        name = p.stem
        try:
            raw = _read_any(p)
            proxies = _to_proxies(raw)
        except Exception as e:
            (out / f"error_{name}.txt").write_text(str(e), encoding="utf-8")
            continue

        rep = build_audit_report(proxies, dataset_name=name, delta_d_window=int(args.window), thresholds=thresholds)
        json_path = out / f"audit_report_{name}.json"
        write_audit_report(rep, json_path)

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        render_audit_html(report_dict=payload, df=proxies, out_html=out / f"audit_report_{name}.html", thresholds=thresholds, delta_d_window=int(args.window))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
