#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from amplification_barometer.real_data_adapters import ensure_datetime_index, finance_ohlcv_to_proxies, has_required_proxies


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert real-world data to Amplification-Barometer proxies CSV.")
    ap.add_argument("--infile", type=str, required=True, help="Input CSV/Parquet (OHLCV or already-proxies).")
    ap.add_argument("--outfile", type=str, required=True, help="Output CSV with proxies and a date index column.")
    ap.add_argument("--price-col", type=str, default="close")
    ap.add_argument("--volume-col", type=str, default="volume")
    args = ap.parse_args()

    in_path = Path(args.infile)
    if not in_path.exists():
        raise SystemExit(f"Missing infile: {in_path}")

    if in_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    if has_required_proxies(df):
        out = ensure_datetime_index(df).reset_index().rename(columns={"index": "date"})
    else:
        # OHLCV
        cols = {c.lower() for c in df.columns}
        if not {"open", "high", "low", args.price_col.lower()}.issubset(cols):
            raise SystemExit("Unsupported input: provide proxies columns or OHLCV columns (open/high/low/close[,volume]).")
        ren = {c: c.lower() for c in df.columns}
        df2 = df.rename(columns=ren)
        df2 = ensure_datetime_index(df2)
        vol_col = args.volume_col.lower() if args.volume_col.lower() in df2.columns else None
        out = finance_ohlcv_to_proxies(df2, price_col=args.price_col.lower(), volume_col=vol_col).reset_index().rename(columns={"index": "date"})

    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
