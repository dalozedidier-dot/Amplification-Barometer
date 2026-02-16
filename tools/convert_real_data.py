#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from amplification_barometer.real_data_adapters import algae_mat_to_proxies, crypto_orderbook_to_proxies


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert real-world datasets to amplification-barometer proxy CSV.")
    p.add_argument("--kind", choices=["crypto_orderbook", "algae_mat"], required=True)
    p.add_argument("--in-path", required=True, help="Input file path (CSV or MAT).")
    p.add_argument("--out-csv", required=True, help="Output CSV path (proxy frame).")
    p.add_argument("--time-col", default="system_time", help="Time column for crypto_orderbook.")
    p.add_argument("--raceway", type=int, default=0, help="Raceway index for algae_mat.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    in_path = Path(args.in_path)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.kind == "crypto_orderbook":
        df = pd.read_csv(in_path)
        prox = crypto_orderbook_to_proxies(df, time_col=args.time_col)
    else:
        prox = algae_mat_to_proxies(in_path, raceway=args.raceway)

    prox.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv} ({len(prox)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
