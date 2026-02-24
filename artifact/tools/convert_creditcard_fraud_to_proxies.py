#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from amplification_barometer.real_data_adapters import univariate_csv_to_proxies


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert creditcard fraud dataset to Amplification-Barometer proxies.")
    ap.add_argument("--input", type=str, required=True, help="Path to creditcard.csv (with Time, Amount, Class).")
    ap.add_argument("--out-csv", type=str, required=True, help="Output proxies CSV.")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Missing input: {in_path}")

    df = pd.read_csv(in_path)
    needed = {"Time", "Amount", "Class"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in {in_path.name}: {missing}")

    # The dataset uses "Time" as seconds elapsed since first transaction.
    # We map it to an epoch-like timestamp for compatibility.
    base = 1_600_000_000  # arbitrary, stable
    out = pd.DataFrame(
        {
            "timestamp": (pd.to_numeric(df["Time"], errors="coerce").fillna(0.0).astype("int64") + base),
            "value": pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0).astype(float),
            "label": pd.to_numeric(df["Class"], errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(float),
        }
    )

    proxies = univariate_csv_to_proxies(out, timestamp_col="timestamp", value_col="value", label_col="label", tz="UTC")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proxies = proxies.copy()
    proxies.index.name = "date"
    proxies.reset_index().to_csv(out_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
