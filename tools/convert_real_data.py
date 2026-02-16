#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from amplification_barometer.real_data_adapters import (
    aiops_phase2_to_proxies,
    binance_aggtrades_to_proxies,
    binance_trades_to_proxies,
    borg_traces_to_proxies,
)


def _read_csv_from_zip(zip_path: Path, *, header: Optional[int]) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as z:
        csvs = [nm for nm in z.namelist() if nm.lower().endswith(".csv")]
        if not csvs:
            raise FileNotFoundError("No .csv file found in zip")
        nm = csvs[0]
        with z.open(nm) as f:
            df = pd.read_csv(f, header=header)
    df.attrs["source_member"] = nm
    return df


def _read_input(path: Path, *, header: Optional[int]) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        return _read_csv_from_zip(path, header=header)
    return pd.read_csv(path, header=header)


def _write_meta(meta_path: Optional[Path], meta: Dict[str, Any]) -> None:
    if meta_path is None:
        return
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Convert real datasets into Amplification-Barometer proxy CSV.")
    p.add_argument("--input", required=True, help="Path to .csv or .zip containing a single .csv")
    p.add_argument(
        "--kind",
        required=True,
        choices=["binance_aggtrades", "binance_trades", "borg_traces", "aiops_phase2"],
        help="Dataset kind / adapter to use",
    )
    p.add_argument("--out-csv", required=True, help="Output proxy CSV path")
    p.add_argument("--meta-json", default=None, help="Optional metadata JSON output path")

    p.add_argument("--limit-rows", type=int, default=0, help="Optional row limit for input parsing")
    p.add_argument("--bar-freq", default="1min", help="Finance aggregation frequency, ex 1min, 5min")
    p.add_argument("--bucket-seconds", type=int, default=60, help="Borg traces bucket size in seconds")
    p.add_argument("--kpi-id", default=None, help="AIOps KPI ID to select (defaults to first)")

    args = p.parse_args()

    in_path = Path(args.input)
    out_csv = Path(args.out_csv)
    meta_path = Path(args.meta_json) if args.meta_json else None

    header: Optional[int] = 0
    if args.kind in {"binance_aggtrades", "binance_trades"}:
        header = None

    df = _read_input(in_path, header=header)
    if args.limit_rows and args.limit_rows > 0:
        df = df.head(int(args.limit_rows)).copy()

    meta: Dict[str, Any] = {
        "input": str(in_path),
        "kind": str(args.kind),
        "rows_in": int(len(df)),
        "columns_in": list(df.columns),
        "source_member": df.attrs.get("source_member"),
    }

    if args.kind == "binance_aggtrades":
        prox = binance_aggtrades_to_proxies(df, bar_freq=str(args.bar_freq))
    elif args.kind == "binance_trades":
        prox = binance_trades_to_proxies(df, bar_freq=str(args.bar_freq))
    elif args.kind == "borg_traces":
        prox = borg_traces_to_proxies(df, bucket_seconds=int(args.bucket_seconds))
    elif args.kind == "aiops_phase2":
        prox = aiops_phase2_to_proxies(df, kpi_id=args.kpi_id)
    else:
        raise ValueError(f"Unknown kind: {args.kind}")

    # Important: tools/run_audit.py expects a 'date' column for parsing + index.
    # Ensure the index is written as a named column "date" in the CSV.
    if prox.index.name is None or str(prox.index.name).strip() == "":
        prox.index.name = "date"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    prox.to_csv(out_csv, index=True)

    meta.update(
        {
            "rows_out": int(len(prox)),
            "columns_out": list(prox.columns),
            "index_name_out": str(prox.index.name),
        }
    )
    _write_meta(meta_path, meta)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
