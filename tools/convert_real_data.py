#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import pandas as pd


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _ensure_tz_index(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(tz)
    out = df.copy()
    out.index = idx
    out.index.name = "date"
    return out


def _call_with_optional_tz(func, df: pd.DataFrame, tz: str):
    sig = inspect.signature(func)
    if "tz" in sig.parameters:
        return func(df, tz=tz)
    return func(df)


def _write_proxies_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if getattr(out.index, "name", None) in (None, ""):
        out.index.name = "date"
    out = out.reset_index()
    if out.columns[0] != "date":
        out = out.rename(columns={out.columns[0]: "date"})
    out.to_csv(out_path, index=False)


def _maybe_import_adapters():
    # Imports kept inside so `--help` works even if optional deps are missing.
    from amplification_barometer import real_data_adapters as rda  # type: ignore

    return rda


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert real-world data to Amplification-Barometer proxies CSV.")

    # New interface (used by CI workflows)
    ap.add_argument("--input", type=str, default="", help="Input CSV/Parquet.")
    ap.add_argument(
        "--kind",
        type=str,
        default="",
        help="Adapter kind: binance_aggtrades, binance_trades, borg_traces, aiops_phase2, finance_ohlcv, auto.",
    )
    ap.add_argument("--out-csv", type=str, default="", help="Output CSV with proxies and a date column.")
    ap.add_argument("--bar-freq", type=str, default="1min", help="Resampling frequency for trade-like feeds.")
    ap.add_argument("--tz", type=str, default="UTC", help="Timezone for output index.")
    ap.add_argument("--price-col", type=str, default="close", help="OHLCV price column (for finance_ohlcv).")
    ap.add_argument("--volume-col", type=str, default="volume", help="OHLCV volume column (optional).")

    # Legacy interface (kept for backward compatibility)
    ap.add_argument("--infile", type=str, default="", help="(legacy) Input CSV/Parquet (OHLCV or already-proxies).")
    ap.add_argument("--outfile", type=str, default="", help="(legacy) Output CSV with proxies and a date column.")

    args = ap.parse_args()

    using_new = bool(args.input or args.out_csv or args.kind)
    if using_new:
        in_path = Path(args.input)
        out_path = Path(args.out_csv)
        if not args.input or not args.out_csv:
            raise SystemExit("convert_real_data.py: --input and --out-csv are required with the new interface.")
        if not in_path.exists():
            raise SystemExit(f"Missing input: {in_path}")
        if not args.kind:
            raise SystemExit("convert_real_data.py: --kind is required with the new interface.")

        df = _read_any(in_path)
        rda = _maybe_import_adapters()

        kind = str(args.kind).strip().lower()

        if kind == "auto":
            cols = {str(c).lower() for c in df.columns}
            if {"p", "q", "t"}.issubset(cols) or {"price", "qty", "ts"}.issubset(cols):
                kind = "binance_aggtrades"
            elif any(c in cols for c in ("job_id", "task_id", "container", "service", "user", "resource_request")):
                kind = "borg_traces"
            else:
                kind = "aiops_phase2"

        tz = str(args.tz)

        if kind == "binance_aggtrades":
            # This adapter usually supports tz, but we keep it tolerant.
            func = rda.binance_aggtrades_to_proxies
            out = func(df, bar_freq=str(args.bar_freq), tz=tz) if "tz" in inspect.signature(func).parameters else func(df, bar_freq=str(args.bar_freq))
            out = _ensure_tz_index(out, tz)
        elif kind == "binance_trades":
            func = rda.binance_trades_to_proxies
            out = func(df, bar_freq=str(args.bar_freq), tz=tz) if "tz" in inspect.signature(func).parameters else func(df, bar_freq=str(args.bar_freq))
            out = _ensure_tz_index(out, tz)
        elif kind == "borg_traces":
            out = _call_with_optional_tz(rda.borg_traces_to_proxies, df, tz)
            out = _ensure_tz_index(out, tz)
        elif kind == "aiops_phase2":
            out = _call_with_optional_tz(rda.aiops_phase2_to_proxies, df, tz)
            out = _ensure_tz_index(out, tz)
        elif kind in ("finance_ohlcv", "ohlcv"):
            # If already in proxy schema, just normalize index. Otherwise build proxies from OHLCV.
            if hasattr(rda, "has_required_proxies") and hasattr(rda, "ensure_datetime_index") and hasattr(rda, "finance_ohlcv_to_proxies"):
                if rda.has_required_proxies(df):
                    out = rda.ensure_datetime_index(df, tz=tz) if "tz" in inspect.signature(rda.ensure_datetime_index).parameters else rda.ensure_datetime_index(df)
                else:
                    df2 = df.rename(columns={c: str(c).lower() for c in df.columns})
                    df2 = rda.ensure_datetime_index(df2, tz=tz) if "tz" in inspect.signature(rda.ensure_datetime_index).parameters else rda.ensure_datetime_index(df2)
                    vol_col = str(args.volume_col).lower()
                    vol_col = vol_col if vol_col in df2.columns else None
                    out = rda.finance_ohlcv_to_proxies(df2, price_col=str(args.price_col).lower(), volume_col=vol_col)
                out = _ensure_tz_index(out, tz)
            else:
                raise SystemExit("finance_ohlcv adapter requires ensure_datetime_index/finance_ohlcv_to_proxies in real_data_adapters.")
        else:
            raise SystemExit(f"Unknown --kind: {args.kind}")

        _write_proxies_csv(out, out_path)
        return 0

    # Legacy interface
    if not args.infile or not args.outfile:
        raise SystemExit("convert_real_data.py: provide --infile and --outfile, or use the new interface (--input/--kind/--out-csv).")

    in_path = Path(args.infile)
    out_path = Path(args.outfile)
    if not in_path.exists():
        raise SystemExit(f"Missing infile: {in_path}")

    df = _read_any(in_path)
    rda = _maybe_import_adapters()
    tz = str(args.tz)

    if hasattr(rda, "has_required_proxies") and rda.has_required_proxies(df):
        out = rda.ensure_datetime_index(df, tz=tz) if "tz" in inspect.signature(rda.ensure_datetime_index).parameters else rda.ensure_datetime_index(df)
        out = _ensure_tz_index(out, tz)
    else:
        cols = {str(c).lower() for c in df.columns}
        if not {"open", "high", "low", str(args.price_col).lower()}.issubset(cols):
            raise SystemExit("Unsupported legacy input: provide proxies columns or OHLCV columns (open/high/low/close[,volume]).")
        df2 = df.rename(columns={c: str(c).lower() for c in df.columns})
        df2 = rda.ensure_datetime_index(df2, tz=tz) if "tz" in inspect.signature(rda.ensure_datetime_index).parameters else rda.ensure_datetime_index(df2)
        vol_col = str(args.volume_col).lower()
        vol_col = vol_col if vol_col in df2.columns else None
        out = rda.finance_ohlcv_to_proxies(df2, price_col=str(args.price_col).lower(), volume_col=vol_col)
        out = _ensure_tz_index(out, tz)

    _write_proxies_csv(out, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
