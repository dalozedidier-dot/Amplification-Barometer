#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Set

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
    # Import inside so `--help` works even if optional deps are missing.
    from amplification_barometer import real_data_adapters as rda  # type: ignore

    return rda


def _cols_lower(df: pd.DataFrame) -> Set[str]:
    return {str(c).strip().lower() for c in df.columns}


def _detect_kind(cols: Set[str], price_col: str) -> str:
    price_col = price_col.strip().lower()

    # Already proxy-like
    if "scale_proxy" in cols and "rule_execution_gap" in cols:
        return "proxies"

    # Binance aggTrades style
    if {"p", "q", "t"}.issubset(cols) or {"price", "qty", "ts"}.issubset(cols):
        return "binance_aggtrades"

    # OHLCV style
    if {"open", "high", "low", price_col}.issubset(cols):
        return "finance_ohlcv"

    # Borg traces style (very heuristic)
    if any(
        c in cols
        for c in (
            "job_id",
            "task_id",
            "container",
            "service",
            "req_cpu",
            "avg_cpu",
            "resource_request",
            "resource_request_cpu",
        )
    ):
        return "borg_traces"

    # AIOps phase2 style (univariate + labels)
    if {"timestamp", "value"}.issubset(cols) and any(
        c in cols for c in ("kpi id", "kpi_id", "label", "is_anomaly", "anomaly", "is_outlier")
    ):
        return "aiops_phase2"

    # Generic univariate
    if {"timestamp", "value"}.issubset(cols) or {"date", "value"}.issubset(cols) or {"time", "value"}.issubset(cols):
        return "univariate_csv"

    raise SystemExit(
        "convert_real_data.py: cannot auto-detect --kind from columns. "
        "Provide --kind explicitly (univariate_csv, aiops_phase2, borg_traces, finance_ohlcv, binance_aggtrades)."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert real-world data to Amplification-Barometer proxies CSV.")

    # New interface (used by CI workflows)
    ap.add_argument("--input", type=str, default="", help="Input CSV/Parquet.")
    ap.add_argument(
        "--kind",
        type=str,
        default="",
        help="Adapter kind: binance_aggtrades, binance_trades, borg_traces, aiops_phase2, univariate_csv, finance_ohlcv, proxies, auto.",
    )
    ap.add_argument("--out-csv", type=str, default="", help="Output CSV with proxies and a date column.")
    ap.add_argument("--bar-freq", type=str, default="1min", help="Resampling frequency for trade-like feeds.")
    ap.add_argument("--tz", type=str, default="UTC", help="Timezone for output index.")
    ap.add_argument("--price-col", type=str, default="close", help="OHLCV price column (for finance_ohlcv).")
    ap.add_argument("--volume-col", type=str, default="volume", help="OHLCV volume column (optional).")

    # Legacy interface (kept for backwards compatibility)
    ap.add_argument("--infile", type=str, default="", help=argparse.SUPPRESS)
    ap.add_argument("--outfile", type=str, default="", help=argparse.SUPPRESS)

    args = ap.parse_args()

    # New interface
    if args.input or args.out_csv:
        in_path = Path(args.input)
        out_path = Path(args.out_csv)

        if not args.input or not args.out_csv:
            raise SystemExit("convert_real_data.py: --input and --out-csv are required.")
        if not in_path.exists():
            raise SystemExit(f"Missing input: {in_path}")
        if not args.kind:
            raise SystemExit("convert_real_data.py: --kind is required (or use --kind auto).")

        df = _read_any(in_path)
        rda = _maybe_import_adapters()
        tz = str(args.tz)
        kind = str(args.kind).strip().lower()

        if kind == "auto":
            kind = _detect_kind(_cols_lower(df), str(args.price_col))

        if kind == "proxies":
            out = df
            if hasattr(rda, "ensure_datetime_index"):
                out = rda.ensure_datetime_index(out, tz=tz) if "tz" in inspect.signature(rda.ensure_datetime_index).parameters else rda.ensure_datetime_index(out)
            out = _ensure_tz_index(out, tz)

        elif kind == "binance_aggtrades":
            func = rda.binance_aggtrades_to_proxies
            sig = inspect.signature(func)
            if "tz" in sig.parameters:
                out = func(df, bar_freq=str(args.bar_freq), tz=tz)
            else:
                out = func(df, bar_freq=str(args.bar_freq))
            out = _ensure_tz_index(out, tz)

        elif kind == "binance_trades":
            func = rda.binance_trades_to_proxies
            sig = inspect.signature(func)
            if "tz" in sig.parameters:
                out = func(df, bar_freq=str(args.bar_freq), tz=tz)
            else:
                out = func(df, bar_freq=str(args.bar_freq))
            out = _ensure_tz_index(out, tz)

        elif kind == "borg_traces":
            out = _call_with_optional_tz(rda.borg_traces_to_proxies, df, tz)
            out = _ensure_tz_index(out, tz)

        elif kind == "aiops_phase2":
            out = _call_with_optional_tz(rda.aiops_phase2_to_proxies, df, tz)
            out = _ensure_tz_index(out, tz)

        elif kind == "univariate_csv":
            out = _call_with_optional_tz(rda.univariate_csv_to_proxies, df, tz)
            out = _ensure_tz_index(out, tz)

        elif kind in ("finance_ohlcv", "ohlcv"):
            if not (hasattr(rda, "has_required_proxies") and hasattr(rda, "ensure_datetime_index") and hasattr(rda, "finance_ohlcv_to_proxies")):
                raise SystemExit("finance_ohlcv adapter requires ensure_datetime_index/finance_ohlcv_to_proxies in real_data_adapters.")

            if rda.has_required_proxies(df):
                out = rda.ensure_datetime_index(df, tz=tz) if "tz" in inspect.signature(rda.ensure_datetime_index).parameters else rda.ensure_datetime_index(df)
            else:
                df2 = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
                df2 = rda.ensure_datetime_index(df2, tz=tz) if "tz" in inspect.signature(rda.ensure_datetime_index).parameters else rda.ensure_datetime_index(df2)
                vol_col = str(args.volume_col).strip().lower()
                vol_col = vol_col if vol_col in df2.columns else None
                out = rda.finance_ohlcv_to_proxies(df2, price_col=str(args.price_col).strip().lower(), volume_col=vol_col)
            out = _ensure_tz_index(out, tz)

        else:
            raise SystemExit(f"Unknown --kind: {args.kind}")

        _write_proxies_csv(out, out_path)
        return 0

    # Legacy interface
    if not args.infile or not args.outfile:
        raise SystemExit("convert_real_data.py: provide --input/--kind/--out-csv (preferred).")

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
        cols = _cols_lower(df)
        if not {"open", "high", "low", str(args.price_col).lower()}.issubset(cols):
            raise SystemExit("Unsupported legacy input: provide proxies columns or OHLCV columns (open/high/low/close[,volume]).")
        df2 = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
        df2 = rda.ensure_datetime_index(df2, tz=tz) if "tz" in inspect.signature(rda.ensure_datetime_index).parameters else rda.ensure_datetime_index(df2)
        vol_col = str(args.volume_col).strip().lower()
        vol_col = vol_col if vol_col in df2.columns else None
        out = rda.finance_ohlcv_to_proxies(df2, price_col=str(args.price_col).strip().lower(), volume_col=vol_col)
        out = _ensure_tz_index(out, tz)

    _write_proxies_csv(out, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
