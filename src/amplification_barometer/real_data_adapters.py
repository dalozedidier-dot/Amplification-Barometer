from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS: Tuple[str, ...] = (
    # P proxies
    "scale_proxy",
    "speed_proxy",
    "leverage_proxy",
    "autonomy_proxy",
    "replicability_proxy",
    # O proxies
    "stop_proxy",
    "threshold_proxy",
    "decision_proxy",
    "execution_proxy",
    "coherence_proxy",
    # E proxies
    "impact_proxy",
    "propagation_proxy",
    "hysteresis_proxy",
    # R proxies
    "margin_proxy",
    "redundancy_proxy",
    "diversity_proxy",
    "recovery_time_proxy",
    # G proxies (non _proxy)
    "exemption_rate",
    "sanction_delay",
    "control_turnover",
    "conflict_interest_proxy",
    "rule_execution_gap",
)


def _mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def robust_z(x: pd.Series, eps: float = 1e-12) -> pd.Series:
    """Robust z-score using median and MAD.
    Returns 0 for constant or empty series.
    """
    arr = x.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return pd.Series([0.0] * len(x), index=x.index)
    med = float(np.median(arr))
    mad = _mad(arr)
    denom = mad + eps
    z = (x.astype(float) - med) / denom
    z = z.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return z


def _detect_epoch_unit(ts: pd.Series) -> str:
    """Detect seconds vs ms vs us using magnitude heuristics."""
    v = float(np.nanmax(ts.to_numpy(dtype=float)))
    if v > 1e14:
        return "us"
    if v > 1e12:
        return "ms"
    return "s"


def _to_datetime_index(ts: pd.Series) -> pd.DatetimeIndex:
    unit = _detect_epoch_unit(ts)
    dt = pd.to_datetime(ts.astype("int64"), unit=unit, utc=True, errors="coerce")
    if dt.isna().all():
        # Fallback: treat as monotonic steps
        dt = pd.date_range("1970-01-01", periods=len(ts), freq="s", tz="UTC")
    return pd.DatetimeIndex(dt)


def _ensure_governance_defaults(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults = {
        "exemption_rate": 0.05,
        "sanction_delay": 60.0,
        "control_turnover": 0.05,
        "conflict_interest_proxy": 0.05,
        "rule_execution_gap": 0.05,
    }
    for k, v in defaults.items():
        if k not in out.columns:
            out[k] = float(v)
    return out


def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = _ensure_governance_defaults(out)
    for c in REQUIRED_COLUMNS:
        if c not in out.columns:
            out[c] = 0.0
    out = out[list(REQUIRED_COLUMNS)]
    return out


def _rolling_autocorr(x: pd.Series, window: int = 10) -> pd.Series:
    x = x.astype(float).fillna(0.0)
    def _ac(a: np.ndarray) -> float:
        if a.size < 3:
            return 0.0
        a0 = a[:-1]
        a1 = a[1:]
        s0 = float(np.std(a0))
        s1 = float(np.std(a1))
        if s0 == 0.0 or s1 == 0.0:
            return 0.0
        return float(np.corrcoef(a0, a1)[0, 1])
    return x.rolling(window, min_periods=max(3, window // 2)).apply(_ac, raw=True).fillna(0.0)


def _run_length_high(flag: pd.Series) -> pd.Series:
    """Return run length of consecutive True values ending at each point."""
    out = np.zeros(len(flag), dtype=float)
    run = 0
    for i, v in enumerate(flag.to_numpy(dtype=bool)):
        if v:
            run += 1
        else:
            run = 0
        out[i] = float(run)
    return pd.Series(out, index=flag.index)


def binance_aggtrades_to_proxies(
    raw: pd.DataFrame,
    *,
    bar_freq: str = "1min",
    tz: str = "UTC",
) -> pd.DataFrame:
    """Convert Binance aggTrades (no headers) into the proxy schema.
    Expected columns: [agg_trade_id, price, quantity, first_trade_id, last_trade_id, timestamp, isBuyerMaker]
    """
    if raw.shape[1] < 7:
        raise ValueError("aggTrades expects 7 columns")
    df = raw.iloc[:, :7].copy()
    df.columns = ["agg_id", "price", "qty", "first_id", "last_id", "ts", "is_buyer_maker"]
    df["price"] = df["price"].astype(float)
    df["qty"] = df["qty"].astype(float)
    df["ts"] = df["ts"].astype("int64")
    idx = _to_datetime_index(df["ts"])
    df = df.set_index(idx).sort_index()
    df.index = df.index.tz_convert(tz)

    bars = pd.DataFrame(index=df.resample(bar_freq).sum().index)
    bars["price_last"] = df["price"].resample(bar_freq).last().ffill()
    bars["price_mean"] = df["price"].resample(bar_freq).mean().ffill()
    bars["vol"] = df["qty"].resample(bar_freq).sum().fillna(0.0)
    bars["n_trades"] = df["qty"].resample(bar_freq).count().fillna(0.0)
    bm = df["is_buyer_maker"].astype(str).str.lower().isin(["true", "1", "t"])
    bars["buyer_maker_frac"] = bm.resample(bar_freq).mean().fillna(0.0)

    return _finance_bars_to_proxies(bars)


def binance_trades_to_proxies(
    raw: pd.DataFrame,
    *,
    bar_freq: str = "1min",
    tz: str = "UTC",
) -> pd.DataFrame:
    """Convert Binance trades (no headers) into the proxy schema.
    Typical 6 columns: [id, price, qty, quoteQty, time, isBuyerMaker]
    Typical 7 columns: [id, price, qty, quoteQty, time, isBuyerMaker, isBestMatch]
    """
    if raw.shape[1] < 6:
        raise ValueError("trades expects at least 6 columns")
    df = raw.copy()
    df = df.iloc[:, :7] if df.shape[1] >= 7 else df.iloc[:, :6]
    if df.shape[1] == 6:
        df.columns = ["trade_id", "price", "qty", "quote_qty", "ts", "is_buyer_maker"]
        df["is_best_match"] = False
    else:
        df.columns = ["trade_id", "price", "qty", "quote_qty", "ts", "is_buyer_maker", "is_best_match"]
    df["price"] = df["price"].astype(float)
    df["qty"] = df["qty"].astype(float)
    df["ts"] = df["ts"].astype("int64")
    idx = _to_datetime_index(df["ts"])
    df = df.set_index(idx).sort_index()
    df.index = df.index.tz_convert(tz)

    bars = pd.DataFrame(index=df.resample(bar_freq).sum().index)
    bars["price_last"] = df["price"].resample(bar_freq).last().ffill()
    bars["price_mean"] = df["price"].resample(bar_freq).mean().ffill()
    bars["vol"] = df["qty"].resample(bar_freq).sum().fillna(0.0)
    bars["n_trades"] = df["qty"].resample(bar_freq).count().fillna(0.0)
    bm = df["is_buyer_maker"].astype(str).str.lower().isin(["true", "1", "t"])
    bars["buyer_maker_frac"] = bm.resample(bar_freq).mean().fillna(0.0)

    return _finance_bars_to_proxies(bars)


def _finance_bars_to_proxies(bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars.copy()
    bars["price_last"] = bars["price_last"].astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    bars["ret"] = np.log(bars["price_last"]).diff().fillna(0.0)
    bars["abs_ret"] = bars["ret"].abs()
    bars["volatility"] = bars["ret"].rolling(20, min_periods=5).std().fillna(0.0)
    bars["vwap_gap"] = (bars["price_mean"] - bars["price_last"]).abs() / (bars["price_last"].abs() + 1e-12)

    high_vol = bars["volatility"] > bars["volatility"].rolling(200, min_periods=20).median().fillna(bars["volatility"].median())
    rec_run = _run_length_high(high_vol)

    # Proxies (keep them numeric, downstream composites will robust-normalize)
    out = pd.DataFrame(index=bars.index)
    out["scale_proxy"] = np.log1p(bars["vol"].astype(float))
    out["speed_proxy"] = np.log1p(bars["n_trades"].astype(float))
    out["leverage_proxy"] = bars["volatility"].astype(float)
    out["autonomy_proxy"] = 1.0 - _rolling_autocorr(bars["ret"], window=20).abs()
    out["replicability_proxy"] = 1.0 - (bars["n_trades"].rolling(50, min_periods=10).std() / (bars["n_trades"].rolling(50, min_periods=10).mean() + 1e-12)).fillna(0.0)

    out["stop_proxy"] = 1.0 - robust_z(bars["volatility"]).clip(lower=-3, upper=3) / 3.0
    out["threshold_proxy"] = 1.0 - robust_z(bars["abs_ret"]).clip(lower=-3, upper=3) / 3.0
    out["decision_proxy"] = 1.0 - robust_z(bars["vwap_gap"]).clip(lower=-3, upper=3) / 3.0
    out["execution_proxy"] = 1.0 - robust_z(bars["buyer_maker_frac"]).clip(lower=-3, upper=3) / 3.0
    out["coherence_proxy"] = 1.0 - robust_z(_rolling_autocorr(bars["abs_ret"], window=30).abs()).clip(lower=-3, upper=3) / 3.0

    out["impact_proxy"] = bars["abs_ret"] * np.log1p(bars["vol"].astype(float))
    out["propagation_proxy"] = bars["volatility"].diff().abs().fillna(0.0)
    out["hysteresis_proxy"] = _rolling_autocorr(bars["abs_ret"], window=50).abs()

    out["margin_proxy"] = 1.0 / (1.0 + bars["volatility"].astype(float))
    out["redundancy_proxy"] = 1.0 / (1.0 + bars["n_trades"].rolling(30, min_periods=10).std().fillna(0.0))
    out["diversity_proxy"] = 1.0 / (1.0 + bars["vwap_gap"].astype(float))
    out["recovery_time_proxy"] = rec_run.astype(float)

    out = _ensure_governance_defaults(out)
    return ensure_required_columns(out)


def borg_traces_to_proxies(
    raw: pd.DataFrame,
    *,
    bucket_seconds: int = 60,
) -> pd.DataFrame:
    """Convert Borg traces rows into proxy schema using time buckets.
    Uses average_usage, maximum_usage, resource_request.cpus, failed, user diversity.
    """
    df = raw.copy()

    if "time" not in df.columns:
        raise ValueError("borg traces require a 'time' column")

    t = pd.to_numeric(df["time"], errors="coerce").fillna(0.0)
    # convert to seconds relative
    unit = "us" if float(np.nanmax(t.to_numpy())) > 1e12 else "ms"
    scale = 1e6 if unit == "us" else 1e3
    t_s = (t - float(t.min())) / scale
    bucket = np.floor(t_s / float(bucket_seconds)).astype(int)
    df["_bucket"] = bucket

    def _parse_req(x: Any) -> Dict[str, float]:
        try:
            d = ast.literal_eval(str(x))
            if isinstance(d, dict):
                return {"cpus": float(d.get("cpus", 0.0)), "memory": float(d.get("memory", 0.0))}
        except Exception:
            pass
        return {"cpus": 0.0, "memory": 0.0}

    req = df["resource_request"].apply(_parse_req) if "resource_request" in df.columns else pd.Series([{"cpus": 0.0, "memory": 0.0}] * len(df))
    df["_req_cpu"] = req.apply(lambda d: float(d.get("cpus", 0.0)))
    df["_req_mem"] = req.apply(lambda d: float(d.get("memory", 0.0)))

    avg = pd.to_numeric(df.get("average_usage", 0.0), errors="coerce").fillna(0.0)
    mx = pd.to_numeric(df.get("maximum_usage", 0.0), errors="coerce").fillna(0.0)
    failed = pd.to_numeric(df.get("failed", 0.0), errors="coerce").fillna(0.0)

    df["_avg_cpu"] = avg
    df["_max_cpu"] = mx
    df["_failed"] = failed

    user = df.get("user", pd.Series(["u"] * len(df))).astype(str)

    g = df.groupby("_bucket", sort=True)
    agg = pd.DataFrame(index=g.size().index.astype(int))
    agg["avg_cpu"] = g["_avg_cpu"].mean().astype(float)
    agg["max_cpu"] = g["_max_cpu"].mean().astype(float)
    agg["req_cpu"] = g["_req_cpu"].mean().astype(float)
    agg["req_mem"] = g["_req_mem"].mean().astype(float)
    agg["failed_rate"] = g["_failed"].mean().astype(float)
    agg["n_rows"] = g.size().astype(float)
    agg["user_diversity"] = g.apply(lambda x: x["user"].nunique()).astype(float)

    agg = agg.sort_index()
    idx = pd.date_range("1970-01-01", periods=len(agg), freq=f"{bucket_seconds}s", tz="UTC")
    agg.index = idx

    overuse = ((agg["avg_cpu"] - agg["req_cpu"]).clip(lower=0.0)) / (agg["req_cpu"] + 1e-12)
    gap = (agg["req_cpu"] - agg["avg_cpu"]).abs() / (agg["req_cpu"] + 1e-12)

    high_fail = agg["failed_rate"] > 0.0
    rec_run = _run_length_high(high_fail)

    out = pd.DataFrame(index=agg.index)
    out["scale_proxy"] = np.log1p(agg["req_cpu"] + agg["req_mem"])
    out["speed_proxy"] = np.log1p(agg["n_rows"])
    out["leverage_proxy"] = (agg["max_cpu"] / (agg["avg_cpu"] + 1e-12)).clip(lower=0.0)
    out["autonomy_proxy"] = 1.0 - _rolling_autocorr(agg["avg_cpu"], window=20).abs()
    out["replicability_proxy"] = 1.0 - (agg["avg_cpu"].rolling(30, min_periods=10).std() / (agg["avg_cpu"].rolling(30, min_periods=10).mean() + 1e-12)).fillna(0.0)

    out["stop_proxy"] = 1.0 - agg["failed_rate"].clip(0.0, 1.0)
    out["threshold_proxy"] = 1.0 - robust_z(overuse).clip(lower=-3, upper=3) / 3.0
    out["decision_proxy"] = 1.0 - robust_z(gap).clip(lower=-3, upper=3) / 3.0
    out["execution_proxy"] = 1.0 - robust_z(agg["max_cpu"]).clip(lower=-3, upper=3) / 3.0
    out["coherence_proxy"] = 1.0 - robust_z(agg["avg_cpu"].diff().abs().fillna(0.0)).clip(lower=-3, upper=3) / 3.0

    out["impact_proxy"] = agg["failed_rate"] + overuse
    out["propagation_proxy"] = agg["failed_rate"].diff().abs().fillna(0.0)
    out["hysteresis_proxy"] = agg["failed_rate"].rolling(10, min_periods=3).mean().fillna(0.0)

    out["margin_proxy"] = 1.0 / (1.0 + agg["avg_cpu"])
    out["redundancy_proxy"] = 1.0 / (1.0 + gap)
    out["diversity_proxy"] = np.log1p(agg["user_diversity"])
    out["recovery_time_proxy"] = rec_run.astype(float)

    out["rule_execution_gap"] = gap.clip(0.0, 1.0)
    out["control_turnover"] = robust_z(agg["user_diversity"]).abs().clip(0.0, 1.0)
    out["sanction_delay"] = 60.0
    out["exemption_rate"] = 0.05
    out["conflict_interest_proxy"] = 0.05

    return ensure_required_columns(out)


def aiops_phase2_to_proxies(
    raw: pd.DataFrame,
    *,
    kpi_id: Optional[str] = None,
) -> pd.DataFrame:
    """Convert AIOps KPI time series (phase2_train style) to proxy schema.
    Columns: timestamp (s), value, label (0/1), KPI ID.
    """
    df = raw.copy()
    for c in ["timestamp", "value"]:
        if c not in df.columns:
            raise ValueError("AIOps phase2 requires timestamp and value columns")

    if kpi_id is None and "KPI ID" in df.columns:
        kpi_id = str(df["KPI ID"].iloc[0])
    if kpi_id is not None and "KPI ID" in df.columns:
        df = df[df["KPI ID"].astype(str) == str(kpi_id)].copy()

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0.0).astype("int64")
    idx = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
    if idx.isna().all():
        idx = pd.date_range("1970-01-01", periods=len(df), freq="min", tz="UTC")
    df = df.set_index(pd.DatetimeIndex(idx)).sort_index()

    v = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    label = pd.to_numeric(df.get("label", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)

    dv = v.diff().abs().fillna(0.0)
    vol = v.rolling(30, min_periods=10).std().fillna(0.0)
    resid = (v - v.rolling(30, min_periods=10).median().fillna(v.median())).abs()

    high = (label > 0.0) | (dv > dv.quantile(0.95))
    rec_run = _run_length_high(high)

    out = pd.DataFrame(index=df.index)
    out["scale_proxy"] = np.log1p(v.abs())
    out["speed_proxy"] = np.log1p(dv + 1e-12)
    out["leverage_proxy"] = vol
    out["autonomy_proxy"] = 1.0 - _rolling_autocorr(v, window=20).abs()
    out["replicability_proxy"] = 1.0 - (vol / (v.abs().rolling(30, min_periods=10).mean() + 1e-12)).fillna(0.0)

    out["stop_proxy"] = 1.0 - label
    out["threshold_proxy"] = 1.0 - robust_z(resid).clip(lower=-3, upper=3) / 3.0
    out["decision_proxy"] = 1.0 - robust_z(dv).clip(lower=-3, upper=3) / 3.0
    out["execution_proxy"] = 1.0 - robust_z(vol).clip(lower=-3, upper=3) / 3.0
    out["coherence_proxy"] = 1.0 - robust_z(_rolling_autocorr(v, window=30).abs()).clip(lower=-3, upper=3) / 3.0

    out["impact_proxy"] = label + robust_z(resid).abs()
    out["propagation_proxy"] = label.diff().abs().fillna(0.0)
    out["hysteresis_proxy"] = label.rolling(10, min_periods=3).mean().fillna(0.0)

    out["margin_proxy"] = 1.0 / (1.0 + vol)
    out["redundancy_proxy"] = 1.0 / (1.0 + robust_z(dv).abs())
    out["diversity_proxy"] = 0.5
    out["recovery_time_proxy"] = rec_run.astype(float)

    out = _ensure_governance_defaults(out)
    return ensure_required_columns(out)
