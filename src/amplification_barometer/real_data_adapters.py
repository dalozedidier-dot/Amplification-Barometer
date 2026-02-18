from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


REQUIRED_PROXIES: Tuple[str, ...] = (
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



# Backwards/forwards compatibility: some code/tests refer to REQUIRED_COLUMNS.
REQUIRED_COLUMNS: Tuple[str, ...] = REQUIRED_PROXIES

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


GOVERNANCE_DEFAULTS = {
    "exemption_rate": 0.05,
    "sanction_delay": 60.0,
    # Defaults are deliberately conservative and compatible with the critical target <0.05.
    "control_turnover": 0.03,
    "conflict_interest_proxy": 0.05,
    "rule_execution_gap": 0.03,
}

def _ensure_governance_defaults(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for k, v in GOVERNANCE_DEFAULTS.items():
        if k not in out.columns:
            out[k] = float(v)
    return out



def _sanitize_governance(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize governance proxies when a known degeneracy pattern is detected.

    Motivation (auditability):
    Some real datasets provide incomplete fields for "rule execution gap" and the
    derived proxy can saturate near 1.0 even when incident-related signals are low.
    This creates a false governance collapse and dominates G(t).

    Guard rule:
    - If rule_execution_gap median is very high (>= 0.95)
      AND impact_proxy median is low (< 0.20),
      then we reinterpret rule_execution_gap as an incident-persistence proxy
      derived from impact_proxy (bounded 0..1) with a short rolling mean.

    This keeps the proxy conservative and bounded, while preventing division-by-zero
    and missing-field artifacts from polluting the audit.
    """
    out = df.copy()
    if "rule_execution_gap" not in out.columns:
        return out
    r = pd.to_numeric(out["rule_execution_gap"], errors="coerce")
    if "impact_proxy" not in out.columns:
        return out
    imp = pd.to_numeric(out["impact_proxy"], errors="coerce")
    r_med = float(np.nanmedian(r.to_numpy(dtype=float))) if r.size else float("nan")
    imp_med = float(np.nanmedian(imp.to_numpy(dtype=float))) if imp.size else float("nan")
    if np.isfinite(r_med) and np.isfinite(imp_med) and (r_med >= 0.95) and (imp_med < 0.20):
        # Degeneracy: prefer recovery persistence if available (more direct "execution gap").
        if "recovery_time_proxy" in out.columns:
            rt = pd.to_numeric(out["recovery_time_proxy"], errors="coerce").fillna(0.0).astype(float)
            # Scale: 10 time steps to reach 1.0 (conservative); then smooth.
            s = np.clip(rt / 10.0, 0.0, 1.0)
        else:
            # Fallback: incident intensity proxy.
            s = np.clip(imp.astype(float), 0.0, 1.0)

        s = s.rolling(10, min_periods=3).mean().fillna(0.0)
        out["rule_execution_gap"] = s.astype(float)
    return out

def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = _ensure_governance_defaults(out)
    out = _sanitize_governance(out)
    for c in REQUIRED_PROXIES:
        if c not in out.columns:
            out[c] = 0.0
    out = out[list(REQUIRED_PROXIES)]
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
    tz: str = "UTC",
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
    agg["user_diversity"] = g["user"].nunique().astype(float)

    agg = agg.sort_index()
    idx = pd.date_range("1970-01-01", periods=len(agg), freq=f"{bucket_seconds}s", tz="UTC")
    idx = idx.tz_convert(str(tz))
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

    # Governance proxy: rule_execution_gap
    # Original implementation used abs(req-avg)/req which explodes when req is missing or near zero,
    # and then gets clipped to 1.0 (false "governance collapse"). We make it robust:
    # - if requests coverage is low, fall back to a conservative, auditable proxy based on failure persistence
    # - otherwise keep the original gap definition.
    req_total = (agg["req_cpu"] + agg["req_mem"]).astype(float)
    req_coverage = float(np.mean((req_total > 0.0).astype(float)))
    if req_coverage < 0.80:
        # Failure persistence as "execution gap" proxy (0..1), small when incidents are rare/short.
        exec_gap = (agg["failed_rate"] > 0.0).astype(float).rolling(10, min_periods=3).mean().fillna(0.0)
    else:
        # Use overuse (violations above request) rather than absolute gap.
        # Underuse is not a governance failure.
        exec_gap = overuse

    out["rule_execution_gap"] = exec_gap.clip(0.0, 1.0)
    out["control_turnover"] = robust_z(agg["user_diversity"]).abs().clip(0.0, 1.0)
    out["sanction_delay"] = 60.0
    out["exemption_rate"] = 0.05
    out["conflict_interest_proxy"] = 0.05

    out.index.name = "date"
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

def univariate_csv_to_proxies(
    raw: pd.DataFrame,
    *,
    timestamp_col: Optional[str] = None,
    value_col: Optional[str] = None,
    label_col: Optional[str] = None,
    tz: str = "UTC",
) -> pd.DataFrame:
    """Convert a generic univariate time series CSV to proxy schema.

    Supported inputs:
    - timestamp/date column as ISO string or epoch seconds/ms/us
    - value column (float)
    - optional label column (0/1 or bool), or is_anomaly/anomaly

    This adapter is intended for datasets like NAB, Yahoo S5, simple KPI exports,
    and CPC result series with columns [timestamp, value].
    """
    df = raw.copy()

    cols_l = {str(c).lower(): c for c in df.columns}
    def _pick(cands, explicit):
        if explicit:
            if explicit in df.columns:
                return explicit
            if explicit.lower() in cols_l:
                return cols_l[explicit.lower()]
        for c in cands:
            if c in df.columns:
                return c
            if c.lower() in cols_l:
                return cols_l[c.lower()]
        return None

    ts_col = _pick(["timestamp", "date", "time", "ds", "datetime"], timestamp_col)
    val_col = _pick(["value", "y", "metric", "count", "val"], value_col)
    lab_col = _pick(["label", "is_anomaly", "anomaly", "is_outlier"], label_col)

    if ts_col is None or val_col is None:
        raise ValueError("univariate_csv_to_proxies: need timestamp/date and value columns")

    ts = df[ts_col]
    # Numeric epoch
    if pd.api.types.is_numeric_dtype(ts):
        idx = _to_datetime_index(pd.to_numeric(ts, errors="coerce").fillna(0.0))
    else:
        dt = pd.to_datetime(ts, utc=True, errors="coerce")
        if dt.isna().mean() > 0.80:
            # Fallback: attempt numeric parsing
            ts2 = pd.to_numeric(ts, errors="coerce")
            if ts2.notna().any():
                idx = _to_datetime_index(ts2.fillna(0.0))
            else:
                idx = pd.date_range("1970-01-01", periods=len(df), freq="min", tz="UTC")
        else:
            idx = pd.DatetimeIndex(dt)

    idx = idx.tz_convert(tz) if idx.tz is not None else idx.tz_localize("UTC").tz_convert(tz)
    df = df.copy()
    df.index = idx
    df = df.sort_index()

    v = pd.to_numeric(df[val_col], errors="coerce").fillna(0.0).astype(float)

    if lab_col is None:
        label = pd.Series([0.0] * len(df), index=df.index, dtype=float)
    else:
        label = pd.to_numeric(df[lab_col], errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(float)

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

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a UTC datetime index named 'date'.

    Accepts either:
    - a 'date' column
    - an existing DatetimeIndex

    Always returns a copy sorted by index, with tz-aware UTC timestamps.
    """
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
        out = out.dropna(subset=["date"]).set_index("date")
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("Input must have a 'date' column or a DatetimeIndex.")
    # Ensure UTC tz-aware
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out = out.sort_index()
    out.index.name = "date"
    return out


def has_required_proxies(df: pd.DataFrame) -> bool:
    """Return True if all required proxy columns are present."""
    cols = set(df.columns)
    return all(c in cols for c in REQUIRED_PROXIES)


def finance_ohlcv_to_proxies(
    df: pd.DataFrame,
    *,
    price_col: str = "close",
    volume_col: Optional[str] = None,
) -> pd.DataFrame:
    """Convert OHLCV-like data to proxy columns.

    This is a lightweight adapter for smoke tests. It produces the REQUIRED_PROXIES
    using robust, scale-aware transforms. Governance columns are set to conservative
    defaults and will be flagged as uninformative by the audit layer.
    """
    df2 = ensure_datetime_index(df)
    if price_col not in df2.columns:
        raise ValueError(f"Missing price column: {price_col}")
    price = pd.to_numeric(df2[price_col], errors="coerce").astype(float)
    price = price.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")
    logp = np.log(np.maximum(price.to_numpy(), 1e-12))
    # Returns and volatility
    ret = np.diff(logp, prepend=logp[0])
    abs_ret = np.abs(ret)
    vol = pd.Series(abs_ret, index=df2.index).rolling(50, min_periods=5).mean().to_numpy()
    vol = np.nan_to_num(vol, nan=float(np.nanmean(vol)) if np.isfinite(np.nanmean(vol)) else 0.0)

    scale_proxy = pd.Series(logp, index=df2.index)
    speed_proxy = pd.Series(abs_ret, index=df2.index)
    leverage_proxy = pd.Series(vol, index=df2.index)

    if volume_col and volume_col in df2.columns:
        volu = pd.to_numeric(df2[volume_col], errors="coerce").astype(float)
        volu = volu.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        leverage_proxy = leverage_proxy + robust_z(volu)

    proxies = pd.DataFrame(index=df2.index)
    proxies["scale_proxy"] = robust_z(scale_proxy)
    proxies["speed_proxy"] = robust_z(speed_proxy)
    proxies["leverage_proxy"] = robust_z(leverage_proxy)

    # P
    proxies["autonomy_proxy"] = 1.0
    proxies["replicability_proxy"] = 1.0

    # O (use volatility as a proxy for execution/coherence pressures)
    proxies["stop_proxy"] = 1.0
    proxies["threshold_proxy"] = robust_z(pd.Series(vol, index=df2.index))
    proxies["decision_proxy"] = robust_z(pd.Series(ret, index=df2.index))
    proxies["execution_proxy"] = robust_z(pd.Series(abs_ret, index=df2.index))
    proxies["coherence_proxy"] = 1.0

    # E
    proxies["impact_proxy"] = robust_z(pd.Series(abs_ret, index=df2.index))
    proxies["propagation_proxy"] = robust_z(pd.Series(vol, index=df2.index))
    proxies["hysteresis_proxy"] = 0.0

    # R
    proxies["margin_proxy"] = 1.0
    proxies["redundancy_proxy"] = 0.5
    proxies["diversity_proxy"] = 0.5
    proxies["recovery_time_proxy"] = 0.0

    # G defaults (unavoidably uninformative for pure OHLCV)
    proxies["exemption_rate"] = 0.05
    proxies["sanction_delay"] = 60.0
    proxies["control_turnover"] = 0.03
    proxies["conflict_interest_proxy"] = 0.05
    proxies["rule_execution_gap"] = 0.03

    return proxies
