from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

REQUIRED_PROXIES: Tuple[str, ...] = (
    "scale_proxy","speed_proxy","leverage_proxy","autonomy_proxy","replicability_proxy",
    "stop_proxy","threshold_proxy","decision_proxy","execution_proxy","coherence_proxy",
    "impact_proxy","propagation_proxy","hysteresis_proxy",
    "margin_proxy","redundancy_proxy","diversity_proxy","recovery_time_proxy",
    "exemption_rate","sanction_delay","control_turnover","conflict_interest_proxy","rule_execution_gap",
)


def has_required_proxies(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    return all(c in cols for c in REQUIRED_PROXIES)


def _robust_unit(x: np.ndarray, *, scale: float = 0.15) -> np.ndarray:
    """Map arbitrary numeric to a 'nice' proxy scale around 1.0 (robust)."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < 1e-12:
        return np.ones_like(x, dtype=float)
    z = (x - med) / (1.4826 * mad)
    return np.clip(1.0 + scale * z, 0.05, 3.0)


def finance_ohlcv_to_proxies(
    ohlcv: pd.DataFrame,
    *,
    price_col: str = "close",
    volume_col: Optional[str] = "volume",
    freq_hint: Optional[str] = None,
) -> pd.DataFrame:
    """Convert a finance OHLCV-like time series to barometer proxies.

    This is a neutral demonstrator: it creates stable, auditable proxies from observable time series,
    without claiming economic causality.
    """
    df = ohlcv.copy()

    if price_col not in df.columns:
        raise ValueError(f"Missing price_col={price_col}")

    price = df[price_col].astype(float)
    ret = price.pct_change().fillna(0.0)
    abs_ret = ret.abs()

    # Rolling stats
    vol = abs_ret.rolling(20, min_periods=5).mean().fillna(abs_ret.mean())
    accel = abs_ret.diff().abs().rolling(10, min_periods=3).mean().fillna(0.0)

    if volume_col and volume_col in df.columns:
        volu = df[volume_col].astype(float).fillna(method="ffill").fillna(0.0)
    else:
        volu = pd.Series(np.ones(len(df), dtype=float), index=df.index)

    logv = np.log1p(volu.to_numpy(dtype=float))

    # Drawdown as hysteresis-ish
    cum = (1.0 + ret).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    dd_abs = dd.abs()

    # Autocorrelation proxy
    autoc = ret.rolling(50, min_periods=10).corr(ret.shift(1)).fillna(0.0).abs()

    out = pd.DataFrame(index=df.index)

    # P proxies (capacity to scale / speed / leverage)
    out["scale_proxy"] = _robust_unit(logv)
    out["speed_proxy"] = _robust_unit(abs_ret.to_numpy(dtype=float))
    out["leverage_proxy"] = _robust_unit(vol.to_numpy(dtype=float))
    out["autonomy_proxy"] = _robust_unit(1.0 - autoc.to_numpy(dtype=float))
    out["replicability_proxy"] = _robust_unit(1.0 / (1.0 + accel.to_numpy(dtype=float)))

    # O proxies (orientation / brakes / coherence). Higher = stronger O.
    out["stop_proxy"] = _robust_unit(1.0 / (1.0 + vol.to_numpy(dtype=float)))
    out["threshold_proxy"] = _robust_unit(1.0 / (1.0 + abs_ret.to_numpy(dtype=float) * 5.0))
    out["decision_proxy"] = _robust_unit(1.0 / (1.0 + accel.to_numpy(dtype=float)))
    out["execution_proxy"] = _robust_unit(1.0 / (1.0 + dd_abs.to_numpy(dtype=float)))
    out["coherence_proxy"] = _robust_unit(1.0 - autoc.to_numpy(dtype=float))

    # E proxies
    out["impact_proxy"] = _robust_unit(abs_ret.to_numpy(dtype=float))
    out["propagation_proxy"] = _robust_unit(vol.to_numpy(dtype=float) + accel.to_numpy(dtype=float))
    out["hysteresis_proxy"] = _robust_unit(dd_abs.to_numpy(dtype=float))

    # R proxies
    out["margin_proxy"] = _robust_unit(1.0 / (1.0 + vol.to_numpy(dtype=float)))
    out["redundancy_proxy"] = _robust_unit(1.0 / (1.0 + autoc.to_numpy(dtype=float)))
    out["diversity_proxy"] = _robust_unit(1.0 / (1.0 + accel.to_numpy(dtype=float)))
    # Recovery time proxy (rolling time-to-recover drawdown approx)
    out["recovery_time_proxy"] = np.clip(_robust_unit(dd_abs.to_numpy(dtype=float), scale=0.30), 0.05, 3.0)

    # G proxies: neutral placeholders (real governance should be measured from actual processes)
    out["exemption_rate"] = 0.10
    out["sanction_delay"] = 30.0
    out["control_turnover"] = 0.10
    out["conflict_interest_proxy"] = 0.10
    out["rule_execution_gap"] = 0.04

    return out


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    for col in ("date", "datetime", "timestamp", "time", "open_time", "close_time"):
        if col in df.columns:
            s = df[col]
            if pd.api.types.is_numeric_dtype(s):
                unit = "ms" if float(s.dropna().iloc[0]) > 1e11 else "s"
                dt = pd.to_datetime(s, unit=unit, utc=True)
            else:
                dt = pd.to_datetime(s, utc=True, errors="coerce")
            df = df.drop(columns=[col])
            df.insert(0, "date", dt)
            df = df.dropna(subset=["date"]).set_index("date")
            break
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("No datetime index or date-like column found")
    return df.sort_index()
