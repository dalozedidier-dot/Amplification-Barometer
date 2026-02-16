from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


EPS = 1e-12


def robust_scale_to_range(x: Sequence[float] | pd.Series, lo: float, hi: float) -> pd.Series:
    """Robustly maps a numeric series into [lo, hi] using median/MAD and tanh squashing.

    This is meant for *adapters* only: it produces sane proxy magnitudes from arbitrary real-world units.
    Baseline comparability is handled later by the barometer baseline and thresholds.
    """
    s = pd.to_numeric(pd.Series(x), errors="coerce").astype(float)
    med = float(np.nanmedian(s))
    mad = float(np.nanmedian(np.abs(s - med)) + EPS)
    z = (s - med) / mad
    y = np.tanh(z / 3.0)  # squash to about [-1, 1]
    return pd.Series(lo + (y + 1.0) * 0.5 * (hi - lo), index=s.index)


def _df_numeric_sum(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[list(cols)].apply(pd.to_numeric, errors="coerce").sum(axis=1)


def _df_numeric_mean_abs(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[list(cols)].apply(pd.to_numeric, errors="coerce").abs().mean(axis=1)


def crypto_orderbook_to_proxies(df: pd.DataFrame, *, time_col: str = "system_time") -> pd.DataFrame:
    """Adapter: crypto orderbook-style CSV -> canonical proxy frame.

    Expected columns (typical): midpoint, spread, buys, sells, bids_notional_k, asks_notional_k,
    bids_distance_k, asks_distance_k, bids_cancel_notional_k, asks_cancel_notional_k.
    """
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[time_col], errors="coerce")

    midpoint = pd.to_numeric(df.get("midpoint"), errors="coerce")
    spread = pd.to_numeric(df.get("spread"), errors="coerce")
    buys = pd.to_numeric(df.get("buys"), errors="coerce")
    sells = pd.to_numeric(df.get("sells"), errors="coerce")

    logp = np.log(midpoint.replace(0, np.nan))
    ret = logp.diff()
    vol = ret.rolling(20, min_periods=5).std()

    depth_cols = [c for c in df.columns if re.match(r"(bids|asks)_notional_\d+$", str(c))]
    dist_cols = [c for c in df.columns if re.match(r"(bids|asks)_distance_\d+$", str(c))]
    depth = _df_numeric_sum(df, depth_cols)
    dist = _df_numeric_mean_abs(df, dist_cols)

    imbalance = (buys - sells) / (buys + sells + 1e-9)

    # P (up-risk)
    out["scale_proxy"] = robust_scale_to_range(logp, 0.0, 2.0)
    out["speed_proxy"] = robust_scale_to_range(ret.abs().fillna(0.0), 0.0, 2.0)
    out["leverage_proxy"] = robust_scale_to_range(imbalance.abs().fillna(0.0), 0.0, 2.0)
    out["autonomy_proxy"] = robust_scale_to_range(1.0 / (1.0 + spread.fillna(spread.median()) * 1000.0), 0.0, 1.0)
    out["replicability_proxy"] = robust_scale_to_range(1.0 / (1.0 + vol.fillna(vol.median()) * 50.0), 0.0, 1.0)

    # O (down-risk): higher is better
    out["stop_proxy"] = robust_scale_to_range(1.0 / (1.0 + vol.fillna(vol.median()) * 80.0), 0.0, 1.0)
    out["threshold_proxy"] = robust_scale_to_range(1.0 / (1.0 + spread.fillna(spread.median()) * 1500.0), 0.0, 1.0)
    out["decision_proxy"] = robust_scale_to_range(1.0 - imbalance.abs().fillna(imbalance.abs().median()), 0.0, 1.0)
    out["execution_proxy"] = robust_scale_to_range(depth.fillna(depth.median()) / (depth.median() + 1e-9), 0.0, 1.0)
    out["coherence_proxy"] = robust_scale_to_range((buys.fillna(buys.median()) * sells.fillna(sells.median())) ** 0.5, 0.0, 1.0)

    # E (up-risk)
    absret = ret.abs().fillna(0.0)

    def _autocorr1(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.size < 3:
            return 0.0
        x0 = x[:-1]
        x1 = x[1:]
        s0 = float(np.std(x0))
        s1 = float(np.std(x1))
        if s0 <= 0.0 or s1 <= 0.0:
            return 0.0
        return float(np.corrcoef(x0, x1)[0, 1])

    hyst = absret.rolling(50, min_periods=10).apply(_autocorr1, raw=True)

    out["impact_proxy"] = robust_scale_to_range(vol.fillna(vol.median()) * 100.0, 0.0, 2.0)
    out["propagation_proxy"] = robust_scale_to_range(dist.fillna(dist.median()) * 1e4, 0.0, 2.0)
    out["hysteresis_proxy"] = robust_scale_to_range(hyst.fillna(0.0).clip(-1, 1) + 1.0, 0.0, 2.0)

    # R (down-risk except recovery_time_proxy is up-risk by protocol)
    out["margin_proxy"] = robust_scale_to_range((depth / (depth.median() + 1e-9)).fillna(1.0), 0.0, 1.0)

    if depth_cols:
        levels = df[list(depth_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
        p = levels / (levels.sum(axis=1, keepdims=True) + EPS)
        ent = -(p * np.log(p + EPS)).sum(axis=1) / np.log(p.shape[1])
        ent = pd.Series(ent, index=df.index)
    else:
        ent = pd.Series(0.5, index=df.index)
    out["redundancy_proxy"] = robust_scale_to_range(ent, 0.0, 1.0)

    bid_cols = [c for c in df.columns if re.match(r"bids_notional_\d+$", str(c))]
    ask_cols = [c for c in df.columns if re.match(r"asks_notional_\d+$", str(c))]
    bid_depth = _df_numeric_sum(df, bid_cols) if bid_cols else depth / 2.0
    ask_depth = _df_numeric_sum(df, ask_cols) if ask_cols else depth / 2.0
    balance = 1.0 - (bid_depth - ask_depth).abs() / (bid_depth + ask_depth + 1e-9)
    out["diversity_proxy"] = robust_scale_to_range(balance.fillna(0.5), 0.0, 1.0)

    roll_max = logp.rolling(100, min_periods=10).max()
    drawdown = (roll_max - logp).fillna(0.0)
    out["recovery_time_proxy"] = robust_scale_to_range(drawdown, 0.0, 1.0)

    # G: neutral defaults unless you provide governance observables
    out["exemption_rate"] = 0.02
    out["sanction_delay"] = 0.03
    out["control_turnover"] = 0.02
    out["conflict_interest_proxy"] = 0.02
    out["rule_execution_gap"] = 0.03

    return out


def algae_mat_to_proxies(mat_path: str | Path, *, raceway: int = 0) -> pd.DataFrame:
    """Adapter: Algae Raceway MATLAB file -> canonical proxy frame.

    Reads the algae_ts struct, extracts a few variables, interpolates temperature to the main grid,
    then maps to the canonical proxy set.
    """
    from scipy.io import loadmat  # optional dependency, imported lazily

    mat = loadmat(str(mat_path))
    algae_ts = mat["algae_ts"]
    race = algae_ts[0, raceway]

    def _extract_series(field: str) -> pd.Series:
        obj = race[field]
        if isinstance(obj, np.ndarray) and obj.shape == (1, 1):
            obj = obj[0, 0]
        data = obj["data"]
        time = obj["time_num"]
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data[0, 0]
        if isinstance(time, np.ndarray) and time.dtype == object:
            time = time[0, 0]
        data = np.asarray(data, dtype=float).reshape(-1)
        time = np.asarray(time, dtype=float).reshape(-1)
        return pd.Series(data, index=time)

    base = _extract_series("density")
    idx = base.index.to_numpy(dtype=float)

    density = base.to_numpy(dtype=float)
    oxygen = _extract_series("oxygen").reindex(base.index).to_numpy(dtype=float)
    ph = _extract_series("pH").reindex(base.index).to_numpy(dtype=float)
    irr = _extract_series("irradiance_amount_smooth").reindex(base.index).to_numpy(dtype=float)
    nitrate = _extract_series("nitrate").reindex(base.index).to_numpy(dtype=float)
    phosphate = _extract_series("phosphate").reindex(base.index).to_numpy(dtype=float)

    # temperature sampled on a different grid, interpolate to density grid
    temp_s = _extract_series("temperature")
    tx = temp_s.index.to_numpy(dtype=float)
    ty = temp_s.to_numpy(dtype=float)
    order = np.argsort(tx)
    tx = tx[order]
    ty = ty[order]
    mask = np.isfinite(tx) & np.isfinite(ty)
    tx = tx[mask]
    ty = ty[mask]
    if tx.size >= 2:
        temperature = np.interp(idx, tx, ty)
    else:
        temperature = np.full_like(idx, np.nan, dtype=float)

    raw = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=len(idx), freq="h"),
            "density": density,
            "oxygen": oxygen,
            "pH": ph,
            "irradiance": irr,
            "nitrate": nitrate,
            "phosphate": phosphate,
            "temperature": temperature,
        }
    )

    dens = pd.Series(raw["density"]).astype(float)
    logd = np.log(dens + 1e-9)
    dlog = logd.diff().fillna(0.0)

    oxy = pd.Series(raw["oxygen"]).astype(float)
    dox = oxy.diff().fillna(0.0)

    phs = pd.Series(raw["pH"]).astype(float)
    ph_dev = (phs - phs.median()).abs()

    irr_s = pd.Series(raw["irradiance"]).astype(float)
    n_s = pd.Series(raw["nitrate"]).astype(float)
    p_s = pd.Series(raw["phosphate"]).astype(float)

    temp = pd.Series(raw["temperature"]).astype(float)
    temp_var = temp.rolling(24, min_periods=6).std().fillna(temp.std())

    out = pd.DataFrame()
    out["date"] = raw["date"]

    out["scale_proxy"] = robust_scale_to_range(logd, 0.0, 2.0)
    out["speed_proxy"] = robust_scale_to_range(dlog.abs(), 0.0, 2.0)
    out["leverage_proxy"] = robust_scale_to_range(irr_s, 0.0, 2.0)
    out["autonomy_proxy"] = robust_scale_to_range(1.0 / (1.0 + temp_var), 0.0, 1.0)
    out["replicability_proxy"] = robust_scale_to_range(1.0 / (1.0 + ph_dev), 0.0, 1.0)

    out["stop_proxy"] = robust_scale_to_range(1.0 / (1.0 + dlog.abs() * 10.0), 0.0, 1.0)
    out["threshold_proxy"] = robust_scale_to_range(1.0 / (1.0 + ph_dev), 0.0, 1.0)
    out["decision_proxy"] = robust_scale_to_range(1.0 / (1.0 + n_s.diff().abs().fillna(0.0) * 2.0), 0.0, 1.0)
    out["execution_proxy"] = robust_scale_to_range(1.0 / (1.0 + (-dox).clip(lower=0.0) * 0.05), 0.0, 1.0)
    coh = temp.rolling(48, min_periods=10).corr(irr_s).fillna(0.0)
    out["coherence_proxy"] = robust_scale_to_range((coh.clip(-1, 1) + 1.0) / 2.0, 0.0, 1.0)

    impact = (-dox).clip(lower=0.0) + ph_dev * 10.0
    out["impact_proxy"] = robust_scale_to_range(impact, 0.0, 2.0)
    out["propagation_proxy"] = robust_scale_to_range((n_s.diff().abs().fillna(0.0) + p_s.diff().abs().fillna(0.0)) * 5.0, 0.0, 2.0)

    absimp = pd.Series(impact).fillna(0.0)

    def _autocorr1(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.size < 3:
            return 0.0
        x0 = x[:-1]
        x1 = x[1:]
        s0 = float(np.std(x0))
        s1 = float(np.std(x1))
        if s0 <= 0.0 or s1 <= 0.0:
            return 0.0
        return float(np.corrcoef(x0, x1)[0, 1])

    hyst = absimp.rolling(50, min_periods=10).apply(_autocorr1, raw=True)
    out["hysteresis_proxy"] = robust_scale_to_range(hyst.fillna(0.0).clip(-1, 1) + 1.0, 0.0, 2.0)

    out["margin_proxy"] = robust_scale_to_range(oxy, 0.0, 1.0)
    var = np.vstack([ph_dev.to_numpy(), n_s.to_numpy(), p_s.to_numpy()]).var(axis=0)
    out["redundancy_proxy"] = robust_scale_to_range(1.0 / (1.0 + var), 0.0, 1.0)

    balance = 1.0 - (n_s - p_s * 20.0).abs() / (n_s.abs() + p_s.abs() * 20.0 + 1e-9)
    out["diversity_proxy"] = robust_scale_to_range(balance.fillna(0.5), 0.0, 1.0)

    oxmax = oxy.rolling(72, min_periods=12).max()
    draw = (oxmax - oxy).fillna(0.0)
    out["recovery_time_proxy"] = robust_scale_to_range(draw, 0.0, 1.0)

    out["exemption_rate"] = 0.02
    out["sanction_delay"] = 0.03
    out["control_turnover"] = 0.02
    out["conflict_interest_proxy"] = 0.02
    out["rule_execution_gap"] = 0.03

    return out
