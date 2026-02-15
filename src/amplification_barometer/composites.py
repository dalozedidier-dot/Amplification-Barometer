
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .baseline import BaselineCalibration, RobustStats, baseline_mad_z, robust_stats


WEIGHTS_VERSION = "v0.5.0"


@dataclass(frozen=True)
class CompositeSpec:
    name: str
    proxies: Sequence[str]
    weights: np.ndarray
    invert: bool = False


def _require_columns(df: pd.DataFrame, cols: Iterable[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes pour {context}: {missing}")


def robust_zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)) + eps)
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        return np.zeros_like(x, dtype=float)
    return (x - med) / scale


def standard_zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if not np.isfinite(sd) or sd < eps:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd


def _proxy_stats_from_baseline(baseline: BaselineCalibration, proxy: str) -> RobustStats | None:
    return baseline.per_proxy.get(proxy)


def _normalize_columns(
    df: pd.DataFrame,
    proxies: Sequence[str],
    *,
    baseline: BaselineCalibration | None = None,
) -> np.ndarray:
    arr = df.loc[:, list(proxies)].astype(float).to_numpy()
    out = np.empty_like(arr, dtype=float)
    for j, p in enumerate(proxies):
        stats = _proxy_stats_from_baseline(baseline, p) if baseline is not None else None
        if stats is None:
            out[:, j] = robust_zscore(arr[:, j])
        else:
            out[:, j] = baseline_mad_z(arr[:, j], stats)
    return out


def _weighted_average(arr: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != arr.shape[1]:
        raise ValueError("Nombre de poids incompatible avec le nombre de proxys")
    s = float(np.sum(w))
    if not np.isfinite(s) or abs(s) < 1e-12:
        w = np.ones_like(w) / float(len(w))
    elif not np.isclose(s, 1.0):
        w = w / s
    return np.average(arr, weights=w, axis=1)



def compute_composite_raw(
    df: pd.DataFrame,
    spec: CompositeSpec,
    *,
    baseline: BaselineCalibration | None = None,
) -> np.ndarray:
    """Compute composite raw score.

    If some proxies are missing, use the available subset and renormalize weights.
    If none are available, return a zero series.
    """
    available = [p for p in spec.proxies if p in df.columns]
    if not available:
        return np.zeros(len(df), dtype=float)

    idx = [spec.proxies.index(p) for p in available]
    w = np.asarray(spec.weights, dtype=float)[idx]
    s = float(np.sum(w))
    if not np.isfinite(s) or abs(s) < 1e-12:
        w = np.ones_like(w) / float(len(w))
    else:
        w = w / s

    normed = _normalize_columns(df, available, baseline=baseline)
    raw = _weighted_average(normed, w)
    if spec.invert:
        raw = -raw
    return raw


def compute_composite(
    df: pd.DataFrame,
    spec: CompositeSpec,
    *,
    baseline: BaselineCalibration | None = None,
    norm: str = "robust",
) -> pd.Series:
    raw = compute_composite_raw(df, spec, baseline=baseline)
    if baseline is None:
        if norm == "robust":
            raw = robust_zscore(raw)
        elif norm == "standard":
            raw = standard_zscore(raw)
        else:
            raise ValueError("norm doit être 'robust' ou 'standard'")
    return pd.Series(raw, index=df.index, name=spec.name)


P_SPEC = CompositeSpec(
    name="P",
    proxies=["scale_proxy", "speed_proxy", "leverage_proxy", "autonomy_proxy", "replicability_proxy"],
    weights=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
)
O_SPEC = CompositeSpec(
    name="O",
    proxies=["stop_proxy", "threshold_proxy", "decision_proxy", "execution_proxy", "coherence_proxy"],
    weights=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
)
E_SPEC = CompositeSpec(
    name="E_LEVEL",
    proxies=["impact_proxy", "propagation_proxy", "hysteresis_proxy"],
    weights=np.array([0.4, 0.3, 0.3]),
)
G_SPEC = CompositeSpec(
    name="G",
    proxies=["exemption_rate", "sanction_delay", "control_turnover", "conflict_interest_proxy", "rule_execution_gap"],
    weights=np.array([0.25, 0.20, 0.15, 0.10, 0.30]),
    invert=True,
)


def _override_weights(spec: CompositeSpec, weights: Sequence[float] | None) -> CompositeSpec:
    if weights is None:
        return spec
    w = np.asarray(list(weights), dtype=float)
    if w.shape[0] != len(spec.proxies):
        raise ValueError(f"Override weights for {spec.name} must have length {len(spec.proxies)}")
    s = float(np.sum(w))
    if not np.isclose(s, 1.0):
        w = w / s
    return CompositeSpec(name=spec.name, proxies=spec.proxies, weights=w, invert=spec.invert)


def compute_p(df: pd.DataFrame, *, baseline: BaselineCalibration | None = None, weights: Sequence[float] | None = None) -> pd.Series:
    return compute_composite(df, _override_weights(P_SPEC, weights), baseline=baseline, norm="robust")


def compute_o(df: pd.DataFrame, *, baseline: BaselineCalibration | None = None, weights: Sequence[float] | None = None) -> pd.Series:
    return compute_composite(df, _override_weights(O_SPEC, weights), baseline=baseline, norm="robust")


def compute_e_metrics(df: pd.DataFrame, *, baseline: BaselineCalibration | None = None, weights: Sequence[float] | None = None) -> Mapping[str, pd.Series]:
    e_level = compute_composite(df, _override_weights(E_SPEC, weights), baseline=baseline, norm="robust")
    stock = pd.Series(np.cumsum(e_level.to_numpy(dtype=float)), index=df.index, name="E_STOCK")
    dE = pd.Series(np.gradient(stock.to_numpy(dtype=float)), index=df.index, name="dE_dt")
    dif = np.diff(stock.to_numpy(dtype=float), prepend=float(stock.to_numpy(dtype=float)[0]) if len(stock) else 0.0)
    pos = float(np.sum(np.clip(dif, 0.0, None)))
    neg = float(np.sum(np.clip(-dif, 0.0, None)))
    irr = float(pos / (pos + neg + 1e-12))
    e_irr = pd.Series(np.full(len(stock), irr, dtype=float), index=df.index, name="E_irreversibility")
    return {"E_level": e_level.rename("E_LEVEL"), "E_stock": stock, "dE_dt": dE, "E_irreversibility": e_irr}


def compute_e(df: pd.DataFrame, *, baseline: BaselineCalibration | None = None, weights: Sequence[float] | None = None) -> pd.Series:
    m = compute_e_metrics(df, baseline=baseline, weights=weights)
    # backward compatible: E as robust z-score of stock if baseline not provided
    arr = m["E_stock"].to_numpy(dtype=float)
    z = robust_zscore(arr) if baseline is None else arr
    return pd.Series(z, index=df.index, name="E")


def compute_r_metrics(df: pd.DataFrame, *, baseline: BaselineCalibration | None = None, weights: Sequence[float] | None = None) -> Mapping[str, pd.Series]:
    proxies = ["margin_proxy", "redundancy_proxy", "diversity_proxy", "recovery_time_proxy"]
    _require_columns(df, proxies, "R")
    arr = _normalize_columns(df, proxies, baseline=baseline)
    arr[:, 3] = -arr[:, 3]  # recovery time is a cost
    w = np.array([0.25, 0.25, 0.25, 0.25]) if weights is None else np.asarray(list(weights), dtype=float)
    if w.shape[0] != 4:
        raise ValueError("Override weights for R must have length 4")
    raw = _weighted_average(arr, w)
    r_level = pd.Series(raw, index=df.index, name="R_LEVEL")
    mttr = pd.Series(pd.to_numeric(df["recovery_time_proxy"], errors="coerce").to_numpy(dtype=float), index=df.index, name="R_mttr_proxy")
    rec_time = mttr.rename("R_recovery_time_est")
    return {"R_level": r_level, "R_mttr_proxy": mttr, "R_recovery_time_est": rec_time}


def compute_r(df: pd.DataFrame, *, baseline: BaselineCalibration | None = None, weights: Sequence[float] | None = None) -> pd.Series:
    m = compute_r_metrics(df, baseline=baseline, weights=weights)
    arr = m["R_level"].to_numpy(dtype=float)
    z = robust_zscore(arr) if baseline is None else arr
    return pd.Series(z, index=df.index, name="R")



def compute_g(df: pd.DataFrame, *, baseline: BaselineCalibration | None = None, weights: Sequence[float] | None = None) -> pd.Series:
    return compute_composite(df, _override_weights(G_SPEC, weights), baseline=baseline, norm="robust")


def compute_p_level(df: pd.DataFrame, *, weights: Sequence[float] | None = None) -> pd.Series:
    """Raw P_level in original units (weighted mean)."""
    available = [p for p in P_SPEC.proxies if p in df.columns]
    if not available:
        return pd.Series(np.ones(len(df), dtype=float), index=df.index, name="P_LEVEL")

    w_full = P_SPEC.weights if weights is None else np.asarray(list(weights), dtype=float)
    w = np.asarray([w_full[list(P_SPEC.proxies).index(p)] for p in available], dtype=float)
    s = float(np.sum(w))
    if not np.isfinite(s) or abs(s) < 1e-12:
        w = np.ones_like(w) / float(len(w))
    else:
        w = w / s
    raw = _weighted_average(df.loc[:, available].astype(float).to_numpy(), w)
    return pd.Series(raw, index=df.index, name="P_LEVEL")


def compute_o_level(df: pd.DataFrame, *, weights: Sequence[float] | None = None) -> pd.Series:
    """Raw O_level in original units (weighted mean)."""
    available = [p for p in O_SPEC.proxies if p in df.columns]
    if not available:
        return pd.Series(np.ones(len(df), dtype=float), index=df.index, name="O_LEVEL")

    w_full = O_SPEC.weights if weights is None else np.asarray(list(weights), dtype=float)
    w = np.asarray([w_full[list(O_SPEC.proxies).index(p)] for p in available], dtype=float)
    s = float(np.sum(w))
    if not np.isfinite(s) or abs(s) < 1e-12:
        w = np.ones_like(w) / float(len(w))
    else:
        w = w / s
    raw = _weighted_average(df.loc[:, available].astype(float).to_numpy(), w)
    return pd.Series(raw, index=df.index, name="O_LEVEL")


def compute_at(
    df: pd.DataFrame,
    eps: float = 1e-8,
    *,
    p_weights: Sequence[float] | None = None,
    o_weights: Sequence[float] | None = None,
) -> pd.Series:
    p_level = compute_p_level(df, weights=p_weights).to_numpy(dtype=float)
    o_level = compute_o_level(df, weights=o_weights).to_numpy(dtype=float)
    at = p_level / (o_level + float(eps))
    return pd.Series(at, index=df.index, name="AT")


def compute_delta_d(
    df: pd.DataFrame,
    *,
    window: int = 5,
    p_weights: Sequence[float] | None = None,
    o_weights: Sequence[float] | None = None,
) -> pd.Series:
    p = compute_p_level(df, weights=p_weights).astype(float)
    o = compute_o_level(df, weights=o_weights).astype(float)

    win = int(max(1, window))
    p_s = p.rolling(win, min_periods=1).mean()
    o_s = o.rolling(win, min_periods=1).mean()

    dp = p_s.diff().fillna(0.0).to_numpy(dtype=float)
    do = o_s.diff().fillna(0.0).to_numpy(dtype=float)
    dd = dp - do
    return pd.Series(dd, index=df.index, name="DELTA_D")
