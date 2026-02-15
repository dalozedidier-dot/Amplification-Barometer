
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from .composites import compute_at, compute_delta_d


def _robust_loc_scale(x: np.ndarray, eps: float = 1e-12) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)) + eps)
    return med, mad


def _robust_z_from_baseline(x: np.ndarray, *, med: float, mad: float, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    scale = 1.4826 * float(mad)
    if not np.isfinite(scale) or scale <= eps:
        return np.zeros_like(x, dtype=float)
    return (x - float(med)) / scale


@dataclass(frozen=True)
class Thresholds:
    risk_thr: float
    at_p95_stable: float
    dd_p95_stable: float
    baseline_at_median: float
    baseline_at_mad: float
    baseline_dd_median: float
    baseline_dd_mad: float


def risk_signature(
    df: pd.DataFrame,
    *,
    thresholds: Thresholds,
    window: int = 5,
    at_series: pd.Series | None = None,
    dd_series: pd.Series | None = None,
) -> pd.Series:
    """Risk(t) = z_baseline(AT) + z_baseline(DELTA_D)."""
    if at_series is None:
        at_series = compute_at(df)
    if dd_series is None:
        dd_series = compute_delta_d(df, window=window)

    at = at_series.to_numpy(dtype=float)
    dd = dd_series.to_numpy(dtype=float)

    z_at = _robust_z_from_baseline(at, med=thresholds.baseline_at_median, mad=thresholds.baseline_at_mad)
    z_dd = _robust_z_from_baseline(dd, med=thresholds.baseline_dd_median, mad=thresholds.baseline_dd_mad)

    risk = z_at + z_dd
    return pd.Series(risk, index=df.index, name="RISK")


def derive_thresholds(
    stable_df: pd.DataFrame,
    *,
    window: int = 5,
    at_series: pd.Series | None = None,
    dd_series: pd.Series | None = None,
) -> Thresholds:
    if at_series is None:
        at_series = compute_at(stable_df)
    if dd_series is None:
        dd_series = compute_delta_d(stable_df, window=window)

    at = at_series.to_numpy(dtype=float)
    dd = dd_series.to_numpy(dtype=float)

    at_med, at_mad = _robust_loc_scale(at)
    dd_med, dd_mad = _robust_loc_scale(dd)

    tmp = Thresholds(
        risk_thr=0.0,
        at_p95_stable=float(np.percentile(at, 95)),
        dd_p95_stable=float(np.percentile(dd, 95)),
        baseline_at_median=float(at_med),
        baseline_at_mad=float(at_mad),
        baseline_dd_median=float(dd_med),
        baseline_dd_mad=float(dd_mad),
    )
    risk = risk_signature(stable_df, thresholds=tmp, window=window, at_series=at_series, dd_series=dd_series).to_numpy(dtype=float)

    return Thresholds(
        risk_thr=float(np.percentile(risk, 95)),
        at_p95_stable=float(np.percentile(at, 95)),
        dd_p95_stable=float(np.percentile(dd, 95)),
        baseline_at_median=float(at_med),
        baseline_at_mad=float(at_mad),
        baseline_dd_median=float(dd_med),
        baseline_dd_mad=float(dd_mad),
    )


def evaluate_dataset(df: pd.DataFrame, *, thresholds: Thresholds, window: int = 5) -> Dict[str, Any]:
    at = compute_at(df)
    dd = compute_delta_d(df, window=window)
    risk = risk_signature(df, thresholds=thresholds, window=window, at_series=at, dd_series=dd)

    mask = risk.to_numpy(dtype=float) > float(thresholds.risk_thr)

    longest = 0
    cur = 0
    for v in mask:
        if bool(v):
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0

    return {
        "n": int(len(df)),
        "mean_at": float(np.mean(at)),
        "mean_dd": float(np.mean(dd)),
        "mean_risk": float(np.mean(risk)),
        "frac_risk_above_thr": float(np.mean(mask)),
        "longest_run_above_thr": int(longest),
        "risk_threshold": float(thresholds.risk_thr),
    }


def discriminate_regimes(
    datasets: Mapping[str, pd.DataFrame],
    *,
    stable_name: str = "stable",
    window: int = 5,
) -> Dict[str, Any]:
    if stable_name not in datasets:
        raise ValueError(f"stable_name missing from datasets: {stable_name}")
    thresholds = derive_thresholds(datasets[stable_name], window=window)

    evals: Dict[str, Any] = {}
    for name, df in datasets.items():
        evals[name] = evaluate_dataset(df, thresholds=thresholds, window=window)

    ordering = sorted(evals.keys(), key=lambda k: (evals[k]["frac_risk_above_thr"], evals[k]["longest_run_above_thr"]))
    return {"thresholds": thresholds.__dict__, "datasets": evals, "ordering_by_severity": ordering}
