from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from .composites import compute_at, compute_delta_d, robust_zscore


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    med = float(np.nanmedian(x))
    return float(np.nanmedian(np.abs(x - med)) + 1e-12)


def _robust_loc_scale(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)) + 1e-12)
    return med, mad


def _robust_z_from_baseline(x: np.ndarray, *, med: float, mad: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    scale = 1.4826 * float(mad)
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - float(med)) / scale


@dataclass(frozen=True)
class Thresholds:
    """Minimal calibration artefacts derived from a stable baseline.

    Important:
    - The baseline location and scale are derived ONLY from the stable regime.
    - Other regimes are evaluated against the same baseline to avoid
      per-dataset renormalisation hiding genuine regime shifts.
    """
    risk_thr: float
    at_p95_stable: float
    dd_p95_stable: float
    baseline_at_median: float
    baseline_at_mad: float
    baseline_dd_median: float
    baseline_dd_mad: float


def risk_signature(df: pd.DataFrame, *, thresholds: Thresholds, window: int = 5) -> pd.Series:
    """Risk signature used for audit calibration and regime discrimination.

    Signature (audit-friendly):
    - compute @(t) and Δd(t)
    - compute robust z-scores using STABLE baseline location/scale
    - risk = z_at + z_dd

    This prevents the "renormalize per dataset" issue that can otherwise
    make stable and bifurcation incomparable.
    """
    at = compute_at(df).to_numpy(dtype=float)
    dd = compute_delta_d(df, window=window).to_numpy(dtype=float)

    z_at = _robust_z_from_baseline(at, med=thresholds.baseline_at_median, mad=thresholds.baseline_at_mad)
    z_dd = _robust_z_from_baseline(dd, med=thresholds.baseline_dd_median, mad=thresholds.baseline_dd_mad)

    risk = z_at + z_dd
    return pd.Series(risk, index=df.index, name="RISK")


def derive_thresholds(stable_df: pd.DataFrame, *, window: int = 5) -> Thresholds:
    """Derives minimal calibration thresholds from the stable regime.

    The point is not to optimize. The point is to produce a transparent,
    reproducible baseline threshold that can be audited and reused.
    """
    at = compute_at(stable_df).to_numpy(dtype=float)
    dd = compute_delta_d(stable_df, window=window).to_numpy(dtype=float)

    at_med, at_mad = _robust_loc_scale(at)
    dd_med, dd_mad = _robust_loc_scale(dd)

    # temporary thresholds object to compute risk signature on stable itself
    tmp = Thresholds(
        risk_thr=0.0,
        at_p95_stable=float(np.percentile(at, 95)),
        dd_p95_stable=float(np.percentile(dd, 95)),
        baseline_at_median=float(at_med),
        baseline_at_mad=float(at_mad),
        baseline_dd_median=float(dd_med),
        baseline_dd_mad=float(dd_mad),
    )
    risk = risk_signature(stable_df, thresholds=tmp, window=window).to_numpy(dtype=float)

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
    """Evaluates a dataset against stable-derived thresholds."""
    at = compute_at(df).to_numpy(dtype=float)
    dd = compute_delta_d(df, window=window).to_numpy(dtype=float)
    risk = risk_signature(df, thresholds=thresholds, window=window).to_numpy(dtype=float)

    mask = risk > float(thresholds.risk_thr)

    # persistence: longest consecutive run above threshold
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
    """Computes minimal calibration report for regime discrimination."""
    if stable_name not in datasets:
        raise ValueError(f"stable_name missing from datasets: {stable_name}")
    thresholds = derive_thresholds(datasets[stable_name], window=window)

    evals: Dict[str, Any] = {}
    for name, df in datasets.items():
        evals[name] = evaluate_dataset(df, thresholds=thresholds, window=window)

    ordering = sorted(evals.keys(), key=lambda k: (evals[k]["frac_risk_above_thr"], evals[k]["mean_risk"]))
    return {
        "thresholds": {
            "risk_thr": float(thresholds.risk_thr),
            "at_p95_stable": float(thresholds.at_p95_stable),
            "dd_p95_stable": float(thresholds.dd_p95_stable),
            "baseline_at_median": float(thresholds.baseline_at_median),
            "baseline_at_mad": float(thresholds.baseline_at_mad),
            "baseline_dd_median": float(thresholds.baseline_dd_median),
            "baseline_dd_mad": float(thresholds.baseline_dd_mad),
        },
        "datasets": evals,
        "ordering_by_severity": ordering,
        "ordering_by_mean_risk": sorted(evals.keys(), key=lambda k: evals[k]["mean_risk"]),
    }
