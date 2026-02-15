from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .composites import compute_at, compute_delta_d, robust_zscore


def risk_signature(df: pd.DataFrame, *, window: int = 5) -> pd.Series:
    """Risk signature used for calibration and regime discrimination.

    Demonstrator signature:
    risk = robust_zscore(@(t)) + robust_zscore(Δd(t))
    """
    at = compute_at(df).to_numpy(dtype=float)
    dd = compute_delta_d(df, window=window).to_numpy(dtype=float)
    risk = robust_zscore(at) + robust_zscore(dd)
    return pd.Series(risk, index=df.index, name="RISK")


@dataclass(frozen=True)
class Thresholds:
    risk_p95_stable: float
    at_p95_stable: float
    dd_p95_stable: float


def derive_thresholds(stable_df: pd.DataFrame, *, window: int = 5) -> Thresholds:
    """Derives minimal calibration thresholds from the stable regime.

    The point is not to optimize. The point is to produce a transparent,
    reproducible baseline threshold that can be audited.
    """
    at = compute_at(stable_df).to_numpy(dtype=float)
    dd = compute_delta_d(stable_df, window=window).to_numpy(dtype=float)
    risk = risk_signature(stable_df, window=window).to_numpy(dtype=float)
    return Thresholds(
        risk_p95_stable=float(np.percentile(risk, 95)),
        at_p95_stable=float(np.percentile(at, 95)),
        dd_p95_stable=float(np.percentile(dd, 95)),
    )


def evaluate_dataset(df: pd.DataFrame, *, thresholds: Thresholds, window: int = 5) -> Dict[str, Any]:
    """Evaluates a dataset against stable-derived thresholds."""
    at = compute_at(df).to_numpy(dtype=float)
    dd = compute_delta_d(df, window=window).to_numpy(dtype=float)
    risk = risk_signature(df, window=window).to_numpy(dtype=float)

    risk_thr = thresholds.risk_p95_stable
    mask = risk > risk_thr

    # persistence: longest consecutive run above threshold
    longest = 0
    cur = 0
    for v in mask:
        if v:
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
        "risk_threshold": float(risk_thr),
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

    # expected ordering: stable lowest risk, bifurcation highest risk
    ordering = sorted(evals.keys(), key=lambda k: (evals[k]["frac_risk_above_thr"], evals[k]["mean_risk"]))
    return {
        "thresholds": {
            "risk_p95_stable": float(thresholds.risk_p95_stable),
            "at_p95_stable": float(thresholds.at_p95_stable),
            "dd_p95_stable": float(thresholds.dd_p95_stable),
        },
        "datasets": evals,
        "ordering_by_severity": ordering,
        "ordering_by_mean_risk": sorted(evals.keys(), key=lambda k: evals[k]["mean_risk"]),
    }
