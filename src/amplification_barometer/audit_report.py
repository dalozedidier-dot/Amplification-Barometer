from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from .audit_tools import anti_gaming_o_bias, audit_score_stability, run_stress_suite
from .composites import (
    WEIGHTS_VERSION,
    compute_at,
    compute_delta_d,
    compute_e,
    compute_g,
    compute_o,
    compute_p,
    compute_r,
    robust_zscore,
    standard_zscore,
)
from .l_operator import MaturityAssessment, assess_maturity, evaluate_l_performance
from .manipulability import run_manipulability_suite


@dataclass(frozen=True)
class AuditReport:
    version: str
    weights_version: str
    dataset_name: str
    created_utc: str
    summary: Dict[str, Any]
    stability: Dict[str, Any]
    stress_suite: Dict[str, Any]
    maturity: Dict[str, Any]
    l_performance: Dict[str, Any]
    manipulability: Dict[str, Any]
    anti_gaming: Dict[str, Any]


def _series_stats(x: pd.Series) -> Dict[str, float]:
    arr = x.to_numpy(dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p05": float(np.percentile(arr, 5)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _risk_series(df: pd.DataFrame, *, window: int = 5) -> pd.Series:
    at = compute_at(df).to_numpy(dtype=float)
    dd = compute_delta_d(df, window=window).to_numpy(dtype=float)
    risk = robust_zscore(at) + robust_zscore(dd)
    return pd.Series(risk, index=df.index, name="RISK")


def _topk_indices(x: pd.Series, k: int) -> Sequence[int]:
    if k <= 0:
        return []
    return [int(i) for i in np.argsort(x.to_numpy(dtype=float))[-k:]]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return obj.total_seconds()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serialisable")


def build_audit_report(
    df: pd.DataFrame,
    *,
    dataset_name: str = "dataset",
    delta_d_window: int = 5,
    stability_windows: Sequence[int] = (3, 5, 8),
    topk_frac: float = 0.10,
    stress_intensity: float = 1.0,
    manipulability_magnitude: float = 0.2,
    o_bias_magnitude: float = 0.15,
) -> AuditReport:
    """Builds a reproducible audit report for a single dataset.

    Scope:
    - measurability: statistics for P,O,E,R,G,@,Δd
    - stability: rank-based stability and Top-K consistency under small perturbations
    - stress suite: standard scenarios including adversarial cases
    - anti-gaming: proxy range checks and falsification detection suite
    - limit operator: L_cap/L_act maturity label and tested performance on the series
    """
    created = datetime.now(timezone.utc).isoformat()

    p = compute_p(df)
    o = compute_o(df)
    e = compute_e(df)
    r = compute_r(df)
    g = compute_g(df)
    at = compute_at(df)
    dd = compute_delta_d(df, window=delta_d_window)
    risk = _risk_series(df, window=delta_d_window)

    k = max(1, int(len(df) * float(topk_frac)))
    topk = _topk_indices(risk, k)

    summary = {
        "P": _series_stats(p),
        "O": _series_stats(o),
        "E": _series_stats(e),
        "R": _series_stats(r),
        "G": _series_stats(g),
        "AT": _series_stats(at),
        "DELTA_D": _series_stats(dd),
        "RISK": _series_stats(risk),
        "topk_frac": float(topk_frac),
        "topk_k": int(k),
        "topk_indices": topk,
    }

    stability = audit_score_stability(df, windows=stability_windows, topk_frac=topk_frac)

    # Compare robust vs standard normalization on the same raw risk series
    risk_std = pd.Series(standard_zscore(risk.to_numpy(dtype=float)), index=risk.index, name="RISK_STD")
    stability["risk_norm_spearman_robust_vs_standard"] = float(
        np.corrcoef(pd.Series(risk).rank().to_numpy(), pd.Series(risk_std).rank().to_numpy())[0, 1]
    )

    suite = run_stress_suite(df, intensity=stress_intensity)
    stress_suite: Dict[str, Any] = {
        name: {"status": r.status, "degradation": r.degradation, "details": r.details} for name, r in suite.items()
    }

    # Maturity label with overload stress
    df_over = None
    try:
        from .audit_tools import _apply_scenario  # type: ignore

        df_over = _apply_scenario(df, "Overload", stress_intensity)
    except Exception:
        df_over = None

    maturity_assessment: MaturityAssessment
    if df_over is not None:
        maturity_assessment = assess_maturity(df, df_stressed=df_over)
    else:
        maturity_assessment = assess_maturity(df)
    maturity = asdict(maturity_assessment)

    l_perf = evaluate_l_performance(df, window=delta_d_window, topk_frac=topk_frac, intensity=stress_intensity)

    manipulability = run_manipulability_suite(df, magnitude=manipulability_magnitude)
    anti_gaming = {
        "o_bias": anti_gaming_o_bias(df, magnitude=o_bias_magnitude, window=delta_d_window),
    }

    return AuditReport(
        version="0.4.0",
        weights_version=WEIGHTS_VERSION,
        dataset_name=dataset_name,
        created_utc=created,
        summary=summary,
        stability=stability,
        stress_suite=stress_suite,
        maturity=maturity,
        l_performance=l_perf,
        manipulability=manipulability,
        anti_gaming=anti_gaming,
    )


def write_audit_report(report: AuditReport, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": report.version,
        "weights_version": report.weights_version,
        "dataset_name": report.dataset_name,
        "created_utc": report.created_utc,
        "summary": report.summary,
        "stability": report.stability,
        "stress_suite": report.stress_suite,
        "maturity": report.maturity,
        "l_performance": report.l_performance,
        "manipulability": report.manipulability,
        "anti_gaming": report.anti_gaming,
    }
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
