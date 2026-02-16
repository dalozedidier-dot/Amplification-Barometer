from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from .audit_tools import anti_gaming_o_bias, audit_score_stability, run_stress_suite
from .calibration import Thresholds, derive_thresholds, risk_signature
from .composites import (
    WEIGHTS_VERSION,
    compute_at,
    compute_delta_d,
    compute_e,
    compute_g,
    compute_o,
    compute_o_level,
    compute_p,
    compute_r,
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
    l_performance_proactive: Dict[str, Any]
    manipulability: Dict[str, Any]
    anti_gaming: Dict[str, Any]
    targets: Dict[str, Any]
    verdict: Dict[str, Any]


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


def _thresholds_from_mapping(m: Mapping[str, Any] | None) -> Thresholds | None:
    if m is None:
        return None
    try:
        return Thresholds(
            risk_thr=float(m["risk_thr"]),
            at_p95_stable=float(m.get("at_p95_stable", 0.0)),
            dd_p95_stable=float(m.get("dd_p95_stable", 0.0)),
            baseline_at_median=float(m["baseline_at_median"]),
            baseline_at_mad=float(m["baseline_at_mad"]),
            baseline_dd_median=float(m["baseline_dd_median"]),
            baseline_dd_mad=float(m["baseline_dd_mad"]),
        )
    except Exception as e:
        raise ValueError(f"Seuils baseline invalides: {e}") from e


def _risk_series(df: pd.DataFrame, *, window: int = 5, thresholds: Thresholds | None = None) -> pd.Series:
    if thresholds is not None:
        return risk_signature(df, thresholds=thresholds, window=window)
    at = compute_at(df).to_numpy(dtype=float)
    dd = compute_delta_d(df, window=window).to_numpy(dtype=float)
    # fallback démonstrateur
    from .composites import robust_zscore  # local import to avoid extra coupling
    risk = robust_zscore(at) + robust_zscore(dd)
    return pd.Series(risk, index=df.index, name="RISK")


def _topk_indices(x: pd.Series, k: int) -> Sequence[int]:
    if k <= 0:
        return []
    return [int(i) for i in np.argsort(x.to_numpy(dtype=float))[-k:]]


def _sanitize_json(obj: Any) -> Any:
    """Remplace NaN et Inf par None, de façon récursive."""
    if obj is None:
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.ndarray,)):
        return [_sanitize_json(v) for v in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return obj.total_seconds()
    if isinstance(obj, Path):
        return str(obj)
    return obj


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
    thresholds: Mapping[str, Any] | Thresholds | None = None,
) -> AuditReport:
    """Builds a reproducible audit report for a single dataset.

    Important:
    - If thresholds is provided, risk and L performance use the stable baseline (no per-dataset renormalisation).
    """
    created = datetime.now(timezone.utc).isoformat()

    thr = thresholds if isinstance(thresholds, Thresholds) else _thresholds_from_mapping(thresholds)

    p = compute_p(df)
    o = compute_o(df)
    e_level = compute_e(df)
    r_level = compute_r(df)
    g_level = compute_g(df)
    at = compute_at(df)
    dd = compute_delta_d(df, window=delta_d_window)
    risk = _risk_series(df, window=delta_d_window, thresholds=thr)

    # E metrics: stock, derivative, irreversibility
    e_stock = e_level.cumsum()
    dE_dt = e_stock.diff().fillna(0.0)
    inc = dE_dt.to_numpy(dtype=float)
    denom = float(np.sum(np.abs(inc))) + 1e-12
    e_irrev = float(np.sum(np.clip(inc, 0.0, None)) / denom)

    # R metrics: MTTR proxy (bounded), recovery estimate (p95)
    from .composites import robust_zscore  # fallback only for MTTR proxy mapping
    r01 = np.clip(0.5 + 0.25 * robust_zscore(r_level.to_numpy(dtype=float)), 0.0, 1.0)
    mttr_proxy = pd.Series(1.0 / (0.05 + r01), index=r_level.index, name="R_mttr_proxy")
    r_recovery_est = float(np.percentile(mttr_proxy.to_numpy(dtype=float), 95))

    k = max(1, int(len(df) * float(topk_frac)))
    topk = _topk_indices(risk, k)

    summary: Dict[str, Any] = {
        "P": _series_stats(p),
        "O": _series_stats(o),
        "E_level": _series_stats(e_level),
        "E_stock": _series_stats(e_stock),
        "dE_dt": _series_stats(dE_dt),
        "E_irreversibility": float(e_irrev),
        "R_level": _series_stats(r_level),
        "R_mttr_proxy": _series_stats(mttr_proxy),
        "R_recovery_time_est": float(r_recovery_est),
        "G_level": _series_stats(g_level),
        "AT": _series_stats(at),
        "DELTA_D": _series_stats(dd),
        "RISK": _series_stats(risk),
        "risk_threshold": float(thr.risk_thr) if thr is not None else None,
        "baseline_used": bool(thr is not None),
        "topk_frac": float(topk_frac),
        "topk_k": int(k),
        "topk_indices": topk,
    }

    stability = audit_score_stability(df, windows=stability_windows, topk_frac=topk_frac)

    suite = run_stress_suite(df, intensity=stress_intensity)
    stress_suite: Dict[str, Any] = {
        name: {"status": r.status, "degradation": r.degradation, "details": r.details} for name, r in suite.items()
    }

    # Maturity label with overload stress, when available
    df_over = None
    try:
        from .audit_tools import _apply_scenario  # type: ignore

        df_over = _apply_scenario(df, "Overload", stress_intensity)
    except Exception:
        df_over = None

    maturity_assessment: MaturityAssessment = assess_maturity(df, df_stressed=df_over) if df_over is not None else assess_maturity(df)
    maturity = asdict(maturity_assessment)

    # L performance uses baseline thresholds when provided
    risk_thr = float(thr.risk_thr) if thr is not None else None
    l_perf = evaluate_l_performance(
        df,
        window=delta_d_window,
        thresholds=thr,
        risk_threshold=risk_thr,
        topk_frac=topk_frac,
        intensity=stress_intensity,
    )

    # Proactive variant: risk OR low orientation
    o_level = compute_o_level(df)
    o_thr = float(np.quantile(o_level.to_numpy(dtype=float), 0.15))

    proactive_topk = float(max(float(topk_frac), 0.20))
    variants: list[dict[str, Any]] = []

    v1 = evaluate_l_performance(
        df,
        window=delta_d_window,
        thresholds=thr,
        risk_threshold=risk_thr,
        topk_frac=proactive_topk,
        o_threshold=o_thr,
        persist=2,
        max_delay=8,
        intensity=stress_intensity,
    )
    v1["variant"] = "proactive_v1_risk_or_low_o"
    variants.append(v1)

    if (float(v1.get("prevented_topk_excess_rel", 0.0)) < 0.10) and (float(v1.get("prevented_exceedance_rel", 0.0)) < 0.10):
        o_thr2 = float(np.quantile(o_level.to_numpy(dtype=float), 0.20))
        v2 = evaluate_l_performance(
            df,
            window=delta_d_window,
            thresholds=thr,
            risk_threshold=risk_thr,
            topk_frac=float(max(float(topk_frac), 0.30)),
            o_threshold=o_thr2,
            persist=1,
            max_delay=6,
            intensity=stress_intensity,
        )
        v2["variant"] = "proactive_v2_autotune"
        variants.append(v2)

    l_perf_pro = max(
        variants,
        key=lambda d: (
            float(d.get("prevented_topk_excess_rel", 0.0)),
            float(d.get("prevented_exceedance_rel", 0.0)),
        ),
    )
    l_perf_pro["variants_tried"] = [
        {
            "variant": v.get("variant"),
            "prevented_exceedance_rel": float(v.get("prevented_exceedance_rel", 0.0)),
            "prevented_topk_excess_rel": float(v.get("prevented_topk_excess_rel", 0.0)),
            "params": v.get("params", {}),
        }
        for v in variants
    ]

    manipulability = run_manipulability_suite(df, magnitude=manipulability_magnitude)
    anti_gaming = {
        "o_bias": anti_gaming_o_bias(df, magnitude=o_bias_magnitude, window=delta_d_window),
    }

    gap_mean = float(np.mean(df["rule_execution_gap"].astype(float).to_numpy())) if "rule_execution_gap" in df.columns else None
    turnover_mean = float(np.mean(df["control_turnover"].astype(float).to_numpy())) if "control_turnover" in df.columns else None

    def meets_min(v: Any, thr_min: float) -> bool:
        try:
            return (v is not None) and (float(v) >= float(thr_min))
        except Exception:
            return False

    def meets_max(v: Any, thr_max: float) -> bool:
        try:
            return (v is not None) and (float(v) <= float(thr_max))
        except Exception:
            return False

    # Targets (demonstrator): governance + L performance + recovery and E reduction
    targets: Dict[str, Any] = {
        "rule_execution_gap_target_max": 0.05,
        "rule_execution_gap_mean": gap_mean,
        "rule_execution_gap_meets_target": meets_max(gap_mean, 0.05),
        "control_turnover_target_max": 0.05,
        "control_turnover_mean": turnover_mean,
        "control_turnover_meets_target": meets_max(turnover_mean, 0.05),
        "prevented_exceedance_rel_target_min": 0.10,
        "prevented_exceedance_rel": float(l_perf_pro.get("prevented_exceedance_rel", 0.0)),
        "prevented_exceedance_rel_meets_target": meets_min(l_perf_pro.get("prevented_exceedance_rel", 0.0), 0.10),
        "prevented_topk_excess_rel_target_min": 0.10,
        "prevented_topk_excess_rel": float(l_perf_pro.get("prevented_topk_excess_rel", 0.0)),
        "prevented_topk_excess_rel_meets_target": meets_min(l_perf_pro.get("prevented_topk_excess_rel", 0.0), 0.10),
        "activation_delay_steps_target_max": 5,
        "activation_delay_steps": l_perf_pro.get("activation_delay_steps"),
        "activation_delay_steps_meets_target": meets_max(l_perf_pro.get("activation_delay_steps"), 5),
        "E_reduction_rel_target_min": 0.20,
        "E_reduction_rel": l_perf_pro.get("E_reduction_rel"),
        "E_reduction_rel_meets_target": meets_min(l_perf_pro.get("E_reduction_rel"), 0.20),
        "recovery_rate_post_stress_target_min": 0.90,
        "recovery_rate_post_stress": l_perf_pro.get("recovery_rate_post_stress"),
        "recovery_rate_post_stress_meets_target": meets_min(l_perf_pro.get("recovery_rate_post_stress"), 0.90),
        "proactive_topk_frac": proactive_topk,
        "proactive_o_threshold": float(o_thr),
    }

    # Multidimensional verdict (audit-friendly)
    stability_score = float(np.clip(0.5 * float(stability.get("spearman_worst_risk", 0.0)) + 0.5 * float(stability.get("topk_jaccard_worst_risk", 0.0)), 0.0, 1.0))
    perf_score = float(np.clip(float(l_perf_pro.get("prevented_topk_excess_rel", 0.0)) / 0.20, 0.0, 1.0))
    gov_score = float(np.clip(1.0 - float(gap_mean or 0.0) / 0.10, 0.0, 1.0))
    stress_deg = float(np.mean([float(v.get("degradation", 0.0)) for v in stress_suite.values()])) if stress_suite else 0.0
    stress_score = float(np.clip(1.0 - stress_deg / 3.0, 0.0, 1.0))

    global_score = 100.0 * float(0.30 * stability_score + 0.35 * perf_score + 0.20 * gov_score + 0.15 * stress_score)
    verdict_label = maturity.get("label", "Immature")

    verdict = {
        "label": verdict_label,
        "global_score": float(global_score),
        "dimensions": {
            "stability": float(stability_score),
            "performance": float(perf_score),
            "governance": float(gov_score),
            "stress": float(stress_score),
        },
        "notes": {
            "baseline_used": bool(thr is not None),
            "risk_thr": float(thr.risk_thr) if thr is not None else None,
        },
    }

    return AuditReport(
        version="0.4.4",
        weights_version=WEIGHTS_VERSION,
        dataset_name=dataset_name,
        created_utc=created,
        summary=summary,
        stability=stability,
        stress_suite=stress_suite,
        maturity=maturity,
        l_performance=l_perf,
        l_performance_proactive=l_perf_pro,
        manipulability=manipulability,
        anti_gaming=anti_gaming,
        targets=targets,
        verdict=verdict,
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
        "l_performance_proactive": report.l_performance_proactive,
        "manipulability": report.manipulability,
        "anti_gaming": report.anti_gaming,
        "targets": report.targets,
        "verdict": report.verdict,
    }

    payload = _sanitize_json(payload)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
