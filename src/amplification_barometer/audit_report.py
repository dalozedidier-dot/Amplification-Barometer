from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from .audit_tools import anti_gaming_o_bias, audit_score_stability, run_stress_suite
from .calibration import Thresholds, derive_thresholds, risk_signature
from .composites import (
    WEIGHTS_VERSION,
    compute_at,
    compute_delta_d,
    compute_e_metrics,
    compute_g,
    compute_o,
    compute_o_level,
    compute_p,
    compute_p_level,
    compute_r_metrics,
)
from .l_operator import assess_maturity, evaluate_l_performance
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
    verdict: Dict[str, Any]
    targets: Dict[str, Any]


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


def _topk_indices(x: pd.Series, k: int) -> Sequence[int]:
    arr = x.to_numpy(dtype=float)
    if k <= 0:
        return []
    idx = np.argpartition(arr, -k)[-k:]
    return [int(i) for i in idx.tolist()]


def _resilience_summary(stress_suite: Dict[str, Any]) -> Dict[str, Any]:
    statuses = {}
    degradations = []
    for name, payload in stress_suite.items():
        st = str(payload.get("status"))
        statuses[name] = st
        degradations.append(float(payload.get("degradation", 0.0)))
    resilient = sum(1 for s in statuses.values() if s == "Résilient")
    total = max(1, len(statuses))
    return {
        "resilient_frac": float(resilient / total),
        "worst_degradation": float(np.max(degradations)) if degradations else 0.0,
        "statuses": statuses,
    }


def _governance_summary(maturity: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, Any]:
    gap = float(maturity.get("governance_rule_execution_gap", 1.0))
    ct = float(maturity.get("governance_control_turnover", 1.0))
    ok = bool((gap <= float(targets["rule_execution_gap_target_max"])) and (ct <= float(targets["control_turnover_target_max"])))
    return {"rule_execution_gap_mean": gap, "control_turnover_mean": ct, "meets_targets": ok}


def _compute_verdict(
    *,
    stability: Dict[str, Any],
    maturity: Dict[str, Any],
    stress_suite: Dict[str, Any],
    anti_gaming: Dict[str, Any],
    targets: Dict[str, Any],
) -> Dict[str, Any]:
    stability_score = 1.0 if bool(stability.get("stable_flag")) else 0.0
    lcap_score = float(np.clip(float(maturity.get("l_cap_bench_score", 0.0)), 0.0, 1.0))
    lact_score = float(np.clip(float(maturity.get("l_act_mean", 0.0)), 0.0, 1.0))

    gov = _governance_summary(maturity, targets)
    governance_score = 1.0 if gov["meets_targets"] else 0.0

    res = _resilience_summary(stress_suite)
    resilience_score = float(np.clip(res["resilient_frac"], 0.0, 1.0))

    anti_gaming_score = 0.0 if bool(anti_gaming.get("red_flag")) else 1.0

    dims = {
        "stability": {"score": stability_score, "stable_flag": bool(stability.get("stable_flag")), "spearman_mean": float(stability.get("spearman_mean_risk", 0.0)), "topk_jaccard_mean": float(stability.get("topk_jaccard_mean_risk", 0.0))},
        "L_cap": {"score": lcap_score, "l_cap_bench_score": float(maturity.get("l_cap_bench_score", 0.0))},
        "L_act": {"score": lact_score, "l_act_mean": float(maturity.get("l_act_mean", 0.0)), "activation_delay_steps": float(maturity.get("activation_delay_steps", float("nan"))), "activation_stability": float(maturity.get("activation_stability", float("nan")))},
        "resilience": {"score": resilience_score, **res},
        "governance": {"score": governance_score, **gov, "targets": {"rule_execution_gap_max": float(targets["rule_execution_gap_target_max"]), "control_turnover_max": float(targets["control_turnover_target_max"])}},
        "anti_gaming": {"score": anti_gaming_score, "red_flag": bool(anti_gaming.get("red_flag")), "delta_risk_mean": float(anti_gaming.get("delta_risk_mean", 0.0))},
        "stress_suite": {"scenarios": stress_suite},
    }

    global_score = float(np.mean([stability_score, lcap_score, lact_score, governance_score, resilience_score, anti_gaming_score]))

    return {
        "label": str(maturity.get("label")),
        "global_score": global_score,
        "dimensions": dims,
    }


def build_audit_report(
    df: pd.DataFrame,
    *,
    dataset_name: str = "dataset",
    window: int = 5,
    thresholds: Thresholds | None = None,
) -> AuditReport:
    if thresholds is None:
        thresholds = derive_thresholds(df, window=window)

    p = compute_p(df)
    o = compute_o(df)
    g = compute_g(df)
    e_m = compute_e_metrics(df)
    r_m = compute_r_metrics(df)
    at = compute_at(df)
    dd = compute_delta_d(df, window=window)
    risk = risk_signature(df, thresholds=thresholds, window=window, at_series=at, dd_series=dd)

    p_level = compute_p_level(df)
    o_level = compute_o_level(df)

    summary = {
        "P": _series_stats(p),
        "O": _series_stats(o),
        "G": _series_stats(g),
        "E_level": _series_stats(e_m["E_level"]),
        "E_stock": _series_stats(e_m["E_stock"]),
        "dE_dt": _series_stats(e_m["dE_dt"]),
        "E_irreversibility": float(np.mean(e_m["E_irreversibility"].to_numpy(dtype=float))),
        "R_level": _series_stats(r_m["R_level"]),
        "R_mttr_proxy": _series_stats(r_m["R_mttr_proxy"]),
        "AT": _series_stats(at),
        "DELTA_D": _series_stats(dd),
        "RISK": _series_stats(risk),
        "P_level": _series_stats(p_level),
        "O_level": _series_stats(o_level),
        "risk_threshold": float(thresholds.risk_thr),
        "topk_risk_indices": _topk_indices(risk, k=max(1, int(len(df) * 0.10))),
    }

    stability = audit_score_stability(df, windows=(3, 5, 8), topk_frac=0.10)
    stress_suite = run_stress_suite(df, window=window, thresholds=thresholds)
    maturity = asdict(assess_maturity(df, window=window, thresholds=thresholds))

    l_perf = evaluate_l_performance(df, window=window, thresholds=thresholds)

    # proactive: O weakness triggers desired activation earlier
    o_thr = float(np.percentile(o_level.to_numpy(dtype=float), 10))
    l_perf_pro = evaluate_l_performance(df, window=window, thresholds=thresholds, o_threshold=o_thr)

    manipulability = run_manipulability_suite(df)
    anti_gaming = anti_gaming_o_bias(df, window=window)

    targets = {
        "recovery_rate_post_stress_target_min": 0.90,
        "activation_delay_steps_target_max": 5,
        "E_reduction_rel_target_min": 0.20,
        "prevented_exceedance_rel_target_min": 0.10,
        "rule_execution_gap_target_max": 0.05,
        "control_turnover_target_max": 0.05,
        # backward compatible aliases
        "recovery_rate_post_stress_min": 0.90,
        "activation_delay_steps_max": 5,
        "E_reduction_rel_min": 0.20,
        "prevented_exceedance_rel_min": 0.10,
    }

    verdict = _compute_verdict(
        stability=stability,
        maturity=maturity,
        stress_suite=stress_suite,
        anti_gaming=anti_gaming,
        targets=targets,
    )

    created_utc = datetime.now(timezone.utc).isoformat()

    return AuditReport(
        version="0.5.1",
        weights_version=WEIGHTS_VERSION,
        dataset_name=str(dataset_name),
        created_utc=created_utc,
        summary=summary,
        stability=stability,
        stress_suite=stress_suite,
        maturity=maturity,
        l_performance=l_perf,
        l_performance_proactive=l_perf_pro,
        manipulability=manipulability,
        anti_gaming=anti_gaming,
        verdict=verdict,
        targets=targets,
    )


def _sanitize_json(obj: Any) -> Any:
    """Convert NaN/Inf to None recursively so JSON is strict and portable."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, bool)):
        return obj
    # numpy scalars
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_sanitize_json(v) for v in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    # last resort
    return str(obj)


def write_audit_report(report: AuditReport, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = _sanitize_json(asdict(report))
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
