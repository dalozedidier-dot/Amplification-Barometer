
from __future__ import annotations

import json
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
    maturity = asdict(assess_maturity(df, window=window))

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

    created_utc = datetime.now(timezone.utc).isoformat()

    return AuditReport(
        version="0.5.0",
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
        targets=targets,
    )


def write_audit_report(report: AuditReport, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
