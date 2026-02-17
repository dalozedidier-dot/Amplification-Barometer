from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

REPORT_VERSION = "0.4.5"

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
    l_performance_proactive: Dict[str, Any]
    manipulability: Dict[str, Any]
    anti_gaming: Dict[str, Any]
    targets: Dict[str, Any]
    thresholds: Optional[Dict[str, Any]] = None


def _series_stats(x: pd.Series) -> Dict[str, float]:
    arr = x.to_numpy(dtype=float)
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "p05": float(np.nanpercentile(arr, 5)),
        "p50": float(np.nanpercentile(arr, 50)),
        "p95": float(np.nanpercentile(arr, 95)),
    }


def _risk_series(df: pd.DataFrame, *, window: int = 5, thresholds: Optional[Thresholds] = None) -> pd.Series:
    if thresholds is not None:
        s = risk_signature(df, thresholds=thresholds, window=window)
        return pd.Series(s.to_numpy(dtype=float), index=df.index, name="RISK")
    at = compute_at(df).to_numpy(dtype=float)
    dd = compute_delta_d(df, window=window).to_numpy(dtype=float)
    risk = robust_zscore(at) + robust_zscore(dd)
    return pd.Series(risk, index=df.index, name="RISK")


def _topk_indices(x: pd.Series, k: int) -> Sequence[int]:
    if k <= 0:
        return []
    arr = x.to_numpy(dtype=float)
    n = arr.size
    k = max(1, min(int(k), int(n)))
    thr = float(np.partition(arr, n - k)[n - k])
    return [int(i) for i, v in enumerate(arr) if float(v) >= thr]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)


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
    thresholds: Optional[Thresholds] = None,
) -> AuditReport:
    """Builds a reproducible audit report for a single dataset.

    Important:
    - if `thresholds` is provided (derived from STABLE baseline), risk normalization is comparable across datasets.
    """
    created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    p = compute_p(df)
    o = compute_o(df)
    e = compute_e(df)
    r = compute_r(df)
    g = compute_g(df)
    at = compute_at(df)
    dd = compute_delta_d(df, window=delta_d_window)

    risk = _risk_series(df, window=delta_d_window, thresholds=thresholds)
    k = max(1, int(np.ceil(float(topk_frac) * float(len(df)))))
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
        "risk_threshold": float(thresholds.risk_thr) if thresholds is not None else float(np.nanpercentile(risk.to_numpy(dtype=float), 95)),
        "baseline_used": bool(thresholds is not None),
        "topk_frac": float(topk_frac),
        "topk_k": int(k),
        "topk_indices": topk,
    }

    stability = audit_score_stability(df, windows=stability_windows, topk_frac=topk_frac)

    # Compare robust vs standard normalization on the same risk series (diagnostic)
    risk_std = pd.Series(standard_zscore(risk.to_numpy(dtype=float)), index=risk.index, name="RISK_STD")
    stability["risk_norm_spearman_robust_vs_standard"] = float(
        np.corrcoef(pd.Series(risk).rank().to_numpy(), pd.Series(risk_std).rank().to_numpy())[0, 1]
    )

    # Stress suite
    stress_suite = run_stress_suite(df, intensity=stress_intensity)
    # Maturity (proxy-based, non-circular)
    maturity_assessment = assess_maturity(df)
    maturity = asdict(maturity_assessment)

    # L performance (reactive)
    l_perf = evaluate_l_performance(
        df,
        window=delta_d_window,
        topk_frac=topk_frac,
        intensity=stress_intensity,
        thresholds=thresholds,
        risk_threshold=float(thresholds.risk_thr) if thresholds is not None else None,
    )

    # Proactive variant: activate on risk OR low O_level
    o_level = compute_o_level(df)
    o_thr = float(np.quantile(o_level.to_numpy(dtype=float), 0.15))

    proactive_topk = float(max(float(topk_frac), 0.20))
    variants: list[dict[str, Any]] = []

    v1 = evaluate_l_performance(
        df,
        window=delta_d_window,
        topk_frac=proactive_topk,
        o_threshold=o_thr,
        persist=2,
        max_delay=8,
        intensity=stress_intensity,
        thresholds=thresholds,
        risk_threshold=float(thresholds.risk_thr) if thresholds is not None else None,
    )
    v1["variant"] = "proactive_v1_risk_or_low_o"
    variants.append(v1)

    if (float(v1.get("prevented_topk_excess_rel", 0.0)) < 0.10) and (float(v1.get("prevented_exceedance_rel", 0.0)) < 0.10):
        o_thr2 = float(np.quantile(o_level.to_numpy(dtype=float), 0.20))
        v2 = evaluate_l_performance(
            df,
            window=delta_d_window,
            topk_frac=float(max(float(topk_frac), 0.30)),
            o_threshold=o_thr2,
            persist=1,
            max_delay=6,
            intensity=stress_intensity,
            thresholds=thresholds,
            risk_threshold=float(thresholds.risk_thr) if thresholds is not None else None,
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

    # Demo targets (auditable, sector-dependent in real deployments)
    gap_mean = float(np.mean(df["rule_execution_gap"].astype(float).to_numpy())) if "rule_execution_gap" in df.columns else float("nan")
    targets: Dict[str, Any] = {
        "rule_execution_gap_target_max": 0.05,
        "rule_execution_gap_mean": gap_mean,
        "rule_execution_gap_meets_target": bool(gap_mean <= 0.05) if np.isfinite(gap_mean) else False,
        "prevented_exceedance_rel_target_min": 0.10,
        "prevented_topk_excess_rel_target_min": 0.10,
        "prevented_exceedance_rel": float(l_perf_pro.get("prevented_exceedance_rel", 0.0)),
        "prevented_topk_excess_rel": float(l_perf_pro.get("prevented_topk_excess_rel", 0.0)),
        "proactive_topk_excess_rel": float(l_perf_pro.get("prevented_topk_excess_rel", 0.0)),
        "prevented_primary_metric": "prevented_topk_excess_rel",
        "prevented_primary_value": float(l_perf_pro.get("prevented_topk_excess_rel", 0.0)),
        "prevented_exceedance_meets_target": bool(float(l_perf_pro.get("prevented_exceedance_rel", 0.0)) >= 0.10),
        "prevented_meets_target": bool(float(l_perf_pro.get("prevented_topk_excess_rel", 0.0)) >= 0.10),
        "proactive_topk_frac": proactive_topk,
        "proactive_o_threshold": float(o_thr),
    }

    thresholds_payload = asdict(thresholds) if thresholds is not None else None

    return AuditReport(
        version=str(REPORT_VERSION),
        weights_version=str(WEIGHTS_VERSION),
        dataset_name=str(dataset_name),
        created_utc=str(created),
        summary=summary,
        stability=stability,
        stress_suite={k: asdict(v) for k, v in stress_suite.items()},
        maturity=maturity,
        l_performance=l_perf,
        l_performance_proactive=l_perf_pro,
        manipulability=manipulability,
        anti_gaming=anti_gaming,
        targets=targets,
        thresholds=thresholds_payload,
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
        "thresholds": report.thresholds,
    }
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )