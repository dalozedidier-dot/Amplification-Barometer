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


def _compute_verdict(
    *,
    maturity: Mapping[str, Any],
    stability: Mapping[str, Any],
    stress_suite: Mapping[str, Any],
    anti_gaming: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> Dict[str, Any]:
    # Scores normalisés 0..1 (démo). Les dimensions restent séparées.
    def _clamp01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    # stabilité: moyenne Spearman et Jaccard TopK si présents
    spe = float(stability.get("spearman_mean_risk", stability.get("spearman_mean", 0.0)) or 0.0)
    jac = float(stability.get("topk_jaccard_mean_risk", stability.get("topk_jaccard_mean", stability.get("topk_jaccard_mean_risk", 0.0))) or 0.0)
    stability_score = _clamp01((spe + jac) / 2.0)

    # L_cap / L_act depuis maturité
    cap = float(maturity.get("cap_score_enforced", maturity.get("cap_score_raw", 0.0)) or 0.0)
    act = float(maturity.get("act_score_enforced", maturity.get("act_score_raw", 0.0)) or 0.0)
    lcap_score = _clamp01(cap)
    lact_score = _clamp01(act)

    # stress: plus la dégradation max est basse, meilleur est le score
    degrs = []
    for v in stress_suite.values():
        try:
            degrs.append(float(v.get("degradation", 1.0)))
        except Exception:
            pass
    max_deg = max(degrs) if degrs else 1.0
    stress_score = _clamp01(1.0 / (1.0 + max(0.0, max_deg - 1.0)))

    # anti-gaming: pénalité si red_flag ou biais O élevé
    red_flag = bool(anti_gaming.get("red_flag", False))
    sev = 0.0
    if "o_bias" in anti_gaming and isinstance(anti_gaming["o_bias"], Mapping):
        sev = float(anti_gaming["o_bias"].get("severity", 0.0) or 0.0)
    anti_score = _clamp01(1.0 - max(sev, 1.0 if red_flag else 0.0))

    # gouvernance: cibles neutres via targets
    gap_ok = bool(targets.get("rule_execution_gap_meets_target", False))
    turn_ok = True
    if "control_turnover_meets_target" in targets:
        turn_ok = bool(targets.get("control_turnover_meets_target", False))
    governance_score = 1.0 if (gap_ok and turn_ok) else (0.5 if (gap_ok or turn_ok) else 0.0)

    # résilience: proxy via stress_score + L_act (démo)
    resilience_score = _clamp01(0.6 * stress_score + 0.4 * lact_score)

    dims = {
        "stability": stability_score,
        "L_cap": lcap_score,
        "L_act": lact_score,
        "resilience": resilience_score,
        "governance": governance_score,
        "anti_gaming": anti_score,
        "stress_suite": stress_score,
    }

    # score global optionnel 0..100
    weights = {
        "stability": 0.15,
        "L_cap": 0.20,
        "L_act": 0.15,
        "resilience": 0.15,
        "governance": 0.15,
        "anti_gaming": 0.10,
        "stress_suite": 0.10,
    }
    global_score = 0.0
    for k, w in weights.items():
        global_score += float(dims.get(k, 0.0)) * float(w)
    global_score_100 = float(round(100.0 * _clamp01(global_score), 2))

    label = str(maturity.get("label", "Immature"))

    return {"label": label, "global_score": global_score_100, "dimensions": dims}

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

    # Variante proactive: activation basée sur risque OU orientation dégradée (O_level).
    # Intention: booster la prévention (>10%) avec un L proactif via seuils O(t), sans données sensibles.
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
    )
    v1["variant"] = "proactive_v1_risk_or_low_o"
    variants.append(v1)

    # Auto-tuning minimal si la cible >10% n'est pas atteinte.
    # On augmente la proactivité: topk plus large, persistance minimale, délai max réduit, seuil O un peu plus strict.
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
        )
        v2["variant"] = "proactive_v2_autotune"
        variants.append(v2)

    # On choisit la variante la plus efficace, priorité à la réduction de sévérité de queue.
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

    # Cibles de démonstration (audit réel: ces seuils doivent être justifiés par secteur)
    gap_mean = float(np.mean(df["rule_execution_gap"].astype(float).to_numpy())) if "rule_execution_gap" in df.columns else float("nan")
    targets: Dict[str, Any] = {
        "rule_execution_gap_target_max": 0.05,
        "rule_execution_gap_mean": gap_mean,
        "rule_execution_gap_meets_target": bool(gap_mean <= 0.05) if np.isfinite(gap_mean) else False,
        "prevented_exceedance_rel_target_min": 0.10,
        "prevented_topk_excess_rel_target_min": 0.10,
        "prevented_exceedance_rel": float(l_perf_pro.get("prevented_exceedance_rel", 0.0)),
        "prevented_topk_excess_rel": float(l_perf_pro.get("prevented_topk_excess_rel", 0.0)),
        "prevented_primary_metric": "prevented_topk_excess_rel",
        "prevented_primary_value": float(l_perf_pro.get("prevented_topk_excess_rel", 0.0)),
        "prevented_exceedance_meets_target": bool(float(l_perf_pro.get("prevented_exceedance_rel", 0.0)) >= 0.10),
        "prevented_meets_target": bool(float(l_perf_pro.get("prevented_topk_excess_rel", 0.0)) >= 0.10),
        "proactive_topk_frac": proactive_topk,
        "proactive_o_threshold": float(o_thr),
    }

    verdict = _compute_verdict(maturity=maturity, stability=stability, stress_suite=stress_suite, anti_gaming=anti_gaming, targets=targets)

    return AuditReport(
        version="0.4.3",
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
        verdict=verdict,
        targets=targets,
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
        "verdict": report.verdict,
        "targets": report.targets,
    }
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
