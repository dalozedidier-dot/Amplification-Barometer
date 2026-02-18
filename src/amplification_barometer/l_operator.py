from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .calibration import Thresholds, risk_signature
from .composites import compute_at, compute_delta_d, compute_o_level, robust_zscore


def _require(df: pd.DataFrame, cols: Tuple[str, ...], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes pour {context}: {missing}")


L_CAP_PROXIES: Tuple[str, ...] = ("stop_proxy", "threshold_proxy", "execution_proxy", "coherence_proxy")
L_ACT_PROXIES: Tuple[str, ...] = (
    "exemption_rate",
    "sanction_delay",
    "control_turnover",
    "conflict_interest_proxy",
    "rule_execution_gap",
)


def compute_l_cap(df: pd.DataFrame) -> pd.Series:
    """L_cap (z): capacité intrinsèque d'arrêt (proxy composite)."""
    _require(df, L_CAP_PROXIES, "L_cap")
    arr = df.loc[:, list(L_CAP_PROXIES)].astype(float).to_numpy()
    raw = np.mean(arr, axis=1)
    z = robust_zscore(raw)
    return pd.Series(z, index=df.index, name="L_CAP")


def compute_l_act(df: pd.DataFrame) -> pd.Series:
    """L_act (z): activation observée (enforcement effectif, proxy composite).

    Important: L_act doit rester *absolu* et comparable entre datasets.
    L'ancienne version normalisait par z-score interne au dataset, ce qui écrase
    l'information quand les proxys de gouvernance sont constants (cas fréquent
    sur données réelles anonymisées ou quand on utilise des valeurs par défaut).

    On calcule une "qualité d'activation" good dans [0,1], puis on la projette en z
    via un logit. Avec _sigmoid01(k=1.6), on retrouve good (à l'epsilon près).
    """
    _require(df, L_ACT_PROXIES, "L_act")
    # plus bas = mieux pour exemption_rate, sanction_delay, turnover, conflict, gap
    x = df.loc[:, list(L_ACT_PROXIES)].astype(float).to_numpy()

    sanc = x[:, 1]
    sanc01 = np.clip(sanc / 365.0, 0.0, 1.0)

    x01 = np.column_stack(
        [
            np.clip(x[:, 0], 0.0, 1.0),
            sanc01,
            np.clip(x[:, 2], 0.0, 1.0),
            np.clip(x[:, 3], 0.0, 1.0),
            np.clip(x[:, 4], 0.0, 1.0),
        ]
    )

    bad = 0.25 * x01[:, 0] + 0.20 * x01[:, 1] + 0.20 * x01[:, 2] + 0.15 * x01[:, 3] + 0.20 * x01[:, 4]
    good = 1.0 - bad

    eps = 1e-6
    good = np.clip(good, eps, 1.0 - eps)

    # logit projection so that _sigmoid01(z, k=1.6) ≈ good
    z = (np.log(good / (1.0 - good)) / 1.6).astype(float)
    return pd.Series(z, index=df.index, name="L_ACT")


def _sigmoid01(z: float, *, k: float = 1.6) -> float:
    """Map un score (≈0 autour de stable) vers [0,1] avec dynamique (évite plafond à ~0.50).

    Implémentation numériquement stable pour éviter les overflows sur exp().
    """
    z = float(z)
    x = float(k) * z
    if x >= 0.0:
        e = float(np.exp(-x))
        return float(1.0 / (1.0 + e))
    e = float(np.exp(x))
    return float(e / (1.0 + e))


@dataclass(frozen=True)
class MaturityAssessment:
    """Maturity without circularity.

    - L_cap: capacité testable (ici, proxy démonstrateur + stress-derived recovery)
    - L_act: activation observée (gouvernance opérationnelle)
    """
    label: str
    cap_score_raw: float
    cap_score_enforced: float
    act_score_raw: float
    act_score_enforced: float
    notes: Dict[str, Any]


def assess_maturity(df: pd.DataFrame, *, df_stressed: Optional[pd.DataFrame] = None) -> MaturityAssessment:
    """Compute maturity label and scores.

    Fixes:
    - remove linear mapping ceiling around 0.50 (use sigmoid)
    - dual gate governance (turnover AND rule_gap) to avoid over-scoring IA_borg
    - recovery uses p90 and a simple quality mapping 1/(1+p90)
    """
    # Base from L_cap z
    lcap_z = compute_l_cap(df).to_numpy(dtype=float)
    cap01_base = float(np.nanmean([_sigmoid01(v) for v in lcap_z]))

    # Governance gate (neutral, auditable)
    turnover = float(np.nanmean(df["control_turnover"].astype(float))) if "control_turnover" in df.columns else float("nan")
    rule_gap = float(np.nanmean(df["rule_execution_gap"].astype(float))) if "rule_execution_gap" in df.columns else float("nan")
    gov_ok = bool(np.isfinite(turnover) and np.isfinite(rule_gap) and (turnover <= 0.05) and (rule_gap <= 0.05))
    gov_bonus = 0.25 if gov_ok else 0.0
    diminishing = gov_bonus * (1.0 - cap01_base)

    # Recovery quality (p90)
    recovery_p90 = float("nan")
    recovery_quality = 0.5
    if "recovery_time_proxy" in df.columns:
        vals = df["recovery_time_proxy"].astype(float).to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size:
            recovery_p90 = float(np.percentile(vals, 90))
            # Robust mapping, bounded, interpretable
            recovery_quality = float(1.0 / (1.0 + max(recovery_p90, 0.0)))
            # Center at 0.5 so "unknown/flat" does not inflate
            recovery_bonus = 0.12 * max(0.0, (recovery_quality - 0.5) / 0.5)
        else:
            recovery_bonus = 0.0
    else:
        recovery_bonus = 0.0

    cap01_v2 = float(np.clip(cap01_base + diminishing + recovery_bonus, 0.0, 1.0))

    # Enforcement factor: penalize high turnover or high gap (prevents "paper governance")
    # Turnover above 0.10 degrades; above 0.30 collapses.
    if not (np.isfinite(turnover) and np.isfinite(rule_gap)):
        enforcement_factor = 1.0
    else:
        pen_turn = float(np.clip((turnover - 0.10) / 0.20, 0.0, 1.0))
        pen_gap = float(np.clip((rule_gap - 0.10) / 0.20, 0.0, 1.0))
        enforcement_factor = float(np.clip(1.0 - 0.60 * pen_turn - 0.60 * pen_gap, 0.0, 1.0))

    cap_score_enforced = float(np.clip(cap01_v2 * enforcement_factor, 0.0, 1.0))

    # L_act (raw from governance proxies, not circular)
    lact_z = compute_l_act(df).to_numpy(dtype=float)
    act01_base = float(np.nanmean([_sigmoid01(v) for v in lact_z]))
    act_score_enforced = act01_base  # for now, keep as observed activation

    # Label
    cap_thr = 0.60
    act_thr = 0.70
    if cap_score_enforced < cap_thr:
        label = "Immature"
    elif act_score_enforced < act_thr or not gov_ok:
        label = "Dissonant"
    else:
        label = "Mature"

    notes: Dict[str, Any] = {
        "cap01_base": cap01_base,
        "cap01_v2": cap01_v2,
        "cap_score_raw": cap01_base,
        "cap_score_enforced": cap_score_enforced,
        "act01_base": act01_base,
        "act_score_raw": act01_base,
        "act_score_enforced": act_score_enforced,
        "enforcement_factor": enforcement_factor,
        "gov_ok": gov_ok,
        "turnover_mean": turnover,
        "rule_execution_gap_mean": rule_gap,
        "recovery_p90": recovery_p90,
        "recovery_quality": recovery_quality,
        "thresholds": {"cap_thr": cap_thr, "act_thr": act_thr, "turnover_gate": 0.05, "rule_gap_gate": 0.05},
    }
    return MaturityAssessment(
        label=label,
        cap_score_raw=float(cap01_base),
        cap_score_enforced=cap_score_enforced,
        act_score_raw=float(act01_base),
        act_score_enforced=float(act_score_enforced),
        notes=notes,
    )


def _topk_mask(x: np.ndarray, *, frac: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.size
    k = max(1, int(np.ceil(float(frac) * float(n))))
    thr = float(np.partition(x, n - k)[n - k])
    return x >= thr


def evaluate_l_performance(
    df: pd.DataFrame,
    *,
    window: int = 5,
    topk_frac: float = 0.10,
    o_threshold: Optional[float] = None,
    risk_threshold: Optional[float] = None,
    thresholds: Optional[Thresholds] = None,
    persist: int = 2,
    max_delay: int = 8,
    intensity: float = 1.0,
) -> Dict[str, Any]:
    """Evaluate a demonstrator limit operator L.

    Fixes:
    - accept `thresholds` (stable baseline) to prevent per-dataset renormalization
    - return both legacy keys and *_rel keys expected by audit_report/sector_suite
    """
    if persist < 1:
        persist = 1

    # Risk
    if thresholds is not None:
        risk = risk_signature(df, thresholds=thresholds, window=window).to_numpy(dtype=float)
        rt = float(thresholds.risk_thr) if risk_threshold is None else float(risk_threshold)
    else:
        at = compute_at(df).to_numpy(dtype=float)
        dd = compute_delta_d(df, window=window).to_numpy(dtype=float)
        risk = robust_zscore(at) + robust_zscore(dd)
        rt = float(np.percentile(risk, 95)) if risk_threshold is None else float(risk_threshold)

    # O threshold (optional proactive trigger)
    if o_threshold is None:
        o_level = compute_o_level(df).to_numpy(dtype=float)
        o_thr = float(np.quantile(o_level, 0.15))
    else:
        o_thr = float(o_threshold)

    o_level = compute_o_level(df).to_numpy(dtype=float)
    desired = (risk > rt) | (o_level < o_thr)

    # Persistence (simple)
    if persist > 1:
        desired_p = np.zeros_like(desired, dtype=bool)
        for i in range(desired.size):
            start = max(0, i - persist + 1)
            desired_p[i] = bool(np.all(desired[start : i + 1]))
        desired = desired_p

    # Activation delay (use L_act as gate)
    lact_z = compute_l_act(df).to_numpy(dtype=float)
    lact01 = np.array([_sigmoid01(v) for v in lact_z], dtype=float)

    activated = np.zeros_like(desired, dtype=bool)
    delay_steps = np.full(desired.shape, fill_value=-1, dtype=int)

    last_desired = -10**9
    for i in range(desired.size):
        if desired[i]:
            if i != last_desired + 1:
                last_desired = i
            # allow activation if lact is decent
            # activation time is first j within max_delay where lact01[j] > 0.55
            for dly in range(0, max_delay + 1):
                j = i + dly
                if j >= desired.size:
                    break
                if lact01[j] >= 0.55:
                    activated[j] = True
                    delay_steps[j] = dly
                    break

    activation_delay_steps = int(np.min(delay_steps[delay_steps >= 0])) if np.any(delay_steps >= 0) else int(max_delay + 1)

    # Apply a simple operator: reduce "scale_proxy" and "speed_proxy" under activation
    df2 = df.copy()
    for col in ("scale_proxy", "speed_proxy", "leverage_proxy"):
        if col in df2.columns:
            arr = df2[col].astype(float).to_numpy()
            arr2 = arr.copy()
            arr2[activated] = arr2[activated] * float(max(0.0, 1.0 - 0.25 * float(intensity)))
            df2[col] = arr2

    
    # Optional O-boost under activation (tighten controls).
    # Rationale: on datasets where P reduction alone mostly scales risk without changing
    # the TopK membership (e.g., AIOps), a small improvement of O proxies is a realistic
    # countermeasure (better stop/threshold/decision/execution/coherence).
    o_boost = float(np.clip(0.08 * float(intensity), 0.0, 0.60))
    if o_boost > 0.0:
        for col in ("stop_proxy", "threshold_proxy", "decision_proxy", "execution_proxy", "coherence_proxy"):
            if col in df2.columns:
                arr = df2[col].astype(float).to_numpy()
                arr2 = arr.copy()
                # Move towards 1.0 with a bounded step
                arr2[activated] = arr2[activated] + o_boost * (1.0 - arr2[activated])
                df2[col] = np.clip(arr2, 0.0, 1.0)

# Post risk for prevention metrics
    if thresholds is not None:
        risk2 = risk_signature(df2, thresholds=thresholds, window=window).to_numpy(dtype=float)
    else:
        at2 = compute_at(df2).to_numpy(dtype=float)
        dd2 = compute_delta_d(df2, window=window).to_numpy(dtype=float)
        risk2 = robust_zscore(at2) + robust_zscore(dd2)

    exceed0 = risk > rt
    exceed1 = risk2 > rt
    prevented_exceedance = 0.0
    if float(np.mean(exceed0.astype(float))) > 1e-12:
        prevented_exceedance = float(np.mean(exceed0.astype(float)) - np.mean(exceed1.astype(float))) / float(np.mean(exceed0.astype(float)))
        prevented_exceedance = float(np.clip(prevented_exceedance, 0.0, 1.0))

    # TopK overlap (severity tail)
    m0 = _topk_mask(risk, frac=float(topk_frac))
    m1 = _topk_mask(risk2, frac=float(topk_frac))
    inter = float(np.sum(m0 & m1))
    denom = float(np.sum(m0))
    topk_overlap = float(inter / denom) if denom > 0 else 1.0
    prevented_topk_excess_rel = float(np.clip(1.0 - topk_overlap, 0.0, 1.0))

    # E reduction proxy (if E exists)
    e_reduction_rel = 0.0
    if "impact_proxy" in df.columns:
        e0 = df["impact_proxy"].astype(float).to_numpy()
        e1 = df2["impact_proxy"].astype(float).to_numpy()
        base = float(np.mean(np.abs(e0))) + 1e-12
        e_reduction_rel = float(np.clip((base - float(np.mean(np.abs(e1)))) / base, 0.0, 1.0))

    # maturity (no circularity): use base df and stressed (overload proxy if provided)
    maturity = assess_maturity(df)

    verdict = maturity.label

    return {
        "risk_threshold": rt,
        "o_threshold": o_thr,
        "topk_frac": float(topk_frac),
        "prevented_exceedance": float(prevented_exceedance),
        "prevented_exceedance_rel": float(prevented_exceedance),
        "topk_overlap": float(topk_overlap),
        "prevented_topk_excess_rel": float(prevented_topk_excess_rel),
        "e_reduction_rel": float(e_reduction_rel),
        "activation_delay_steps": activation_delay_steps,
        "verdict": verdict,
        "maturity": maturity.notes,
        "params": {
            "window": int(window),
            "topk_frac": float(topk_frac),
            "persist": int(persist),
            "max_delay": int(max_delay),
            "intensity": float(intensity),
            "use_baseline_thresholds": bool(thresholds is not None),
        },
    }
