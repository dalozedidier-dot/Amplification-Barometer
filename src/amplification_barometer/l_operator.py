from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .calibration import Thresholds, derive_thresholds, risk_signature
from .composites import compute_at, compute_delta_d, compute_e_metrics, compute_o_level, robust_zscore


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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def _to_unit_interval(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.clip(_sigmoid(z), 0.0, 1.0)


def compute_l_cap(df: pd.DataFrame) -> pd.Series:
    """L_cap: capacité intrinsèque (proxy technique)."""
    _require(df, L_CAP_PROXIES, "L_cap")
    arr = df.loc[:, list(L_CAP_PROXIES)].astype(float).to_numpy()
    raw = np.mean(arr, axis=1)
    z = robust_zscore(raw)
    return pd.Series(z, index=df.index, name="L_CAP")


def compute_l_act(df: pd.DataFrame) -> pd.Series:
    """L_act: activation observée, neutre, mesurable."""
    _require(df, L_ACT_PROXIES, "L_act")
    ex = df["exemption_rate"].astype(float).to_numpy()
    sd = df["sanction_delay"].astype(float).to_numpy()
    sd_norm = np.clip(sd / 365.0, 0.0, 1.0)
    ct = df["control_turnover"].astype(float).to_numpy()
    ci = df["conflict_interest_proxy"].astype(float).to_numpy()
    gap = df["rule_execution_gap"].astype(float).to_numpy()

    risk = 0.25 * ex + 0.20 * sd_norm + 0.20 * ct + 0.15 * ci + 0.20 * gap
    raw = 1.0 - risk  # higher is better activation
    z = robust_zscore(raw)
    return pd.Series(z, index=df.index, name="L_ACT")


def desired_activation(risk: pd.Series, *, threshold: float, persist: int = 3) -> pd.Series:
    """Desired activation signal based on persistent exceedance."""
    r = risk.to_numpy(dtype=float)
    mask = r > float(threshold)
    out = np.zeros_like(mask, dtype=bool)
    run = 0
    for i, v in enumerate(mask):
        run = run + 1 if v else 0
        if run >= int(persist):
            out[i] = True
    return pd.Series(out, index=risk.index, name="DESIRED")


def desired_activation_proactive(
    risk: pd.Series,
    *,
    o_level: pd.Series,
    risk_threshold: float,
    o_threshold: float,
    persist: int = 3,
) -> pd.Series:
    """Proactive desired activation: combine risk exceedance and O weakness."""
    r = risk.to_numpy(dtype=float) > float(risk_threshold)
    o = o_level.to_numpy(dtype=float) < float(o_threshold)
    mask = r | o
    out = np.zeros_like(mask, dtype=bool)
    run = 0
    for i, v in enumerate(mask):
        run = run + 1 if v else 0
        if run >= int(persist):
            out[i] = True
    return pd.Series(out, index=risk.index, name="DESIRED")


def realize_activation(desired: pd.Series, lact: pd.Series, *, max_delay: int = 12) -> pd.Series:
    """Real activation: desired delayed depending on L_act strength."""
    d = desired.to_numpy(dtype=bool)
    a = np.zeros_like(d, dtype=bool)
    lact01 = _to_unit_interval(lact.to_numpy(dtype=float))
    delays = np.round((1.0 - lact01) * float(max_delay)).astype(int)

    last_desired = -10**9
    for i, v in enumerate(d):
        if v and (i != last_desired):
            last_desired = i
            delay = int(delays[i])
            j = min(len(a) - 1, i + delay)
            a[j] = True

    hold = max(1, int(max_delay // 3))
    for i in range(len(a)):
        if a[i]:
            a[i:min(len(a), i + hold)] = True
    return pd.Series(a, index=desired.index, name="ACTIVATED")


def apply_limit_action(
    df: pd.DataFrame,
    activation: pd.Series,
    lcap: pd.Series,
    *,
    intensity: float = 1.0,
) -> pd.DataFrame:
    """Stylized limit action applied on proxies after activation."""
    out = df.copy()
    act = activation.to_numpy(dtype=bool)
    cap01 = _to_unit_interval(lcap.to_numpy(dtype=float))
    strength = float(np.nanmean(cap01)) * float(intensity)

    if not np.any(act):
        return out

    idx = out.index[act]

    for c in ("scale_proxy", "speed_proxy", "leverage_proxy", "autonomy_proxy", "replicability_proxy"):
        if c in out.columns:
            out.loc[idx, c] = out.loc[idx, c].astype(float) * (1.0 - 0.15 * strength)

    for c in ("stop_proxy", "threshold_proxy", "decision_proxy", "execution_proxy", "coherence_proxy"):
        if c in out.columns:
            out.loc[idx, c] = out.loc[idx, c].astype(float) * (1.0 + 0.10 * strength)

    for c in ("impact_proxy", "propagation_proxy", "hysteresis_proxy"):
        if c in out.columns:
            out.loc[idx, c] = out.loc[idx, c].astype(float) * (1.0 - 0.08 * strength)

    for c in ("margin_proxy", "redundancy_proxy", "diversity_proxy"):
        if c in out.columns:
            out.loc[idx, c] = out.loc[idx, c].astype(float) * (1.0 + 0.08 * strength)

    if "recovery_time_proxy" in out.columns:
        out.loc[idx, "recovery_time_proxy"] = out.loc[idx, "recovery_time_proxy"].astype(float) * (1.0 - 0.06 * strength)

    return out


@dataclass(frozen=True)
class MaturityAssessment:
    label: str
    l_cap_bench_score: float
    l_act_mean: float
    activation_delay_steps: float
    activation_stability: float
    prevented_exceedance_rel: float
    e_reduction_rel: float
    governance_rule_execution_gap: float
    governance_control_turnover: float
    notes: Dict[str, float]


def _risk_series(df: pd.DataFrame, *, window: int = 5, thresholds: Thresholds | None = None) -> pd.Series:
    if thresholds is None:
        thresholds = derive_thresholds(df, window=window)
    at = compute_at(df)
    dd = compute_delta_d(df, window=window)
    return risk_signature(df, thresholds=thresholds, window=window, at_series=at, dd_series=dd)


def evaluate_l_performance(
    df: pd.DataFrame,
    *,
    window: int = 5,
    risk_threshold: float | None = None,
    o_threshold: float | None = None,
    topk_frac: float = 0.10,
    persist: int = 3,
    max_delay: int = 12,
    intensity: float = 1.0,
    thresholds: Thresholds | None = None,
    lact_override: pd.Series | None = None,
    lcap_override: pd.Series | None = None,
) -> Dict[str, Any]:
    """
    Évalue un opérateur de limite L en utilisant un seuil de risque.

    Règle baseline:
    - Si thresholds est fourni, risk_threshold par défaut doit être thresholds.risk_thr
    - Interdire un recalcul par dataset quand une baseline est fournie
    """
    local_thr = thresholds if thresholds is not None else derive_thresholds(df, window=window)
    risk = _risk_series(df, window=window, thresholds=local_thr)

    if risk_threshold is None:
        risk_threshold = float(local_thr.risk_thr)

    lact = compute_l_act(df) if lact_override is None else lact_override
    lcap = compute_l_cap(df) if lcap_override is None else lcap_override

    if o_threshold is None:
        desired = desired_activation(risk, threshold=float(risk_threshold), persist=persist)
    else:
        o_level = compute_o_level(df)
        desired = desired_activation_proactive(
            risk,
            o_level=o_level,
            risk_threshold=float(risk_threshold),
            o_threshold=float(o_threshold),
            persist=persist,
        )

    activated = realize_activation(desired, lact, max_delay=max_delay)
    df_limited = apply_limit_action(df, activated, lcap, intensity=intensity)
    risk_limited = _risk_series(df_limited, window=window, thresholds=local_thr)

    base_mask = risk.to_numpy(dtype=float) > float(risk_threshold)
    lim_mask = risk_limited.to_numpy(dtype=float) > float(risk_threshold)

    exceed_base = float(np.mean(base_mask))
    exceed_limited = float(np.mean(lim_mask))
    prevented = float(max(0.0, exceed_base - exceed_limited))
    prevented_rel = float(prevented / exceed_base) if exceed_base > 1e-12 else 0.0

    eps = 1e-12
    r_arr = risk.to_numpy(dtype=float)
    rl_arr = risk_limited.to_numpy(dtype=float)
    excess_base = np.clip(r_arr - float(risk_threshold), 0.0, None)
    excess_limited = np.clip(rl_arr - float(risk_threshold), 0.0, None)
    k_tail = max(1, int(len(r_arr) * float(topk_frac)))
    tail_base = np.partition(excess_base, -k_tail)[-k_tail:]
    tail_limited = np.partition(excess_limited, -k_tail)[-k_tail:]
    mean_excess_base = float(np.mean(tail_base))
    mean_excess_limited = float(np.mean(tail_limited))
    prevented_topk_excess_rel = float(max(0.0, mean_excess_base - mean_excess_limited) / (mean_excess_base + eps))

    d = desired.to_numpy(dtype=bool)
    a = activated.to_numpy(dtype=bool)
    first_d = int(np.argmax(d)) if np.any(d) else -1
    first_a = int(np.argmax(a)) if np.any(a) else -1
    activation_delay_steps = float(first_a - first_d) if (first_d >= 0 and first_a >= 0) else float("nan")

    # E reduction relative, bornée dans [0, 1]
    e_base = compute_e_metrics(df)["E_level"].to_numpy(dtype=float)
    e_lim = compute_e_metrics(df_limited)["E_level"].to_numpy(dtype=float)
    base_abs = float(np.mean(np.abs(e_base))) + 1e-12
    lim_abs = float(np.mean(np.abs(e_lim)))
    e_reduction_rel = float(np.clip(max(0.0, base_abs - lim_abs) / base_abs, 0.0, 1.0))

    cap01 = float(np.mean(_to_unit_interval(lcap.to_numpy(dtype=float))))
    act01 = float(np.mean(_to_unit_interval(lact.to_numpy(dtype=float))))

    # Verdict minimal (sans circularité): basé sur performance + activation + garde-fous gouvernance.
    g_rule_gap = float(np.mean(pd.to_numeric(df.get("rule_execution_gap"), errors="coerce").to_numpy(dtype=float)))
    g_turnover = float(np.mean(pd.to_numeric(df.get("control_turnover"), errors="coerce").to_numpy(dtype=float)))

    verdict = "Dissonant"
    if (cap01 < 0.40) or (prevented_rel < 0.05):
        verdict = "Immature"
    elif (
        (cap01 >= 0.60)
        and (act01 >= 0.60)
        and (np.isfinite(activation_delay_steps) and activation_delay_steps <= 5.0)
        and (prevented_rel >= 0.10)
        and (e_reduction_rel >= 0.20)
        and (g_rule_gap <= 0.05)
        and (g_turnover <= 0.05)
    ):
        verdict = "Mature"
    return {
        "prevented_exceedance": float(prevented),
        "prevented_exceedance_rel": float(prevented_rel),
        "verdict": verdict,
        "prevented_topk_excess_rel": float(prevented_topk_excess_rel),
        "activation_delay_steps": float(activation_delay_steps),
        "e_reduction_rel": float(e_reduction_rel),
        "risk_threshold": float(risk_threshold),
        "l_cap_proxy_mean": float(cap01),
        "l_act_proxy_mean": float(act01),
    }


def assess_maturity(df: pd.DataFrame, *, window: int = 5, thresholds: Thresholds | None = None) -> MaturityAssessment:
    """Maturity without circularity: L_cap bench + L_act observed."""
    local_thr = thresholds if thresholds is not None else derive_thresholds(df, window=window)

    lcap = compute_l_cap(df)
    lact = compute_l_act(df)

    # Bench: idéal (activation toujours active)
    ideal_lact = pd.Series(np.ones(len(df), dtype=float) * 10.0, index=df.index, name="L_ACT_IDEAL")
    perf_ideal = evaluate_l_performance(df, window=window, thresholds=local_thr, lact_override=ideal_lact, lcap_override=lcap)

    l_cap_bench_score = float(0.6 * perf_ideal["prevented_exceedance_rel"] + 0.4 * perf_ideal["prevented_topk_excess_rel"])

    perf_real = evaluate_l_performance(df, window=window, thresholds=local_thr)
    activation_delay_steps = float(perf_real["activation_delay_steps"])

    act01 = _to_unit_interval(lact.to_numpy(dtype=float))
    activation_stability = float(1.0 - np.std(act01))

    gap = (
        float(np.mean(pd.to_numeric(df.get("rule_execution_gap", 0.0), errors="coerce").to_numpy(dtype=float)))
        if "rule_execution_gap" in df.columns
        else 1.0
    )
    ct = (
        float(np.mean(pd.to_numeric(df.get("control_turnover", 0.0), errors="coerce").to_numpy(dtype=float)))
        if "control_turnover" in df.columns
        else 1.0
    )

    prevented_exceedance_rel = float(perf_real["prevented_exceedance_rel"])
    e_reduction_rel = float(perf_real["e_reduction_rel"])

    # Typologie conforme
    label = "Mature"
    if l_cap_bench_score < 0.45:
        label = "Immature"
    else:
        # dissonant si activation faible ou gouvernance hors cibles
        if (float(np.mean(act01)) < 0.45) or (gap > 0.05) or (ct > 0.05):
            label = "Dissonant"

    notes = {
        "cap01": float(np.mean(_to_unit_interval(lcap.to_numpy(dtype=float)))),
        "act01": float(np.mean(act01)),
        "gap_mean": gap,
        "ct_mean": ct,
    }

    return MaturityAssessment(
        label=label,
        l_cap_bench_score=float(l_cap_bench_score),
        l_act_mean=float(np.mean(act01)),
        activation_delay_steps=activation_delay_steps,
        activation_stability=float(activation_stability),
        prevented_exceedance_rel=prevented_exceedance_rel,
        e_reduction_rel=e_reduction_rel,
        governance_rule_execution_gap=gap,
        governance_control_turnover=ct,
        notes=notes,
    )
