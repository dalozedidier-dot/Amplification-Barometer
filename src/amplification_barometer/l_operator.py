from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import pandas as pd

from .calibration import Thresholds, risk_signature
from .composites import compute_at, compute_delta_d, compute_e, compute_o_level, robust_zscore


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


def compute_l_cap(df: pd.DataFrame) -> pd.Series:
    """L_cap: capacité intrinsèque d'arrêt.

    Démonstrateur: agrégat sur proxys d'arrêt et d'exécution, puis z-score robuste.
    """
    _require(df, L_CAP_PROXIES, "L_cap")
    arr = df.loc[:, list(L_CAP_PROXIES)].astype(float).to_numpy()
    raw = np.mean(arr, axis=1)
    z = robust_zscore(raw)
    return pd.Series(z, index=df.index, name="L_CAP")


def compute_l_act(df: pd.DataFrame) -> pd.Series:
    """L_act: activation effective.

    Démonstrateur: agrégat sur exemptions, délai de sanction et signaux de capture.
    Sens: plus exemptions et délais montent, plus L_act doit baisser.
    """
    _require(df, L_ACT_PROXIES, "L_act")
    ex = df["exemption_rate"].astype(float).to_numpy()
    sd = df["sanction_delay"].astype(float).to_numpy()
    sd_norm = np.clip(sd / 365.0, 0.0, 1.0)
    ct = df["control_turnover"].astype(float).to_numpy()
    ci = df["conflict_interest_proxy"].astype(float).to_numpy()
    gap = df["rule_execution_gap"].astype(float).to_numpy()

    risk = 0.25 * ex + 0.20 * sd_norm + 0.20 * ct + 0.15 * ci + 0.20 * gap
    raw = 1.0 - risk
    z = robust_zscore(raw)
    return pd.Series(z, index=df.index, name="L_ACT")


def _risk_series(
    df: pd.DataFrame,
    *,
    window: int = 5,
    thresholds: Thresholds | None = None,
) -> pd.Series:
    """Risk signature, baseline-aware.

    Si thresholds est fourni, on utilise la baseline stable (pas de renormalisation par dataset).
    Sinon, démonstrateur: robust z-score par dataset.
    """
    if thresholds is not None:
        return risk_signature(df, thresholds=thresholds, window=window)

    at = compute_at(df).to_numpy(dtype=float)
    dd = compute_delta_d(df, window=window).to_numpy(dtype=float)
    risk = robust_zscore(at) + robust_zscore(dd)
    return pd.Series(risk, index=df.index, name="RISK")


def _to_unit_interval(z: np.ndarray) -> np.ndarray:
    """Maps z-scores to [0,1] in a bounded, monotonic way."""
    z = np.asarray(z, dtype=float)
    return np.clip(0.5 + 0.25 * z, 0.0, 1.0)


def desired_activation(risk: pd.Series, *, threshold: float, persist: int = 3) -> pd.Series:
    """Returns desired activation signal based on persistence above a threshold."""
    persist = max(1, int(persist))
    above = (risk.to_numpy(dtype=float) > float(threshold)).astype(int)
    roll = np.convolve(above, np.ones(persist, dtype=int), mode="same")
    desired = roll >= persist
    return pd.Series(desired, index=risk.index, name="L_DESIRED")


def desired_activation_proactive(
    risk: pd.Series,
    *,
    o_level: pd.Series,
    risk_threshold: float,
    o_threshold: float,
    persist: int = 3,
) -> pd.Series:
    """Desired activation when risk is high OR orientation is low."""
    persist = max(1, int(persist))
    r = risk.to_numpy(dtype=float)
    o = o_level.to_numpy(dtype=float)
    trig = ((r > float(risk_threshold)) | (o < float(o_threshold))).astype(int)
    roll = np.convolve(trig, np.ones(persist, dtype=int), mode="same")
    desired = roll >= persist
    return pd.Series(desired, index=risk.index, name="L_DESIRED_PROACTIVE")


def realize_activation(
    desired: pd.Series,
    lact: pd.Series,
    *,
    max_delay: int = 12,
    min_on: int = 3,
) -> pd.Series:
    """Realizes activation with delay that depends on L_act."""
    max_delay = max(0, int(max_delay))
    min_on = max(1, int(min_on))

    d = desired.to_numpy(dtype=bool)
    a = lact.to_numpy(dtype=float)
    a01 = _to_unit_interval(a)

    n = len(d)
    out = np.zeros(n, dtype=bool)

    i = 0
    while i < n:
        if not d[i]:
            i += 1
            continue

        j = i
        while j < n and d[j]:
            j += 1

        delay = int(round((1.0 - float(np.mean(a01[i:j]))) * max_delay))
        start = min(n, i + delay)
        end = j

        if start < end:
            on_end = min(n, end + min_on)
            out[start:on_end] = True

        i = j

    return pd.Series(out, index=desired.index, name="L_ACTIVATED")


def apply_limit_action(
    df: pd.DataFrame,
    activation: pd.Series,
    lcap: pd.Series,
    *,
    intensity: float = 1.0,
) -> pd.DataFrame:
    """Applies a stylized limit action to proxies after activation."""
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
    cap_score_raw: float
    cap_score_enforced: float
    act_score_raw: float
    act_score_enforced: float
    act_drop_under_stress: float
    notes: Dict[str, float]


def assess_maturity(df: pd.DataFrame, *, df_stressed: pd.DataFrame | None = None) -> MaturityAssessment:
    """Typologie simple: Mature, Immature, Dissonant."""
    _require(df, L_CAP_PROXIES, "L_cap")
    _require(df, L_ACT_PROXIES, "L_act")

    cap_raw = float(np.mean(df.loc[:, list(L_CAP_PROXIES)].astype(float).to_numpy()))
    act_raw = float(np.mean(df.loc[:, list(L_ACT_PROXIES)].astype(float).to_numpy()))

    # enforcement factor: penalize turnover above 5%
    turnover_mean = float(np.mean(df["control_turnover"].astype(float).to_numpy())) if "control_turnover" in df.columns else 0.0
    enforcement_factor = float(max(0.0, 1.0 - max(0.0, turnover_mean - 0.05) / 0.25))

    cap_enf = cap_raw * enforcement_factor
    act_enf = (1.0 - act_raw) * enforcement_factor  # higher is better (less capture)

    act_drop = 0.0
    if df_stressed is not None:
        try:
            act_s = float(np.mean(df_stressed.loc[:, list(L_ACT_PROXIES)].astype(float).to_numpy()))
            act_drop = float(max(0.0, act_s - act_raw))
        except Exception:
            act_drop = 0.0

    label = "Mature"
    if cap_enf < 0.45:
        label = "Immature"
    elif (cap_enf >= 0.45) and (act_enf < 0.45):
        label = "Dissonant"

    return MaturityAssessment(
        label=label,
        cap_score_raw=float(cap_raw),
        cap_score_enforced=float(cap_enf),
        act_score_raw=float(1.0 - act_raw),
        act_score_enforced=float(act_enf),
        act_drop_under_stress=float(act_drop),
        notes={
            "turnover_mean": float(turnover_mean),
            "enforcement_factor": float(enforcement_factor),
        },
    )


def evaluate_l_performance(
    df: pd.DataFrame,
    *,
    window: int = 5,
    thresholds: Thresholds | Mapping[str, Any] | None = None,
    risk_threshold: float | None = None,
    o_threshold: float | None = None,
    topk_frac: float = 0.10,
    persist: int = 3,
    max_delay: int = 12,
    intensity: float = 1.0,
) -> Dict[str, Any]:
    """Evaluates L(t) performance with reproducible tests, baseline-aware."""

    thr = thresholds if isinstance(thresholds, Thresholds) else _thresholds_from_mapping(thresholds)
    risk = _risk_series(df, window=window, thresholds=thr)

    if risk_threshold is None:
        if thr is not None:
            risk_threshold = float(thr.risk_thr)
        else:
            k = max(1, int(len(df) * float(topk_frac)))
            risk_threshold = float(np.partition(risk.to_numpy(dtype=float), -k)[-k])

    lact = compute_l_act(df)
    lcap = compute_l_cap(df)

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
    risk_limited = _risk_series(df_limited, window=window, thresholds=thr)

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
    delay_steps: int | None = int(first_a - first_d) if (first_d >= 0 and first_a >= 0) else None

    drop: float | None = None
    if np.any(a):
        idxs = np.where(a)[0]
        t0 = int(idxs[0])
        pre = risk.to_numpy(dtype=float)[max(0, t0 - 10):t0]
        post = risk_limited.to_numpy(dtype=float)[t0:min(len(risk), t0 + 10)]
        if len(pre) > 0 and len(post) > 0:
            drop = float(np.mean(pre) - np.mean(post))

    # E reduction proxy (end-of-series stock reduction)
    e_base = compute_e(df).to_numpy(dtype=float)
    e_lim = compute_e(df_limited).to_numpy(dtype=float)
    e_base_end = float(e_base[-1]) if len(e_base) else 0.0
    e_lim_end = float(e_lim[-1]) if len(e_lim) else 0.0
    e_reduction_rel = float(max(0.0, e_base_end - e_lim_end) / (abs(e_base_end) + eps)) if abs(e_base_end) > eps else 0.0

    # Recovery rate proxy after first activation
    recovery_rate: float | None = None
    if np.any(a):
        t0 = int(np.where(a)[0][0])
        post = risk_limited.to_numpy(dtype=float)[t0:]
        if len(post) > 0:
            recovery_rate = float(np.mean(post < float(risk_threshold)))

    cap01 = float(np.mean(_to_unit_interval(lcap.to_numpy(dtype=float))))
    act01 = float(np.mean(_to_unit_interval(lact.to_numpy(dtype=float))))

    verdict = "Mature"
    if cap01 < 0.45:
        verdict = "Immature"
    elif (cap01 >= 0.45) and (act01 < 0.45):
        verdict = "Dissonant"

    return {
        "risk_threshold": float(risk_threshold),
        "exceedance_base": exceed_base,
        "exceedance_limited": exceed_limited,
        "prevented_exceedance": prevented,
        "prevented_exceedance_rel": prevented_rel,
        "prevented_topk_excess_rel": prevented_topk_excess_rel,
        "topk_mean_excess_base": mean_excess_base,
        "topk_mean_excess_limited": mean_excess_limited,
        "first_activation_delay_steps": delay_steps,
        "activation_delay_steps": delay_steps,
        "risk_drop_around_activation": drop,
        "E_reduction_rel": float(e_reduction_rel),
        "recovery_rate_post_stress": recovery_rate,
        "mean_l_cap_unit": cap01,
        "mean_l_act_unit": act01,
        "verdict": verdict,
        "params": {
            "window": int(window),
            "topk_frac": float(topk_frac),
            "o_threshold": (float(o_threshold) if o_threshold is not None else None),
            "persist": int(persist),
            "max_delay": int(max_delay),
            "intensity": float(intensity),
        },
    }
