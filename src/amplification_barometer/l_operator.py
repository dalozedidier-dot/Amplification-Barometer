from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .composites import compute_at, compute_delta_d, robust_zscore


def _require(df: pd.DataFrame, cols: Tuple[str, ...], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes pour {context}: {missing}")


L_CAP_PROXIES: Tuple[str, ...] = ("stop_proxy", "threshold_proxy", "execution_proxy", "coherence_proxy")
L_ACT_PROXIES: Tuple[str, ...] = ("exemption_rate", "sanction_delay", "control_turnover", "conflict_interest_proxy")


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

    risk = 0.30 * ex + 0.25 * sd_norm + 0.25 * ct + 0.20 * ci
    raw = 1.0 - risk
    z = robust_zscore(raw)
    return pd.Series(z, index=df.index, name="L_ACT")


def _risk_series(df: pd.DataFrame, *, window: int = 5) -> pd.Series:
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
    # rolling sum over the last `persist` points
    roll = np.convolve(above, np.ones(persist, dtype=int), mode="same")
    desired = roll >= persist
    return pd.Series(desired, index=risk.index, name="L_DESIRED")


def realize_activation(
    desired: pd.Series,
    lact: pd.Series,
    *,
    max_delay: int = 12,
    min_on: int = 3,
) -> pd.Series:
    """Realizes activation with delay that depends on L_act.

    - If L_act is low, delay increases.
    - min_on enforces a minimal persistence once activated.
    """
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

        # start of a desired episode
        j = i
        while j < n and d[j]:
            j += 1

        # delay is higher when L_act is low
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
    """Applies a stylized limit action to proxies after activation.

    This does not claim realism. It is a minimal demonstrator that allows
    performance tests without sensitive data.
    """
    out = df.copy()
    act = activation.to_numpy(dtype=bool)
    cap01 = _to_unit_interval(lcap.to_numpy(dtype=float))
    strength = float(np.nanmean(cap01)) * float(intensity)

    if not np.any(act):
        return out

    idx = out.index[act]

    # Reduce amplification proxies when L is active
    for c in ("scale_proxy", "speed_proxy", "leverage_proxy", "autonomy_proxy", "replicability_proxy"):
        if c in out.columns:
            out.loc[idx, c] = out.loc[idx, c].astype(float) * (1.0 - 0.15 * strength)

    # Increase orientation and stop proxies when L is active
    for c in ("stop_proxy", "threshold_proxy", "decision_proxy", "execution_proxy", "coherence_proxy"):
        if c in out.columns:
            out.loc[idx, c] = out.loc[idx, c].astype(float) * (1.0 + 0.10 * strength)

    # Reduce externalities proxies modestly (mitigation)
    for c in ("impact_proxy", "propagation_proxy", "hysteresis_proxy"):
        if c in out.columns:
            out.loc[idx, c] = out.loc[idx, c].astype(float) * (1.0 - 0.08 * strength)

    # Improve resilience proxies modestly (margins, redundancy, diversity)
    for c in ("margin_proxy", "redundancy_proxy", "diversity_proxy"):
        if c in out.columns:
            out.loc[idx, c] = out.loc[idx, c].astype(float) * (1.0 + 0.08 * strength)

    # Recovery time should go down under effective limit actions
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
    """Typologie simple: Mature, Immature, Dissonant.

    Principe d'audit:
    - L_cap mesure une capacité intrinsèque sur proxys techniques (arrêt, seuils, exécution, cohérence).
    - L_act mesure l'activation effective sous contraintes institutionnelles (exemptions, délais, turnover, conflits).
    - Pour éviter la circularité, on applique une règle d'enforcement explicite:
      si control_turnover moyen dépasse 5%, la capacité est considérée comme moins fiable.

    Cette règle matérialise le point du document: "viser L_cap > 0.95 via enforcement"
    en rendant l'objectif conditionnel à une stabilité organisationnelle minimale.
    """
    _require(df, L_CAP_PROXIES, "L_cap")
    _require(df, L_ACT_PROXIES, "L_act")

    cap_raw = np.mean(df.loc[:, list(L_CAP_PROXIES)].astype(float).to_numpy(), axis=1)
    cap_score_raw = float(np.mean(cap_raw))

    ex = df["exemption_rate"].astype(float).to_numpy()
    sd_norm = np.clip(df["sanction_delay"].astype(float).to_numpy() / 365.0, 0.0, 1.0)
    ct = df["control_turnover"].astype(float).to_numpy()
    ci = df["conflict_interest_proxy"].astype(float).to_numpy()
    act_raw = 1.0 - (0.30 * ex + 0.25 * sd_norm + 0.25 * ct + 0.20 * ci)
    act_score_raw = float(np.mean(act_raw))
    act_score_enforced = act_score_raw

    turnover_mean = float(np.mean(ct))
    turnover_target = 0.05

    if turnover_mean <= turnover_target:
        enforcement_factor = 1.0
    else:
        # degrade smoothly: at +15% turnover above target, factor reaches 0
        enforcement_factor = float(max(0.0, 1.0 - (turnover_mean - turnover_target) / 0.15))

    cap_score_enforced = float(cap_score_raw * enforcement_factor)

    act_drop = 0.0
    if df_stressed is not None:
        _require(df_stressed, L_ACT_PROXIES, "L_act stressed")
        ex_s = df_stressed["exemption_rate"].astype(float).to_numpy()
        sd_s = np.clip(df_stressed["sanction_delay"].astype(float).to_numpy() / 365.0, 0.0, 1.0)
        ct_s = df_stressed["control_turnover"].astype(float).to_numpy()
        ci_s = df_stressed["conflict_interest_proxy"].astype(float).to_numpy()
        act_raw_s = 1.0 - (0.30 * ex_s + 0.25 * sd_s + 0.25 * ct_s + 0.20 * ci_s)
        act_drop = float(np.mean(act_raw) - np.mean(act_raw_s))

    cap_high = cap_score_enforced >= 0.95
    act_high = act_score_enforced >= 0.70
    label = "Mature" if (cap_high and act_high) else ("Immature" if not cap_high else "Dissonant")

    notes = {
        "cap_high_threshold": 0.95,
        "act_high_threshold": 0.70,
        "turnover_mean": float(turnover_mean),
        "turnover_target": float(turnover_target),
        "enforcement_factor": float(enforcement_factor),
    }
    return MaturityAssessment(
        label=label,
        cap_score_raw=cap_score_raw,
        cap_score_enforced=cap_score_enforced,
        act_score_raw=act_score_raw,
        act_score_enforced=act_score_enforced,
        act_drop_under_stress=act_drop,
        notes=notes,
    )


def evaluate_l_performance(
    df: pd.DataFrame,
    *,
    window: int = 5,
    risk_threshold: float | None = None,
    topk_frac: float = 0.10,
    persist: int = 3,
    max_delay: int = 12,
    intensity: float = 1.0,
) -> Dict[str, Any]:
    """Evaluates L(t) performance with reproducible, non-sensitive tests.

    Performance definition (demonstrator):
    - Desired activation is defined by persistent exceedance of a risk threshold.
    - L_act controls delay (capture/exemptions reduce effective activation).
    - L_cap controls effect magnitude once activated.
    - We measure reduction of exceedance and risk after activation, plus delay.
    """
    risk = _risk_series(df, window=window)
    if risk_threshold is None:
        k = max(1, int(len(df) * float(topk_frac)))
        risk_threshold = float(np.partition(risk.to_numpy(dtype=float), -k)[-k])

    lact = compute_l_act(df)
    lcap = compute_l_cap(df)

    desired = desired_activation(risk, threshold=float(risk_threshold), persist=persist)
    activated = realize_activation(desired, lact, max_delay=max_delay)

    df_limited = apply_limit_action(df, activated, lcap, intensity=intensity)
    risk_limited = _risk_series(df_limited, window=window)

    base_mask = risk.to_numpy(dtype=float) > float(risk_threshold)
    lim_mask = risk_limited.to_numpy(dtype=float) > float(risk_threshold)

    exceed_base = float(np.mean(base_mask))
    exceed_limited = float(np.mean(lim_mask))
    prevented = float(max(0.0, exceed_base - exceed_limited))

    # delay estimate: first desired episode start vs first activation
    d = desired.to_numpy(dtype=bool)
    a = activated.to_numpy(dtype=bool)
    first_d = int(np.argmax(d)) if np.any(d) else -1
    first_a = int(np.argmax(a)) if np.any(a) else -1
    delay = float(first_a - first_d) if (first_d >= 0 and first_a >= 0) else float("nan")

    # risk drop around activation: compare mean risk pre/post (window of 10 points)
    drop = float("nan")
    if np.any(a):
        idxs = np.where(a)[0]
        t0 = int(idxs[0])
        pre = risk.to_numpy(dtype=float)[max(0, t0 - 10):t0]
        post = risk_limited.to_numpy(dtype=float)[t0:min(len(risk), t0 + 10)]
        if len(pre) > 0 and len(post) > 0:
            drop = float(np.mean(pre) - np.mean(post))

    # simple verdict
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
        "first_activation_delay_steps": delay,
        "risk_drop_around_activation": drop,
        "mean_l_cap_unit": cap01,
        "mean_l_act_unit": act01,
        "verdict": verdict,
        "params": {
            "window": int(window),
            "topk_frac": float(topk_frac),
            "persist": int(persist),
            "max_delay": int(max_delay),
            "intensity": float(intensity),
        },
    }
