from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

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
    """L_cap: capacité intrinsèque d'arrêt, en score robuste.

    Démonstrateur:
    - agrégat sur proxys d'arrêt et d'exécution
    - z-score robuste pour obtenir une série comparable inter datasets
    """
    _require(df, L_CAP_PROXIES, "L_cap")
    arr = df.loc[:, list(L_CAP_PROXIES)].astype(float).to_numpy()
    raw = np.mean(arr, axis=1)
    z = robust_zscore(raw)
    return pd.Series(z, index=df.index, name="L_CAP")


def compute_l_act(df: pd.DataFrame) -> pd.Series:
    """L_act: activation effective, en score robuste.

    Démonstrateur:
    - agrégat sur exemptions, délai de sanction et signaux de capture
    - z-score robuste pour obtenir une série comparable inter datasets
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


def _risk_series(df: pd.DataFrame, *, window: int = 5) -> pd.Series:
    at = compute_at(df).to_numpy(dtype=float)
    dd = compute_delta_d(df, window=window).to_numpy(dtype=float)
    risk = robust_zscore(at) + robust_zscore(dd)
    return pd.Series(risk, index=df.index, name="RISK")


def _sigmoid01(z: np.ndarray, *, k: float = 1.35) -> np.ndarray:
    """Map z-scores to [0,1] with a sigmoid.

    Motivation:
    - avoid the "glass ceiling" created by 0.5 + 0.25*z
    - keep 0.50 at z=0
    - still bounded and monotonic
    """
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-float(k) * z))


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
    """Desired activation when risk is high OR orientation is low.

    Implements a proactive trigger:
    - risk_threshold captures quantitative risk exceedance
    - o_threshold captures degraded orientation (weak stop/steer)
    - persistence reduces false positives
    """
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
    min_on: int = 2,
) -> pd.Series:
    """Realize activation based on desired signal and L_act availability."""
    max_delay = max(0, int(max_delay))
    min_on = max(1, int(min_on))

    d = desired.to_numpy(dtype=bool)
    a = lact.to_numpy(dtype=float)
    a01 = _sigmoid01(a)

    n = len(d)
    out = np.zeros(n, dtype=bool)

    i = 0
    while i < n:
        if not d[i]:
            i += 1
            continue

        # find first index in [i, i+max_delay] where lact is "available"
        j_max = min(n - 1, i + max_delay)
        j = i
        found = False
        while j <= j_max:
            if a01[j] >= 0.50:
                found = True
                break
            j += 1

        if not found:
            i = j_max + 1
            continue

        # once on, stay on for at least min_on
        end = min(n, j + min_on)
        out[j:end] = True
        i = end

    return pd.Series(out, index=desired.index, name="L_ACTIVE")


def apply_l_operator(
    df: pd.DataFrame,
    *,
    activation: pd.Series,
    lcap: pd.Series,
    intensity: float = 1.0,
) -> pd.DataFrame:
    """Apply L operator to a dataset.

    This is a demonstrator:
    - if activated, apply a damping on the amplification proxy @(t)
    - strength is proportional to mean L_cap in unit space and stress intensity
    """
    out = df.copy()
    act = activation.to_numpy(dtype=bool)
    cap01 = _sigmoid01(lcap.to_numpy(dtype=float))
    strength = float(np.nanmean(cap01)) * float(intensity)

    if not np.isfinite(strength):
        strength = 0.0
    strength = float(np.clip(strength, 0.0, 1.0))

    if "stop_proxy" in out.columns:
        sp = out["stop_proxy"].astype(float).to_numpy()
        sp2 = sp.copy()
        sp2[act] = sp2[act] * (1.0 + 0.50 * strength)
        out["stop_proxy"] = sp2

    return out


@dataclass(frozen=True)
class MaturityAssessment:
    label: str
    cap_score_raw: float
    cap_score_enforced: float
    act_score_raw: float
    act_score_enforced: float
    act_drop_under_stress: float
    notes: Dict[str, Any]


def _recovery_p90(df: pd.DataFrame) -> float:
    if "recovery_time_proxy" not in df.columns:
        return 0.0
    x = df["recovery_time_proxy"].astype(float).to_numpy()
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    return float(np.percentile(x, 90))


def _recovery_quality(recovery_p90: float) -> float:
    if recovery_p90 <= 0.0 or not np.isfinite(recovery_p90):
        return 0.50
    q = 1.0 / (1.0 + float(recovery_p90))
    return float(np.clip(q, 0.0, 1.0))


def assess_maturity(df: pd.DataFrame, *, df_stressed: pd.DataFrame | None = None) -> MaturityAssessment:
    """Typologie: Mature, Immature, Dissonant.

    Version v0.4.5:
    - L_cap en unité utilise une sigmoïde pour éviter le plafonnement près de 0.50
    - Bonus gouvernance uniquement si turnover ET gap sont sous cibles
    - Bonus recovery basé sur p90 de recovery_time_proxy, centré pour ne pas gonfler IA_borg
    - Enforcement factor plus tolérant: turnover cible 10%
    """
    _require(df, L_CAP_PROXIES, "L_cap")
    _require(df, L_ACT_PROXIES, "L_act")

    # L_cap base: mean in unit space, via sigmoid mapping of robust z-series
    lcap_z = compute_l_cap(df).to_numpy(dtype=float)
    cap01_base = float(np.mean(_sigmoid01(lcap_z)))

    # Governance gates
    ct = df["control_turnover"].astype(float).to_numpy()
    gap = df["rule_execution_gap"].astype(float).to_numpy()
    turnover_mean = float(np.mean(ct))
    rule_gap_mean = float(np.mean(gap))

    gov_ok = (turnover_mean <= 0.05) and (rule_gap_mean <= 0.05)
    gov_bonus = 0.25 if gov_ok else 0.0
    cap01_gov = cap01_base + gov_bonus * (1.0 - cap01_base)

    # Recovery bonus, only above neutral quality to avoid inflating all datasets
    rec_p90 = _recovery_p90(df)
    rec_q = _recovery_quality(rec_p90)
    rec_bonus = 0.0
    if rec_q > 0.50:
        rec_bonus = 0.12 * (rec_q - 0.50) / 0.50
    cap01_v2 = float(np.clip(cap01_gov + rec_bonus, 0.0, 1.0))

    # L_act raw and enforced
    ex = df["exemption_rate"].astype(float).to_numpy()
    sd_norm = np.clip(df["sanction_delay"].astype(float).to_numpy() / 365.0, 0.0, 1.0)
    ci = df["conflict_interest_proxy"].astype(float).to_numpy()
    act_raw = 1.0 - (0.25 * ex + 0.20 * sd_norm + 0.20 * ct + 0.15 * ci + 0.20 * gap)
    act_score_raw = float(np.mean(act_raw))
    act_score_enforced = act_score_raw

    # Enforcement factor: tolerant turnover target
    turnover_target = 0.10
    ramp = 0.20
    if turnover_mean <= turnover_target:
        enforcement_factor = 1.0
    else:
        enforcement_factor = float(max(0.0, 1.0 - (turnover_mean - turnover_target) / ramp))

    cap_score_raw = float(cap01_base)
    cap_score_enforced = float(np.clip(cap01_v2 * enforcement_factor, 0.0, 1.0))

    act_drop = 0.0
    if df_stressed is not None:
        _require(df_stressed, L_ACT_PROXIES, "L_act stressed")
        ex_s = df_stressed["exemption_rate"].astype(float).to_numpy()
        sd_s = np.clip(df_stressed["sanction_delay"].astype(float).to_numpy() / 365.0, 0.0, 1.0)
        ct_s = df_stressed["control_turnover"].astype(float).to_numpy()
        ci_s = df_stressed["conflict_interest_proxy"].astype(float).to_numpy()
        gap_s = df_stressed["rule_execution_gap"].astype(float).to_numpy()
        act_raw_s = 1.0 - (0.25 * ex_s + 0.20 * sd_s + 0.20 * ct_s + 0.15 * ci_s + 0.20 * gap_s)
        act_drop = float(np.mean(act_raw) - np.mean(act_raw_s))

    # Maturity thresholds (v0.4.5)
    cap_high_thr = 0.60
    act_high_thr = 0.70
    cap_high = cap_score_enforced >= cap_high_thr
    act_high = act_score_enforced >= act_high_thr
    label = "Mature" if (cap_high and act_high) else ("Immature" if not cap_high else "Dissonant")

    notes: Dict[str, Any] = {
        "cap_high_threshold": float(cap_high_thr),
        "act_high_threshold": float(act_high_thr),
        "turnover_mean": float(turnover_mean),
        "rule_execution_gap_mean": float(rule_gap_mean),
        "gov_ok": bool(gov_ok),
        "gov_bonus": float(gov_bonus),
        "recovery_p90": float(rec_p90),
        "recovery_quality": float(rec_q),
        "recovery_bonus": float(rec_bonus),
        "turnover_target": float(turnover_target),
        "enforcement_factor": float(enforcement_factor),
        "cap01_base": float(cap01_base),
        "cap01_v2": float(cap01_v2),
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
    thresholds: Thresholds | None = None,
    risk_threshold: float | None = None,
    o_threshold: float | None = None,
    topk_frac: float = 0.10,
    persist: int = 3,
    max_delay: int = 12,
    intensity: float = 1.0,
) -> Dict[str, Any]:
    """Evaluates L(t) performance with reproducible, non-sensitive tests.

    Compatibility:
    - Some pipelines pass thresholds=<stable baseline thresholds>.
      This function accepts it, and uses baseline risk if provided.
    """
    df0 = df.copy()

    if thresholds is not None:
        risk = risk_signature(df0, thresholds=thresholds, window=window)
        if risk_threshold is None:
            risk_threshold = float(thresholds.risk_thr)
    else:
        risk = _risk_series(df0, window=window)
        if risk_threshold is None:
            risk_threshold = float(np.quantile(risk.to_numpy(dtype=float), 0.95))

    # desired activation
    if o_threshold is not None:
        o_level = compute_o_level(df0)
        desired = desired_activation_proactive(
            risk,
            o_level=o_level,
            risk_threshold=float(risk_threshold),
            o_threshold=float(o_threshold),
            persist=persist,
        )
    else:
        desired = desired_activation(risk, threshold=float(risk_threshold), persist=persist)

    lcap = compute_l_cap(df0)
    lact = compute_l_act(df0)

    active = realize_activation(desired, lact, max_delay=max_delay, min_on=2)
    df_post = apply_l_operator(df0, activation=active, lcap=lcap, intensity=float(intensity))

    # compute prevention metrics: fraction of exceedances prevented
    r0 = risk.to_numpy(dtype=float)
    if thresholds is not None:
        r1 = risk_signature(df_post, thresholds=thresholds, window=window).to_numpy(dtype=float)
    else:
        r1 = _risk_series(df_post, window=window).to_numpy(dtype=float)

    thr = float(risk_threshold)
    pre_ex = r0 > thr
    post_ex = r1 > thr
    prevented_exceedance = 0.0
    if float(np.mean(pre_ex)) > 1e-12:
        prevented_exceedance = float(np.mean(pre_ex) - np.mean(post_ex))
        prevented_exceedance = float(np.clip(prevented_exceedance / float(np.mean(pre_ex)), 0.0, 1.0))

    # top-k prevention
    topk_frac = float(np.clip(float(topk_frac), 0.01, 0.50))
    k = max(1, int(round(len(r0) * topk_frac)))
    idx0 = np.argsort(r0)[-k:]
    idx1 = np.argsort(r1)[-k:]
    topk_overlap = float(len(set(idx0.tolist()).intersection(set(idx1.tolist()))) / float(k))

    # E reduction proxy: compare mean risk above threshold (tail intensity)
    pre_tail = r0[pre_ex]
    post_tail = r1[post_ex]
    e_reduction_rel = 0.0
    if len(pre_tail) > 0:
        pre_m = float(np.mean(pre_tail))
        post_m = float(np.mean(post_tail)) if len(post_tail) > 0 else 0.0
        if pre_m > 1e-12:
            e_reduction_rel = float(np.clip((pre_m - post_m) / pre_m, 0.0, 1.0))

    maturity_assessment = assess_maturity(df0)
    verdict = maturity_assessment.label

    cap01 = float(np.mean(_sigmoid01(lcap.to_numpy(dtype=float))))
    act01 = float(np.mean(_sigmoid01(lact.to_numpy(dtype=float))))

    return {
        "prevented_exceedance": float(prevented_exceedance),
        "prevented_exceedance_rel": float(prevented_exceedance),
        "topk_overlap": float(topk_overlap),
        "prevented_topk_excess_rel": float(np.clip(1.0 - float(topk_overlap), 0.0, 1.0)),
        "e_reduction_rel": float(e_reduction_rel),
        "risk_threshold": float(thr),
        "cap01": float(cap01),
        "act01": float(act01),
        "verdict": str(verdict),
        "maturity": maturity_assessment.notes,
    }
