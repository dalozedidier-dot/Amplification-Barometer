from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .composites import robust_zscore


def _require(df: pd.DataFrame, cols: Tuple[str, ...], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes pour {context}: {missing}")


L_CAP_PROXIES: Tuple[str, ...] = ("stop_proxy", "threshold_proxy", "execution_proxy", "coherence_proxy")
L_ACT_PROXIES: Tuple[str, ...] = ("exemption_rate", "sanction_delay", "control_turnover", "conflict_interest_proxy")


def compute_l_cap(df: pd.DataFrame) -> pd.Series:
    """L_cap: capacité d'arrêt testée (proxy).

    Démo: agrégat sur proxys d'arrêt et d'exécution.
    """
    _require(df, L_CAP_PROXIES, "L_cap")
    arr = df.loc[:, list(L_CAP_PROXIES)].astype(float).to_numpy()
    raw = np.mean(arr, axis=1)
    z = robust_zscore(raw)
    return pd.Series(z, index=df.index, name="L_CAP")


def compute_l_act(df: pd.DataFrame) -> pd.Series:
    """L_act: activation effective (proxy).

    Démo: agrégat sur proxys d'exemptions, délais de sanction et signaux de capture.
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


@dataclass(frozen=True)
class MaturityAssessment:
    label: str
    cap_score: float
    act_score: float
    act_drop_under_stress: float
    notes: Dict[str, float]


def assess_maturity(df: pd.DataFrame, *, df_stressed: pd.DataFrame | None = None) -> MaturityAssessment:
    """Typologie simple: Mature, Immature, Dissonant.

    Règle de démo:
    - cap_score basé sur moyenne brute des proxys L_cap
    - act_score basé sur 1 - risque (exemptions, délais, capture)
    - si df_stressed est fourni, on mesure la chute d'activation.
    """
    _require(df, L_CAP_PROXIES, "L_cap")
    _require(df, L_ACT_PROXIES, "L_act")

    cap_raw = np.mean(df.loc[:, list(L_CAP_PROXIES)].astype(float).to_numpy(), axis=1)
    cap_score = float(np.mean(cap_raw))

    ex = df["exemption_rate"].astype(float).to_numpy()
    sd_norm = np.clip(df["sanction_delay"].astype(float).to_numpy() / 365.0, 0.0, 1.0)
    ct = df["control_turnover"].astype(float).to_numpy()
    ci = df["conflict_interest_proxy"].astype(float).to_numpy()
    act_raw = 1.0 - (0.30 * ex + 0.25 * sd_norm + 0.25 * ct + 0.20 * ci)
    act_score = float(np.mean(act_raw))

    act_drop = 0.0
    if df_stressed is not None:
        _require(df_stressed, L_ACT_PROXIES, "L_act stressed")
        ex_s = df_stressed["exemption_rate"].astype(float).to_numpy()
        sd_s = np.clip(df_stressed["sanction_delay"].astype(float).to_numpy() / 365.0, 0.0, 1.0)
        ct_s = df_stressed["control_turnover"].astype(float).to_numpy()
        ci_s = df_stressed["conflict_interest_proxy"].astype(float).to_numpy()
        act_raw_s = 1.0 - (0.30 * ex_s + 0.25 * sd_s + 0.25 * ct_s + 0.20 * ci_s)
        act_drop = float(np.mean(act_raw) - np.mean(act_raw_s))

    cap_high = cap_score >= 0.95
    act_high = act_score >= 0.70
    label = "Mature" if (cap_high and act_high) else ("Immature" if not cap_high else "Dissonant")

    notes = {
        "cap_high_threshold": 0.95,
        "act_high_threshold": 0.70,
    }
    return MaturityAssessment(label=label, cap_score=cap_score, act_score=act_score, act_drop_under_stress=act_drop, notes=notes)
