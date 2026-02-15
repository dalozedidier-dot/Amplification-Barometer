from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# Version des pondérations et conventions (auditabilité)
WEIGHTS_VERSION = "v0.1.0"


@dataclass(frozen=True)
class CompositeSpec:
    name: str
    proxies: Sequence[str]
    weights: np.ndarray
    invert: bool = False  # utile pour G(t) où certains proxys sont "sens risque"


def _require_columns(df: pd.DataFrame, cols: Iterable[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes pour {context}: {missing}")


def robust_zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalisation robuste: (x - médiane) / (1.4826 * MAD)

    Retourne 0 si la dispersion est nulle.
    """
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        return np.zeros_like(x, dtype=float)
    return (x - med) / scale


def _weighted_average(df: pd.DataFrame, proxies: Sequence[str], weights: np.ndarray) -> np.ndarray:
    arr = df.loc[:, proxies].astype(float).to_numpy()
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != arr.shape[1]:
        raise ValueError("Nombre de poids incompatible avec le nombre de proxys")
    if not np.isclose(w.sum(), 1.0):
        w = w / w.sum()
    return np.average(arr, weights=w, axis=1)


def compute_composite(df: pd.DataFrame, spec: CompositeSpec) -> pd.Series:
    """Construit un composite auditables via proxys + pondérations versionnées."""
    _require_columns(df, spec.proxies, spec.name)
    raw = _weighted_average(df, spec.proxies, spec.weights)
    if spec.invert:
        raw = -raw
    z = robust_zscore(raw)
    return pd.Series(z, index=df.index, name=spec.name)


# Spécifications par défaut (calibrables, mais fixées pour la démonstration)
P_SPEC = CompositeSpec(
    name="P",
    proxies=["scale_proxy", "speed_proxy", "leverage_proxy", "autonomy_proxy", "replicability_proxy"],
    weights=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
)
O_SPEC = CompositeSpec(
    name="O",
    proxies=["stop_proxy", "threshold_proxy", "decision_proxy", "execution_proxy", "coherence_proxy"],
    weights=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
)
E_SPEC = CompositeSpec(
    name="E",
    proxies=["impact_proxy", "propagation_proxy", "hysteresis_proxy"],
    weights=np.array([0.4, 0.3, 0.3]),
)
R_SPEC = CompositeSpec(
    name="R",
    proxies=["margin_proxy", "redundancy_proxy", "diversity_proxy", "recovery_time_proxy"],
    weights=np.array([0.25, 0.25, 0.25, 0.25]),
    invert=True,  # recovery_time_proxy est un coût, mais ici on traite l'agrégat comme "sens risque"
)
G_SPEC = CompositeSpec(
    name="G",
    proxies=["exemption_rate", "sanction_delay", "control_turnover", "conflict_interest_proxy"],
    weights=np.array([0.3, 0.25, 0.25, 0.2]),
    invert=True,  # sens risque: plus ces proxys montent, plus G doit baisser
)


def compute_p(df: pd.DataFrame) -> pd.Series:
    """Puissance P(t): échelle, vitesse, levier, autonomie, réplicabilité."""
    return compute_composite(df, P_SPEC)


def compute_o(df: pd.DataFrame) -> pd.Series:
    """Orientation O(t): arrêt, seuils, décision, exécution, cohérence."""
    return compute_composite(df, O_SPEC)


def compute_e(df: pd.DataFrame) -> pd.Series:
    """Externalités E(t): traité comme stock/quasi-stock.

    Pour la démo: on construit un flux composite puis on cumule.
    """
    flow = compute_composite(df, E_SPEC).to_numpy()
    stock = np.cumsum(flow)
    z = robust_zscore(stock)
    return pd.Series(z, index=df.index, name="E")


def compute_r(df: pd.DataFrame) -> pd.Series:
    """Résilience R(t): marges, redondances, diversité, temps de récupération (proxy)."
    """
    return compute_composite(df, R_SPEC)


def compute_g(df: pd.DataFrame) -> pd.Series:
    """Stabilité narrative G(t): composite auditables (conformité, sanctions, intégrité)."""
    return compute_composite(df, G_SPEC)


def compute_at(df: pd.DataFrame, eps: float = 1e-8) -> pd.Series:
    """@(t) = P(t) / O(t) avec garde-fou numérique."""
    p = compute_p(df)
    o = compute_o(df)
    at = p / (o + eps)
    at.name = "AT"
    return at


def compute_delta_d(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Δd(t) = dP/dt - dO/dt, dérivées approximées via lissage + différences finies."""
    if window < 1:
        raise ValueError("window doit être >= 1")
    p = compute_p(df).rolling(window=window, min_periods=1).mean()
    o = compute_o(df).rolling(window=window, min_periods=1).mean()
    dp = p.diff().fillna(0.0)
    do = o.diff().fillna(0.0)
    dd = dp - do
    dd.name = "DELTA_D"
    return dd
