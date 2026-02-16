from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# Version des pondérations et conventions (auditabilité)
WEIGHTS_VERSION = "v0.4.3"


@dataclass(frozen=True)
class CompositeSpec:
    name: str
    proxies: Sequence[str]
    weights: np.ndarray
    invert: bool = False


def _require_columns(df: pd.DataFrame, cols: Iterable[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes pour {context}: {missing}")


def robust_zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalisation robuste: (x - médiane) / (1.4826 * MAD)

    Si la dispersion est nulle, retourne un vecteur de zéros.
    """
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        return np.zeros_like(x, dtype=float)
    return (x - med) / scale


def standard_zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalisation classique: (x - moyenne) / écart-type."""
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < eps:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd


def _normalize_columns_robust(df: pd.DataFrame, proxies: Sequence[str]) -> np.ndarray:
    """Normalise chaque proxy séparément via z-score robuste (médiane/MAD).

    Cela rend les proxys comparables même s'ils ont des unités différentes.
    """
    arr = df.loc[:, proxies].astype(float).to_numpy()
    out = np.empty_like(arr, dtype=float)
    for j in range(arr.shape[1]):
        out[:, j] = robust_zscore(arr[:, j])
    return out


def _weighted_average(arr: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != arr.shape[1]:
        raise ValueError("Nombre de poids incompatible avec le nombre de proxys")
    if not np.isclose(w.sum(), 1.0):
        w = w / w.sum()
    return np.average(arr, weights=w, axis=1)


def compute_composite_raw(df: pd.DataFrame, spec: CompositeSpec) -> np.ndarray:
    """Construit un agrégat brut (dimensionless) sur proxys normalisés.

    Retourne un tableau 1D, sans normalisation finale.
    """
    _require_columns(df, spec.proxies, spec.name)
    normed = _normalize_columns_robust(df, spec.proxies)
    raw = _weighted_average(normed, spec.weights)
    if spec.invert:
        raw = -raw
    return raw


def compute_composite(df: pd.DataFrame, spec: CompositeSpec, *, norm: str = "robust") -> pd.Series:
    """Construit un composite auditable via proxys + pondérations versionnées."""
    raw = compute_composite_raw(df, spec)
    if norm == "robust":
        z = robust_zscore(raw)
    elif norm == "standard":
        z = standard_zscore(raw)
    else:
        raise ValueError("norm doit être 'robust' ou 'standard'")
    return pd.Series(z, index=df.index, name=spec.name)


# Spécifications par défaut (calibrables, fixées pour la démonstration)
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
G_SPEC = CompositeSpec(
    name="G",
    proxies=["exemption_rate", "sanction_delay", "control_turnover", "conflict_interest_proxy", "rule_execution_gap"],
    # Le proxy "rule_execution_gap" est un objectif clé (cible < 5%).
    weights=np.array([0.25, 0.20, 0.15, 0.10, 0.30]),
    invert=True,
)



def _override_weights(spec: CompositeSpec, weights: Sequence[float] | None) -> CompositeSpec:
    if weights is None:
        return spec
    w = np.asarray(list(weights), dtype=float)
    if w.shape[0] != len(spec.proxies):
        raise ValueError(f"Override weights for {spec.name} must have length {len(spec.proxies)}")
    if not np.isclose(float(np.sum(w)), 1.0):
        w = w / float(np.sum(w))
    return CompositeSpec(name=spec.name, proxies=spec.proxies, weights=w, invert=spec.invert)

def compute_p(df: pd.DataFrame, *, weights: Sequence[float] | None = None) -> pd.Series:
    """Puissance P(t): échelle, vitesse, levier, autonomie, réplicabilité."""
    return compute_composite(df, _override_weights(P_SPEC, weights), norm="robust")


def compute_o(df: pd.DataFrame, *, weights: Sequence[float] | None = None) -> pd.Series:
    """Orientation O(t): arrêt, seuils, décision, exécution, cohérence."""
    return compute_composite(df, _override_weights(O_SPEC, weights), norm="robust")


def compute_e(df: pd.DataFrame, *, weights: Sequence[float] | None = None) -> pd.Series:
    """Externalités E(t): traité comme stock ou quasi-stock.

    Pour la démo, on calcule un flux composite puis on cumule.
    """
    flow = compute_composite(df, _override_weights(E_SPEC, weights), norm="robust").to_numpy()
    stock = np.cumsum(flow)
    z = robust_zscore(stock)
    return pd.Series(z, index=df.index, name="E")


def compute_r(df: pd.DataFrame, *, weights: Sequence[float] | None = None) -> pd.Series:
    """Résilience R(t): marges, redondances, diversité, temps de récupération.

    Les proxys margin, redundancy, diversity sont "bénéfiques".
    Le proxy recovery_time est un coût (plus grand, plus mauvais), il est donc inversé.
    """
    proxies = ["margin_proxy", "redundancy_proxy", "diversity_proxy", "recovery_time_proxy"]
    _require_columns(df, proxies, "R")

    arr = _normalize_columns_robust(df, proxies)
    # inversion du coût recovery_time_proxy
    arr[:, 3] = -arr[:, 3]

    w = np.array([0.25, 0.25, 0.25, 0.25]) if weights is None else np.asarray(list(weights), dtype=float)
    if w.shape[0] != 4:
        raise ValueError("Override weights for R must have length 4")
    raw = _weighted_average(arr, w)
    z = robust_zscore(raw)
    return pd.Series(z, index=df.index, name="R")


def compute_g(df: pd.DataFrame, *, weights: Sequence[float] | None = None) -> pd.Series:
    """Stabilité narrative G(t): conformité effective, délais de sanction, intégrité contrôle.

    G est construit en "sens risque", puis inversé.
    """
    return compute_composite(df, _override_weights(G_SPEC, weights), norm="robust")



def compute_p_level(df: pd.DataFrame, *, weights: Sequence[float] | None = None) -> pd.Series:
    """P_level(t): moyenne pondérée des proxys bruts de P(t).

    Nota: ceci est un niveau (unités de proxy), pas un z-score.
    On l'utilise pour construire @(t) = P_level / O_level sans ambiguïté de signe.
    """
    _require_columns(df, P_SPEC.proxies, "P_level")
    w = P_SPEC.weights if weights is None else np.asarray(list(weights), dtype=float)
    if w.shape[0] != len(P_SPEC.proxies):
        raise ValueError(f"Override weights for P_level must have length {len(P_SPEC.proxies)}")
    raw = _weighted_average(df.loc[:, list(P_SPEC.proxies)].astype(float).to_numpy(), w)
    return pd.Series(raw, index=df.index, name="P_LEVEL")


def compute_o_level(df: pd.DataFrame, *, weights: Sequence[float] | None = None) -> pd.Series:
    """O_level(t): moyenne pondérée des proxys bruts de O(t)."""
    _require_columns(df, O_SPEC.proxies, "O_level")
    w = O_SPEC.weights if weights is None else np.asarray(list(weights), dtype=float)
    if w.shape[0] != len(O_SPEC.proxies):
        raise ValueError(f"Override weights for O_level must have length {len(O_SPEC.proxies)}")
    raw = _weighted_average(df.loc[:, list(O_SPEC.proxies)].astype(float).to_numpy(), w)
    return pd.Series(raw, index=df.index, name="O_LEVEL")


def compute_at(df: pd.DataFrame, eps: float = 1e-8, *, p_weights: Sequence[float] | None = None, o_weights: Sequence[float] | None = None) -> pd.Series:
    """@(t) = P_level(t) / O_level(t).

    Important pour l'audit:
    - évite l'ambiguïté de signe induite par des z-scores centrés (P ou O négatifs)
    - la normalisation (robust_zscore) se fait ensuite, au niveau du risque ou des seuils
    """
    p = compute_p_level(df, weights=p_weights)
    o = compute_o_level(df, weights=o_weights)
    at = p / (o + eps)
    at.name = "AT"
    return at


def compute_delta_d(df: pd.DataFrame, window: int = 5, *, p_weights: Sequence[float] | None = None, o_weights: Sequence[float] | None = None) -> pd.Series:
    """Δd(t) = dP_level/dt - dO_level/dt via lissage + différences finies."""
    if window < 1:
        raise ValueError("window doit être >= 1")
    p = compute_p_level(df, weights=p_weights).rolling(window=window, min_periods=1).mean()
    o = compute_o_level(df, weights=o_weights).rolling(window=window, min_periods=1).mean()
    dp = p.diff().fillna(0.0)
    do = o.diff().fillna(0.0)
    dd = dp - do
    dd.name = "DELTA_D"
    return dd
