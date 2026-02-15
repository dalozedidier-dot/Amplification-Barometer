from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .composites import compute_at, compute_delta_d, compute_e, compute_g, compute_o, compute_p, compute_r, robust_zscore
from .manipulability import O_PROXIES, detect_falsification, inject_bias_o


@dataclass(frozen=True)
class StressResult:
    status: str
    degradation: float
    details: Dict[str, float]


def _numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _risk_signature(df: pd.DataFrame, *, window: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retourne (AT, DELTA_D, RISK) comme vecteurs numpy."""
    at = compute_at(df).to_numpy()
    dd = compute_delta_d(df, window=window).to_numpy()
    # Signature risque: normaliser AT et Δd séparément puis sommer
    risk = robust_zscore(at) + robust_zscore(dd)
    return at, dd, risk


def run_stress_test(
    df: pd.DataFrame,
    *,
    shock_magnitude: float = 2.0,
    shock_col: str = "scale_proxy",
    shock_start_frac: float = 0.5,
    window: int = 5,
) -> StressResult:
    """Stress test reproductible (démo).

    Injecte un choc exogène sur un proxy de P, recalcule @(t), Δd, E, R, G
    et mesure la dégradation sur @(t).
    """
    if shock_col not in df.columns:
        raise ValueError(f"Colonne {shock_col} absente")

    base_at = compute_at(df)
    base_std = float(np.std(base_at)) or 1.0

    stressed = df.copy()
    start = int(len(stressed) * shock_start_frac)
    stressed.loc[stressed.index[start:], shock_col] = stressed.loc[stressed.index[start:], shock_col].astype(float) + shock_magnitude

    at_s = compute_at(stressed)
    dd_s = compute_delta_d(stressed, window=window)
    e_s = compute_e(stressed)
    r_s = compute_r(stressed)
    g_s = compute_g(stressed)

    degradation = float(np.mean(np.abs(at_s - base_at)) / base_std)
    status = "Résilient" if degradation <= 1.5 else "Instable sous stress"

    details = {
        "std_at_base": float(np.std(base_at)),
        "std_at_stressed": float(np.std(at_s)),
        "mean_abs_delta_at": float(np.mean(np.abs(at_s - base_at))),
        "std_delta_d_stressed": float(np.std(dd_s)),
        "mean_e_stressed": float(np.mean(e_s)),
        "mean_r_stressed": float(np.mean(r_s)),
        "mean_g_stressed": float(np.mean(g_s)),
    }
    return StressResult(status=status, degradation=degradation, details=details)


def _spearman_rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Corrélation sur rangs (approx Spearman)."""
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 1.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _topk_jaccard(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Jaccard entre ensembles Top-K (plus grands)."""
    if k <= 0:
        return 1.0
    ia = set(np.argsort(a)[-k:])
    ib = set(np.argsort(b)[-k:])
    inter = len(ia.intersection(ib))
    union = len(ia.union(ib)) or 1
    return float(inter / union)


def audit_score_stability(
    df: pd.DataFrame,
    *,
    windows: Sequence[int] = (3, 5, 8),
    noise_eps: float = 0.02,
    weight_eps: float = 0.05,
    topk_frac: float = 0.10,
    seed: int = 7,
) -> Dict[str, float]:
    """Audit de stabilité des signatures @(t) et Δd(t).

    Le document demande un test explicite: si de faibles variations de fenêtre,
    de normalisation ou de paramètres inversent le classement des risques,
    le score est déclaré instable.

    Implémentation (démo):
    1) variations de fenêtre sur Δd
    2) bruit multiplicatif léger sur proxys (anti-gaming basique)
    3) perturbation légère des pondérations implicites via bruit sur colonnes numériques

    Mesures:
    - spearman_mean_risk: moyenne corrélation sur rangs entre risque baseline et variantes
    - topk_jaccard_mean_risk: moyenne Jaccard sur Top-K points les plus risqués
    - spearman_worst_risk, topk_jaccard_worst_risk
    - sensitivity_risk: écart-type moyen (risk_variant - risk_base)
    """
    rng = np.random.default_rng(seed)

    _, _, risk_base = _risk_signature(df, window=int(windows[0]))
    k = max(1, int(len(risk_base) * float(topk_frac)))

    spears: List[float] = []
    jaccs: List[float] = []
    sens: List[float] = []

    num_cols = _numeric_cols(df)

    for w in windows:
        dff = df.copy()

        # bruit multiplicatif sur les proxys
        noise = rng.normal(0.0, noise_eps, size=(len(dff), len(num_cols)))
        dff.loc[:, num_cols] = dff.loc[:, num_cols].astype(float).to_numpy() * (1.0 + noise)

        # "weight jitter" simple: rescale aléatoire par colonne
        col_scale = 1.0 + rng.normal(0.0, weight_eps, size=len(num_cols))
        dff.loc[:, num_cols] = dff.loc[:, num_cols].astype(float).to_numpy() * col_scale.reshape(1, -1)

        _, _, risk_v = _risk_signature(dff, window=int(w))

        spears.append(_spearman_rank_corr(risk_base, risk_v))
        jaccs.append(_topk_jaccard(risk_base, risk_v, k=k))
        sens.append(float(np.std(risk_v - risk_base)))

    return {
        "spearman_mean_risk": float(np.mean(spears)),
        "topk_jaccard_mean_risk": float(np.mean(jaccs)),
        "spearman_worst_risk": float(np.min(spears)),
        "topk_jaccard_worst_risk": float(np.min(jaccs)),
        "sensitivity_risk": float(np.mean(sens)),
        "topk_k": float(k),
    }


def _apply_scenario(df: pd.DataFrame, name: str, intensity: float) -> pd.DataFrame:
    """Applique un scénario standardisé de stress test.

    Scénarios inspirés des tests Q2: Shock-P, Automation, Coupling, Lag-O,
    et des tests adversariaux: exception, sanctions, surcharge.
    """
    out = df.copy()

    def add(col: str, delta: float):
        if col in out.columns:
            out[col] = out[col].astype(float) + delta

    def mul(col: str, factor: float):
        if col in out.columns:
            out[col] = out[col].astype(float) * factor

    start = int(len(out) * 0.5)
    idx = out.index[start:]

    if name == "Shock-P":
        for c in ("scale_proxy", "speed_proxy", "leverage_proxy"):
            if c in out.columns:
                out.loc[idx, c] = out.loc[idx, c].astype(float) + intensity
    elif name == "Automation":
        if "autonomy_proxy" in out.columns:
            out.loc[idx, "autonomy_proxy"] = out.loc[idx, "autonomy_proxy"].astype(float) + intensity
    elif name == "Coupling":
        for c in ("propagation_proxy", "replicability_proxy"):
            if c in out.columns:
                out.loc[idx, c] = out.loc[idx, c].astype(float) + intensity
    elif name == "Lag-O":
        # latence et friction d'orientation: baisse exécution/cohérence + hausse délais
        for c in ("execution_proxy", "coherence_proxy", "decision_proxy"):
            if c in out.columns:
                out.loc[idx, c] = out.loc[idx, c].astype(float) - abs(intensity) * 0.5
        if "sanction_delay" in out.columns:
            out.loc[idx, "sanction_delay"] = out.loc[idx, "sanction_delay"].astype(float) + abs(intensity) * 20.0
    elif name == "Exception":
        if "exemption_rate" in out.columns:
            out.loc[idx, "exemption_rate"] = np.clip(out.loc[idx, "exemption_rate"].astype(float) + abs(intensity) * 0.15, 0.0, 1.0)
    elif name == "SanctionDelay":
        if "sanction_delay" in out.columns:
            out.loc[idx, "sanction_delay"] = out.loc[idx, "sanction_delay"].astype(float) + abs(intensity) * 30.0
    elif name == "Capture":
        for c in ("control_turnover", "conflict_interest_proxy"):
            if c in out.columns:
                out.loc[idx, c] = np.clip(out.loc[idx, c].astype(float) + abs(intensity) * 0.10, 0.0, 1.0)
    elif name == "Overload":
        # surcharge organisationnelle: O baisse, exemptions et délais augmentent
        for c in ("execution_proxy", "coherence_proxy", "stop_proxy"):
            if c in out.columns:
                out.loc[idx, c] = out.loc[idx, c].astype(float) - abs(intensity) * 0.6
        if "exemption_rate" in out.columns:
            out.loc[idx, "exemption_rate"] = np.clip(out.loc[idx, "exemption_rate"].astype(float) + abs(intensity) * 0.12, 0.0, 1.0)
        if "sanction_delay" in out.columns:
            out.loc[idx, "sanction_delay"] = out.loc[idx, "sanction_delay"].astype(float) + abs(intensity) * 25.0
    else:
        raise ValueError(f"Scénario inconnu: {name}")

    return out


def run_stress_suite(
    df: pd.DataFrame,
    *,
    intensity: float = 1.0,
    scenarios: Sequence[str] = ("Shock-P", "Automation", "Coupling", "Lag-O", "Exception", "SanctionDelay", "Capture", "Overload"),
    window: int = 5,
) -> Mapping[str, StressResult]:
    """Exécute une suite de stress tests standardisés."""
    results: Dict[str, StressResult] = {}
    base_at = compute_at(df)
    base_std = float(np.std(base_at)) or 1.0

    for sc in scenarios:
        stressed = _apply_scenario(df, sc, intensity)
        at_s = compute_at(stressed)
        degradation = float(np.mean(np.abs(at_s - base_at)) / base_std)
        status = "Résilient" if degradation <= 1.5 else "Instable sous stress"

        _, _, risk_s = _risk_signature(stressed, window=window)

        details = {
            "mean_abs_delta_at": float(np.mean(np.abs(at_s - base_at))),
            "std_at_stressed": float(np.std(at_s)),
            "mean_risk_stressed": float(np.mean(risk_s)),
            "std_risk_stressed": float(np.std(risk_s)),
        }
        results[sc] = StressResult(status=status, degradation=degradation, details=details)

    return results


def anti_gaming_o_bias(
    df: pd.DataFrame,
    *,
    magnitude: float = 0.15,
    start_frac: float = 0.5,
    clamp_volatility: bool = True,
    window: int = 5,
) -> Dict[str, float | str]:
    """Test anti-gaming ciblé O(t).

    1) on calcule une signature risque baseline (robust_zscore(@) + robust_zscore(Δd))
    2) on injecte un biais O (gonflement + option lissage)
    3) on mesure la baisse de risque obtenue
    4) on vérifie qu'au moins un détecteur de falsification sur proxys O déclenche

    Verdict:
    - Resistant si la manipulation est détectée ou si le gain de risque est faible
    - Vulnerable si le gain est grand sans détection
    """
    base_at = compute_at(df).to_numpy(dtype=float)
    base_dd = compute_delta_d(df, window=window).to_numpy(dtype=float)
    base_risk = robust_zscore(base_at) + robust_zscore(base_dd)

    df_biased = inject_bias_o(df, magnitude=magnitude, start_frac=start_frac, clamp_volatility=clamp_volatility)
    bias_at = compute_at(df_biased).to_numpy(dtype=float)
    bias_dd = compute_delta_d(df_biased, window=window).to_numpy(dtype=float)
    bias_risk = robust_zscore(bias_at) + robust_zscore(bias_dd)

    base_mean = float(np.mean(base_risk))
    bias_mean = float(np.mean(bias_risk))
    base_sd = float(np.std(base_risk)) or 1.0

    risk_drop = float(base_mean - bias_mean)
    risk_drop_sd = float(risk_drop / base_sd)

    detected_any = False
    triggered = 0
    available = 0
    for proxy in O_PROXIES:
        if proxy not in df.columns:
            continue
        available += 1
        det = detect_falsification(df_biased, proxy=proxy)
        if det.detected:
            detected_any = True
            triggered += 1

    # heuristic threshold: a drop larger than 0.75 std is considered meaningful
    meaningful_drop = risk_drop_sd >= 0.75

    status = "Resistant"
    if meaningful_drop and not detected_any:
        status = "Vulnerable"

    return {
        "status": status,
        "risk_drop_mean": float(risk_drop),
        "risk_drop_std": float(risk_drop_sd),
        "detected_any": float(1.0 if detected_any else 0.0),
        "o_proxies_available": float(available),
        "o_proxies_triggered": float(triggered),
        "params_magnitude": float(magnitude),
        "params_clamp_volatility": float(1.0 if clamp_volatility else 0.0),
    }
