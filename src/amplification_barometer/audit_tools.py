from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .composites import (
    compute_at,
    compute_delta_d,
    compute_e,
    compute_g,
    compute_o,
    compute_p,
    compute_r,
    robust_zscore,
)


@dataclass(frozen=True)
class StressResult:
    status: str
    degradation: float
    details: Dict[str, float]


def run_stress_test(
    df: pd.DataFrame,
    *,
    shock_magnitude: float = 2.0,
    shock_col: str = "scale_proxy",
    shock_start_frac: float = 0.5,
    window: int = 5,
) -> StressResult:
    """Stress test reproductible (démo).

    Injecte un choc exogène sur un proxy de P, recalcule @(t), Δd, E, R, G et mesure la dégradation.
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


def _rank_consistency(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Mesure simple de stabilité de classement (Spearman approx via corrélation sur rangs)."""
    ra = pd.Series(scores_a).rank().to_numpy()
    rb = pd.Series(scores_b).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 1.0
    return float(np.corrcoef(ra, rb)[0, 1])


def audit_score_stability(
    df: pd.DataFrame,
    *,
    windows: Sequence[int] = (3, 5, 8),
    noise_eps: float = 0.02,
    seed: int = 7,
) -> Dict[str, float]:
    """Audit de stabilité des signatures.

    Idée: si de petites variations de fenêtre ou de bruit inversent les classements des risques,
    la signature est instable et ne doit pas piloter une décision automatique.

    Retourne:
    - rank_consistency_at: moyenne corrélation sur rangs entre variantes
    - rank_consistency_delta_d
    - sensitivity_at: amplitude typique (écart-type) sous petites perturbations
    """
    rng = np.random.default_rng(seed)

    at_base = compute_at(df).to_numpy()
    dd_base = compute_delta_d(df, window=int(windows[0])).to_numpy()

    at_variants: List[np.ndarray] = []
    dd_variants: List[np.ndarray] = []

    # variantes de fenêtres + bruit léger sur les proxys (anti-gaming basique)
    for w in windows:
        dff = df.copy()
        numeric_cols = [c for c in dff.columns if pd.api.types.is_numeric_dtype(dff[c])]
        noise = rng.normal(0.0, noise_eps, size=(len(dff), len(numeric_cols)))
        dff.loc[:, numeric_cols] = dff.loc[:, numeric_cols].astype(float).to_numpy() * (1.0 + noise)
        at_variants.append(compute_at(dff).to_numpy())
        dd_variants.append(compute_delta_d(dff, window=int(w)).to_numpy())

    at_cons = float(np.mean([_rank_consistency(at_base, v) for v in at_variants]))
    dd_cons = float(np.mean([_rank_consistency(dd_base, v) for v in dd_variants]))

    sens_at = float(np.mean([np.std(v - at_base) for v in at_variants]))
    sens_dd = float(np.mean([np.std(v - dd_base) for v in dd_variants]))

    return {
        "rank_consistency_at": at_cons,
        "rank_consistency_delta_d": dd_cons,
        "sensitivity_at": sens_at,
        "sensitivity_delta_d": sens_dd,
    }
