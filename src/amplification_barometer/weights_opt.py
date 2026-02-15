
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class WeightOptimizationResult:
    weights: Dict[str, List[float]]
    objective: float
    details: Dict[str, float]


def _project_simplex(w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0.0, None)
    s = float(np.sum(w))
    if s < eps:
        return np.ones_like(w) / float(len(w))
    return w / s


def optimize_weights_simplex(
    *,
    df_stable: pd.DataFrame,
    compute_score: Callable[[pd.DataFrame, Mapping[str, Sequence[float]]], pd.Series],
    proxy_groups: Mapping[str, Sequence[str]],
    initial: Mapping[str, Sequence[float]] | None = None,
    windows: Sequence[int] = (3, 5, 8),
    topk_frac: float = 0.10,
    lam_stability: float = 1.0,
    lam_variance: float = 0.2,
    seed: int = 7,
) -> WeightOptimizationResult:
    """Constrained optimization of weights targeting ranking stability.

    This is a lightweight demonstrator:
    - objective: maximize Spearman stability across windows (converted to minimization)
    - penalty: variance of the score series (avoid degenerate flat solutions)
    """
    rng = np.random.default_rng(seed)

    group_names = list(proxy_groups.keys())
    dims = [len(proxy_groups[g]) for g in group_names]
    offsets = np.cumsum([0] + dims)

    def pack(wmap: Mapping[str, Sequence[float]]) -> np.ndarray:
        vec = []
        for g, d in zip(group_names, dims):
            ww = np.asarray(list(wmap.get(g, rng.random(d))), dtype=float)
            ww = _project_simplex(ww)
            vec.append(ww)
        return np.concatenate(vec)

    def unpack(vec: np.ndarray) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        for i, g in enumerate(group_names):
            sl = vec[offsets[i]:offsets[i+1]]
            out[g] = _project_simplex(sl).tolist()
        return out

    x0 = pack(initial or {})

    def stability_objective(vec: np.ndarray) -> float:
        wmap = unpack(vec)
        scores_by_w: List[np.ndarray] = []
        for w in windows:
            s = compute_score(df_stable, wmap)
            scores_by_w.append(s.to_numpy(dtype=float))
        ref = scores_by_w[0]
        # Spearman (fallback to Pearson on ranks)
        ref_rank = ref.argsort().argsort().astype(float)
        corrs = []
        for arr in scores_by_w[1:]:
            r_rank = arr.argsort().argsort().astype(float)
            denom = np.std(ref_rank) * np.std(r_rank)
            corr = 0.0 if denom <= 1e-12 else float(np.cov(ref_rank, r_rank)[0, 1] / denom)
            corrs.append(abs(corr))
        spearman_mean = float(np.mean(corrs)) if corrs else 1.0
        var_pen = float(np.var(ref))
        # minimize
        return -lam_stability * spearman_mean + lam_variance * (1.0 / (var_pen + 1e-9))

    res = minimize(stability_objective, x0, method="Nelder-Mead", options={"maxiter": 400, "xatol": 1e-5, "fatol": 1e-5})
    w_best = unpack(res.x)
    obj = float(res.fun)

    return WeightOptimizationResult(
        weights=w_best,
        objective=obj,
        details={"success": float(bool(res.success)), "nfev": float(res.nfev), "nit": float(res.nit)},
    )
