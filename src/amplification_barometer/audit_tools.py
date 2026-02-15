
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .calibration import Thresholds, derive_thresholds, risk_signature
from .composites import compute_at, compute_delta_d
from .manipulability import O_PROXIES, detect_falsification, inject_bias_o


@dataclass(frozen=True)
class StressResult:
    status: str
    degradation: float
    details: Dict[str, float]


def _numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _risk(df: pd.DataFrame, *, window: int, thresholds: Thresholds | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if thresholds is None:
        thresholds = derive_thresholds(df, window=window)
    at = compute_at(df)
    dd = compute_delta_d(df, window=window)
    r = risk_signature(df, thresholds=thresholds, window=window, at_series=at, dd_series=dd)
    return at.to_numpy(dtype=float), dd.to_numpy(dtype=float), r.to_numpy(dtype=float)


def run_stress_test(
    df: pd.DataFrame,
    *,
    shock_magnitude: float = 2.0,
    shock_col: str = "scale_proxy",
    shock_start_frac: float = 0.5,
    window: int = 5,
    thresholds: Thresholds | None = None,
) -> StressResult:
    if shock_col not in df.columns:
        raise ValueError(f"Colonne {shock_col} absente")

    _, _, risk0 = _risk(df, window=window, thresholds=thresholds)
    base_std = float(np.std(risk0)) or 1.0

    stressed = df.copy()
    start = int(len(stressed) * float(shock_start_frac))
    stressed.loc[stressed.index[start:], shock_col] = stressed.loc[stressed.index[start:], shock_col].astype(float) + float(shock_magnitude)

    _, _, risk1 = _risk(stressed, window=window, thresholds=thresholds)

    degradation = float(np.mean(np.abs(risk1 - risk0)) / base_std)
    status = "Résilient" if degradation <= 1.5 else "Instable sous stress"

    details = {
        "std_risk_base": float(np.std(risk0)),
        "std_risk_stressed": float(np.std(risk1)),
        "mean_abs_delta_risk": float(np.mean(np.abs(risk1 - risk0))),
        "shock_col": 0.0,
    }
    return StressResult(status=status, degradation=degradation, details=details)


def run_stress_suite(
    df: pd.DataFrame,
    *,
    window: int = 5,
    thresholds: Thresholds | None = None,
) -> Dict[str, Any]:
    """Suite alignée Type I/II/III (démo)."""
    res: Dict[str, Any] = {}

    # Type I: bruit / surcharge
    res["Overload"] = run_stress_test(df, shock_magnitude=2.0, shock_col="scale_proxy", shock_start_frac=0.55, window=window, thresholds=thresholds).__dict__

    # Type II: oscillations (inject a sinusoidal perturbation)
    osc = df.copy()
    if "speed_proxy" in osc.columns:
        n = len(osc)
        t = np.linspace(0.0, 2.0 * np.pi, n)
        osc["speed_proxy"] = pd.to_numeric(osc["speed_proxy"], errors="coerce").to_numpy(dtype=float) + 0.35 * np.sin(6.0 * t)
    res["Oscillation"] = run_stress_test(osc, shock_magnitude=0.0, shock_col="speed_proxy" if "speed_proxy" in df.columns else "scale_proxy", shock_start_frac=0.0, window=window, thresholds=thresholds).__dict__

    # Type III: bifurcation like ramp
    bif = df.copy()
    if "leverage_proxy" in bif.columns:
        start = int(len(bif) * 0.5)
        ramp = np.linspace(0.0, 2.0, len(bif) - start)
        bif.loc[bif.index[start:], "leverage_proxy"] = pd.to_numeric(bif.loc[bif.index[start:], "leverage_proxy"], errors="coerce").to_numpy(dtype=float) + ramp
    res["Bifurcation"] = run_stress_test(bif, shock_magnitude=0.0, shock_col="leverage_proxy" if "leverage_proxy" in df.columns else "scale_proxy", shock_start_frac=0.0, window=window, thresholds=thresholds).__dict__

    return res


def _rank(x: np.ndarray) -> np.ndarray:
    # stable rank transform
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = _rank(a)
    rb = _rank(b)
    denom = float(np.std(ra) * np.std(rb))
    if denom <= 1e-12:
        return 0.0
    return float(np.cov(ra, rb)[0, 1] / denom)


def _topk_idx(x: np.ndarray, k: int) -> set[int]:
    if k <= 0:
        return set()
    idx = np.argpartition(x, -k)[-k:]
    return set(int(i) for i in idx.tolist())


def audit_score_stability(
    df: pd.DataFrame,
    *,
    windows: Sequence[int] = (3, 5, 8),
    topk_frac: float = 0.10,
) -> Dict[str, Any]:
    """Audit stability of the risk ranking under window changes."""
    windows = [int(w) for w in windows]
    risks: Dict[int, np.ndarray] = {}
    for w in windows:
        thr = derive_thresholds(df, window=w)
        _, _, r = _risk(df, window=w, thresholds=thr)
        risks[w] = r

    ref = risks[windows[0]]
    corrs = []
    jaccs = []

    k = max(1, int(len(ref) * float(topk_frac)))
    ref_top = _topk_idx(ref, k)

    for w in windows[1:]:
        r = risks[w]
        corrs.append(abs(_spearman(ref, r)))
        top = _topk_idx(r, k)
        inter = len(ref_top & top)
        uni = len(ref_top | top)
        jaccs.append(float(inter / uni) if uni else 1.0)

    spearman_mean = float(np.mean(corrs)) if corrs else 1.0
    topk_jaccard_mean = float(np.mean(jaccs)) if jaccs else 1.0
    stable_flag = bool((spearman_mean >= 0.90) and (topk_jaccard_mean >= 0.70))

    return {
        "windows": windows,
        "spearman_mean_risk": spearman_mean,
        "topk_jaccard_mean_risk": topk_jaccard_mean,
        "stable_flag": stable_flag,
    }


def anti_gaming_o_bias(
    df: pd.DataFrame,
    *,
    window: int = 5,
) -> Dict[str, Any]:
    """Checks whether gaming O proxies unduly reduces risk signature."""
    thr = derive_thresholds(df, window=window)
    _, _, r0 = _risk(df, window=window, thresholds=thr)
    biased = inject_bias_o(df, magnitude=0.25, start_frac=0.55, clamp_volatility=True)
    _, _, r1 = _risk(biased, window=window, thresholds=thr)

    delta_mean = float(np.mean(r1) - np.mean(r0))
    # if risk decreases strongly after O gaming, that's a red flag
    red_flag = bool(delta_mean < -0.25)

    det = {}
    for p in O_PROXIES:
        if p in biased.columns:
            det[p] = detect_falsification(biased, proxy=p).__dict__

    return {"delta_risk_mean": delta_mean, "red_flag": red_flag, "detections": det}
