from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .calibration import Thresholds, derive_thresholds, risk_signature
from .composites import compute_at, compute_delta_d, compute_e_metrics, compute_r_metrics, compute_o_level
from .manipulability import detect_falsification, inject_bias_o


@dataclass(frozen=True)
class StressResult:
    status: str
    degradation: float
    details: Dict[str, float]


def _risk(df: pd.DataFrame, *, window: int, thresholds: Thresholds | None = None) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if thresholds is None:
        thresholds = derive_thresholds(df, window=window)
    at = compute_at(df)
    dd = compute_delta_d(df, window=window)
    r = risk_signature(df, thresholds=thresholds, window=window, at_series=at, dd_series=dd)
    return at, dd, r


def _safe_std(x: np.ndarray) -> float:
    s = float(np.std(x))
    return s if s > 1e-12 else 1.0


def _stress_start_idx(n: int, frac: float) -> int:
    return int(max(0, min(n - 1, round(n * float(frac)))))


def run_stress_test(
    df: pd.DataFrame,
    *,
    shock_magnitude: float = 2.0,
    shock_col: str = "scale_proxy",
    shock_start_frac: float = 0.5,
    window: int = 5,
    thresholds: Thresholds | None = None,
) -> StressResult:
    """
    Stress test aligné signatures Type I/II/III.

    Mesure:
    - Dégradation du risque (risk_signature) normalisée
    - Persistance dE/dt (ratio d'amplitude après choc)
    - Récupération R (pas jusqu'au retour proche de la médiane de base)
    - Saturation O (fraction sous p10 base après choc)
    - Divergence @(t) et Δd(t) (normée par l'écart type base)
    - Irréversibilité E (delta de la métrique d'irréversibilité)
    """
    if shock_col not in df.columns:
        raise ValueError(f"Colonne {shock_col} absente")

    local_thr = thresholds if thresholds is not None else derive_thresholds(df, window=window)

    at0, dd0, risk0 = _risk(df, window=window, thresholds=local_thr)
    base_std = _safe_std(risk0.to_numpy(dtype=float))

    n = len(df)
    start = _stress_start_idx(n, shock_start_frac)

    stressed = df.copy()
    stressed.loc[stressed.index[start:], shock_col] = stressed.loc[stressed.index[start:], shock_col].astype(float) + float(shock_magnitude)

    at1, dd1, risk1 = _risk(stressed, window=window, thresholds=local_thr)

    degradation = float(np.mean(np.abs(risk1.to_numpy(dtype=float) - risk0.to_numpy(dtype=float))) / base_std)
    status = "Résilient" if degradation <= 1.5 else "Instable sous stress"

    # E metrics
    e0 = compute_e_metrics(df)
    e1 = compute_e_metrics(stressed)
    de_ratio = float(np.mean(np.abs(e1["dE_dt"].to_numpy(dtype=float)[start:])) / (np.mean(np.abs(e0["dE_dt"].to_numpy(dtype=float)[start:])) + 1e-12))
    e_irrev_delta = float(np.mean(e1["E_irreversibility"].to_numpy(dtype=float)) - np.mean(e0["E_irreversibility"].to_numpy(dtype=float)))

    # R recovery
    r0 = compute_r_metrics(df)["R_level"].to_numpy(dtype=float)
    r1 = compute_r_metrics(stressed)["R_level"].to_numpy(dtype=float)
    base_med = float(np.median(r0))
    target = 0.95 * base_med
    rec_steps = float("nan")
    for i in range(start, n):
        if r1[i] >= target:
            rec_steps = float(i - start)
            break

    # O saturation
    o0 = compute_o_level(df).to_numpy(dtype=float)
    o1 = compute_o_level(stressed).to_numpy(dtype=float)
    o_p10 = float(np.percentile(o0, 10))
    o_sat_frac = float(np.mean(o1[start:] <= o_p10))

    # Divergence AT and Δd
    at_div = float(np.mean(np.abs(at1.to_numpy(dtype=float)[start:] - at0.to_numpy(dtype=float)[start:])) / _safe_std(at0.to_numpy(dtype=float)))
    dd_div = float(np.mean(np.abs(dd1.to_numpy(dtype=float)[start:] - dd0.to_numpy(dtype=float)[start:])) / _safe_std(dd0.to_numpy(dtype=float)))

    details = {
        "std_risk_base": float(np.std(risk0.to_numpy(dtype=float))),
        "std_risk_stressed": float(np.std(risk1.to_numpy(dtype=float))),
        "mean_abs_delta_risk": float(np.mean(np.abs(risk1.to_numpy(dtype=float) - risk0.to_numpy(dtype=float)))),
        "shock_col": 0.0,
        "dE_dt_persist_ratio": de_ratio,
        "E_irreversibility_delta": e_irrev_delta,
        "R_recovery_steps": float(rec_steps),
        "O_saturation_frac": o_sat_frac,
        "AT_divergence_norm": at_div,
        "DELTA_D_divergence_norm": dd_div,
        "stress_start_index": float(start),
    }
    return StressResult(status=status, degradation=degradation, details=details)


def run_stress_suite(
    df: pd.DataFrame,
    *,
    window: int = 5,
    thresholds: Thresholds | None = None,
) -> Dict[str, Any]:
    """Suite alignée Type I/II/III."""
    res: Dict[str, Any] = {}

    # Type I: bruit ou surcharge
    res["Overload"] = run_stress_test(
        df,
        shock_magnitude=2.0,
        shock_col="scale_proxy" if "scale_proxy" in df.columns else df.columns[0],
        shock_start_frac=0.55,
        window=window,
        thresholds=thresholds,
    ).__dict__

    # Type II: oscillations
    osc = df.copy()
    if "speed_proxy" in osc.columns:
        n = len(osc)
        t = np.linspace(0.0, 2.0 * np.pi, n)
        osc["speed_proxy"] = pd.to_numeric(osc["speed_proxy"], errors="coerce").to_numpy(dtype=float) + 0.35 * np.sin(6.0 * t)
        shock_col = "speed_proxy"
    else:
        shock_col = "scale_proxy" if "scale_proxy" in osc.columns else osc.columns[0]
    res["Oscillation"] = run_stress_test(
        osc,
        shock_magnitude=0.0,
        shock_col=shock_col,
        shock_start_frac=0.0,
        window=window,
        thresholds=thresholds,
    ).__dict__

    # Type III: bifurcation
    bif = df.copy()
    if "leverage_proxy" in bif.columns:
        start = int(len(bif) * 0.5)
        ramp = np.linspace(0.0, 2.0, len(bif) - start)
        bif.loc[bif.index[start:], "leverage_proxy"] = pd.to_numeric(bif.loc[bif.index[start:], "leverage_proxy"], errors="coerce").to_numpy(dtype=float) + ramp
        shock_col = "leverage_proxy"
    else:
        shock_col = "scale_proxy" if "scale_proxy" in bif.columns else bif.columns[0]
    res["Bifurcation"] = run_stress_test(
        bif,
        shock_magnitude=0.0,
        shock_col=shock_col,
        shock_start_frac=0.0,
        window=window,
        thresholds=thresholds,
    ).__dict__

    return res


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = _rank(a)
    rb = _rank(b)
    ra = ra - float(np.mean(ra))
    rb = rb - float(np.mean(rb))
    denom = float(np.linalg.norm(ra) * np.linalg.norm(rb)) + 1e-12
    return float(np.dot(ra, rb) / denom)


def _topk_idx(x: np.ndarray, k: int) -> set[int]:
    if k <= 0:
        return set()
    idx = np.argpartition(x, -k)[-k:]
    return {int(i) for i in idx.tolist()}


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
        risks[w] = r.to_numpy(dtype=float)

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

    delta_mean = float(np.mean(r1.to_numpy(dtype=float) - r0.to_numpy(dtype=float)))
    red_flag = bool(delta_mean < -0.25 * float(np.std(r0.to_numpy(dtype=float)) + 1e-9))

    detections = detect_falsification(biased, cols=[c for c in biased.columns if c.endswith("_proxy")])

    return {
        "delta_risk_mean": float(delta_mean),
        "red_flag": red_flag,
        "detections": detections,
    }
