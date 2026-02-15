from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .proxy_protocol import PROXY_PROTOCOL


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-12)


def validate_proxy_ranges(df: pd.DataFrame, *, protocol: Mapping[str, Mapping[str, Any]] = PROXY_PROTOCOL) -> Dict[str, Dict[str, float]]:
    """Checks basic measurability constraints: range compliance and missingness.

    Returns per-proxy stats that are easy to audit and aggregate.
    """
    out: Dict[str, Dict[str, float]] = {}
    for proxy, meta in protocol.items():
        if proxy not in df.columns:
            out[proxy] = {"present": 0.0, "missing_rate": 1.0, "out_of_range_rate": 1.0}
            continue
        s = pd.to_numeric(df[proxy], errors="coerce")
        miss = float(s.isna().mean())
        lo, hi = meta["expected_range"]
        oor = float(((s < lo) | (s > hi)).mean())
        out[proxy] = {
            "present": 1.0,
            "missing_rate": miss,
            "out_of_range_rate": oor,
            "min": float(np.nanmin(s.to_numpy(dtype=float))),
            "max": float(np.nanmax(s.to_numpy(dtype=float))),
        }
    return out


def inject_falsification(
    df: pd.DataFrame,
    *,
    proxy: str,
    kind: str = "shift",
    magnitude: float = 0.2,
    start_frac: float = 0.5,
    seed: int = 7,
) -> pd.DataFrame:
    """Simulates a simple falsification attempt on a proxy.

    kind:
    - shift: adds a constant delta from start point
    - spike: injects rare large spikes after start
    - clamp: clamps values to a narrow range to reduce apparent volatility
    """
    if proxy not in df.columns:
        raise ValueError(f"Proxy absent: {proxy}")
    out = df.copy()
    s = pd.to_numeric(out[proxy], errors="coerce").to_numpy(dtype=float, copy=True)
    n = len(s)
    start = int(n * float(start_frac))
    rng = np.random.default_rng(seed)

    if kind == "shift":
        s[start:] = s[start:] + float(magnitude)
    elif kind == "spike":
        # 3 spikes on average
        k = max(1, n // 40)
        idx = rng.choice(np.arange(start, n), size=k, replace=False)
        s[idx] = s[idx] + float(magnitude) * 5.0
    elif kind == "clamp":
        lo = np.quantile(s[:start], 0.45)
        hi = np.quantile(s[:start], 0.55)
        s[start:] = np.clip(s[start:], lo, hi)
    else:
        raise ValueError(f"Unknown falsification kind: {kind}")

    out[proxy] = s
    return out



# Ciblage O(t): manipulation de l'orientation pour faire baisser @(t)
O_PROXIES: Tuple[str, ...] = ("stop_proxy", "threshold_proxy", "decision_proxy", "execution_proxy", "coherence_proxy")


def inject_bias_o(
    df: pd.DataFrame,
    *,
    magnitude: float = 0.15,
    start_frac: float = 0.5,
    clamp_volatility: bool = False,
    seed: int = 7,
) -> pd.DataFrame:
    """Injecte un biais artificiel sur les proxys de O(t).

    But: simuler une tentative de gaming où l'acteur "fait monter" les indicateurs
    d'orientation (arrêt, seuils, exécution) pour faire baisser @(t).

    Options:
    - clamp_volatility: réduit artificiellement la variance après start (cas de "reporting" lissé)
    """
    out = df.copy()
    n = len(out)
    start = int(n * float(start_frac))
    rng = np.random.default_rng(seed)

    for proxy in O_PROXIES:
        if proxy not in out.columns:
            continue
        s = pd.to_numeric(out[proxy], errors="coerce").to_numpy(dtype=float, copy=True)

        if clamp_volatility:
            base = float(np.nanmedian(s[:start]))
            noise = rng.normal(0.0, 0.02 * (np.nanstd(s[:start]) + 1e-12), size=n - start)
            s[start:] = base + noise

        # gonflement "positif" (conserve l'ordre de grandeur)
        s[start:] = s[start:] * (1.0 + float(magnitude))

        out[proxy] = s

    return out

@dataclass(frozen=True)
class DetectionResult:
    detected: bool
    out_of_range_rate: float
    jump_rate: float
    shift_score: float
    notes: Dict[str, float]


def detect_falsification(
    df: pd.DataFrame,
    *,
    proxy: str,
    protocol: Mapping[str, Mapping[str, Any]] = PROXY_PROTOCOL,
    jump_mad_mult: Optional[float] = None,
    shift_mad_mult: float = 6.0,
) -> DetectionResult:
    """Detects basic falsification patterns with transparent heuristics.

    This is a demonstrator. Real deployments should replace these heuristics
    with sector-specific detectors and independent data sources.
    """
    if proxy not in protocol:
        raise ValueError(f"Unknown proxy in protocol: {proxy}")
    meta = protocol[proxy]
    if proxy not in df.columns:
        return DetectionResult(detected=True, out_of_range_rate=1.0, jump_rate=1.0, shift_score=1e9, notes={"missing": 1.0})

    s = pd.to_numeric(df[proxy], errors="coerce").to_numpy(dtype=float, copy=False)
    lo, hi = meta["expected_range"]
    oor = float(np.mean((s < lo) | (s > hi)))

    ds = np.diff(s)
    mad_ds = _mad(ds)
    if jump_mad_mult is None:
        jump_mad_mult = float(meta.get("falsification_flags", {}).get("jump_mad_mult", 8.0))
    jump_rate = float(np.mean(np.abs(ds) > jump_mad_mult * mad_ds))

    # shift detection: compare medians pre/post mid point
    mid = len(s) // 2
    pre = s[:mid]
    post = s[mid:]
    mad_pre = _mad(pre)
    shift = float(np.median(post) - np.median(pre))
    shift_score = float(np.abs(shift) / mad_pre)

    range_flag = oor > float(meta.get("falsification_flags", {}).get("range_violation_rate", 0.01))
    jump_flag = jump_rate > 0.02
    shift_flag = shift_score > float(shift_mad_mult)

    detected = bool(range_flag or jump_flag or shift_flag)

    notes = {
        "range_flag": float(range_flag),
        "jump_flag": float(jump_flag),
        "shift_flag": float(shift_flag),
        "jump_mad_mult": float(jump_mad_mult),
        "shift_mad_mult": float(shift_mad_mult),
    }
    return DetectionResult(detected=detected, out_of_range_rate=oor, jump_rate=jump_rate, shift_score=shift_score, notes=notes)


def run_manipulability_suite(
    df: pd.DataFrame,
    *,
    proxies: Optional[Sequence[str]] = None,
    kinds: Sequence[str] = ("shift", "spike", "clamp"),
    magnitude: float = 0.2,
    seed: int = 7,
) -> Dict[str, Any]:
    """Runs a reproducible anti-gaming suite over proxies.

    For each proxy and falsification kind:
    - inject falsification
    - run detection on the modified series
    The output is JSON-friendly.
    """
    if proxies is None:
        proxies = list(PROXY_PROTOCOL.keys())

    out: Dict[str, Any] = {
        "range_checks": validate_proxy_ranges(df),
        "falsification": {},
    }

    for proxy in proxies:
        if proxy not in df.columns:
            continue
        out["falsification"][proxy] = {}
        for kind in kinds:
            df_f = inject_falsification(df, proxy=proxy, kind=kind, magnitude=magnitude, seed=seed)
            det = detect_falsification(df_f, proxy=proxy)
            out["falsification"][proxy][kind] = {
                "detected": bool(det.detected),
                "out_of_range_rate": float(det.out_of_range_rate),
                "jump_rate": float(det.jump_rate),
                "shift_score": float(det.shift_score),
                "notes": {k: float(v) for k, v in det.notes.items()},
            }

    # summary: detection rate across all injected scenarios
    flags = []
    for proxy, d in out["falsification"].items():
        for kind, res in d.items():
            flags.append(bool(res.get("detected")))
    out["summary"] = {
        "n_scenarios": int(len(flags)),
        "detected_rate": float(np.mean(flags) if flags else 0.0),
    }
    return out
