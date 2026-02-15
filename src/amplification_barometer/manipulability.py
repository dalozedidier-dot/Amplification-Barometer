
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .proxy_protocol import DEFAULT_PROTOCOL, PROXY_PROTOCOL, ProxyProtocol


def _mad(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    med = float(np.nanmedian(arr))
    return float(np.nanmedian(np.abs(arr - med)) + 1e-12)


def _as_mapping(protocol: ProxyProtocol | Mapping[str, Mapping[str, Any]]) -> Mapping[str, Mapping[str, Any]]:
    if isinstance(protocol, ProxyProtocol):
        return protocol.as_mapping()
    return protocol


def _defaults(protocol: ProxyProtocol | Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    if isinstance(protocol, ProxyProtocol):
        return dict(protocol.falsification_defaults)
    return {"jump_mad_mult": 8.0, "shift_mad_mult": 1.8, "range_violation_rate": 0.01, "clamp_quantile_width": 0.10}


def validate_proxy_ranges(
    df: pd.DataFrame,
    *,
    protocol: ProxyProtocol | Mapping[str, Mapping[str, Any]] = DEFAULT_PROTOCOL,
) -> Dict[str, Dict[str, float]]:
    """Basic measurability checks: presence, missingness, range compliance."""
    proto = _as_mapping(protocol)
    out: Dict[str, Dict[str, float]] = {}
    for proxy, meta in proto.items():
        if proxy not in df.columns:
            out[proxy] = {"present": 0.0, "missing_rate": 1.0, "out_of_range_rate": 1.0}
            continue
        s = pd.to_numeric(df[proxy], errors="coerce")
        miss = float(s.isna().mean())
        lo, hi = meta.get("expected_range", (0.0, 1.0))
        lo = float(lo)
        hi = float(hi)
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
      - clamp: clamps values to a narrow quantile band to reduce apparent volatility
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


# Targeting O(t): gaming orientation proxies to lower AT
O_PROXIES: Tuple[str, ...] = ("stop_proxy", "threshold_proxy", "decision_proxy", "execution_proxy", "coherence_proxy")


def inject_bias_o(
    df: pd.DataFrame,
    *,
    magnitude: float = 0.15,
    start_frac: float = 0.5,
    clamp_volatility: bool = False,
    seed: int = 7,
) -> pd.DataFrame:
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
    protocol: ProxyProtocol | Mapping[str, Mapping[str, Any]] = DEFAULT_PROTOCOL,
    jump_mad_mult: Optional[float] = None,
    shift_mad_mult: Optional[float] = None,
) -> DetectionResult:
    proto = _as_mapping(protocol)
    defaults = _defaults(protocol)
    jump_mad_mult = float(jump_mad_mult if jump_mad_mult is not None else defaults["jump_mad_mult"])
    shift_mad_mult = float(shift_mad_mult if shift_mad_mult is not None else defaults["shift_mad_mult"])

    if proxy not in df.columns:
        return DetectionResult(detected=False, out_of_range_rate=1.0, jump_rate=0.0, shift_score=0.0, notes={"reason": -1.0})

    s = pd.to_numeric(df[proxy], errors="coerce")
    arr = s.to_numpy(dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.nanmedian(arr))

    meta = proto.get(proxy, {})
    lo, hi = meta.get("expected_range", (float(np.nanmin(arr)), float(np.nanmax(arr))))
    lo = float(lo)
    hi = float(hi)

    oor = float(np.mean((arr < lo) | (arr > hi)))

    dif = np.diff(arr)
    mad_d = _mad(dif) if dif.size else 1.0
    jump_rate = float(np.mean(np.abs(dif) > jump_mad_mult * mad_d)) if dif.size else 0.0

    # shift: compare second half median to first half median
    mid = len(arr) // 2
    med1 = float(np.nanmedian(arr[:mid])) if mid > 0 else float(np.nanmedian(arr))
    med2 = float(np.nanmedian(arr[mid:])) if mid > 0 else float(np.nanmedian(arr))
    mad = _mad(arr)
    shift_score = float(abs(med2 - med1) / (mad + 1e-12))

    detected = bool((oor > defaults["range_violation_rate"]) or (jump_rate > 0.02) or (shift_score >= shift_mad_mult))

    return DetectionResult(
        detected=detected,
        out_of_range_rate=oor,
        jump_rate=jump_rate,
        shift_score=shift_score,
        notes={"lo": lo, "hi": hi, "mad": float(mad), "mid": float(mid)},
    )


def run_manipulability_suite(
    df: pd.DataFrame,
    *,
    protocol: ProxyProtocol | Mapping[str, Mapping[str, Any]] = DEFAULT_PROTOCOL,
    proxies: Sequence[str] = ("exemption_rate", "rule_execution_gap"),
) -> Dict[str, Any]:
    range_stats = validate_proxy_ranges(df, protocol=protocol)
    det: Dict[str, Any] = {}
    for p in proxies:
        det[p] = detect_falsification(df, proxy=p, protocol=protocol).__dict__
    return {"range_stats": range_stats, "detections": det}
