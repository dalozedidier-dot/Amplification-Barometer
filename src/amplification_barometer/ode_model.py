
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class BarometerParams:
    # growth and coupling
    a_p: float = 0.60
    a_o: float = 0.45
    a_e: float = 0.30
    a_r: float = 0.35
    a_g: float = 0.25

    k_p: float = 2.5
    k_o: float = 2.5
    k_e: float = 3.0
    k_r: float = 2.5

    # couplings (non-linear)
    b_po: float = 0.35  # O reduces P
    b_op: float = 0.20  # P increases O demand/pressure
    b_pe: float = 0.08  # P amplifies E
    b_oe: float = 0.12  # E degrades O
    b_er: float = 0.25  # R mitigates E
    b_re: float = 0.18  # E reduces R
    b_eg: float = 0.22  # G mitigates E
    b_rg: float = 0.15  # G helps R
    b_go: float = 0.22  # O supports G
    b_ge: float = 0.10  # E erodes G

    g_target: float = 0.85  # long-run governance target

    noise_scale: float = 0.0  # optional additive noise per derivative


def simulate_minimal_po(
    *,
    t: np.ndarray,
    p0: float = 1.0,
    o0: float = 1.0,
    a_p: float = 0.6,
    a_o: float = 0.45,
    b_po: float = 0.35,
) -> pd.DataFrame:
    t = np.asarray(t, dtype=float)

    def f(_t: float, y: np.ndarray) -> np.ndarray:
        p, o = y
        dp = a_p * p * (1.0 - p / 2.5) - b_po * o * p
        do = a_o * o * (1.0 - o / 2.5) + 0.2 * p - 0.1 * o
        return np.array([dp, do], dtype=float)

    sol = solve_ivp(f, t_span=(float(t[0]), float(t[-1])), y0=[p0, o0], t_eval=t, method="RK45")
    out = pd.DataFrame({"P": sol.y[0], "O": sol.y[1]}, index=np.arange(len(t)))
    return out


def _poisson_shocks(
    t: np.ndarray,
    *,
    rate: float,
    magnitude: float,
    seed: int = 7,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = float(np.mean(np.diff(t))) if len(t) > 1 else 1.0
    p = 1.0 - np.exp(-rate * dt)
    events = rng.random(len(t)) < p
    shocks = np.zeros(len(t), dtype=float)
    shocks[events] = magnitude * rng.normal(1.0, 0.15, size=int(np.sum(events)))
    return shocks


def simulate_barometer_ode(
    *,
    t: np.ndarray,
    params: BarometerParams = BarometerParams(),
    y0: Tuple[float, float, float, float, float] = (1.0, 1.0, 0.6, 1.0, 0.8),
    shocks: bool = False,
    shock_rate: float = 0.05,
    shock_magnitude: float = 0.6,
    seed: int = 7,
    adjacency: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """5D non-linear barometer ODE: P, O, E, R, G.

    shocks: if True, Poisson shocks applied on P and E channels.
    adjacency: optional NxN matrix to model network propagation on E (demo).
    """
    t = np.asarray(t, dtype=float)

    shock_p = _poisson_shocks(t, rate=shock_rate, magnitude=shock_magnitude, seed=seed) if shocks else np.zeros(len(t), dtype=float)
    shock_e = _poisson_shocks(t, rate=shock_rate, magnitude=shock_magnitude, seed=seed + 1) if shocks else np.zeros(len(t), dtype=float)

    # network effect on E: scalar approximation (mean-field)
    net_gain = 0.0
    if adjacency is not None:
        adj = np.asarray(adjacency, dtype=float)
        if adj.ndim == 2 and adj.shape[0] == adj.shape[1] and adj.size > 0:
            net_gain = float(np.clip(np.mean(adj), 0.0, 1.0)) * 0.15

    def f(tt: float, y: np.ndarray) -> np.ndarray:
        p, o, e, r, g = y
        # interpolate shocks
        idx = int(np.clip(np.searchsorted(t, tt), 0, len(t) - 1))
        sp = float(shock_p[idx])
        se = float(shock_e[idx])

        dp = params.a_p * p * (1.0 - p / params.k_p) - params.b_po * o * p + params.b_pe * p * e + sp
        do = params.a_o * o * (1.0 - o / params.k_o) + params.b_op * p - params.b_oe * e * o + params.b_go * g * (1.0 - o)
        de = params.a_e * p + net_gain * e - params.b_er * r * e - params.b_eg * g * e + se
        dr = params.a_r * r * (1.0 - r / params.k_r) - params.b_re * e * r + params.b_rg * g * (1.0 - r)
        dg = params.a_g * (params.g_target - g) + params.b_go * o * (1.0 - g) - params.b_ge * e * g

        if params.noise_scale > 0.0:
            rng = np.random.default_rng(seed + idx)
            noise = rng.normal(0.0, params.noise_scale, size=5)
            dp, do, de, dr, dg = (dp + noise[0], do + noise[1], de + noise[2], dr + noise[3], dg + noise[4])

        return np.array([dp, do, de, dr, dg], dtype=float)

    sol = solve_ivp(
        f,
        t_span=(float(t[0]), float(t[-1])),
        y0=list(y0),
        t_eval=t,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )

    y = sol.y
    out = pd.DataFrame(
        {
            "P": y[0],
            "O": y[1],
            "E": y[2],
            "R": y[3],
            "G": y[4],
        },
        index=np.arange(len(t)),
    )
    return out


def simulate_endogenous_g(
    *,
    t: np.ndarray,
    params: BarometerParams = BarometerParams(),
    seed: int = 7,
) -> pd.DataFrame:
    """Backward compatible helper returning only P,O,E,R with endogenous G folded in."""
    df = simulate_barometer_ode(t=np.asarray(t, dtype=float), params=params, seed=seed, shocks=False)
    return df[["P", "O", "E", "R"]].copy()
