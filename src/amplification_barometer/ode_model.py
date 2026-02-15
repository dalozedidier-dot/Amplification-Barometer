from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.integrate import odeint


def minimal_po_ode(y, t, alpha: float = 0.10, beta: float = 0.05, gamma: float = 0.05, delta: float = 0.10):
    """Système minimal P/O (démo).

    dP/dt = alpha * P - beta * O
    dO/dt = gamma * P - delta * O
    """
    p, o = y
    dpdt = alpha * p - beta * o
    dodt = gamma * p - delta * o
    return [dpdt, dodt]


def simulate_minimal_po(
    initial_state=(1.0, 1.0),
    t: np.ndarray | None = None,
    *,
    alpha: float = 0.10,
    beta: float = 0.05,
    gamma: float = 0.05,
    delta: float = 0.10,
    shock_u: Optional[Callable[[float], float]] = None,
    tau: int = 0,
) -> pd.DataFrame:
    """Simule le système minimal (P,O) avec extensions simples.

    shock_u: fonction u(t) ajoutée à P (choc exogène)
    tau: retard discret sur O (shift de tau pas)
    """
    if t is None:
        t = np.linspace(0.0, 10.0, 200)
    t = np.asarray(t, dtype=float)

    args = (alpha, beta, gamma, delta)
    sol = odeint(minimal_po_ode, initial_state, t, args=args)

    df = pd.DataFrame(sol, columns=["P", "O"], index=t)

    if shock_u is not None:
        u = np.array([float(shock_u(tt)) for tt in t], dtype=float)
        df["P"] = df["P"] + np.cumsum(u) * (t[1] - t[0])

    if tau > 0:
        tau = int(tau)
        if tau >= len(df):
            raise ValueError("tau trop grand pour la longueur de la série")
        df.loc[df.index[tau:], "O"] = df.loc[df.index[:-tau], "O"].to_numpy()

    return df


@dataclass(frozen=True)
class BarometerParams:
    # équations (5.1) à (5.4) (classe de modèle de démo)
    a: float = 0.8
    b: float = 0.4
    c: float = 0.2
    u: float = 0.6
    v: float = 0.4
    n: float = 0.3
    m: float = 0.2
    alpha: float = 0.6
    beta: float = 0.4
    lam: float = 0.3
    delta: float = 0.4
    gamma: float = 0.5
    xi: float = 0.3


def barometer_ode_4d(x, t, p: BarometerParams, shock_u: float = 0.0):
    """Modèle Baromètre d’amplification 4D de démonstration.

    dP/dt = P * (a - b*O - c*P)
    dO/dt = u - v*P - n*O - m*E
    dE/dt = alpha*P - beta*O - lam*E
    dR/dt = delta*O - gamma*E - xi*R

    shock_u peut servir à perturber P via un terme additif (exogène).
    """
    P, O, E, R = x
    dPdt = P * (p.a - p.b * O - p.c * P) + shock_u
    dOdt = p.u - p.v * P - p.n * O - p.m * E
    dEdt = p.alpha * P - p.beta * O - p.lam * E
    dRdt = p.delta * O - p.gamma * E - p.xi * R
    return [dPdt, dOdt, dEdt, dRdt]


def simulate_barometer_ode(
    initial_state=(0.6, 0.6, 0.0, 0.8),
    t: np.ndarray | None = None,
    *,
    params: BarometerParams = BarometerParams(),
    shock_profile: Optional[Callable[[float], float]] = None,
) -> pd.DataFrame:
    """Simule le modèle Baromètre d’amplification 4D (démo, non jumeau numérique).

    shock_profile: fonction s(t) ajoutée à dP/dt.
    """
    if t is None:
        t = np.linspace(0.0, 40.0, 600)
    t = np.asarray(t, dtype=float)

    if shock_profile is None:
        def shock_profile(_tt: float) -> float:
            return 0.0

    # odeint ne supporte pas directement une fonction s(t) sans wrapper
    def f(x, tt):
        return barometer_ode_4d(x, tt, params, shock_u=float(shock_profile(tt)))

    sol = odeint(f, initial_state, t)
    return pd.DataFrame(sol, columns=["P", "O", "E", "R"], index=t)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


def endogenous_g_ode(x, t, *, alpha0: float, beta: float, gamma: float, delta: float, eta: float, xi: float, g_set: float, shock_u: float = 0.0):
    """Système (P,O,G) non linéaire minimaliste pour illustrer G(t) endogène.

    Intuition (démonstrateur):
    - G augmente vers g_set si le système est "calme" (relaxation).
    - Lorsque P dépasse O durablement (pression d'amplification), G tend à baisser.
    - G module l'accélération de P (boucle culturelle/institutionnelle simplifiée).
    """
    P, O, G = x
    eff_alpha = alpha0 * (1.0 + 0.5 * max(0.0, min(1.0, G)))
    dPdt = eff_alpha * P - beta * O + shock_u
    dOdt = gamma * P - delta * O + 0.10 * (1.0 - G)

    pressure = _sigmoid(P - O)  # dans [0,1]
    dGdt = eta * (g_set - G) - xi * pressure

    return [dPdt, dOdt, dGdt]


def simulate_endogenous_g(
    initial_state=(1.0, 1.0, 0.8),
    t: np.ndarray | None = None,
    *,
    alpha0: float = 0.12,
    beta: float = 0.05,
    gamma: float = 0.06,
    delta: float = 0.10,
    eta: float = 0.15,
    xi: float = 0.35,
    g_set: float = 0.85,
    shock_profile: Optional[Callable[[float], float]] = None,
) -> pd.DataFrame:
    """Simule un modèle (P,O,G) avec G(t) endogène (démo).

    Ce module n'est pas une "preuve". Il sert à générer des régimes et des chocs
    sans données sensibles, en cohérence avec la section "extensions ODE" du document.
    """
    if t is None:
        t = np.linspace(0.0, 40.0, 800)
    t = np.asarray(t, dtype=float)

    if shock_profile is None:
        def shock_profile(_tt: float) -> float:
            return 0.0

    def f(x, tt):
        return endogenous_g_ode(
            x,
            tt,
            alpha0=alpha0,
            beta=beta,
            gamma=gamma,
            delta=delta,
            eta=eta,
            xi=xi,
            g_set=g_set,
            shock_u=float(shock_profile(tt)),
        )

    sol = odeint(f, initial_state, t)
    return pd.DataFrame(sol, columns=["P", "O", "G"], index=t)
