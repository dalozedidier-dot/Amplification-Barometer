from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Any

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


@dataclass(frozen=True)
class ActiveEquilibrium:
    """Équilibre actif du modèle 4D, quand il existe dans le domaine positif."""

    P: float
    O: float
    E: float
    R: float
    feasible: bool
    reason: str = ""


@dataclass(frozen=True)
class HurwitzDiagnostics:
    """Diagnostic de stabilité locale du bloc (P, O, E).

    Ce diagnostic est strictement local: il vaut au voisinage de l'équilibre actif
    linéarisé. Il ne mesure ni le bassin d'attraction, ni les effets réseau, ni la dérive
    temporelle des paramètres.
    """

    A1: float
    A2: float
    A3: float
    A1A2_minus_A3: float
    locally_stable: bool
    conditions: dict[str, bool]
    equilibrium: dict[str, float | bool | str]
    status: str
    scope_note: str


def _require_positive_params(params: BarometerParams, names: tuple[str, ...]) -> None:
    bad = [name for name in names if float(getattr(params, name)) <= 0.0]
    if bad:
        raise ValueError(f"Paramètres strictement positifs requis pour le diagnostic: {bad}")


def compute_active_equilibrium(params: BarometerParams = BarometerParams(), *, eps: float = 1e-12) -> ActiveEquilibrium:
    """Calcule l'équilibre actif P*>0 du modèle 4D, si le cas linéaire est faisable.

    Conditions de validité minimales: b > 0, lam > 0, xi > 0. Le résultat est marqué
    infeasible si P*, O*, E* ou R* sortent de R_+^4.
    """
    _require_positive_params(params, ("b", "lam", "xi"))

    def f_of_p(P: float) -> float:
        O = (params.a - params.c * P) / params.b
        E = (params.alpha * P - params.beta * O) / params.lam
        return params.u - params.v * P - params.n * O - params.m * E

    f0 = float(f_of_p(0.0))
    f1 = float(f_of_p(1.0))
    slope = f1 - f0
    if abs(slope) <= float(eps):
        return ActiveEquilibrium(float("nan"), float("nan"), float("nan"), float("nan"), False, "singular_equilibrium_equation")

    P = -f0 / slope
    O = (params.a - params.c * P) / params.b
    E = (params.alpha * P - params.beta * O) / params.lam
    R = (params.delta * O - params.gamma * E) / params.xi
    vals = np.asarray([P, O, E, R], dtype=float)
    if not np.all(np.isfinite(vals)):
        return ActiveEquilibrium(float(P), float(O), float(E), float(R), False, "non_finite_equilibrium")
    if np.any(vals < -float(eps)) or P <= float(eps):
        return ActiveEquilibrium(float(P), float(O), float(E), float(R), False, "outside_positive_domain")
    return ActiveEquilibrium(float(P), float(O), float(E), float(R), True, "")


def jacobian_3d_at_active_equilibrium(
    params: BarometerParams = BarometerParams(),
    equilibrium: ActiveEquilibrium | None = None,
) -> np.ndarray:
    """Jacobienne du bloc (P, O, E) au voisinage de l'équilibre actif."""
    eq = equilibrium if equilibrium is not None else compute_active_equilibrium(params)
    if not eq.feasible:
        raise ValueError(f"Équilibre actif non faisable: {eq.reason}")
    P = float(eq.P)
    return np.array(
        [
            [-params.c * P, -params.b * P, 0.0],
            [-params.v, -params.n, -params.m],
            [params.alpha, -params.beta, -params.lam],
        ],
        dtype=float,
    )


def hurwitz_coefficients(
    params: BarometerParams = BarometerParams(),
    equilibrium: ActiveEquilibrium | None = None,
) -> dict[str, float]:
    """Retourne A1, A2, A3 pour le polynôme μ^3 + A1 μ^2 + A2 μ + A3."""
    eq = equilibrium if equilibrium is not None else compute_active_equilibrium(params)
    if not eq.feasible:
        raise ValueError(f"Équilibre actif non faisable: {eq.reason}")
    P = float(eq.P)
    A1 = params.c * P + params.n + params.lam
    A2 = P * (params.c * params.n - params.b * params.v) + params.c * P * params.lam + (params.n * params.lam - params.m * params.beta)
    A3 = P * (params.c * (params.n * params.lam - params.m * params.beta) - params.b * (params.v * params.lam + params.m * params.alpha))
    return {"A1": float(A1), "A2": float(A2), "A3": float(A3), "A1A2_minus_A3": float(A1 * A2 - A3)}


def assess_hurwitz_local_stability(
    params: BarometerParams = BarometerParams(),
    equilibrium: ActiveEquilibrium | None = None,
) -> HurwitzDiagnostics:
    """Évalue les conditions de Hurwitz pour le bloc local (P, O, E).

    Interprétation: STABLE_LOCAL signifie stabilité asymptotique locale du modèle
    linéarisé autour de l'équilibre actif. UNSTABLE_LOCAL signale une zone rouge
    structurale dans ce cadre paramétré, sans prétendre prédire un événement réel.
    """
    eq = equilibrium if equilibrium is not None else compute_active_equilibrium(params)
    if not eq.feasible:
        return HurwitzDiagnostics(
            A1=float("nan"),
            A2=float("nan"),
            A3=float("nan"),
            A1A2_minus_A3=float("nan"),
            locally_stable=False,
            conditions={"A1_positive": False, "A2_positive": False, "A3_positive": False, "A1A2_gt_A3": False},
            equilibrium={"P": float(eq.P), "O": float(eq.O), "E": float(eq.E), "R": float(eq.R), "feasible": False, "reason": str(eq.reason)},
            status="INFEASIBLE_EQUILIBRIUM",
            scope_note="Diagnostic local non applicable: l'équilibre actif n'est pas faisable dans R_+^4.",
        )

    coeffs = hurwitz_coefficients(params, eq)
    conditions = {
        "A1_positive": bool(coeffs["A1"] > 0.0),
        "A2_positive": bool(coeffs["A2"] > 0.0),
        "A3_positive": bool(coeffs["A3"] > 0.0),
        "A1A2_gt_A3": bool(coeffs["A1A2_minus_A3"] > 0.0),
    }
    stable = bool(all(conditions.values()))
    return HurwitzDiagnostics(
        A1=float(coeffs["A1"]),
        A2=float(coeffs["A2"]),
        A3=float(coeffs["A3"]),
        A1A2_minus_A3=float(coeffs["A1A2_minus_A3"]),
        locally_stable=stable,
        conditions=conditions,
        equilibrium={"P": float(eq.P), "O": float(eq.O), "E": float(eq.E), "R": float(eq.R), "feasible": True, "reason": ""},
        status="STABLE_LOCAL" if stable else "UNSTABLE_LOCAL",
        scope_note="Valable seulement au voisinage de l'équilibre actif, pour les paramètres fournis. Ne mesure pas le bassin d'attraction, les réseaux, ni la dérive des coefficients.",
    )


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



@dataclass(frozen=True)
class BarometerParams5(BarometerParams):
    """Extension 5D: ajout d'un état G(t) (gouvernance).

    Objectif: fournir un système couplé P,O,E,R,G pour stress tests et Monte Carlo.
    Ce modèle est démonstratif et ne prétend pas être un jumeau numérique.
    """
    # dynamique gouvernance
    g_target: float = 0.9
    k_g: float = 0.25
    p_g: float = 0.15
    o_g: float = 0.10
    e_g: float = 0.10

    # couplages faibles (optionnels)
    k_go: float = 0.10
    k_gr: float = 0.05
    k_gp: float = 0.05


def barometer_ode_5d(x, t, p: BarometerParams5, shock_u: float = 0.0, shock_g: float = 0.0):
    """Modèle Baromètre 5D (démo) avec gouvernance endogène.

    dP/dt = P * (a - b*O - c*P) + shock_u + k_gp*(1-G)*P
    dO/dt = u - v*P - n*O - m*E + k_go*(G-0.5)
    dE/dt = alpha*P - beta*O - lam*E
    dR/dt = delta*O - gamma*E - xi*R + k_gr*(G-0.5)
    dG/dt = k_g*(g_target - G) - p_g*P - e_g*E + o_g*O + shock_g
    """
    P, O, E, R, G = x
    dPdt = P * (p.a - p.b * O - p.c * P) + float(shock_u) + p.k_gp * (1.0 - G) * P
    dOdt = p.u - p.v * P - p.n * O - p.m * E + p.k_go * (G - 0.5)
    dEdt = p.alpha * P - p.beta * O - p.lam * E
    dRdt = p.delta * O - p.gamma * E - p.xi * R + p.k_gr * (G - 0.5)
    dGdt = p.k_g * (p.g_target - G) - p.p_g * P - p.e_g * E + p.o_g * O + float(shock_g)
    return [dPdt, dOdt, dEdt, dRdt, dGdt]


def poisson_shock_on_grid(
    t: np.ndarray,
    *,
    rate: float,
    scale: float,
    seed: int = 7,
    positive: bool = True,
) -> np.ndarray:
    """Chocs exogènes rares sur grille temporelle (Poisson approx discret).

    rate: intensité (événements par unité de temps)
    scale: amplitude typique
    """
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return np.zeros_like(t, dtype=float)
    dt = float(np.mean(np.diff(t)))
    p_evt = float(np.clip(rate * dt, 0.0, 1.0))
    rng = np.random.default_rng(int(seed))
    events = rng.random(t.size) < p_evt
    amps = rng.normal(0.0, float(scale), size=t.size)
    if positive:
        amps = np.abs(amps)
    shock = amps * events.astype(float)
    return shock


def simulate_barometer_ode5(
    initial_state=(0.6, 0.6, 0.0, 0.8, 0.8),
    t: np.ndarray | None = None,
    *,
    params: BarometerParams5 = BarometerParams5(),
    shock_profile_p: Optional[Callable[[float], float]] = None,
    shock_profile_g: Optional[Callable[[float], float]] = None,
) -> pd.DataFrame:
    """Simule le modèle 5D (P,O,E,R,G).

    shock_profile_p: sP(t) ajouté à dP/dt
    shock_profile_g: sG(t) ajouté à dG/dt
    """
    if t is None:
        t = np.linspace(0.0, 40.0, 600)
    t = np.asarray(t, dtype=float)

    if shock_profile_p is None:
        def shock_profile_p(_tt: float) -> float:
            return 0.0

    if shock_profile_g is None:
        def shock_profile_g(_tt: float) -> float:
            return 0.0

    def f(x, tt):
        return barometer_ode_5d(
            x,
            tt,
            params,
            shock_u=float(shock_profile_p(tt)),
            shock_g=float(shock_profile_g(tt)),
        )

    sol = odeint(f, initial_state, t)
    return pd.DataFrame(sol, columns=["P", "O", "E", "R", "G"], index=t)


def simulate_barometer_monte_carlo(
    *,
    runs: int = 200,
    t: np.ndarray | None = None,
    params: BarometerParams5 = BarometerParams5(),
    shock_rate: float = 0.08,
    shock_scale_p: float = 0.35,
    shock_scale_g: float = 0.10,
    seed: int = 7,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """Monte Carlo sur le modèle 5D avec chocs rares.

    Sortie: table de résumés par run (max, p95, temps de dépassement, etc).
    """
    if t is None:
        t = np.linspace(0.0, 40.0, 600)
    t = np.asarray(t, dtype=float)

    rng = np.random.default_rng(int(seed))
    rows = []
    for i in range(int(runs)):
        # Chocs discrets puis interpolation linéaire
        sp = poisson_shock_on_grid(t, rate=shock_rate, scale=shock_scale_p, seed=int(rng.integers(0, 2**31 - 1)), positive=True)
        sg = poisson_shock_on_grid(t, rate=shock_rate * 0.5, scale=shock_scale_g, seed=int(rng.integers(0, 2**31 - 1)), positive=False)

        def shock_p(tt: float) -> float:
            return float(np.interp(tt, t, sp))

        def shock_g(tt: float) -> float:
            return float(np.interp(tt, t, sg))

        df = simulate_barometer_ode5(t=t, params=params, shock_profile_p=shock_p, shock_profile_g=shock_g)
        at = df["P"].to_numpy(dtype=float) / (df["O"].to_numpy(dtype=float) + eps)
        risk = at + np.gradient(at)  # proxy simple, distinct des scores data-driven

        row = {
            "run": int(i),
            "at_max": float(np.nanmax(at)),
            "at_p95": float(np.nanpercentile(at, 95)),
            "risk_p95": float(np.nanpercentile(risk, 95)),
            "risk_max": float(np.nanmax(risk)),
            "e_max": float(np.nanmax(df["E"].to_numpy(dtype=float))),
            "g_min": float(np.nanmin(df["G"].to_numpy(dtype=float))),
            "time_high_risk_frac": float(np.mean(risk > float(np.nanpercentile(risk, 95)))),
            "shock_p_total": float(np.nansum(sp)),
            "shock_g_total": float(np.nansum(sg)),
        }
        rows.append(row)

    return pd.DataFrame(rows)

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
