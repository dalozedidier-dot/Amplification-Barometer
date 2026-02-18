from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROXIES = [
    # P
    "scale_proxy",
    "speed_proxy",
    "leverage_proxy",
    "autonomy_proxy",
    "replicability_proxy",
    # O
    "stop_proxy",
    "threshold_proxy",
    "decision_proxy",
    "execution_proxy",
    "coherence_proxy",
    # E
    "impact_proxy",
    "propagation_proxy",
    "hysteresis_proxy",
    # R
    "margin_proxy",
    "redundancy_proxy",
    "diversity_proxy",
    "recovery_time_proxy",
    # G (endogenized)
    "exemption_rate",
    "sanction_delay",
    "control_turnover",
    "conflict_interest_proxy",
    "rule_execution_gap",
]

P_PROXIES = ["scale_proxy", "speed_proxy", "leverage_proxy", "autonomy_proxy", "replicability_proxy"]
O_PROXIES = ["stop_proxy", "threshold_proxy", "decision_proxy", "execution_proxy", "coherence_proxy"]
E_PROXIES = ["impact_proxy", "propagation_proxy", "hysteresis_proxy"]


def _sigmoid(x: np.ndarray, k: float = 2.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-k * x))


def _endogenize_g(
    df: pd.DataFrame,
    rng: np.random.Generator,
    *,
    base_gap: float = 0.02,
    pressure_scale: float = 0.40,
) -> pd.DataFrame:
    """Derive G proxies from other proxies (endogenous governance drift).

    Contrainte de démo: on veut une gouvernance saine par défaut, compatible avec
    l'objectif critique rule_execution_gap_mean < 0.05, sauf dérives explicites.
    Les proxys G sont donc des fonctions bornées de la pression P/O/E.
    """
    p = df[P_PROXIES].astype(float).mean(axis=1).to_numpy()
    o = df[O_PROXIES].astype(float).mean(axis=1).to_numpy()
    e = df[E_PROXIES].astype(float).mean(axis=1).to_numpy()

    pressure_raw = (p - 1.0) + 0.9 * (1.0 - o) + 0.8 * (e - 1.0)
    pressure = np.clip(_sigmoid(pressure_raw) * float(pressure_scale), 0.0, 1.0)

    df["exemption_rate"] = np.clip(0.04 + 0.22 * pressure + rng.normal(0.0, 0.015, size=len(df)), 0.0, 1.0)
    df["sanction_delay"] = np.clip(12.0 + 210.0 * pressure + rng.normal(0.0, 8.0, size=len(df)), 0.0, 365.0)
    df["control_turnover"] = np.clip(0.03 + 0.07 * pressure + rng.normal(0.0, 0.006, size=len(df)), 0.0, 1.0)
    df["conflict_interest_proxy"] = np.clip(0.06 + 0.14 * pressure + rng.normal(0.0, 0.012, size=len(df)), 0.0, 1.0)

    # gap cible <5% dans les régimes mûrs, dérive sous pression
    df["rule_execution_gap"] = np.clip(base_gap + 0.08 * pressure + rng.normal(0.0, 0.004, size=len(df)), 0.0, 0.20)
    return df


def make_stable(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 1.0 + rng.normal(0.0, 0.05, size=(n, len(PROXIES)))
    df = pd.DataFrame(base, columns=PROXIES)

    # recovery_time_proxy est un coût, plus bas dans le régime stable
    df["recovery_time_proxy"] = np.clip(0.6 + rng.normal(0.0, 0.05, size=n), 0.0, 2.0)

    df = _endogenize_g(df, rng, base_gap=0.02, pressure_scale=0.35)
    return df


def make_oscillating(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 8 * np.pi, n)
    osc = 1.0 + 0.25 * np.sin(t)[:, None] + rng.normal(0.0, 0.05, size=(n, len(PROXIES)))
    df = pd.DataFrame(osc, columns=PROXIES)
    df["recovery_time_proxy"] = np.clip(0.8 + 0.15 * (1 + np.sin(t + 0.5)) / 2 + rng.normal(0.0, 0.05, size=n), 0.0, 2.5)

    df = _endogenize_g(df, rng, base_gap=0.025, pressure_scale=0.50)
    return df


def make_bifurcation(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 1.0 + rng.normal(0.0, 0.05, size=(n, len(PROXIES)))
    df = pd.DataFrame(base, columns=PROXIES)

    # après t0: montée rapide de P (scale, speed, leverage) et dégradation de O
    t0 = n // 2
    growth = np.exp((t - t0) / (n / 10))
    growth[:t0] = 1.0

    df["scale_proxy"] = df["scale_proxy"] * growth
    df["speed_proxy"] = df["speed_proxy"] * (0.8 + 0.4 * growth)
    df["leverage_proxy"] = df["leverage_proxy"] * (0.9 + 0.5 * growth)

    df["stop_proxy"] = df["stop_proxy"] * (1.0 / (0.9 + 0.3 * growth))
    df["execution_proxy"] = df["execution_proxy"] * (1.0 / (0.9 + 0.25 * growth))
    df["coherence_proxy"] = df["coherence_proxy"] * (1.0 / (0.9 + 0.20 * growth))

    # Externalités deviennent cumulatives plus fortes
    df["impact_proxy"] = df["impact_proxy"] * (0.9 + 0.6 * growth)
    df["propagation_proxy"] = df["propagation_proxy"] * (0.8 + 0.8 * growth)
    df["hysteresis_proxy"] = df["hysteresis_proxy"] * (0.8 + 0.8 * growth)

    # Résilience baisse: marges et redondance diminuent, temps de récupération augmente
    df["margin_proxy"] = df["margin_proxy"] * (1.0 / (0.9 + 0.4 * growth))
    df["redundancy_proxy"] = df["redundancy_proxy"] * (1.0 / (0.9 + 0.35 * growth))
    df["diversity_proxy"] = df["diversity_proxy"] * (1.0 / (0.9 + 0.30 * growth))
    df["recovery_time_proxy"] = np.clip(
        0.7 + 0.8 * ((growth - 1) / (growth.max() - 1 + 1e-9)) + rng.normal(0.0, 0.05, size=n),
        0.0,
        5.0,
    )

    df = _endogenize_g(df, rng, base_gap=0.05, pressure_scale=1.15)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic multi-regime datasets (stable/oscillating/bifurcation).")
    ap.add_argument("--out-dir", type=str, default="data/synthetic")
    ap.add_argument("--n", type=int, default=365)
    ap.add_argument("--start-date", type=str, default="2020-01-01")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(args.start_date, periods=int(args.n), freq="D")

    for name, maker in [
        ("stable_regime.csv", make_stable),
        ("oscillating_regime.csv", make_oscillating),
        ("bifurcation_regime.csv", make_bifurcation),
    ]:
        df = maker(len(dates), args.seed)
        out_df = df.copy()
        out_df.insert(0, "date", dates)
        out_df.to_csv(out / name, index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())