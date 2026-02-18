from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_PROXIES = [
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


def _endogenize_g(df: pd.DataFrame, rng: np.random.Generator, *, base_gap: float = 0.02, pressure_scale: float = 0.40) -> pd.DataFrame:
    """Derive governance proxies endogenously from P/O/E proxies.

    Contrainte de démo: on veut un régime "gouvernance saine" par défaut, donc des
    valeurs compatibles avec l'objectif critique rule_execution_gap_mean < 0.05,
    sauf dérives explicites.
    """
    p = df[P_PROXIES].astype(float).mean(axis=1).to_numpy()
    o = df[O_PROXIES].astype(float).mean(axis=1).to_numpy()
    e = df[E_PROXIES].astype(float).mean(axis=1).to_numpy()

    pressure_raw = (p - 1.0) + 0.9 * (1.0 - o) + 0.8 * (e - 1.0)
    pressure = np.clip(_sigmoid(pressure_raw) * float(pressure_scale), 0.0, 1.0)

    df["exemption_rate"] = np.clip(0.04 + 0.22 * pressure + rng.normal(0.0, 0.015, size=len(df)), 0.0, 1.0)
    df["sanction_delay"] = np.clip(12.0 + 210.0 * pressure + rng.normal(0.0, 8.0, size=len(df)), 0.0, 365.0)
    df["control_turnover"] = np.clip(0.025 + 0.040 * pressure + rng.normal(0.0, 0.006, size=len(df)), 0.0, 1.0)
    df["conflict_interest_proxy"] = np.clip(0.06 + 0.14 * pressure + rng.normal(0.0, 0.012, size=len(df)), 0.0, 1.0)

    # gap cible <5% dans les régimes mûrs, dérive sous pression
    df["rule_execution_gap"] = np.clip(base_gap + 0.08 * pressure + rng.normal(0.0, 0.004, size=len(df)), 0.0, 0.20)
    return df


def _base_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(1.0 + rng.normal(0.0, 0.05, size=(n, len(BASE_PROXIES))), columns=BASE_PROXIES)

    # recovery time proxy is a cost
    df["recovery_time_proxy"] = np.clip(0.8 + rng.normal(0.0, 0.08, size=n), 0.0, 5.0)

    return df


def make_finance_2026(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = _base_frame(n, seed)

    # exogenous shocks mainly on P proxies
    u_exog = np.zeros(n, dtype=float)
    drift_exempt = np.zeros(n, dtype=float)
    drift_sanc = np.zeros(n, dtype=float)
    drift_gap = np.zeros(n, dtype=float)

    n_events = 10
    event_starts = rng.choice(np.arange(10, n - 20), size=n_events, replace=False)
    for s in sorted(event_starts):
        dur = int(rng.integers(3, 10))
        amp = float(rng.uniform(0.6, 1.2))
        u_exog[s : s + dur] = amp

        df.loc[df.index[s : s + dur], "scale_proxy"] *= (1.0 + 0.25 * amp)
        df.loc[df.index[s : s + dur], "speed_proxy"] *= (1.0 + 0.35 * amp)
        df.loc[df.index[s : s + dur], "leverage_proxy"] *= (1.0 + 0.45 * amp)

        # mild spillover in impacts
        df.loc[df.index[s : s + dur], "impact_proxy"] *= (1.0 + 0.18 * amp)
        df.loc[df.index[s : s + dur], "hysteresis_proxy"] *= (1.0 + 0.10 * amp)

        # orientation lags under shock
        df.loc[df.index[s : s + dur], "execution_proxy"] *= (1.0 - 0.10 * amp)
        df.loc[df.index[s : s + dur], "stop_proxy"] *= (1.0 - 0.08 * amp)

        # governance drift in severe episodes (applied after endogenization)
        if amp > 1.0:
            drift_exempt[s : s + dur] += 0.03
            drift_sanc[s : s + dur] += 10.0
            drift_gap[s : s + dur] += 0.02

    df["u_exog"] = u_exog
    df["sector_tag"] = "finance"

    # Endogenize governance from the final P/O/E/O state, then add explicit drift.
    df = _endogenize_g(df, rng, base_gap=0.02, pressure_scale=0.40)

    df["exemption_rate"] = np.clip(df["exemption_rate"].to_numpy(dtype=float) + drift_exempt, 0.0, 1.0)
    df["sanction_delay"] = np.clip(df["sanction_delay"].to_numpy(dtype=float) + drift_sanc, 0.0, 365.0)
    df["rule_execution_gap"] = np.clip(df["rule_execution_gap"].to_numpy(dtype=float) + drift_gap, 0.0, 1.0)

    return df


def make_ia_2026(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 100)
    df = _base_frame(n, seed + 100)

    # trend to higher autonomy and replicability
    t = np.linspace(0.0, 1.0, n)
    df["autonomy_proxy"] *= (1.0 + 0.30 * t)
    df["replicability_proxy"] *= (1.0 + 0.25 * t)

    # network waves affect propagation and hysteresis
    u_exog = np.zeros(n, dtype=float)
    n_waves = 6
    wave_starts = np.linspace(20, n - 60, n_waves).astype(int)
    for s in wave_starts:
        dur = int(rng.integers(15, 35))
        amp = float(rng.uniform(0.8, 1.5))
        u_exog[s : s + dur] = amp

        df.loc[df.index[s : s + dur], "propagation_proxy"] *= (1.0 + 0.55 * amp)
        df.loc[df.index[s : s + dur], "hysteresis_proxy"] *= (1.0 + 0.45 * amp)
        df.loc[df.index[s : s + dur], "impact_proxy"] *= (1.0 + 0.20 * amp)

        # resilience drops under sustained network waves
        df.loc[df.index[s : s + dur], "margin_proxy"] *= (1.0 - 0.12 * amp)
        df.loc[df.index[s : s + dur], "redundancy_proxy"] *= (1.0 - 0.10 * amp)
        df.loc[df.index[s : s + dur], "recovery_time_proxy"] *= (1.0 + 0.20 * amp)

    df["u_exog"] = u_exog
    df["sector_tag"] = "ia"

    df = _endogenize_g(df, rng, base_gap=0.022, pressure_scale=0.45)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="data/sector_2026")
    ap.add_argument("--n", type=int, default=365)
    ap.add_argument("--start-date", type=str, default="2026-01-01")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    idx = pd.date_range(args.start_date, periods=args.n, freq="D")

    fin = make_finance_2026(args.n, args.seed)
    fin.insert(0, "date", idx)
    fin.to_csv(out / "finance_2026_synth.csv", index=False)

    ia = make_ia_2026(args.n, args.seed)
    ia.insert(0, "date", idx)
    ia.to_csv(out / "ia_2026_synth.csv", index=False)

    (out / "README.md").write_text(
        """# Sector 2026 synthetic datasets

Ces CSV sont synthétiques, destinés à la démonstration et aux stress tests reproductibles.

- finance_2026_synth.csv: chocs P(t) dominants avec dérives G explicites sur épisodes sévères.
- ia_2026_synth.csv: externalités réseau dominantes, G endogène par pression P/O/E.
""",
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
