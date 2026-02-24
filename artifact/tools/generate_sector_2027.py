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

P_PROXIES = ["scale_proxy","speed_proxy","leverage_proxy","autonomy_proxy","replicability_proxy"]
O_PROXIES = ["stop_proxy","threshold_proxy","decision_proxy","execution_proxy","coherence_proxy"]
E_PROXIES = ["impact_proxy","propagation_proxy","hysteresis_proxy"]

def _sigmoid(x: np.ndarray, k: float = 2.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-k * x))

def _endogenize_g(df: pd.DataFrame, rng: np.random.Generator, *, base_gap: float = 0.03, pressure_scale: float = 1.0) -> pd.DataFrame:
    """Derive governance proxies endogenously from P/O/E proxies."""
    p = df[P_PROXIES].astype(float).mean(axis=1).to_numpy()
    o = df[O_PROXIES].astype(float).mean(axis=1).to_numpy()
    e = df[E_PROXIES].astype(float).mean(axis=1).to_numpy()

    pressure_raw = (p - 1.0) + 0.9 * (1.0 - o) + 0.8 * (e - 1.0)
    pressure = np.clip(_sigmoid(pressure_raw) * float(pressure_scale), 0.0, 1.0)

    df["exemption_rate"] = np.clip(0.08 + 0.35 * pressure + rng.normal(0.0, 0.02, size=len(df)), 0.0, 1.0)
    df["sanction_delay"] = np.clip(18.0 + 280.0 * pressure + rng.normal(0.0, 10.0, size=len(df)), 0.0, 365.0)
    df["control_turnover"] = np.clip(0.04 + 0.08 * pressure + rng.normal(0.0, 0.01, size=len(df)), 0.0, 1.0)
    df["conflict_interest_proxy"] = np.clip(0.10 + 0.22 * pressure + rng.normal(0.0, 0.02, size=len(df)), 0.0, 1.0)
    df["rule_execution_gap"] = np.clip(base_gap + 0.18 * pressure + rng.normal(0.0, 0.008, size=len(df)), 0.0, 0.50)
    return df


def _base_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(1.0 + rng.normal(0.0, 0.05, size=(n, len(BASE_PROXIES))), columns=BASE_PROXIES)
    # recovery time proxy is a cost
    df["recovery_time_proxy"] = np.clip(0.8 + rng.normal(0.0, 0.08, size=n), 0.0, 6.0)
    return df


def make_finance_2027(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = _base_frame(n, seed)

    # slightly higher leverage baseline vs 2026
    df["leverage_proxy"] *= 1.08

    # exogenous shocks mainly on P proxies (liquidity / leverage bursts)
    u_exog = np.zeros(n, dtype=float)
    n_events = 12
    event_starts = rng.choice(np.arange(10, n - 20), size=n_events, replace=False)
    for s in sorted(event_starts):
        dur = int(rng.integers(6, 18))
        amp = float(rng.uniform(0.6, 1.7))
        u_exog[s : s + dur] = amp
        df.loc[df.index[s : s + dur], "scale_proxy"] *= (1.0 + 0.35 * amp)
        df.loc[df.index[s : s + dur], "speed_proxy"] *= (1.0 + 0.30 * amp)
        df.loc[df.index[s : s + dur], "leverage_proxy"] *= (1.0 + 0.45 * amp)

        # mild O lag under high pressure
        df.loc[df.index[s : s + dur], "execution_proxy"] *= (1.0 - 0.08 * amp)
        df.loc[df.index[s : s + dur], "coherence_proxy"] *= (1.0 - 0.06 * amp)

        # externalities and resilience respond weakly
        df.loc[df.index[s : s + dur], "impact_proxy"] *= (1.0 + 0.10 * amp)
        df.loc[df.index[s : s + dur], "margin_proxy"] *= (1.0 - 0.06 * amp)
        df.loc[df.index[s : s + dur], "recovery_time_proxy"] *= (1.0 + 0.10 * amp)

    df["u_exog"] = u_exog
    df["sector_tag"] = "finance"
    df = _endogenize_g(df, rng, base_gap=0.03, pressure_scale=0.95)
    return df


def make_ia_2027(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 100)
    df = _base_frame(n, seed + 100)

    # trend to higher autonomy and replicability
    t = np.linspace(0.0, 1.0, n)
    df["autonomy_proxy"] *= (1.0 + 0.35 * t)
    df["replicability_proxy"] *= (1.0 + 0.30 * t)

    # network waves affect E proxies (propagation/hysteresis), this is the suggested real-world calibration axis
    u_exog = np.zeros(n, dtype=float)
    n_waves = 8
    wave_starts = np.linspace(18, n - 50, n_waves).astype(int)
    for s in wave_starts:
        dur = int(rng.integers(12, 28))
        amp = float(rng.uniform(0.7, 1.8))
        u_exog[s : s + dur] = amp

        df.loc[df.index[s : s + dur], "propagation_proxy"] *= (1.0 + 0.60 * amp)
        df.loc[df.index[s : s + dur], "hysteresis_proxy"] *= (1.0 + 0.55 * amp)
        df.loc[df.index[s : s + dur], "impact_proxy"] *= (1.0 + 0.25 * amp)

        # resilience drops under sustained network waves
        df.loc[df.index[s : s + dur], "margin_proxy"] *= (1.0 - 0.14 * amp)
        df.loc[df.index[s : s + dur], "redundancy_proxy"] *= (1.0 - 0.12 * amp)
        df.loc[df.index[s : s + dur], "recovery_time_proxy"] *= (1.0 + 0.22 * amp)

        # O drifts slightly (decision/execution load)
        df.loc[df.index[s : s + dur], "decision_proxy"] *= (1.0 - 0.06 * amp)
        df.loc[df.index[s : s + dur], "execution_proxy"] *= (1.0 - 0.08 * amp)

    df["u_exog"] = u_exog
    df["sector_tag"] = "ia"
    df = _endogenize_g(df, rng, base_gap=0.035, pressure_scale=1.05)
    return df


def make_biotech_2027(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 200)
    df = _base_frame(n, seed + 200)

    # biotech: moderate P, stronger O baseline, but high E hysteresis when failures occur
    df["stop_proxy"] *= 1.12
    df["threshold_proxy"] *= 1.10
    df["coherence_proxy"] *= 1.08

    u_exog = np.zeros(n, dtype=float)
    n_events = 7
    event_starts = rng.choice(np.arange(20, n - 40), size=n_events, replace=False)
    for s in sorted(event_starts):
        dur = int(rng.integers(10, 30))
        amp = float(rng.uniform(0.6, 1.6))
        u_exog[s : s + dur] = amp

        # E shock: impact + hysteresis rise strongly, propagation moderate
        df.loc[df.index[s : s + dur], "impact_proxy"] *= (1.0 + 0.40 * amp)
        df.loc[df.index[s : s + dur], "hysteresis_proxy"] *= (1.0 + 0.75 * amp)
        df.loc[df.index[s : s + dur], "propagation_proxy"] *= (1.0 + 0.25 * amp)

        # R hit: recovery time increases, margins shrink
        df.loc[df.index[s : s + dur], "recovery_time_proxy"] *= (1.0 + 0.35 * amp)
        df.loc[df.index[s : s + dur], "margin_proxy"] *= (1.0 - 0.10 * amp)

        # occasional P bursts (scale/speed) when accelerating deployment
        if amp > 1.2:
            df.loc[df.index[s : s + dur], "scale_proxy"] *= (1.0 + 0.20 * amp)
            df.loc[df.index[s : s + dur], "speed_proxy"] *= (1.0 + 0.18 * amp)

    df["u_exog"] = u_exog
    df["sector_tag"] = "biotech"
    # biotech governance drift: keep base gap low, but allow pressure-induced increases
    df = _endogenize_g(df, rng, base_gap=0.025, pressure_scale=0.90)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate sector datasets for 2027+ (finance, IA, biotech).")
    ap.add_argument("--out-dir", type=str, default="data/sector_2027")
    ap.add_argument("--n", type=int, default=365)
    ap.add_argument("--start", type=str, default="2027-01-01")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(args.start, periods=int(args.n), freq="D")

    df_fin = make_finance_2027(len(dates), args.seed)
    df_ia = make_ia_2027(len(dates), args.seed)
    df_bio = make_biotech_2027(len(dates), args.seed)

    for name, df in [
        ("finance_2027_synth.csv", df_fin),
        ("ia_2027_synth.csv", df_ia),
        ("biotech_2027_synth.csv", df_bio),
    ]:
        out_df = df.copy()
        out_df.insert(0, "date", dates)
        out_df.to_csv(out / name, index=False)

    readme = out / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Sector datasets 2027+ (synthetic)\n\n"
            "Ces datasets sont purement synthétiques et servent à démontrer:\n"
            "- chocs P(t) (finance)\n"
            "- chocs E(t) réseau (IA)\n"
            "- chocs E(t) + hystérèse (biotech)\n\n"
            "G(t) est endogénéisé: les proxys de gouvernance dérivent des proxys P/O/E (pression).\n"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
