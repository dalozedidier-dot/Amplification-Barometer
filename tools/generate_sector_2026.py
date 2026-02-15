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
    # G
    "exemption_rate",
    "sanction_delay",
    "control_turnover",
    "conflict_interest_proxy",
    "rule_execution_gap",
]


def _base_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(1.0 + rng.normal(0.0, 0.05, size=(n, len(BASE_PROXIES))), columns=BASE_PROXIES)

    # clamp G proxies to realistic ranges
    df["exemption_rate"] = np.clip(0.10 + rng.normal(0.0, 0.02, size=n), 0.0, 1.0)
    df["sanction_delay"] = np.clip(25 + rng.normal(0.0, 6.0, size=n), 0.0, 365.0)
    df["control_turnover"] = np.clip(0.045 + rng.normal(0.0, 0.012, size=n), 0.0, 1.0)
    df["conflict_interest_proxy"] = np.clip(0.10 + rng.normal(0.0, 0.03, size=n), 0.0, 1.0)
    df["rule_execution_gap"] = np.clip(0.035 + rng.normal(0.0, 0.01, size=n), 0.0, 0.25)

    # recovery time proxy is a cost
    df["recovery_time_proxy"] = np.clip(0.8 + rng.normal(0.0, 0.08, size=n), 0.0, 5.0)

    return df


def make_finance_2026(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = _base_frame(n, seed)

    # exogenous shocks mainly on P proxies
    u_exog = np.zeros(n, dtype=float)
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

        # governance drift risk: exemptions increase in severe episodes
        if amp > 1.0:
            df.loc[df.index[s : s + dur], "exemption_rate"] = np.clip(
                df.loc[df.index[s : s + dur], "exemption_rate"].to_numpy(dtype=float) + 0.03,
                0.0,
                1.0,
            )
            df.loc[df.index[s : s + dur], "sanction_delay"] = np.clip(
                df.loc[df.index[s : s + dur], "sanction_delay"].to_numpy(dtype=float) + 10.0,
                0.0,
                365.0,
            )
            df.loc[df.index[s : s + dur], "rule_execution_gap"] = np.clip(
                df.loc[df.index[s : s + dur], "rule_execution_gap"].to_numpy(dtype=float) + 0.02,
                0.0,
                1.0,
            )

    df["u_exog"] = u_exog
    df["sector_tag"] = "finance"

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

        # governance drift episode: rule/execution gap increases and sanctions delay
        if amp > 1.2:
            df.loc[df.index[s : s + dur], "rule_execution_gap"] = np.clip(
                df.loc[df.index[s : s + dur], "rule_execution_gap"].to_numpy(dtype=float) + 0.04,
                0.0,
                1.0,
            )
            df.loc[df.index[s : s + dur], "sanction_delay"] = np.clip(
                df.loc[df.index[s : s + dur], "sanction_delay"].to_numpy(dtype=float) + 18.0,
                0.0,
                365.0,
            )
            df.loc[df.index[s : s + dur], "exemption_rate"] = np.clip(
                df.loc[df.index[s : s + dur], "exemption_rate"].to_numpy(dtype=float) + 0.05,
                0.0,
                1.0,
            )

    df["u_exog"] = u_exog
    df["sector_tag"] = "ia"

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
        """# Sector 2026 synthetic datasets\n\nCes CSV sont synthétiques, destinés à la démonstration et aux stress tests reproductibles.\n\n- finance_2026_synth.csv: chocs P(t) dominants.\n- ia_2026_synth.csv: externalités réseau dominantes.\n""",
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
