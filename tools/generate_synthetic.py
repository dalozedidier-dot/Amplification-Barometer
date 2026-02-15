from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROXIES = [
    # P
    "scale_proxy","speed_proxy","leverage_proxy","autonomy_proxy","replicability_proxy",
    # O
    "stop_proxy","threshold_proxy","decision_proxy","execution_proxy","coherence_proxy",
    # E
    "impact_proxy","propagation_proxy","hysteresis_proxy",
    # R
    "margin_proxy","redundancy_proxy","diversity_proxy","recovery_time_proxy",
    # G
    "exemption_rate","sanction_delay","control_turnover","conflict_interest_proxy","rule_execution_gap",
]


def make_stable(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 1.0 + rng.normal(0.0, 0.05, size=(n, len(PROXIES)))
    df = pd.DataFrame(base, columns=PROXIES)
    # bornes plausibles pour G
    df["exemption_rate"] = np.clip(0.10 + rng.normal(0.0, 0.02, size=n), 0.0, 1.0)
    df["sanction_delay"] = np.clip(20 + rng.normal(0.0, 5.0, size=n), 0.0, 365.0)
    df["control_turnover"] = np.clip(0.04 + rng.normal(0.0, 0.01, size=n), 0.0, 1.0)
    df["conflict_interest_proxy"] = np.clip(0.10 + rng.normal(0.0, 0.03, size=n), 0.0, 1.0)
    # Écart règle/exécution (cible < 5%)
    df["rule_execution_gap"] = np.clip(0.03 + rng.normal(0.0, 0.01, size=n), 0.0, 0.12)
    # recovery_time_proxy est un coût, plus bas dans le régime stable
    df["recovery_time_proxy"] = np.clip(0.6 + rng.normal(0.0, 0.05, size=n), 0.0, 2.0)
    return df


def make_oscillating(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 8 * np.pi, n)
    osc = 1.0 + 0.25 * np.sin(t)[:, None] + rng.normal(0.0, 0.05, size=(n, len(PROXIES)))
    df = pd.DataFrame(osc, columns=PROXIES)
    # G se dégrade par cycles et retard (plus d'exemptions, délais)
    df["exemption_rate"] = np.clip(0.15 + 0.08 * (1 + np.sin(t)) / 2 + rng.normal(0.0, 0.02, size=n), 0.0, 1.0)
    df["sanction_delay"] = np.clip(40 + 20 * (1 + np.sin(t + 0.7)) / 2 + rng.normal(0.0, 4.0, size=n), 0.0, 365.0)
    df["control_turnover"] = np.clip(0.10 + 0.06 * (1 + np.sin(t + 1.1)) / 2 + rng.normal(0.0, 0.02, size=n), 0.0, 1.0)
    df["conflict_interest_proxy"] = np.clip(0.12 + 0.05 * (1 + np.sin(t + 2.0)) / 2 + rng.normal(0.0, 0.02, size=n), 0.0, 1.0)
    df["rule_execution_gap"] = np.clip(0.04 + 0.03 * (1 + np.sin(t + 1.4)) / 2 + rng.normal(0.0, 0.01, size=n), 0.0, 0.25)
    df["recovery_time_proxy"] = np.clip(0.8 + 0.15 * (1 + np.sin(t + 0.5)) / 2 + rng.normal(0.0, 0.05, size=n), 0.0, 2.5)
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
    df["recovery_time_proxy"] = np.clip(0.7 + 0.8 * ((growth - 1) / (growth.max() - 1 + 1e-9)) + rng.normal(0.0, 0.05, size=n), 0.0, 5.0)

    # G se dégrade durablement: exemptions et délais montent
    df["exemption_rate"] = np.clip(0.12 + 0.35 * ((growth - 1) / (growth.max() - 1 + 1e-9)) + rng.normal(0.0, 0.02, size=n), 0.0, 1.0)
    df["sanction_delay"] = np.clip(20 + 120 * ((growth - 1) / (growth.max() - 1 + 1e-9)) + rng.normal(0.0, 6.0, size=n), 0.0, 365.0)
    df["control_turnover"] = np.clip(0.08 + 0.20 * ((growth - 1) / (growth.max() - 1 + 1e-9)) + rng.normal(0.0, 0.02, size=n), 0.0, 1.0)
    df["conflict_interest_proxy"] = np.clip(0.10 + 0.25 * ((growth - 1) / (growth.max() - 1 + 1e-9)) + rng.normal(0.0, 0.03, size=n), 0.0, 1.0)
    df["rule_execution_gap"] = np.clip(0.03 + 0.20 * ((growth - 1) / (growth.max() - 1 + 1e-9)) + rng.normal(0.0, 0.015, size=n), 0.0, 0.60)

    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="data/synthetic")
    ap.add_argument("--n", type=int, default=365)
    ap.add_argument("--start-date", type=str, default="2026-01-01")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    idx = pd.date_range(args.start_date, periods=args.n, freq="D")

    stable = make_stable(args.n, args.seed)
    stable.insert(0, "date", idx)
    stable.to_csv(out / "stable_regime.csv", index=False)

    osc = make_oscillating(args.n, args.seed + 1)
    osc.insert(0, "date", idx)
    osc.to_csv(out / "oscillating_regime.csv", index=False)

    bif = make_bifurcation(args.n, args.seed + 2)
    bif.insert(0, "date", idx)
    bif.to_csv(out / "bifurcation_regime.csv", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
