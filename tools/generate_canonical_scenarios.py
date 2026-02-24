#!/usr/bin/env python3
"""
Generate canonical stress scenarios for barometer calibration.

Produces 4 deterministic scenarios × 5 noise levels = 20 CSV files.
All scenarios use fixed seeds for reproducibility.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _base_proxies(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create baseline proxy DataFrame with all 22 columns at neutral values."""
    rng = np.random.default_rng(seed)

    # P family
    scale_proxy = np.ones(n) * 1.0
    speed_proxy = np.ones(n) * 1.0
    leverage_proxy = np.ones(n) * 1.0
    autonomy_proxy = np.ones(n) * 0.5
    replicability_proxy = np.ones(n) * 0.5

    # O family
    stop_proxy = np.ones(n) * 0.8
    threshold_proxy = np.ones(n) * 0.8
    decision_proxy = np.ones(n) * 0.8
    execution_proxy = np.ones(n) * 0.8
    coherence_proxy = np.ones(n) * 0.8

    # E family
    impact_proxy = np.ones(n) * 0.3
    propagation_proxy = np.ones(n) * 0.3
    hysteresis_proxy = np.ones(n) * 0.2

    # R family
    margin_proxy = np.ones(n) * 0.9
    redundancy_proxy = np.ones(n) * 0.9
    diversity_proxy = np.ones(n) * 0.8
    recovery_time_proxy = np.ones(n) * 0.1

    # G family
    exemption_rate = np.ones(n) * 0.05
    sanction_delay = np.ones(n) * 0.1
    control_turnover = np.ones(n) * 0.05
    conflict_interest_proxy = np.ones(n) * 0.05
    rule_execution_gap = np.ones(n) * 0.04

    df = pd.DataFrame({
        "scale_proxy": scale_proxy,
        "speed_proxy": speed_proxy,
        "leverage_proxy": leverage_proxy,
        "autonomy_proxy": autonomy_proxy,
        "replicability_proxy": replicability_proxy,
        "stop_proxy": stop_proxy,
        "threshold_proxy": threshold_proxy,
        "decision_proxy": decision_proxy,
        "execution_proxy": execution_proxy,
        "coherence_proxy": coherence_proxy,
        "impact_proxy": impact_proxy,
        "propagation_proxy": propagation_proxy,
        "hysteresis_proxy": hysteresis_proxy,
        "margin_proxy": margin_proxy,
        "redundancy_proxy": redundancy_proxy,
        "diversity_proxy": diversity_proxy,
        "recovery_time_proxy": recovery_time_proxy,
        "exemption_rate": exemption_rate,
        "sanction_delay": sanction_delay,
        "control_turnover": control_turnover,
        "conflict_interest_proxy": conflict_interest_proxy,
        "rule_execution_gap": rule_execution_gap,
    })

    df.insert(0, "date", pd.date_range("2026-01-01", periods=n, freq="D"))
    return df


def _add_noise(df: pd.DataFrame, noise_level: float = 0.0, seed: int = 99) -> pd.DataFrame:
    """Add Gaussian noise to all proxy columns (except date)."""
    if noise_level == 0.0:
        return df.copy()

    rng = np.random.default_rng(seed)
    df_noisy = df.copy()

    for col in df.columns:
        if col == "date":
            continue
        x = df_noisy[col].to_numpy(dtype=float, copy=True)
        std_x = np.nanstd(x) if np.nanstd(x) > 0 else 1.0
        noise = rng.normal(0.0, noise_level * std_x, size=len(x))
        df_noisy[col] = np.clip(x + noise, 0.0, 2.0)

    return df_noisy


def generate_type_i_noise(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Type I: Pure stochastic noise.

    All proxies remain near baseline with small oscillations.
    No trends, no shocks.
    """
    df = _base_proxies(n=n, seed=seed)
    rng = np.random.default_rng(seed)

    # Add small noise to all proxies
    for col in df.columns:
        if col == "date":
            continue
        x = df[col].to_numpy(dtype=float, copy=True)
        x += rng.normal(0.0, 0.05, size=len(x))
        df[col] = np.clip(x, 0.0, 2.0)

    return df


def generate_type_ii_oscillations(n: int = 200, seed: int = 43) -> pd.DataFrame:
    """
    Type II: Cyclical stress with recovery.

    P oscillates, O lags and responds, E grows slowly, R oscillates.
    """
    df = _base_proxies(n=n, seed=seed)
    rng = np.random.default_rng(seed)

    period = 40.0  # Oscillation period
    t = np.arange(n, dtype=float)

    # P oscillates: scale + speed boost
    df["scale_proxy"] = df["scale_proxy"] + 0.3 * np.sin(2.0 * np.pi * t / period)
    df["speed_proxy"] = df["speed_proxy"] + 0.3 * np.sin(2.0 * np.pi * t / period)

    # O responds with lag (10 steps)
    lag = 10
    o_response = -0.15 * np.sin(2.0 * np.pi * (t - lag) / period)
    df["stop_proxy"] = df["stop_proxy"] + o_response
    df["threshold_proxy"] = df["threshold_proxy"] + o_response * 0.7

    # E grows slowly (linear trend)
    df["impact_proxy"] = df["impact_proxy"] + 0.02 * t / n
    df["propagation_proxy"] = df["propagation_proxy"] + 0.01 * t / n

    # R oscillates
    df["recovery_time_proxy"] = df["recovery_time_proxy"] + 0.05 * np.sin(2.0 * np.pi * t / period)

    # Clip to valid ranges
    for col in df.columns:
        if col != "date":
            df[col] = np.clip(df[col], 0.0, 2.0)

    return df


def generate_type_iii_bifurcation(n: int = 200, seed: int = 44) -> pd.DataFrame:
    """
    Type III: Irreversible bifurcation.

    Phase 1 (t=0-80): oscillate like Type II
    Phase 2 (t=80-200): structural shift with no recovery
    """
    df = _base_proxies(n=n, seed=seed)
    rng = np.random.default_rng(seed)

    period = 40.0
    t = np.arange(n, dtype=float)

    # Phase 1: oscillations (t < 80)
    phase1 = t < 80
    df.loc[phase1, "scale_proxy"] = df.loc[phase1, "scale_proxy"] + 0.3 * np.sin(2.0 * np.pi * t[phase1] / period)
    df.loc[phase1, "speed_proxy"] = df.loc[phase1, "speed_proxy"] + 0.3 * np.sin(2.0 * np.pi * t[phase1] / period)
    df.loc[phase1, "stop_proxy"] = df.loc[phase1, "stop_proxy"] + (-0.15 * np.sin(2.0 * np.pi * (t[phase1] - 10) / period))

    # Phase 2: structural shift (t >= 80)
    phase2 = t >= 80
    phase2_t = t[phase2] - 80

    # P jumps up and stays high (step)
    df.loc[phase2, "scale_proxy"] = df.loc[phase2, "scale_proxy"] + 0.5
    df.loc[phase2, "speed_proxy"] = df.loc[phase2, "speed_proxy"] + 0.4
    df.loc[phase2, "leverage_proxy"] = df.loc[phase2, "leverage_proxy"] + 0.3 * phase2_t / (n - 80)

    # O collapses (no recovery)
    df.loc[phase2, "stop_proxy"] = np.maximum(0.2, df.loc[phase2, "stop_proxy"].values - 0.15 * phase2_t / (n - 80))
    df.loc[phase2, "threshold_proxy"] = np.maximum(0.2, df.loc[phase2, "threshold_proxy"].values - 0.1 * phase2_t / (n - 80))

    # E accumulates (no reversion)
    df.loc[phase2, "impact_proxy"] = df.loc[phase2, "impact_proxy"] + 0.3 + 0.1 * phase2_t / (n - 80)
    df.loc[phase2, "hysteresis_proxy"] = df.loc[phase2, "hysteresis_proxy"] + 0.4

    # R degrades
    df.loc[phase2, "margin_proxy"] = np.maximum(0.1, df.loc[phase2, "margin_proxy"].values - 0.2 * phase2_t / (n - 80))
    df.loc[phase2, "recovery_time_proxy"] = 0.7 * np.ones(len(phase2_t))

    # Clip to valid ranges
    for col in df.columns:
        if col != "date":
            df[col] = np.clip(df[col], 0.0, 2.0)

    return df


def generate_hybrid_ii_to_iii(n: int = 300, seed: int = 45) -> pd.DataFrame:
    """
    Hybrid: Type II → Type III transition.

    Phase 1 (t=0-150): oscillations like Type II
    Phase 2 (t=150-300): gradually morph to Type III
    """
    df = _base_proxies(n=n, seed=seed)
    rng = np.random.default_rng(seed)

    period = 40.0
    t = np.arange(n, dtype=float)

    # Phase 1: Type II oscillations (t < 150)
    phase1 = t < 150
    df.loc[phase1, "scale_proxy"] = df.loc[phase1, "scale_proxy"] + 0.3 * np.sin(2.0 * np.pi * t[phase1] / period)
    df.loc[phase1, "speed_proxy"] = df.loc[phase1, "speed_proxy"] + 0.3 * np.sin(2.0 * np.pi * t[phase1] / period)
    df.loc[phase1, "stop_proxy"] = df.loc[phase1, "stop_proxy"] + (-0.15 * np.sin(2.0 * np.pi * (t[phase1] - 10) / period))

    # Phase 2: gradual transition to Type III (t >= 150)
    phase2 = t >= 150
    phase2_t = (t[phase2] - 150) / (n - 150)  # Normalized 0..1

    # P amplitude increases, then drifts
    df.loc[phase2, "scale_proxy"] = df.loc[phase2, "scale_proxy"] + 0.5 * phase2_t
    df.loc[phase2, "speed_proxy"] = df.loc[phase2, "speed_proxy"] + 0.4 * phase2_t
    df.loc[phase2, "leverage_proxy"] = df.loc[phase2, "leverage_proxy"] + 0.3 * phase2_t

    # O recovery weakens
    o_base = df.loc[phase2, "stop_proxy"].values
    df.loc[phase2, "stop_proxy"] = np.maximum(0.2, o_base - 0.3 * phase2_t)

    # E accumulation accelerates
    df.loc[phase2, "impact_proxy"] = df.loc[phase2, "impact_proxy"] + 0.2 * phase2_t + 0.1 * phase2_t**2
    df.loc[phase2, "hysteresis_proxy"] = df.loc[phase2, "hysteresis_proxy"] + 0.3 * phase2_t

    # R structural degradation
    df.loc[phase2, "recovery_time_proxy"] = 0.1 + 0.6 * phase2_t

    # Clip to valid ranges
    for col in df.columns:
        if col != "date":
            df[col] = np.clip(df[col], 0.0, 2.0)

    return df


def generate_all_scenarios(out_dir: str | Path) -> None:
    """Generate all 4 scenarios × 5 noise levels = 20 CSVs."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    scenarios = {
        "type_i_noise": generate_type_i_noise,
        "type_ii_oscillations": generate_type_ii_oscillations,
        "type_iii_bifurcation": generate_type_iii_bifurcation,
        "hybrid_ii_to_iii": generate_hybrid_ii_to_iii,
    }

    noise_levels = [0.05, 0.10, 0.15, 0.20, 0.25]

    for scenario_name, generator in scenarios.items():
        # Base scenario (no added noise)
        df_base = generator()
        base_path = out / f"{scenario_name}_base.csv"
        df_base.to_csv(base_path, index=False)
        print(f"✓ {base_path.name}")

        # Noise variants
        for noise_level in noise_levels:
            df_noisy = _add_noise(df_base, noise_level=noise_level, seed=9999)
            noise_path = out / f"{scenario_name}_{noise_level:.2f}.csv"
            df_noisy.to_csv(noise_path, index=False)
            print(f"✓ {noise_path.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate canonical calibration scenarios.")
    ap.add_argument("--out-dir", default="data/canonical_scenarios", help="Output directory.")
    args = ap.parse_args()

    print(f"Generating canonical scenarios to {args.out_dir}...")
    generate_all_scenarios(args.out_dir)
    print(f"\n✓ Generated 20 canonical scenarios (4 types × 5 noise levels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
