"""Energy E(t) and Recovery R(t) - Active Implementation for Bifurcation Detection

This module implements real, testable formulas for:
- E(t): Irreversibility energy (accumulates under stress, depends on system state)
- R(t): Recovery capacity (depends on L_act governance response)
- e_reduction_rel: Relative energy reduction (proven only if R > 0 AND L_act sufficient)

Key property: e_reduction_rel > 0 ONLY if governance (L_act) is sufficient.
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def compute_stress_accumulation(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Compute cumulative stress metric from control proxies.

    When stop_proxy, threshold_proxy, execution_proxy are low, stress accumulates.
    Higher value = more accumulated stress.

    Formula: Σ(1 - O_proxies) normalized
    """
    o_proxies = ["stop_proxy", "threshold_proxy", "execution_proxy", "coherence_proxy"]
    missing = [col for col in o_proxies if col not in df.columns]

    if missing:
        # Fallback: use generic stress from impact_proxy if available
        if "impact_proxy" in df.columns:
            return df["impact_proxy"].astype(float).rolling(window=window, center=False).mean()
        return pd.Series(0.0, index=df.index)

    # O-proxies: higher = better. So stress = 1 - mean(O-proxies)
    o_vals = df.loc[:, o_proxies].astype(float).to_numpy()
    o_mean = np.nanmean(o_vals, axis=1)  # mean of [stop, threshold, execution, coherence]
    stress = 1.0 - np.clip(o_mean, 0.0, 1.0)  # higher when O-proxies are low

    # Accumulation: smoothed over window
    stress_series = pd.Series(stress, index=df.index)
    stress_accum = stress_series.rolling(window=window, center=False, min_periods=1).mean()

    return stress_accum


def compute_irreversibility_metric(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Compute irreversibility metric: how much the system state persists despite control attempts.

    Higher = more irreversible (harder to recover).

    Formula:
      - High when: scale_proxy, speed_proxy, leverage_proxy are high
      - High when: O-proxies (stop, threshold, execution) fail to respond

    Irreversibility = persistent_deviation² × (1 - responsive_o_proxies)
    """
    # Deviation from baseline (using speed_proxy as proxy for volatility)
    if "speed_proxy" not in df.columns:
        return pd.Series(0.0, index=df.index)

    speed = df["speed_proxy"].astype(float).to_numpy()
    speed_dev = np.abs(speed - np.nanmedian(speed))  # deviation from median

    # O-responsiveness: how well controls can reduce deviation
    o_proxies = ["stop_proxy", "threshold_proxy", "execution_proxy"]
    o_vals = df.loc[:, [col for col in o_proxies if col in df.columns]].astype(float).to_numpy()
    o_mean = np.nanmean(o_vals, axis=1) if o_vals.size > 0 else np.zeros(len(df))
    o_responsive = np.clip(o_mean, 0.0, 1.0)

    # Irreversibility: high when deviation is large AND controls are weak
    irreversibility = (speed_dev ** 2) * (1.0 - np.clip(o_responsive, 0.0, 1.0))

    # Smooth over window
    irr_series = pd.Series(irreversibility, index=df.index)
    irr_smoothed = irr_series.rolling(window=window, center=False, min_periods=1).mean()

    return irr_smoothed


def compute_energy_budget(
    df: pd.DataFrame,
    window: int = 5,
    baseline_capacity: float = 1.0,
) -> pd.Series:
    """Compute E(t): Total energy budget (irreversibility potential).

    E(t) = baseline_capacity - cumulative_dissipation

    Where:
    - baseline_capacity = 1.0 (fixed, normalized)
    - cumulative_dissipation = integrated stress × irreversibility

    When E > threshold_e, bifurcation risk is high.
    """
    stress = compute_stress_accumulation(df, window=window).to_numpy()
    irr = compute_irreversibility_metric(df, window=window).to_numpy()

    # Energy dissipation rate: product of stress and irreversibility
    dissipation_rate = np.clip(stress * irr, 0.0, 1.0)

    # Cumulative dissipation
    dissipation_series = pd.Series(dissipation_rate, index=df.index)
    cumul_dissipation = dissipation_series.rolling(window=window * 2, center=False, min_periods=1).sum()

    # Normalize to [0, baseline_capacity]
    max_cumul = np.nanpercentile(cumul_dissipation.to_numpy(), 95) if len(cumul_dissipation) > 0 else 1.0
    max_cumul = max(max_cumul, 1e-6)
    cumul_dissipation_normalized = baseline_capacity * np.clip(cumul_dissipation.to_numpy() / max_cumul, 0.0, 1.0)

    # E(t) = baseline - dissipation (when E < 0, bifurcation occurs)
    energy = baseline_capacity - cumul_dissipation_normalized

    return pd.Series(energy, index=df.index)


def compute_recovery_response(
    df: pd.DataFrame,
    window: int = 5,
    l_act_threshold: float = 0.50,
) -> pd.Series:
    """Compute R(t): Recovery response strength (depends on L_act).

    R(t) = L_act × O_responsiveness × recovery_speed

    Key: R > 0 ONLY if L_act ≥ threshold.

    Where:
    - L_act = governance activation (from l_operator)
    - O_responsiveness = ability of O-proxies to act
    - recovery_speed = inverse of recovery_time_proxy
    """
    # Get L_act (if available)
    if "l_act_score" in df.columns:
        l_act = df["l_act_score"].astype(float).to_numpy()
    elif "L_ACT" in df.columns:
        l_act = df["L_ACT"].astype(float).to_numpy()
    else:
        # Fallback: use average of governance proxies
        g_proxies = ["exemption_rate", "sanction_delay", "control_turnover",
                     "conflict_interest_proxy", "rule_execution_gap"]
        g_vals = df.loc[:, [col for col in g_proxies if col in df.columns]].astype(float).to_numpy()
        if g_vals.size == 0:
            l_act = np.zeros(len(df))
        else:
            # Good governance = low exemption_rate, sanction_delay, etc.
            # Invert: good_governance = 1 - mean(bad_indicators)
            bad_gov = np.nanmean(g_vals, axis=1)
            l_act = np.clip(1.0 - bad_gov, 0.0, 1.0)

    # O-responsiveness
    o_proxies = ["stop_proxy", "threshold_proxy", "execution_proxy", "coherence_proxy"]
    o_vals = df.loc[:, [col for col in o_proxies if col in df.columns]].astype(float).to_numpy()
    o_responsive = np.nanmean(o_vals, axis=1) if o_vals.size > 0 else np.zeros(len(df))

    # Recovery speed (inverse of recovery time)
    if "recovery_time_proxy" in df.columns:
        recovery_time = df["recovery_time_proxy"].astype(float).to_numpy()
        recovery_speed = 1.0 / (1.0 + np.clip(recovery_time, 0.0, None))
    else:
        recovery_speed = np.ones(len(df)) * 0.5  # default: moderate recovery speed

    # R(t) = L_act × O_responsive × recovery_speed, gated by L_act threshold
    l_act_binary_gate = (l_act >= l_act_threshold).astype(float)
    recovery = l_act_binary_gate * l_act * o_responsive * recovery_speed

    return pd.Series(recovery, index=df.index)


def compute_energy_reduction_relative(
    df: pd.DataFrame,
    window: int = 5,
    l_act_threshold: float = 0.50,
) -> Tuple[pd.Series, Dict[str, float]]:
    """Compute e_reduction_rel: Relative energy reduction from governance response.

    Returns:
    - e_reduction_rel: Series of relative reduction values (0.0 if no response)
    - metrics_dict: Summary statistics

    Key insight: e_reduction_rel > 0 ONLY if:
    1. E(t) is sufficiently high (stress exists)
    2. R(t) > 0 (governance activated)
    3. L_act ≥ threshold
    """
    energy = compute_energy_budget(df, window=window).to_numpy()
    recovery = compute_recovery_response(df, window=window, l_act_threshold=l_act_threshold).to_numpy()

    # Baseline energy (pre-recovery)
    energy_max = np.nanmax(np.abs(energy)) if np.any(np.isfinite(energy)) else 1.0
    energy_max = max(energy_max, 1e-6)

    # Energy reduction = recovery effort × energy level
    # Only reduce if both energy and recovery are present
    energy_reduction = np.clip(recovery * np.clip(energy / energy_max, 0.0, 1.0), 0.0, 1.0)

    # Relative reduction (what fraction of max energy is recovered?)
    e_reduction_rel = energy_reduction / 1.0 if energy_max > 0.0 else np.zeros_like(energy)
    e_reduction_rel = np.clip(e_reduction_rel, 0.0, 1.0)

    # Metrics
    metrics = {
        "energy_mean": float(np.nanmean(energy)),
        "energy_max": float(np.nanmax(energy)) if np.any(np.isfinite(energy)) else 0.0,
        "energy_p90": float(np.nanpercentile(energy, 90)) if np.any(np.isfinite(energy)) else 0.0,
        "recovery_mean": float(np.nanmean(recovery)),
        "recovery_max": float(np.nanmax(recovery)) if np.any(np.isfinite(recovery)) else 0.0,
        "e_reduction_rel_mean": float(np.nanmean(e_reduction_rel)),
        "e_reduction_rel_max": float(np.nanmax(e_reduction_rel)) if np.any(np.isfinite(e_reduction_rel)) else 0.0,
    }

    return pd.Series(e_reduction_rel, index=df.index), metrics


def assess_bifurcation_energy_state(
    df: pd.DataFrame,
    window: int = 5,
    energy_bifurcation_threshold: float = 0.75,
    recovery_sufficiency_threshold: float = 0.40,
) -> Dict[str, any]:
    """Comprehensive bifurcation energy assessment.

    Returns multidimensional verdict on:
    - E_t: Current energy level
    - R_t: Recovery capacity
    - e_reduction_rel: Effectiveness of governance
    - bifurcation_risk: Boolean (E > threshold AND R < recovery_sufficiency)
    """
    energy = compute_energy_budget(df, window=window).to_numpy()
    recovery, energy_metrics = compute_energy_reduction_relative(df, window=window)
    recovery = recovery.to_numpy()

    # Latest state
    e_current = float(energy[-1]) if len(energy) > 0 else 0.0
    r_current = float(recovery[-1]) if len(recovery) > 0 else 0.0
    e_reduc_current = energy_metrics["e_reduction_rel_mean"]

    # Bifurcation risk: high energy + low recovery
    bifurcation_risk = bool(
        (e_current > energy_bifurcation_threshold) and
        (r_current < recovery_sufficiency_threshold)
    )

    # Trend: is energy increasing or recovering?
    if len(energy) >= 2:
        energy_slope = float(energy[-1] - energy[-5:].mean())  # last vs. 5-point average
    else:
        energy_slope = 0.0

    if len(recovery) >= 2:
        recovery_slope = float(recovery[-1] - recovery[-5:].mean())
    else:
        recovery_slope = 0.0

    return {
        "e_current": e_current,
        "r_current": r_current,
        "e_reduction_rel": e_reduc_current,
        "bifurcation_risk": bifurcation_risk,
        "energy_slope": energy_slope,
        "recovery_slope": recovery_slope,
        "energy_metrics": energy_metrics,
        "assessment": (
            "BIFURCATION_IMMINENT" if bifurcation_risk and energy_slope > 0.1
            else "STRESS_ELEVATED" if e_current > energy_bifurcation_threshold
            else "RECOVERING" if recovery_slope > 0.1
            else "STABLE"
        ),
    }
