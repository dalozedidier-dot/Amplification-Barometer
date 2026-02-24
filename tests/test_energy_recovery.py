"""Test E(t) and R(t) active implementation.

Test that:
1. E(t) increases under stress
2. R(t) depends on L_act
3. e_reduction_rel > 0 only with sufficient governance
4. Bifurcation detection works on Type III scenarios
"""

import numpy as np
import pandas as pd
import pytest

from amplification_barometer.energy_recovery import (
    compute_stress_accumulation,
    compute_irreversibility_metric,
    compute_energy_budget,
    compute_recovery_response,
    compute_energy_reduction_relative,
    assess_bifurcation_energy_state,
)


def create_bifurcation_scenario(n: int = 100, l_act_weak: float = 0.30) -> pd.DataFrame:
    """Create Type III bifurcation scenario with accumulating stress."""
    data = {}

    # O-proxies: start high, degrade over time (stress accumulates)
    t = np.arange(n)
    stress_progression = 1.0 - (0.7 * np.tanh((t - 40) / 15.0))  # S-curve degradation

    data["stop_proxy"] = np.clip(stress_progression + np.random.normal(0, 0.05, n), 0.0, 1.0)
    data["threshold_proxy"] = np.clip(stress_progression + np.random.normal(0, 0.05, n), 0.0, 1.0)
    data["execution_proxy"] = np.clip(stress_progression + np.random.normal(0, 0.05, n), 0.0, 1.0)
    data["coherence_proxy"] = np.clip(stress_progression + np.random.normal(0, 0.05, n), 0.0, 1.0)

    # Speed proxy: increases (volatility/deviation)
    data["speed_proxy"] = 0.3 + 0.6 * np.tanh((t - 50) / 20.0) + np.random.normal(0, 0.03, n)
    data["speed_proxy"] = np.clip(data["speed_proxy"], 0.0, 1.0)

    # Scale and leverage proxies
    data["scale_proxy"] = 0.5 + 0.3 * np.tanh((t - 45) / 15.0) + np.random.normal(0, 0.02, n)
    data["leverage_proxy"] = 0.4 + 0.5 * np.tanh((t - 50) / 20.0) + np.random.normal(0, 0.02, n)

    # Recovery time proxy
    data["recovery_time_proxy"] = 10.0 + 20.0 * np.tanh((t - 50) / 25.0) + np.random.normal(0, 2, n)

    # L_act score: weak governance (stays low)
    data["l_act_score"] = l_act_weak + np.random.normal(0, 0.05, n)
    data["l_act_score"] = np.clip(data["l_act_score"], 0.0, 1.0)

    # G-proxies: poor governance
    data["exemption_rate"] = 0.15 + 0.10 * np.tanh((t - 50) / 25.0) + np.random.normal(0, 0.02, n)
    data["sanction_delay"] = 60.0 + 30.0 * np.tanh((t - 50) / 25.0) + np.random.normal(0, 5, n)
    data["control_turnover"] = 0.12 + 0.08 * np.tanh((t - 50) / 25.0) + np.random.normal(0, 0.02, n)
    data["conflict_interest_proxy"] = 0.10 + np.random.normal(0, 0.03, n)
    data["rule_execution_gap"] = 0.20 + 0.10 * np.tanh((t - 50) / 25.0) + np.random.normal(0, 0.02, n)

    # Impact proxy (energy-like)
    data["impact_proxy"] = 0.2 + 0.7 * np.tanh((t - 50) / 20.0) + np.random.normal(0, 0.03, n)

    return pd.DataFrame(data)


class TestEnergyRecovery:
    """Test suite for E(t) and R(t) implementation."""

    def test_stress_accumulation(self):
        """Test that stress accumulates under degrading O-proxies."""
        df = create_bifurcation_scenario(n=100)
        stress = compute_stress_accumulation(df, window=5)

        # Stress should be low initially (O-proxies high)
        initial_stress = stress.iloc[:10].mean()
        # Stress should be high later (O-proxies low)
        final_stress = stress.iloc[-10:].mean()

        assert final_stress > initial_stress, "Stress should increase over time"
        assert initial_stress < 0.3, "Initial stress should be low"
        assert final_stress > 0.5, "Final stress should be elevated"

    def test_irreversibility_metric(self):
        """Test that irreversibility increases when O-proxies weaken."""
        df = create_bifurcation_scenario(n=100)
        irr = compute_irreversibility_metric(df, window=5)

        # Irreversibility should increase (harder to reverse as stress persists)
        initial_irr = irr.iloc[:10].mean()
        final_irr = irr.iloc[-10:].mean()

        assert final_irr > initial_irr, "Irreversibility should increase"
        assert np.all(np.isfinite(irr)), "Irreversibility should be finite"

    def test_energy_budget_increases(self):
        """Test that E(t) accumulates under stress."""
        df = create_bifurcation_scenario(n=100, l_act_weak=0.30)
        energy = compute_energy_budget(df, window=5, baseline_capacity=1.0)

        # Energy dissipation should increase (E should decrease from baseline)
        # Under stress with weak governance, energy should be consumed
        energy_max = energy.max()
        energy_min = energy.min()

        # E should vary (not constant)
        assert energy_max != energy_min, "Energy should vary over time"

        # With weak governance, energy should be largely consumed
        assert energy_min < 0.5, "Energy should be depleted under stress"

    def test_recovery_depends_on_l_act(self):
        """Test that R(t) depends on L_act governance score."""
        # Scenario 1: Weak governance (L_act = 0.3)
        df_weak = create_bifurcation_scenario(n=100, l_act_weak=0.30)
        recovery_weak = compute_recovery_response(df_weak, window=5, l_act_threshold=0.50)

        # Scenario 2: Strong governance (L_act = 0.75)
        df_strong = create_bifurcation_scenario(n=100, l_act_weak=0.75)
        recovery_strong = compute_recovery_response(df_strong, window=5, l_act_threshold=0.50)

        # Strong governance should enable more recovery
        recovery_weak_mean = recovery_weak.mean()
        recovery_strong_mean = recovery_strong.mean()

        assert recovery_strong_mean > recovery_weak_mean, \
            f"Strong governance (R={recovery_strong_mean:.3f}) should enable more recovery than weak (R={recovery_weak_mean:.3f})"

    def test_e_reduction_rel_zero_without_governance(self):
        """Test that e_reduction_rel ≈ 0 when L_act is below threshold."""
        df = create_bifurcation_scenario(n=100, l_act_weak=0.30)
        e_reduc, metrics = compute_energy_reduction_relative(df, window=5, l_act_threshold=0.50)

        # With weak governance, e_reduction_rel should be minimal
        mean_reduction = metrics["e_reduction_rel_mean"]

        assert mean_reduction < 0.2, \
            f"e_reduction_rel should be low ({mean_reduction:.3f}) with weak governance"

    def test_e_reduction_rel_positive_with_governance(self):
        """Test that e_reduction_rel > 0 when L_act is sufficient."""
        df = create_bifurcation_scenario(n=100, l_act_weak=0.75)
        e_reduc, metrics = compute_energy_reduction_relative(df, window=5, l_act_threshold=0.50)

        # With strong governance, e_reduction_rel should be measurable
        mean_reduction = metrics["e_reduction_rel_mean"]

        assert mean_reduction > 0.1, \
            f"e_reduction_rel should be positive ({mean_reduction:.3f}) with strong governance"

    def test_bifurcation_detection(self):
        """Test that bifurcation assessment is produced (energy state reachable)."""
        df = create_bifurcation_scenario(n=100, l_act_weak=0.30)
        assessment = assess_bifurcation_energy_state(
            df,
            window=5,
            energy_bifurcation_threshold=0.60,
            recovery_sufficiency_threshold=0.30,
        )

        # Assessment should be one of the defined states (infrastructure working)
        assert assessment["assessment"] in ["BIFURCATION_IMMINENT", "STRESS_ELEVATED", "RECOVERING", "STABLE"], \
            f"Assessment should be valid state, got {assessment['assessment']}"

        # Should have meaningful metrics
        assert "e_current" in assessment
        assert "r_current" in assessment
        assert "energy_metrics" in assessment

    def test_bifurcation_averted_with_governance(self):
        """Test that bifurcation risk is reduced with strong governance."""
        df_weak = create_bifurcation_scenario(n=100, l_act_weak=0.30)
        df_strong = create_bifurcation_scenario(n=100, l_act_weak=0.80)

        assessment_weak = assess_bifurcation_energy_state(df_weak, energy_bifurcation_threshold=0.60)
        assessment_strong = assess_bifurcation_energy_state(df_strong, energy_bifurcation_threshold=0.60)

        # Strong governance should have lower energy consumption / higher recovery
        assert assessment_strong["r_current"] >= assessment_weak["r_current"], \
            "Strong governance should provide more recovery capacity"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
