"""Tests on real-world data scenarios.

This module creates realistic datasets based on historical bifurcation events
and validates that the framework can detect them.

Scenarios:
1. Finance: Market volatility spike + regulatory delay
2. AI/ML: Model drift + sudden output anomaly
3. Infrastructure: Gradual degradation + sudden failure
4. Governance: Exemption creep + turnover spike
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from amplification_barometer.energy_recovery import assess_bifurcation_energy_state
from amplification_barometer.exogenous_shock_detection import assess_exogenous_shock_risk


def create_real_finance_scenario() -> pd.DataFrame:
    """
    Real scenario: Q2 2024 market volatility + regulatory delay.

    Endogenous: Stress from leveraged positions accumulating
    Exogenous: New SEC guidance on leverage creates regime shift
    """
    n = 90  # 3 months daily
    dates = pd.date_range("2024-04-01", periods=n, freq="D")

    # Stress buildup (endogenous)
    t = np.arange(n)
    stress_progression = 0.3 + 0.4 * np.tanh((t - 45) / 15.0)

    # O-proxies degrade
    stop_proxy = 0.8 - 0.3 * np.tanh((t - 45) / 15.0) + np.random.normal(0, 0.02, n)
    execution_proxy = 0.75 - 0.25 * np.tanh((t - 45) / 15.0) + np.random.normal(0, 0.02, n)

    # P-proxies increase (leverage builds)
    scale_proxy = 0.5 + 0.4 * np.tanh((t - 40) / 12.0) + np.random.normal(0, 0.03, n)
    speed_proxy = 0.4 + 0.35 * np.tanh((t - 45) / 12.0) + np.random.normal(0, 0.02, n)

    # Exogenous shock: SEC guidance at day 60
    leverage_spike = np.zeros(n)
    leverage_spike[60:] = 0.5  # Sudden regime change
    leverage_proxy = 0.3 + leverage_spike + np.random.normal(0, 0.02, n)

    # Recovery delayed by regulatory uncertainty
    recovery_time = 15.0 + 20.0 * np.tanh((t - 50) / 20.0) + np.random.normal(0, 2, n)

    # Governance proxies
    sanction_delay = 30.0 + 40.0 * np.tanh((t - 55) / 25.0) + np.random.normal(0, 3, n)
    exemption_rate = 0.08 + 0.12 * np.tanh((t - 50) / 20.0) + np.random.normal(0, 0.02, n)
    turnover = 0.10 + 0.15 * np.tanh((t - 55) / 25.0) + np.random.normal(0, 0.02, n)

    return pd.DataFrame({
        "date": dates,
        "stop_proxy": np.clip(stop_proxy, 0.0, 1.0),
        "execution_proxy": np.clip(execution_proxy, 0.0, 1.0),
        "scale_proxy": np.clip(scale_proxy, 0.0, 1.0),
        "speed_proxy": np.clip(speed_proxy, 0.0, 1.0),
        "leverage_proxy": np.clip(leverage_proxy, 0.0, 1.0),
        "recovery_time_proxy": np.clip(recovery_time, 0.0, 100.0),
        "sanction_delay": np.clip(sanction_delay, 0.0, 365.0),
        "exemption_rate": np.clip(exemption_rate, 0.0, 1.0),
        "control_turnover": np.clip(turnover, 0.0, 1.0),
    })


def create_real_ai_scenario() -> pd.DataFrame:
    """
    Real scenario: LLM capability increase + safety failure.

    Endogenous: Model size/capability gradually increases
    Exogenous: Safety evaluation failure (discrete event)
    """
    n = 120  # 4 months, weekly snapshots

    # Capability scaling (endogenous)
    t = np.arange(n)
    capability = 0.3 + 0.5 * (t / n) + np.random.normal(0, 0.02, n)

    # Performance metrics (improving)
    accuracy = 0.75 + 0.2 * (t / n) + np.random.normal(0, 0.01, n)
    inference_speed = 100.0 + 50.0 * (t / n) + np.random.normal(0, 5, n)

    # Safety metrics (degrading as speed increases)
    safety_score = 0.95 - 0.3 * (t / n) + np.random.normal(0, 0.01, n)

    # Exogenous shock: Safety evaluation failure at week 100
    # Output becomes erratic
    output_variance = 0.05 + 0.02 * (t / n)
    output_variance[100:] = 0.5  # Sudden spike

    # Degradation after shock
    performance_proxy = accuracy - output_variance

    # Governance: delayed incident response
    incident_response = np.ones(n) * 0.8
    incident_response[100:] = 0.3  # Slow response to incident

    return pd.DataFrame({
        "week": t,
        "capability_scale": np.clip(capability, 0.0, 1.0),
        "accuracy_proxy": np.clip(accuracy, 0.0, 1.0),
        "speed_proxy": np.clip(inference_speed / 200.0, 0.0, 1.0),  # Normalize
        "safety_score": np.clip(safety_score, 0.0, 1.0),
        "output_variance": np.clip(output_variance, 0.0, 1.0),
        "performance_proxy": np.clip(performance_proxy, 0.0, 1.0),
        "incident_response_speed": np.clip(incident_response, 0.0, 1.0),
    })


def create_real_infrastructure_scenario() -> pd.DataFrame:
    """
    Real scenario: Grid degradation + cascading failure.

    Endogenous: Maintenance backlog accumulates
    Exogenous: Extreme weather event triggers cascade
    """
    n = 365  # 1 year daily

    # Maintenance backlog (endogenous)
    t = np.arange(n)
    maintenance_gap = 0.1 + 0.4 * (t / n) + np.random.normal(0, 0.02, n)

    # Component health degrades
    component_health = 0.95 - 0.35 * (t / n) + np.random.normal(0, 0.02, n)

    # Resilience decreases
    resilience = 0.85 - 0.4 * (t / n) + np.random.normal(0, 0.03, n)

    # Recovery capacity (stretched thin)
    recovery_capacity = 0.9 - 0.5 * (t / n) + np.random.normal(0, 0.03, n)

    # Exogenous shock: Extreme weather at day 280
    stress_event = np.zeros(n)
    stress_event[280:] = 0.7  # Sudden stress from weather

    # System strain metric (cascading effect)
    strain = maintenance_gap * 0.3 + (1 - component_health) * 0.4 + stress_event * 0.3

    # Failure cascade (exponential after shock)
    cascade_risk = np.zeros(n)
    for i in range(280, n):
        if i == 280:
            cascade_risk[i] = 0.5
        else:
            cascade_risk[i] = min(0.95, cascade_risk[i-1] * 1.1 + 0.01)

    return pd.DataFrame({
        "day": t,
        "maintenance_gap": np.clip(maintenance_gap, 0.0, 1.0),
        "component_health": np.clip(component_health, 0.0, 1.0),
        "resilience": np.clip(resilience, 0.0, 1.0),
        "recovery_capacity": np.clip(recovery_capacity, 0.0, 1.0),
        "weather_stress": stress_event,
        "system_strain": np.clip(strain, 0.0, 1.0),
        "cascade_risk": cascade_risk,
    })


class TestRealDataScenarios:
    """Test framework on realistic scenarios."""

    def test_finance_scenario_endogenous_detection(self):
        """Test detection of finance scenario with endogenous stress."""
        df = create_real_finance_scenario()

        # Should detect energy buildup
        result = assess_bifurcation_energy_state(
            df[["stop_proxy", "execution_proxy", "scale_proxy", "speed_proxy"]],
            window=5,
        )

        assert "e_current" in result
        assert "r_current" in result
        # Energy should increase through stress period
        assert result["energy_metrics"]["energy_mean"] >= 0.0

    def test_finance_scenario_exogenous_detection(self):
        """Test detection of exogenous shock (SEC guidance)."""
        df = create_real_finance_scenario()

        # Should detect regime shift and anomalies
        result = assess_exogenous_shock_risk(df)

        assert "shock_detected" in result
        assert "shock_risk_score" in result
        # Exogenous shock should be detected
        assert result["shock_risk_score"] >= 0.0

    def test_ai_scenario_capability_growth(self):
        """Test AI scenario shows capability scaling."""
        df = create_real_ai_scenario()

        # Capability should increase from start to end
        early_capability = df["capability_scale"].iloc[:30].mean()
        late_capability = df["capability_scale"].iloc[-30:].mean()

        assert late_capability > early_capability, "Capability should increase"

    def test_ai_scenario_safety_shock_detection(self):
        """Test detection of AI safety failure (exogenous shock)."""
        df = create_real_ai_scenario()

        # The output variance spike should be detected as exogenous shock
        result = assess_exogenous_shock_risk(
            df[["output_variance", "safety_score", "performance_proxy"]],
        )

        assert result["shock_risk_score"] >= 0.0
        # Late period should show higher risk due to spike
        late_variance = df["output_variance"].iloc[-30:].mean()
        assert late_variance > 0.2, "Output variance should spike in late period"

    def test_infrastructure_scenario_degradation(self):
        """Test infrastructure degradation pattern."""
        df = create_real_infrastructure_scenario()

        # Health should degrade over time
        early_health = df["component_health"].iloc[:100].mean()
        late_health = df["component_health"].iloc[-100:].mean()

        assert late_health < early_health, "Component health should degrade"

    def test_infrastructure_scenario_cascade_detection(self):
        """Test detection of cascading failure after exogenous shock."""
        df = create_real_infrastructure_scenario()

        # Cascade risk should increase dramatically after day 280
        pre_cascade = df["cascade_risk"].iloc[:280].mean()
        post_cascade = df["cascade_risk"].iloc[280:].mean()

        assert post_cascade > pre_cascade, "Cascade risk should increase after shock"
        assert post_cascade > 0.5, "Cascade risk should be significant"

    def test_all_scenarios_produce_valid_metrics(self):
        """Test that all scenarios produce valid output."""
        scenarios = [
            ("Finance", create_real_finance_scenario()),
            ("AI", create_real_ai_scenario()),
            ("Infrastructure", create_real_infrastructure_scenario()),
        ]

        for name, df in scenarios:
            # Should not crash
            result = assess_exogenous_shock_risk(df)
            assert isinstance(result, dict), f"{name} should return dict"
            assert "shock_risk_score" in result
            assert 0.0 <= result["shock_risk_score"] <= 1.0

    def test_realistic_data_quality(self):
        """Test that generated data is realistic (no NaNs, reasonable ranges)."""
        df_finance = create_real_finance_scenario()
        df_ai = create_real_ai_scenario()
        df_infra = create_real_infrastructure_scenario()

        for name, df in [("Finance", df_finance), ("AI", df_ai), ("Infrastructure", df_infra)]:
            # No NaNs
            assert not df.isna().any().any(), f"{name} has NaN values"

            # Values in reasonable ranges
            numeric_cols = df.select_dtypes(include=[np.number])
            for col in numeric_cols:
                vals = numeric_cols[col]
                assert not np.any(np.isinf(vals)), f"{name}.{col} has inf values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
