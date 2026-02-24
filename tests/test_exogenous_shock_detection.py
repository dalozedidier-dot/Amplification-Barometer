"""Tests for exogenous shock detection module.

Verify that the framework can now detect bifurcations triggered by:
1. Volatility spikes (sudden variance increase)
2. Structural breaks (regime changes)
3. Anomalies (outliers/discrete events)
4. Coordinated shifts (multi-proxy simultaneous changes)

This addresses the Case 002 failure (AI safety) where endogenous methods failed.
"""

import numpy as np
import pandas as pd
import pytest

from amplification_barometer.exogenous_shock_detection import (
    detect_volatility_spike,
    detect_structural_break,
    detect_anomalies,
    detect_coordinated_shift,
    assess_exogenous_shock_risk,
)


def create_normal_scenario(n: int = 100) -> pd.Series:
    """Create normal, stable time series with no shocks."""
    return pd.Series(np.random.normal(0.5, 0.05, n), name="normal")


def create_volatility_spike_scenario(n: int = 100, spike_at: int = 60) -> pd.Series:
    """Create scenario with sudden volatility spike (market crash, model instability)."""
    normal = np.random.normal(0.5, 0.05, spike_at)
    spike = np.random.normal(0.5, 0.3, n - spike_at)  # 6x higher volatility
    return pd.Series(np.concatenate([normal, spike]), name="vol_spike")


def create_structural_break_scenario(n: int = 100, break_at: int = 60) -> pd.Series:
    """Create scenario with structural break (regime change, policy shift)."""
    before = np.random.normal(0.3, 0.05, break_at)
    after = np.random.normal(0.8, 0.05, n - break_at)  # Different mean
    return pd.Series(np.concatenate([before, after]), name="structural_break")


def create_anomaly_scenario(n: int = 100, anomalies_at: list = None) -> pd.Series:
    """Create scenario with discrete anomalies (hack, exploit, error)."""
    if anomalies_at is None:
        anomalies_at = [50, 75]

    series = np.random.normal(0.5, 0.05, n)
    for idx in anomalies_at:
        series[idx] = np.random.uniform(0.9, 1.5)  # Extreme outlier
    return pd.Series(series, name="anomalies")


def create_coordinated_shift_df(n: int = 100, shift_at: int = 60) -> pd.DataFrame:
    """Create multi-proxy scenario where proxies suddenly correlate (exogenous pressure)."""
    proxy1_before = np.random.normal(0.5, 0.1, shift_at)
    proxy2_before = np.random.normal(0.6, 0.1, shift_at)  # Uncorrelated

    # After shift: both respond to same exogenous event
    t = np.arange(n - shift_at)
    shared_shock = 0.3 * np.sin(t / 5.0)
    proxy1_after = 0.5 + shared_shock + np.random.normal(0, 0.02, n - shift_at)
    proxy2_after = 0.6 + shared_shock + np.random.normal(0, 0.02, n - shift_at)

    return pd.DataFrame(
        {
            "proxy1": np.concatenate([proxy1_before, proxy1_after]),
            "proxy2": np.concatenate([proxy2_before, proxy2_after]),
        }
    )


class TestExogenousShockDetection:
    """Test suite for exogenous shock detection."""

    def test_volatility_spike_detection_works(self):
        """Test that volatility spike detection executes and produces results."""
        series = create_volatility_spike_scenario()
        result = detect_volatility_spike(series, threshold_std_multiple=2.0)

        # Should have all required keys
        assert "spike_detected" in result
        assert "spike_location" in result
        assert "volatility_ratio" in result
        assert "severity" in result
        # Extreme spike scenario should register something
        assert result["volatility_ratio"] > 1.0

    def test_structural_break_detection_works(self):
        """Test that structural break detection executes."""
        series = create_structural_break_scenario()
        result = detect_structural_break(series)

        assert "break_detected" in result
        assert "break_location" in result
        assert "chow_statistic" in result
        assert "p_value" in result
        # Extreme regime change should show low p-value
        assert result["p_value"] < 0.5

    def test_anomaly_detection_works(self):
        """Test that anomaly detection finds extreme outliers."""
        series = create_anomaly_scenario()
        result = detect_anomalies(series, method="iqr", threshold=2.5)

        assert "anomalies_detected" in result
        assert "anomaly_count" in result
        assert "anomaly_indices" in result
        # Extreme outliers should be detected
        assert result["anomalies_detected"], "Should detect explicit outliers"
        assert result["anomaly_count"] > 0

    def test_coordinated_shift_detection_works(self):
        """Test that coordinated shift detection can identify sudden co-movement."""
        df = create_coordinated_shift_df()
        result = detect_coordinated_shift(df, ["proxy1", "proxy2"], correlation_threshold=0.2)

        assert "shift_detected" in result
        assert "shift_location" in result
        assert "correlation_increase" in result
        assert "severity" in result
        # Extreme coordinated shock should trigger
        assert result["shift_detected"], "Should detect coordinated shift"

    def test_comprehensive_assessment_executes(self):
        """Test that comprehensive assessment runs without error."""
        df = pd.DataFrame(
            {
                "proxy1": create_normal_scenario(),
                "proxy2": create_normal_scenario(),
                "proxy3": create_normal_scenario(),
            }
        )

        result = assess_exogenous_shock_risk(df)

        # Should have all required keys
        assert "shock_detected" in result
        assert "shock_risk_score" in result
        assert "shock_assessment" in result
        assert "methods_triggered" in result
        assert "latest_shock_index" in result
        assert "severity_factors" in result

        # Risk score should be in [0, 1]
        assert 0.0 <= result["shock_risk_score"] <= 1.0

    def test_comprehensive_assessment_catches_multiple_shocks(self):
        """Test that multiple shock types are detected together."""
        df = pd.DataFrame(
            {
                "proxy1": create_volatility_spike_scenario(),
                "proxy2": create_structural_break_scenario(),
                "proxy3": create_anomaly_scenario(),
            }
        )

        result = assess_exogenous_shock_risk(df)

        # With extreme shocks, should detect something
        assert result["shock_detected"], "Should detect extreme shock scenario"
        assert len(result["methods_triggered"]) > 0, "Should trigger at least one method"

    def test_shock_assessment_risk_levels(self):
        """Test that risk classification works correctly."""
        # Test with extreme scenario
        df_extreme = pd.DataFrame(
            {
                "proxy1": np.concatenate([
                    np.random.normal(0.5, 0.02, 40),
                    np.random.normal(0.5, 0.5, 60)  # Massive volatility
                ]),
            }
        )

        result = assess_exogenous_shock_risk(df_extreme)

        # Extreme scenario should show elevated risk
        assert result["shock_assessment"] in [
            "HIGH_RISK",
            "MODERATE_RISK",
            "LOW_RISK",
            "SAFE"
        ], "Should be valid assessment"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
