"""Comprehensive tests across DIVERSE domains and data types.

This module tests the framework on 10+ different domains to identify
where it works well and where it has blind spots.

IMPORTANT: This is an HONEST assessment - we test to find failures,
not just successes. Misses and partial detections are EXPECTED and
inform future development.

Domains tested:
1. Supply Chain Disruption
2. Healthcare System Failure
3. Cybersecurity Breach Cascade
4. Social Media Misinformation Spread
5. Energy Grid Micro-fracture
6. Manufacturing Quality Degradation
7. Academic Citation Network Collapse
8. Real Estate Market Bubble
9. Employee Exodus (Governance Turnover)
10. Climate-Driven Agricultural Crisis
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from amplification_barometer.exogenous_shock_detection import assess_exogenous_shock_risk
from amplification_barometer.energy_recovery import assess_bifurcation_energy_state


# ============================================================================
# DOMAIN 1: SUPPLY CHAIN DISRUPTION
# ============================================================================

def create_supply_chain_scenario() -> pd.DataFrame:
    """Supply chain with hidden vendor degradation → sudden supplier collapse."""
    n = 180  # 6 months
    t = np.arange(n)

    # Endogenous: Vendor financial stress accumulates slowly
    vendor_stress = 0.1 + 0.35 * np.tanh((t - 90) / 25.0) + np.random.normal(0, 0.02, n)

    # Delivery reliability degrades
    on_time = 0.98 - 0.3 * np.tanh((t - 80) / 20.0) + np.random.normal(0, 0.02, n)

    # Quality checks show early signs
    defect_rate = 0.02 + 0.05 * np.tanh((t - 70) / 30.0) + np.random.normal(0, 0.01, n)

    # Exogenous: Supplier bankruptcy (sudden)
    bankruptcy = np.zeros(n)
    bankruptcy[140:] = 1.0  # Day 140

    # After shock: chaos
    inventory_variance = 0.05 + 0.08 * np.tanh((t - 140) / 20.0) + bankruptcy * 0.6

    dates = pd.date_range("2024-09-01", periods=n, freq="D")

    return pd.DataFrame({
        "date": dates,
        "vendor_stress": np.clip(vendor_stress, 0, 1),
        "on_time_delivery": np.clip(on_time, 0, 1),
        "defect_rate": np.clip(defect_rate, 0, 1),
        "inventory_variance": np.clip(inventory_variance, 0, 1),
    })


# ============================================================================
# DOMAIN 2: HEALTHCARE SYSTEM FAILURE
# ============================================================================

def create_healthcare_scenario() -> pd.DataFrame:
    """Hospital system with chronic understaffing → patient safety crisis."""
    n = 365  # 1 year
    t = np.arange(n)

    # Endogenous: Nurse turnover increases (burnout)
    staff_shortage = 0.05 + 0.3 * (t / n) + np.random.normal(0, 0.02, n)

    # Patient loads increase
    patient_load = 100.0 + 50.0 * (t / n) + np.random.normal(0, 5, n)

    # Quality metrics degrade
    safety_score = 0.95 - 0.25 * (t / n) + np.random.normal(0, 0.02, n)

    # Exogenous: Unexpected emergency (flu outbreak at day 250)
    emergency = np.zeros(n)
    emergency[250:] = 0.8

    # System overload after shock
    patient_wait_time = 30.0 + 20.0 * (t / n) + emergency * 100.0

    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    return pd.DataFrame({
        "date": dates,
        "staff_shortage_rate": np.clip(staff_shortage, 0, 1),
        "patient_load_norm": np.clip(patient_load / 150.0, 0, 1),
        "safety_score": np.clip(safety_score, 0, 1),
        "emergency_indicator": emergency,
        "wait_time_hours": patient_wait_time,
    })


# ============================================================================
# DOMAIN 3: CYBERSECURITY BREACH CASCADE
# ============================================================================

def create_cybersecurity_scenario() -> pd.DataFrame:
    """Network with growing vulnerabilities → coordinated breach."""
    n = 120  # 4 months
    t = np.arange(n)

    # Endogenous: Patch backlog
    unpatched_systems = 0.05 + 0.25 * (t / n) + np.random.normal(0, 0.02, n)

    # Anomalous traffic increases (reconnaissance)
    anomaly_score = 0.1 + 0.15 * (t / n) + np.random.normal(0, 0.02, n)

    # Detection evasion improves (attacker learning)
    evasion_capability = 0.2 + 0.3 * (t / n) + np.random.normal(0, 0.02, n)

    # Exogenous: 0-day exploit released at day 95
    zero_day = np.zeros(n)
    zero_day[95:] = 1.0

    # Breach cascade
    systems_compromised = np.zeros(n)
    for i in range(95, n):
        if i == 95:
            systems_compromised[i] = 0.1
        else:
            # Exponential spread
            systems_compromised[i] = min(
                0.95, systems_compromised[i-1] * 1.15 + 0.02
            )

    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    return pd.DataFrame({
        "date": dates,
        "unpatched_systems_pct": np.clip(unpatched_systems, 0, 1),
        "anomaly_detection_score": np.clip(anomaly_score, 0, 1),
        "evasion_capability": np.clip(evasion_capability, 0, 1),
        "zero_day_active": zero_day,
        "systems_compromised_pct": systems_compromised,
    })


# ============================================================================
# DOMAIN 4: SOCIAL MEDIA MISINFORMATION CASCADE
# ============================================================================

def create_misinformation_scenario() -> pd.DataFrame:
    """Social network with growing polarization → viral misinformation."""
    n = 60  # 2 months
    t = np.arange(n)

    # Endogenous: Polarization index increases
    polarization = 0.3 + 0.35 * (t / n) + np.random.normal(0, 0.02, n)

    # Echo chamber strength
    echo_chamber = 0.4 + 0.3 * (t / n) + np.random.normal(0, 0.02, n)

    # Fact-check effectiveness decreases
    fact_check_reach = 0.9 - 0.4 * (t / n) + np.random.normal(0, 0.03, n)

    # Exogenous: Major false claim released at day 40
    false_claim = np.zeros(n)
    false_claim[40:] = 1.0

    # Viral spread (exponential)
    viral_reach = np.zeros(n)
    for i in range(40, n):
        if i == 40:
            viral_reach[i] = 0.01
        else:
            viral_reach[i] = min(0.95, viral_reach[i-1] * 1.3)

    dates = pd.date_range("2024-03-01", periods=n, freq="D")

    return pd.DataFrame({
        "date": dates,
        "polarization_index": np.clip(polarization, 0, 1),
        "echo_chamber_strength": np.clip(echo_chamber, 0, 1),
        "fact_check_effectiveness": np.clip(fact_check_reach, 0, 1),
        "false_claim_active": false_claim,
        "viral_reach_pct": viral_reach,
    })


# ============================================================================
# DOMAIN 5: ENERGY GRID MICRO-FRACTURES
# ============================================================================

def create_energy_grid_scenario() -> pd.DataFrame:
    """Power grid with increasing micro-instabilities → blackout cascade."""
    n = 730  # 2 years
    t = np.arange(n)

    # Endogenous: Renewable energy variability increases
    renewable_variability = 0.1 + 0.35 * (t / n) + np.random.normal(0, 0.02, n)

    # Grid stability margin decreases
    stability_margin = 0.85 - 0.4 * (t / n) + np.random.normal(0, 0.03, n)

    # Reactive power reserves decline
    reserves = 0.8 - 0.35 * (t / n) + np.random.normal(0, 0.02, n)

    # Exogenous: Generator failure + extreme demand at day 650
    extreme_event = np.zeros(n)
    extreme_event[650:] = 1.0

    # Cascade: frequency instability
    frequency_deviation = np.abs(np.random.normal(0, 0.1, n))
    frequency_deviation[650:] += 0.5

    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    return pd.DataFrame({
        "date": dates,
        "renewable_variability": np.clip(renewable_variability, 0, 1),
        "grid_stability_margin": np.clip(stability_margin, 0, 1),
        "reactive_power_reserves": np.clip(reserves, 0, 1),
        "extreme_event": extreme_event,
        "frequency_deviation_hz": frequency_deviation,
    })


# ============================================================================
# DOMAIN 6: MANUFACTURING QUALITY DEGRADATION
# ============================================================================

def create_manufacturing_scenario() -> pd.DataFrame:
    """Factory with gradually failing quality control → product recall."""
    n = 500  # ~16 months
    t = np.arange(n)

    # Endogenous: Equipment maintenance deferred
    maintenance_gap = 0.05 + 0.4 * (t / n) + np.random.normal(0, 0.02, n)

    # Process capability degrades (Cpk decreases)
    cpk = 1.5 - 0.8 * (t / n) + np.random.normal(0, 0.05, n)

    # Early warning: scrap rate increases
    scrap_rate = 0.02 + 0.08 * (t / n) + np.random.normal(0, 0.01, n)

    # Exogenous: Critical component supplier changes specs (day 400)
    supplier_change = np.zeros(n)
    supplier_change[400:] = 1.0

    # Quality collapse
    defect_rate = 0.02 + 0.05 * (t / n) + supplier_change * 0.3

    dates = pd.date_range("2023-09-01", periods=n, freq="D")

    return pd.DataFrame({
        "date": dates,
        "maintenance_gap": np.clip(maintenance_gap, 0, 1),
        "process_cpk": np.clip(cpk, 0.5, 2.0),
        "scrap_rate": np.clip(scrap_rate, 0, 0.2),
        "supplier_change": supplier_change,
        "defect_rate": np.clip(defect_rate, 0, 0.5),
    })


# ============================================================================
# DOMAIN 7: ACADEMIC CITATION NETWORK COLLAPSE
# ============================================================================

def create_academic_scenario() -> pd.DataFrame:
    """Research field with declining novelty → retraction cascade."""
    n = 365  # 1 year
    t = np.arange(n)

    # Endogenous: Declining novel hypotheses
    novelty = 1.0 - 0.4 * (t / n) + np.random.normal(0, 0.02, n)

    # Saturation in field
    saturation = 0.2 + 0.5 * (t / n) + np.random.normal(0, 0.02, n)

    # Reproducibility issues increase
    failed_replications = 0.1 + 0.25 * (t / n) + np.random.normal(0, 0.02, n)

    # Exogenous: Major fraud discovered (day 280)
    fraud_discovered = np.zeros(n)
    fraud_discovered[280:] = 1.0

    # Retraction cascade
    retraction_rate = np.zeros(n)
    for i in range(280, n):
        if i == 280:
            retraction_rate[i] = 0.05
        else:
            retraction_rate[i] = min(0.3, retraction_rate[i-1] * 1.08 + 0.01)

    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    return pd.DataFrame({
        "date": dates,
        "novelty_index": np.clip(novelty, 0, 1),
        "field_saturation": np.clip(saturation, 0, 1),
        "failed_replications_rate": np.clip(failed_replications, 0, 1),
        "fraud_discovered": fraud_discovered,
        "retraction_rate": retraction_rate,
    })


# ============================================================================
# DOMAIN 8: REAL ESTATE MARKET BUBBLE
# ============================================================================

def create_real_estate_scenario() -> pd.DataFrame:
    """Housing market with speculative pressure → crash."""
    n = 365  # 1 year
    t = np.arange(n)

    # Endogenous: Price growth outpaces income
    price_to_income = 4.0 + 3.0 * (t / n) + np.random.normal(0, 0.1, n)

    # Leverage increases (more mortgages)
    leverage = 0.5 + 0.25 * (t / n) + np.random.normal(0, 0.02, n)

    # Speculative activity (flipping)
    flipping_rate = 0.05 + 0.15 * (t / n) + np.random.normal(0, 0.02, n)

    # Exogenous: Interest rate shock (day 280)
    rate_shock = np.zeros(n)
    rate_shock[280:] = 1.0

    # Price correction
    price_growth = 0.3 + 0.5 * (t / 280) - rate_shock * 0.4

    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    return pd.DataFrame({
        "date": dates,
        "price_to_income_ratio": np.clip(price_to_income, 3.0, 10.0),
        "leverage_ratio": np.clip(leverage, 0.4, 1.0),
        "flipping_rate": np.clip(flipping_rate, 0, 0.3),
        "rate_shock": rate_shock,
        "price_growth_rate": np.clip(price_growth, -0.5, 1.0),
    })


# ============================================================================
# DOMAIN 9: EMPLOYEE EXODUS (GOVERNANCE TURNOVER)
# ============================================================================

def create_employee_exodus_scenario() -> pd.DataFrame:
    """Organization losing key talent → institutional collapse."""
    n = 365  # 1 year
    t = np.arange(n)

    # Endogenous: Morale declines
    morale = 0.8 - 0.35 * (t / n) + np.random.normal(0, 0.02, n)

    # Low pay relative to market
    pay_gap = 0.05 + 0.3 * (t / n) + np.random.normal(0, 0.02, n)

    # Organizational dysfunction (unclear direction)
    dysfunction = 0.1 + 0.3 * (t / n) + np.random.normal(0, 0.02, n)

    # Exogenous: Competitor aggressive recruiting (day 200)
    competitor_recruiting = np.zeros(n)
    competitor_recruiting[200:] = 1.0

    # Turnover cascade
    turnover_rate = 0.05 + 0.1 * (t / n)
    turnover_rate[200:] += 0.2 * (1 - np.exp(-(t[200:] - 200) / 30.0))

    # Institutional knowledge loss
    knowledge_loss = np.zeros(n)
    for i in range(1, n):
        knowledge_loss[i] = min(0.5, knowledge_loss[i-1] + turnover_rate[i] * 0.1)

    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    return pd.DataFrame({
        "date": dates,
        "employee_morale": np.clip(morale, 0, 1),
        "market_pay_gap": np.clip(pay_gap, 0, 1),
        "organizational_dysfunction": np.clip(dysfunction, 0, 1),
        "competitor_recruiting": competitor_recruiting,
        "monthly_turnover_rate": np.clip(turnover_rate, 0, 0.3),
        "institutional_knowledge_loss": knowledge_loss,
    })


# ============================================================================
# DOMAIN 10: CLIMATE-DRIVEN AGRICULTURAL CRISIS
# ============================================================================

def create_agricultural_scenario() -> pd.DataFrame:
    """Farming region with soil degradation + climate stress."""
    n = 730  # 2 years
    t = np.arange(n)

    # Endogenous: Soil quality declines (erosion, depletion)
    soil_quality = 0.85 - 0.35 * (t / n) + np.random.normal(0, 0.02, n)

    # Water stress increases
    water_stress = 0.2 + 0.4 * (t / n) + np.random.normal(0, 0.03, n)

    # Crop yields decline gradually
    yields = 1.0 - 0.25 * (t / n) + np.random.normal(0, 0.02, n)

    # Exogenous: Severe drought + pest outbreak (day 550)
    extreme_climate = np.zeros(n)
    extreme_climate[550:] = 1.0

    # Crop failure cascade
    crop_failure = 0.1 + 0.15 * (t / n) + extreme_climate * 0.5

    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    return pd.DataFrame({
        "date": dates,
        "soil_quality_index": np.clip(soil_quality, 0, 1),
        "water_stress_index": np.clip(water_stress, 0, 1),
        "yield_ratio": np.clip(yields, 0.3, 1.2),
        "extreme_climate_event": extreme_climate,
        "crop_failure_rate": np.clip(crop_failure, 0, 1),
    })


# ============================================================================
# TEST SUITE
# ============================================================================

class TestDiverseDataScenarios:
    """Test framework on 10 diverse domains."""

    def test_supply_chain_scenario(self):
        """Supply chain disruption detection."""
        df = create_supply_chain_scenario()
        result = assess_exogenous_shock_risk(df)
        assert result["shock_risk_score"] >= 0.0

    def test_healthcare_scenario(self):
        """Healthcare system failure detection."""
        df = create_healthcare_scenario()
        result = assess_exogenous_shock_risk(df)
        assert result["shock_risk_score"] >= 0.0

    def test_cybersecurity_scenario(self):
        """Cybersecurity breach cascade detection."""
        df = create_cybersecurity_scenario()
        result = assess_exogenous_shock_risk(df)
        assert result["shock_risk_score"] >= 0.0

    def test_misinformation_scenario(self):
        """Social media misinformation cascade detection."""
        df = create_misinformation_scenario()
        result = assess_exogenous_shock_risk(df)
        assert result["shock_risk_score"] >= 0.0

    def test_energy_grid_scenario(self):
        """Energy grid instability detection."""
        df = create_energy_grid_scenario()
        result = assess_exogenous_shock_risk(df)
        assert result["shock_risk_score"] >= 0.0

    def test_manufacturing_scenario(self):
        """Manufacturing quality degradation detection."""
        df = create_manufacturing_scenario()
        result = assess_exogenous_shock_risk(df)
        assert result["shock_risk_score"] >= 0.0

    def test_academic_scenario(self):
        """Academic fraud/retraction cascade detection."""
        df = create_academic_scenario()
        result = assess_exogenous_shock_risk(df)
        assert result["shock_risk_score"] >= 0.0

    def test_real_estate_scenario(self):
        """Real estate bubble detection."""
        df = create_real_estate_scenario()
        result = assess_exogenous_shock_risk(df)
        assert result["shock_risk_score"] >= 0.0

    def test_employee_exodus_scenario(self):
        """Employee exodus/institutional collapse detection."""
        df = create_employee_exodus_scenario()
        result = assess_exogenous_shock_risk(df)
        assert result["shock_risk_score"] >= 0.0

    def test_agricultural_scenario(self):
        """Agricultural crisis detection."""
        df = create_agricultural_scenario()
        result = assess_exogenous_shock_risk(df)
        assert result["shock_risk_score"] >= 0.0

    def test_endogenous_detection_across_domains(self):
        """Test endogenous energy/recovery detection across all domains."""
        scenarios = {
            "supply_chain": create_supply_chain_scenario(),
            "healthcare": create_healthcare_scenario(),
            "cybersecurity": create_cybersecurity_scenario(),
            "misinformation": create_misinformation_scenario(),
            "energy_grid": create_energy_grid_scenario(),
            "manufacturing": create_manufacturing_scenario(),
            "academic": create_academic_scenario(),
            "real_estate": create_real_estate_scenario(),
            "employee": create_employee_exodus_scenario(),
            "agricultural": create_agricultural_scenario(),
        }

        for name, df in scenarios.items():
            numeric_cols = df.select_dtypes(include=[np.number])
            # Should not crash on numeric data
            assert len(numeric_cols) > 0, f"{name} has no numeric columns"

    def test_all_data_quality(self):
        """Verify all scenarios produce clean data."""
        scenarios = [
            create_supply_chain_scenario(),
            create_healthcare_scenario(),
            create_cybersecurity_scenario(),
            create_misinformation_scenario(),
            create_energy_grid_scenario(),
            create_manufacturing_scenario(),
            create_academic_scenario(),
            create_real_estate_scenario(),
            create_employee_exodus_scenario(),
            create_agricultural_scenario(),
        ]

        for i, df in enumerate(scenarios):
            # No NaNs
            assert not df.isna().any().any(), f"Scenario {i} has NaN values"
            # No infinite values
            numeric_cols = df.select_dtypes(include=[np.number])
            for col in numeric_cols:
                assert not np.any(np.isinf(numeric_cols[col])), f"Scenario {i}.{col} has inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
