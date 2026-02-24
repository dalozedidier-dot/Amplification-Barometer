"""Test anti-gaming verdict binding.

Verify that:
1. Gaming robustness score is computed correctly
2. Gaming gate modifies verdict when vulnerabilities exist
3. Credibility score includes gaming dimension
4. Multidimensional verdict is binding
"""

import pytest
from amplification_barometer.anti_gaming_verdict import (
    AttackResult,
    compute_attack_robustness_score,
    apply_gaming_gate_to_verdict,
    compute_credibility_score,
    build_multidimensional_verdict,
)


class TestAntiGamingVerdictBinding:
    """Test suite for anti-gaming verdict binding."""

    def test_all_attacks_detected_score_1_0(self):
        """Test that all detected attacks = 1.0 robustness."""
        attacks = [
            AttackResult(name="o_bias", detected=True, severity="high", details={}),
            AttackResult(name="volatility_clamp", detected=True, severity="high", details={}),
            AttackResult(name="out_of_range", detected=True, severity="critical", details={}),
            AttackResult(name="coordinated_multi", detected=True, severity="critical", details={}),
            AttackResult(name="reporting_delay", detected=True, severity="medium", details={}),
        ]

        dimension = compute_attack_robustness_score(attacks)

        assert dimension.attack_robustness_score == 1.0, "All detected = 1.0"
        assert dimension.attacks_passed == 5, "5 attacks passed"
        assert dimension.binding_verdict == "PASS"
        assert len(dimension.undetected_attacks) == 0

    def test_some_attacks_undetected_score_intermediate(self):
        """Test that some undetected attacks = intermediate score."""
        attacks = [
            AttackResult(name="o_bias", detected=True, severity="high", details={}),
            AttackResult(name="volatility_clamp", detected=True, severity="high", details={}),
            AttackResult(name="out_of_range", detected=True, severity="critical", details={}),
            AttackResult(name="coordinated_multi", detected=False, severity="medium", details={}),  # FAIL
            AttackResult(name="reporting_delay", detected=True, severity="low", details={}),
        ]

        dimension = compute_attack_robustness_score(attacks)

        # 4 out of 5 detected, only low-severity attack missed = CAUTION
        assert dimension.attack_robustness_score < 1.0, "Some attacks undetected < 1.0"
        assert dimension.attacks_passed == 4, "4 attacks passed"
        assert len(dimension.undetected_attacks) == 1, "1 attack undetected"
        assert "coordinated_multi" in dimension.undetected_attacks
        assert dimension.binding_verdict == "CAUTION", "Partial detection = CAUTION"

    def test_all_attacks_undetected_score_0_0(self):
        """Test that no detected attacks = 0.0 robustness."""
        attacks = [
            AttackResult(name="o_bias", detected=False, severity="high", details={}),
            AttackResult(name="volatility_clamp", detected=False, severity="high", details={}),
            AttackResult(name="out_of_range", detected=False, severity="critical", details={}),
            AttackResult(name="coordinated_multi", detected=False, severity="critical", details={}),
            AttackResult(name="reporting_delay", detected=False, severity="medium", details={}),
        ]

        dimension = compute_attack_robustness_score(attacks)

        assert dimension.attack_robustness_score == 0.0, "All undetected = 0.0"
        assert dimension.attacks_passed == 0
        assert len(dimension.undetected_attacks) == 5
        assert dimension.binding_verdict == "FAIL"

    def test_gaming_gate_mature_vulnerable_downgraded(self):
        """Test that Mature verdict is downgraded if gaming vulnerabilities exist."""
        original = "Mature"
        gaming_dim = compute_attack_robustness_score([
            AttackResult(name="o_bias", detected=False, severity="high", details={}),
            AttackResult(name="volatility_clamp", detected=True, severity="high", details={}),
            AttackResult(name="out_of_range", detected=True, severity="critical", details={}),
            AttackResult(name="coordinated_multi", detected=True, severity="critical", details={}),
            AttackResult(name="reporting_delay", detected=True, severity="medium", details={}),
        ])

        modified, reason = apply_gaming_gate_to_verdict(original, gaming_dim)

        assert modified == "Dissonant", "Mature with gaming vulnerabilities → Dissonant"
        assert "o_bias" in reason, "Reason should mention vulnerability"

    def test_gaming_gate_mature_robust_unchanged(self):
        """Test that Mature verdict stays if gaming robust."""
        original = "Mature"
        gaming_dim = compute_attack_robustness_score([
            AttackResult(name="o_bias", detected=True, severity="high", details={}),
            AttackResult(name="volatility_clamp", detected=True, severity="high", details={}),
            AttackResult(name="out_of_range", detected=True, severity="critical", details={}),
            AttackResult(name="coordinated_multi", detected=True, severity="critical", details={}),
            AttackResult(name="reporting_delay", detected=True, severity="medium", details={}),
        ])

        modified, reason = apply_gaming_gate_to_verdict(original, gaming_dim)

        assert modified == "Mature", "Mature with robust gaming → unchanged"
        assert "passed" in reason.lower()

    def test_gaming_gate_immature_stays_immature(self):
        """Test that Immature verdict stays if gaming fails."""
        original = "Immature"
        gaming_dim = compute_attack_robustness_score([
            AttackResult(name="o_bias", detected=False, severity="high", details={}),
            AttackResult(name="volatility_clamp", detected=False, severity="high", details={}),
            AttackResult(name="out_of_range", detected=False, severity="critical", details={}),
            AttackResult(name="coordinated_multi", detected=False, severity="critical", details={}),
            AttackResult(name="reporting_delay", detected=False, severity="medium", details={}),
        ])

        modified, reason = apply_gaming_gate_to_verdict(original, gaming_dim)

        assert modified == "Immature", "Immature with gaming failures → unchanged"

    def test_credibility_score_includes_gaming(self):
        """Test that credibility score is reduced by gaming vulnerabilities."""
        # Scenario 1: No gaming vulnerabilities
        credibility_robust = compute_credibility_score(
            cap_score=0.8,
            act_score=0.8,
            gaming_robustness=1.0,  # Robust
            e_reduction_rel=0.5,
            stability_score=0.8,
        )

        # Scenario 2: Gaming vulnerabilities
        credibility_vulnerable = compute_credibility_score(
            cap_score=0.8,
            act_score=0.8,
            gaming_robustness=0.4,  # Vulnerable
            e_reduction_rel=0.5,
            stability_score=0.8,
        )

        assert credibility_robust > credibility_vulnerable, \
            "Robust system should have higher credibility than vulnerable"

    def test_multidimensional_verdict_synthesis(self):
        """Test complete multidimensional verdict synthesis."""
        attacks = [
            AttackResult(name="o_bias", detected=True, severity="high", details={}),
            AttackResult(name="volatility_clamp", detected=True, severity="high", details={}),
            AttackResult(name="out_of_range", detected=True, severity="critical", details={}),
            AttackResult(name="coordinated_multi", detected=True, severity="critical", details={}),
            AttackResult(name="reporting_delay", detected=True, severity="medium", details={}),
        ]

        verdict = build_multidimensional_verdict(
            original_label="Mature",
            cap_score=0.75,
            act_score=0.72,
            attacks=attacks,
            e_reduction_rel=0.5,
            stability_score=0.85,
        )

        # Should be fully credible
        assert verdict.final_verdict == "Mature", "Should stay Mature with robust gaming"
        assert verdict.gaming_dimension.binding_verdict == "PASS"
        assert verdict.credibility_score > 0.7, "Should be highly credible"

        # Check dict conversion
        d = verdict.to_dict()
        assert d["final_verdict"] == "Mature"
        assert d["gaming"]["robustness_score"] == 1.0
        assert d["credibility_score"] > 0.7

    def test_vulnerable_mature_becomes_dissonant(self):
        """Test that vulnerable Mature becomes Dissonant."""
        attacks = [
            AttackResult(name="o_bias", detected=False, severity="high", details={}),  # FAIL
            AttackResult(name="volatility_clamp", detected=True, severity="high", details={}),
            AttackResult(name="out_of_range", detected=True, severity="critical", details={}),
            AttackResult(name="coordinated_multi", detected=True, severity="critical", details={}),
            AttackResult(name="reporting_delay", detected=True, severity="medium", details={}),
        ]

        verdict = build_multidimensional_verdict(
            original_label="Mature",
            cap_score=0.75,
            act_score=0.72,
            attacks=attacks,
            e_reduction_rel=0.5,
            stability_score=0.85,
        )

        # Should be downgraded
        assert verdict.final_verdict == "Dissonant", "Vulnerable Mature → Dissonant"
        assert verdict.gaming_dimension.binding_verdict == "CAUTION"
        assert "o_bias" in verdict.final_verdict_reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
