"""Anti-Gaming Verdict Binding + Exogenous Shock Detection

This module integrates:
1. The 5-attack anti-gaming suite (endogenous gaming detection)
2. Exogenous shock detection (detects bifurcations from external shocks)

Making BOTH dimensions BINDING to the final verdict.

Key principle: A "Mature" verdict is only credible if:
- The system cannot be gamed (anti-gaming pass)
- The system is not under exogenous shock (shock risk low)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AttackResult:
    """Result of a single attack."""
    name: str
    detected: bool
    severity: str  # "critical", "high", "medium", "low"
    details: Dict[str, Any]

    @property
    def weight(self) -> float:
        """Weight of this attack in final scoring."""
        severity_map = {"critical": 1.0, "high": 0.75, "medium": 0.5, "low": 0.25}
        return severity_map.get(self.severity, 0.5)


@dataclass(frozen=True)
class GamingVerdictDimension:
    """Gaming dimension of multidimensional verdict."""
    attack_robustness_score: float  # 0.0 (vulnerable) to 1.0 (robust)
    attacks_passed: int  # Count of attacks defeated
    attacks_total: int  # Total attacks tested
    undetected_attacks: List[str]  # Names of attacks that passed undetected
    vulnerabilities: List[str]  # Human-readable vulnerability list
    binding_verdict: str  # "PASS" | "CAUTION" | "FAIL"


@dataclass(frozen=True)
class ExogenousShockVerdictDimension:
    """Exogenous shock detection dimension (catches bifurcations from external shocks)."""
    shock_risk_score: float  # 0.0 (safe) to 1.0 (high shock risk)
    shock_detected: bool  # Any shock detected
    shock_assessment: str  # "SAFE" | "LOW_RISK" | "MODERATE_RISK" | "HIGH_RISK"
    methods_triggered: List[str]  # Which detection methods fired
    binding_verdict: str  # "SAFE" | "CAUTION" | "ALERT"

    @property
    def risk_level(self) -> str:
        """Human-readable risk level."""
        if self.shock_risk_score > 0.7:
            return "CRITICAL"
        elif self.shock_risk_score > 0.5:
            return "HIGH"
        elif self.shock_risk_score > 0.3:
            return "MODERATE"
        else:
            return "LOW"


def compute_attack_robustness_score(
    attacks: List[AttackResult],
) -> GamingVerdictDimension:
    """
    Compute attack robustness score from a list of attack results.

    Scoring:
    - Each attack that is DETECTED = +points (system defended well)
    - Each attack that PASSES UNDETECTED = -points (vulnerability)
    - Weighted by severity

    Result:
    - 1.0 = All attacks detected, system robust
    - 0.0 = No attacks detected, system vulnerable
    - Intermediate = Some attacks slip through

    Binding verdict:
    - attack_robustness >= 0.95 → PASS (gaming risk minimal)
    - 0.75 <= attack_robustness < 0.95 → CAUTION (some vulnerabilities exist)
    - attack_robustness < 0.75 → FAIL (gaming risk high)
    """
    if not attacks:
        # No attacks run = unknown robustness
        return GamingVerdictDimension(
            attack_robustness_score=0.5,
            attacks_passed=0,
            attacks_total=0,
            undetected_attacks=[],
            vulnerabilities=["No anti-gaming tests executed"],
            binding_verdict="UNKNOWN",
        )

    total_weight = sum(a.weight for a in attacks)
    detected_weight = sum(a.weight for a in attacks if a.detected)

    # Score: weighted fraction of detected attacks
    if total_weight > 0:
        attack_robustness = float(detected_weight / total_weight)
    else:
        attack_robustness = 0.5

    # Count
    attacks_passed = sum(1 for a in attacks if a.detected)
    attacks_total = len(attacks)

    # Undetected (vulnerabilities)
    undetected = [a.name for a in attacks if not a.detected]

    # Human-readable vulnerabilities
    vulnerabilities = []
    for a in attacks:
        if not a.detected:
            severity_label = f"[{a.severity.upper()}]"
            vulnerabilities.append(f"{severity_label} {a.name}")

    # Binding verdict
    if attack_robustness >= 0.95:
        binding_verdict = "PASS"
    elif attack_robustness >= 0.75:
        binding_verdict = "CAUTION"
    else:
        binding_verdict = "FAIL"

    return GamingVerdictDimension(
        attack_robustness_score=attack_robustness,
        attacks_passed=attacks_passed,
        attacks_total=attacks_total,
        undetected_attacks=undetected,
        vulnerabilities=vulnerabilities,
        binding_verdict=binding_verdict,
    )


def apply_gaming_gate_to_verdict(
    original_verdict: str,
    gaming_dimension: GamingVerdictDimension,
) -> tuple[str, str]:
    """
    Apply anti-gaming binding to final verdict.

    Rules:
    - If gaming_binding_verdict == "PASS" → verdict unchanged
    - If gaming_binding_verdict == "CAUTION" → verdict becomes "Dissonant" (if "Mature")
    - If gaming_binding_verdict == "FAIL" → verdict becomes "Immature"

    Returns:
    - modified_verdict: Verdict after gaming gate
    - reason: Explanation for any change
    """
    if gaming_dimension.binding_verdict == "PASS":
        # No gaming vulnerabilities, verdict stands
        return original_verdict, "Anti-gaming tests passed"

    elif gaming_dimension.binding_verdict == "CAUTION":
        # Some vulnerabilities exist
        if original_verdict == "Mature":
            # Downgrade: "Mature" is only credible if fully robust against gaming
            reason = f"Downgraded due to gaming vulnerabilities: {', '.join(gaming_dimension.undetected_attacks)}"
            return "Dissonant", reason
        else:
            # Other verdicts stay as-is (already conservative)
            return original_verdict, "Gaming cautions noted but verdict already conservative"

    elif gaming_dimension.binding_verdict == "FAIL":
        # Major vulnerabilities
        if original_verdict in ("Mature", "Dissonant"):
            # Downgrade to "Immature"
            reason = f"System vulnerable to gaming: {', '.join(gaming_dimension.undetected_attacks)}"
            return "Immature", reason
        else:
            # Already "Immature"
            return original_verdict, "Gaming vulnerabilities confirmed"

    else:
        # Unknown binding verdict
        return original_verdict, "Gaming verdict unknown"


def create_exogenous_shock_dimension(
    shock_risk_score: float,
    shock_detected: bool,
    shock_assessment: str,
    methods_triggered: List[str],
) -> ExogenousShockVerdictDimension:
    """Create exogenous shock dimension from detection results."""

    # Binding verdict based on risk
    if shock_risk_score > 0.7:
        binding_verdict = "ALERT"
    elif shock_risk_score > 0.4:
        binding_verdict = "CAUTION"
    else:
        binding_verdict = "SAFE"

    return ExogenousShockVerdictDimension(
        shock_risk_score=float(shock_risk_score),
        shock_detected=bool(shock_detected),
        shock_assessment=shock_assessment,
        methods_triggered=methods_triggered,
        binding_verdict=binding_verdict,
    )


@dataclass(frozen=True)
class MultiDimensionalVerdict:
    """Complete multidimensional verdict with ALL dimensions scored.

    Dimensions:
    1. L_cap/L_act (capacity + activation)
    2. E/R (energy + recovery - endogenous stress)
    3. Gaming (can system be gamed?)
    4. Exogenous shocks (is system under attack?)
    5. Stability (data quality)
    """

    # Original dimensions (from l_operator)
    label: str  # Original verdict
    cap_score: float
    act_score: float

    # Dimensions
    gaming_dimension: GamingVerdictDimension
    shock_dimension: ExogenousShockVerdictDimension
    e_reduction_rel: float  # From energy_recovery
    stability_score: float  # Spearman correlation

    # Final verdict after all gates
    final_verdict: str
    final_verdict_reason: str

    # Overall credibility (0.0 = not credible, 1.0 = fully credible)
    credibility_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "original_label": self.label,
            "cap_score": float(self.cap_score),
            "act_score": float(self.act_score),
            "gaming": {
                "robustness_score": float(self.gaming_dimension.attack_robustness_score),
                "attacks_passed": int(self.gaming_dimension.attacks_passed),
                "attacks_total": int(self.gaming_dimension.attacks_total),
                "undetected": self.gaming_dimension.undetected_attacks,
                "binding_verdict": self.gaming_dimension.binding_verdict,
                "vulnerabilities": self.gaming_dimension.vulnerabilities,
            },
            "exogenous_shocks": {
                "shock_risk_score": float(self.shock_dimension.shock_risk_score),
                "shock_detected": bool(self.shock_dimension.shock_detected),
                "shock_assessment": self.shock_dimension.shock_assessment,
                "methods_triggered": self.shock_dimension.methods_triggered,
                "binding_verdict": self.shock_dimension.binding_verdict,
                "risk_level": self.shock_dimension.risk_level,
            },
            "energy": {
                "e_reduction_rel": float(self.e_reduction_rel),
            },
            "stability": {
                "spearman_correlation": float(self.stability_score),
            },
            "final_verdict": self.final_verdict,
            "final_verdict_reason": self.final_verdict_reason,
            "credibility_score": float(self.credibility_score),
        }


def compute_credibility_score(
    cap_score: float,
    act_score: float,
    gaming_robustness: float,
    e_reduction_rel: float,
    stability_score: float,
) -> float:
    """
    Compute overall credibility score combining all dimensions.

    Formula: weighted average of all dimensions
    - L_cap: 20% (intrinsic capacity)
    - L_act: 20% (governance activation)
    - gaming_robustness: 25% (HEAVY: gaming breaks credibility)
    - e_reduction_rel: 15% (governance effectiveness)
    - stability: 20% (data quality)

    Result: 0.0–1.0
    """
    weights = {
        "cap": 0.20,
        "act": 0.20,
        "gaming": 0.25,
        "e_reduction": 0.15,
        "stability": 0.20,
    }

    # Normalize all to [0, 1]
    cap_01 = np.clip(float(cap_score), 0.0, 1.0)
    act_01 = np.clip(float(act_score), 0.0, 1.0)
    gaming_01 = np.clip(float(gaming_robustness), 0.0, 1.0)
    e_01 = np.clip(float(e_reduction_rel), 0.0, 1.0)
    # Stability (Spearman): -1 to 1, map to 0–1
    stab_01 = np.clip((float(stability_score) + 1.0) / 2.0, 0.0, 1.0)

    credibility = (
        weights["cap"] * cap_01
        + weights["act"] * act_01
        + weights["gaming"] * gaming_01
        + weights["e_reduction"] * e_01
        + weights["stability"] * stab_01
    )

    return float(np.clip(credibility, 0.0, 1.0))


def build_multidimensional_verdict(
    original_label: str,
    cap_score: float,
    act_score: float,
    attacks: List[AttackResult],
    e_reduction_rel: float = 0.0,
    stability_score: float = 0.5,
    shock_risk_score: float = 0.0,
    shock_detected: bool = False,
    shock_assessment: str = "SAFE",
    methods_triggered: List[str] = None,
) -> MultiDimensionalVerdict:
    """
    Build complete multidimensional verdict with ALL dimensions:
    - Gaming binding
    - Exogenous shock detection
    - Energy/Recovery
    - Stability

    Returns: Fully scored MultiDimensionalVerdict with final verdict applied.
    """
    if methods_triggered is None:
        methods_triggered = []

    # Compute gaming dimension
    gaming_dim = compute_attack_robustness_score(attacks)

    # Compute exogenous shock dimension
    shock_dim = create_exogenous_shock_dimension(
        shock_risk_score=shock_risk_score,
        shock_detected=shock_detected,
        shock_assessment=shock_assessment,
        methods_triggered=methods_triggered,
    )

    # Apply gaming gate to original verdict
    gated_verdict, gate_reason = apply_gaming_gate_to_verdict(original_label, gaming_dim)

    # Apply shock gate to verdict (if HIGH shock risk, downgrade)
    if shock_dim.shock_risk_score > 0.7:
        if gated_verdict == "Mature":
            gated_verdict = "Dissonant"
            gate_reason += " | Downgraded due to HIGH exogenous shock risk"
        elif gated_verdict == "Dissonant":
            gated_verdict = "Immature"
            gate_reason += " | Downgraded to Immature due to HIGH exogenous shock risk"

    # Compute credibility (now including shock factor)
    # Shock risk reduces credibility: 0.2 weight
    shock_01 = 1.0 - np.clip(shock_risk_score, 0.0, 1.0)  # Invert: high shock = low credibility

    credibility = compute_credibility_score(
        cap_score=cap_score,
        act_score=act_score,
        gaming_robustness=gaming_dim.attack_robustness_score,
        e_reduction_rel=e_reduction_rel,
        stability_score=stability_score,
    )
    # Adjust credibility down if high shock risk
    credibility = credibility * (0.8 + 0.2 * shock_01)  # Shock can reduce credibility up to 20%

    return MultiDimensionalVerdict(
        label=original_label,
        cap_score=cap_score,
        act_score=act_score,
        gaming_dimension=gaming_dim,
        shock_dimension=shock_dim,
        e_reduction_rel=e_reduction_rel,
        stability_score=stability_score,
        final_verdict=gated_verdict,
        final_verdict_reason=gate_reason,
        credibility_score=credibility,
    )
