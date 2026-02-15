
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml


@dataclass(frozen=True)
class ProxySpec:
    definition: str
    expected_range: Tuple[float, float]
    risk_direction: str
    source: str
    frequency: str
    manipulability_test: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "definition": self.definition,
            "expected_range": tuple(self.expected_range),
            "risk_direction": self.risk_direction,
            "source": self.source,
            "frequency": self.frequency,
            "manipulability_test": self.manipulability_test,
        }


@dataclass(frozen=True)
class ProxyProtocol:
    version: str
    normalization_default: str
    missing_policy: str
    falsification_defaults: Dict[str, float]
    specs: Dict[str, ProxySpec]

    def as_mapping(self) -> Dict[str, Dict[str, Any]]:
        return {k: v.to_dict() for k, v in self.specs.items()}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_specs() -> Dict[str, ProxySpec]:
    # Minimal fallback compatible with tests and CI.
    def ps(defn: str, lo: float, hi: float, risk: str, src: str, freq: str, mt: str | None = None) -> ProxySpec:
        return ProxySpec(definition=defn, expected_range=(lo, hi), risk_direction=risk, source=src, frequency=freq, manipulability_test=mt)

    specs: Dict[str, ProxySpec] = {}
    # P
    specs["scale_proxy"] = ps("Relative ability to scale throughput/capacity", 0.0, 2.0, "up_risk", "internal_metrics", "daily", "detect_step_jump")
    specs["speed_proxy"] = ps("Execution speed and decision cycle acceleration", 0.0, 2.0, "up_risk", "internal_metrics", "daily", "detect_volatility_clamp")
    specs["leverage_proxy"] = ps("Leverage per unit resource", 0.0, 2.0, "up_risk", "internal_metrics", "daily", "detect_out_of_range")
    specs["autonomy_proxy"] = ps("Autonomy of operation without external human gating", 0.0, 1.0, "up_risk", "controls_registry", "weekly", "detect_out_of_range")
    specs["replicability_proxy"] = ps("Replicability across contexts", 0.0, 1.0, "up_risk", "controls_registry", "weekly", "detect_out_of_range")
    # O
    specs["stop_proxy"] = ps("Stop/rollback capacity", 0.0, 2.0, "down_risk", "controls_registry", "weekly", "detect_out_of_range")
    specs["threshold_proxy"] = ps("Thresholding discipline", 0.0, 2.0, "down_risk", "controls_registry", "weekly", "detect_out_of_range")
    specs["decision_proxy"] = ps("Decision latency control", 0.0, 2.0, "down_risk", "controls_registry", "weekly", "detect_out_of_range")
    specs["execution_proxy"] = ps("Execution and enforcement strength", 0.0, 2.0, "down_risk", "controls_registry", "weekly", "detect_out_of_range")
    specs["coherence_proxy"] = ps("Operational coherence", 0.0, 2.0, "down_risk", "internal_metrics", "daily", "detect_step_jump")
    # E
    specs["impact_proxy"] = ps("Impact level", 0.0, 3.0, "up_risk", "incident_metrics", "daily", "detect_step_jump")
    specs["propagation_proxy"] = ps("Propagation intensity", 0.0, 3.0, "up_risk", "incident_metrics", "daily", "detect_step_jump")
    specs["hysteresis_proxy"] = ps("Hysteresis / irreversibility proxy", 0.0, 3.0, "up_risk", "incident_metrics", "weekly", "detect_out_of_range")
    # R
    specs["margin_proxy"] = ps("Safety margin", 0.0, 2.0, "down_risk", "ops_metrics", "weekly", "detect_out_of_range")
    specs["redundancy_proxy"] = ps("Redundancy", 0.0, 2.0, "down_risk", "ops_metrics", "weekly", "detect_out_of_range")
    specs["diversity_proxy"] = ps("Diversity", 0.0, 2.0, "down_risk", "ops_metrics", "weekly", "detect_out_of_range")
    specs["recovery_time_proxy"] = ps("Recovery time", 0.0, 365.0, "up_risk", "incident_metrics", "monthly", "detect_step_jump")
    # G
    specs["exemption_rate"] = ps("Exemption rate", 0.0, 1.0, "up_risk", "audit_logs", "monthly", "detect_step_jump")
    specs["sanction_delay"] = ps("Sanction delay (days)", 0.0, 365.0, "up_risk", "audit_logs", "monthly", "detect_step_jump")
    specs["control_turnover"] = ps("Control turnover", 0.0, 1.0, "up_risk", "audit_logs", "monthly", "detect_step_jump")
    specs["conflict_interest_proxy"] = ps("Conflict of interest proxy", 0.0, 1.0, "up_risk", "audit_logs", "monthly", "detect_step_jump")
    specs["rule_execution_gap"] = ps("Rule execution gap", 0.0, 1.0, "up_risk", "audit_logs", "monthly", "detect_step_jump")
    return specs


def load_proxy_protocol(yaml_path: str | Path | None = None) -> ProxyProtocol:
    p = Path(yaml_path) if yaml_path is not None else (_repo_root() / "docs" / "proxies.yaml")
    if not p.exists():
        return ProxyProtocol(
            version="v1.0",
            normalization_default="baseline_mad_z",
            missing_policy="raise",
            falsification_defaults={"jump_mad_mult": 8.0, "shift_mad_mult": 1.8, "range_violation_rate": 0.01, "clamp_quantile_width": 0.10},
            specs=_default_specs(),
        )

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    notes = raw.get("notes") or {}
    version = str(raw.get("version", "v1.0"))
    normalization_default = str(notes.get("normalization_default", "baseline_mad_z"))
    missing_policy = str(notes.get("missing_policy", "raise"))
    fals_defaults = raw.get("falsification_defaults") or {}
    falsification_defaults = {
        "jump_mad_mult": float(fals_defaults.get("jump_mad_mult", 8.0)),
        "shift_mad_mult": float(fals_defaults.get("shift_mad_mult", 1.8)),
        "range_violation_rate": float(fals_defaults.get("range_violation_rate", 0.01)),
        "clamp_quantile_width": float(fals_defaults.get("clamp_quantile_width", 0.10)),
    }

    specs: Dict[str, ProxySpec] = {}
    families = raw.get("families") or {}
    for fam in families.values():
        proxies = (fam or {}).get("proxies") or {}
        for name, spec in proxies.items():
            if not isinstance(spec, dict):
                continue
            definition = str(spec.get("definition", ""))
            er = spec.get("expected_range", [0.0, 1.0])
            lo = float(er[0]) if isinstance(er, list) and len(er) >= 2 else 0.0
            hi = float(er[1]) if isinstance(er, list) and len(er) >= 2 else 1.0
            risk_direction = str(spec.get("risk_direction", "up_risk"))
            source = str(spec.get("source", "unknown"))
            frequency = str(spec.get("frequency", "unknown"))
            manipulability_test = spec.get("manipulability_test")
            specs[str(name)] = ProxySpec(
                definition=definition,
                expected_range=(lo, hi),
                risk_direction=risk_direction,
                source=source,
                frequency=frequency,
                manipulability_test=str(manipulability_test) if manipulability_test is not None else None,
            )

    # ensure fallback minimal set for tests
    fb = _default_specs()
    for k, v in fb.items():
        specs.setdefault(k, v)

    return ProxyProtocol(
        version=version,
        normalization_default=normalization_default,
        missing_policy=missing_policy,
        falsification_defaults=falsification_defaults,
        specs=specs,
    )


DEFAULT_PROTOCOL = load_proxy_protocol()

# Backward compatible mapping used by existing code
PROXY_PROTOCOL: Dict[str, Dict[str, Any]] = DEFAULT_PROTOCOL.as_mapping()


def required_proxies() -> List[str]:
    """List of required proxies for the default audit pipeline."""
    return sorted(PROXY_PROTOCOL.keys())
