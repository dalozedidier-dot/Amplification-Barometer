from __future__ import annotations

"""Proxy protocol definitions for auditability.

The intent is not to claim ontological truth. The goal is operational measurability,
inter-temporal consistency, and explicit falsification conditions.

Each proxy entry documents:
- definition: operational meaning
- expected_range: inclusive [min, max]
- risk_direction: 'up' means higher values increase risk, 'down' means higher values decrease risk
- source: indicative source type (placeholder for real deployments)
- falsification_flags: minimal conditions that should raise an audit flag
"""

from typing import Any, Dict, List, Mapping, Tuple

PROXY_PROTOCOL: Dict[str, Dict[str, Any]] = {
    # P proxies
    "scale_proxy": {
        "definition": "Effective scale of deployment or coverage",
        "expected_range": (0.0, 10.0),
        "risk_direction": "up",
        "source": "operational telemetry",
        "falsification_flags": {"jump_mad_mult": 8.0, "range_violation_rate": 0.01},
    },
    "speed_proxy": {
        "definition": "Velocity of propagation or iteration",
        "expected_range": (0.0, 10.0),
        "risk_direction": "up",
        "source": "telemetry",
        "falsification_flags": {"jump_mad_mult": 8.0, "range_violation_rate": 0.01},
    },
    "leverage_proxy": {
        "definition": "Amplification leverage per unit input",
        "expected_range": (0.0, 10.0),
        "risk_direction": "up",
        "source": "telemetry",
        "falsification_flags": {"jump_mad_mult": 8.0, "range_violation_rate": 0.01},
    },
    "autonomy_proxy": {
        "definition": "Degree of automation and autonomy",
        "expected_range": (0.0, 10.0),
        "risk_direction": "up",
        "source": "system configuration",
        "falsification_flags": {"jump_mad_mult": 8.0, "range_violation_rate": 0.01},
    },
    "replicability_proxy": {
        "definition": "Ease of replication and scaling to new contexts",
        "expected_range": (0.0, 10.0),
        "risk_direction": "up",
        "source": "deployment metadata",
        "falsification_flags": {"jump_mad_mult": 8.0, "range_violation_rate": 0.01},
    },
    # O proxies
    "stop_proxy": {
        "definition": "Ability to stop or pause operations on demand",
        "expected_range": (0.0, 10.0),
        "risk_direction": "down",
        "source": "controls telemetry",
        "falsification_flags": {"jump_mad_mult": 8.0, "range_violation_rate": 0.01},
    },
    "threshold_proxy": {
        "definition": "Quality of thresholds and triggers",
        "expected_range": (0.0, 10.0),
        "risk_direction": "down",
        "source": "controls configuration",
        "falsification_flags": {"jump_mad_mult": 8.0, "range_violation_rate": 0.01},
    },
    "decision_proxy": {
        "definition": "Decision latency and clarity",
        "expected_range": (0.0, 10.0),
        "risk_direction": "down",
        "source": "process metrics",
        "falsification_flags": {"jump_mad_mult": 8.0, "range_violation_rate": 0.01},
    },
    "execution_proxy": {
        "definition": "Execution capability of controls",
        "expected_range": (0.0, 10.0),
        "risk_direction": "down",
        "source": "controls telemetry",
        "falsification_flags": {"jump_mad_mult": 8.0, "range_violation_rate": 0.01},
    },
    "coherence_proxy": {
        "definition": "Consistency between policy and actions",
        "expected_range": (0.0, 10.0),
        "risk_direction": "down",
        "source": "audit records",
        "falsification_flags": {"jump_mad_mult": 8.0, "range_violation_rate": 0.01},
    },
    # E proxies
    "impact_proxy": {
        "definition": "Magnitude of impacts per unit time",
        "expected_range": (0.0, 10.0),
        "risk_direction": "up",
        "source": "incident metrics",
        "falsification_flags": {"jump_mad_mult": 10.0, "range_violation_rate": 0.01},
    },
    "propagation_proxy": {
        "definition": "Propagation of externalities through a network",
        "expected_range": (0.0, 10.0),
        "risk_direction": "up",
        "source": "network telemetry",
        "falsification_flags": {"jump_mad_mult": 10.0, "range_violation_rate": 0.01},
    },
    "hysteresis_proxy": {
        "definition": "Persistence of impacts after shock removal",
        "expected_range": (0.0, 10.0),
        "risk_direction": "up",
        "source": "longitudinal metrics",
        "falsification_flags": {"jump_mad_mult": 10.0, "range_violation_rate": 0.01},
    },
    # R proxies
    "margin_proxy": {
        "definition": "Safety margin and slack",
        "expected_range": (0.0, 10.0),
        "risk_direction": "down",
        "source": "ops metrics",
        "falsification_flags": {"jump_mad_mult": 8.0, "range_violation_rate": 0.01},
    },
    "redundancy_proxy": {
        "definition": "Redundancy of critical components",
        "expected_range": (0.0, 10.0),
        "risk_direction": "down",
        "source": "architecture inventory",
        "falsification_flags": {"jump_mad_mult": 8.0, "range_violation_rate": 0.01},
    },
    "diversity_proxy": {
        "definition": "Diversity to avoid monoculture fragility",
        "expected_range": (0.0, 10.0),
        "risk_direction": "down",
        "source": "architecture inventory",
        "falsification_flags": {"jump_mad_mult": 8.0, "range_violation_rate": 0.01},
    },
    "recovery_time_proxy": {
        "definition": "Time to recover after stress, higher is worse",
        "expected_range": (0.0, 10.0),
        "risk_direction": "up",
        "source": "incident metrics",
        "falsification_flags": {"jump_mad_mult": 10.0, "range_violation_rate": 0.01},
    },
    # G proxies
    "exemption_rate": {
        "definition": "Share of exemptions granted against baseline rules",
        "expected_range": (0.0, 1.0),
        "risk_direction": "up",
        "source": "policy logs",
        "falsification_flags": {"jump_mad_mult": 10.0, "range_violation_rate": 0.005},
    },
    "sanction_delay": {
        "definition": "Median delay (days) between violation and sanction",
        "expected_range": (0.0, 365.0),
        "risk_direction": "up",
        "source": "sanctions database",
        "falsification_flags": {"jump_mad_mult": 10.0, "range_violation_rate": 0.005},
    },
    "control_turnover": {
        "definition": "Turnover rate in control functions",
        "expected_range": (0.0, 1.0),
        "risk_direction": "up",
        "source": "HR metrics",
        "falsification_flags": {"jump_mad_mult": 10.0, "range_violation_rate": 0.005},
    },
    "conflict_interest_proxy": {
        "definition": "Capture or conflict-of-interest signal",
        "expected_range": (0.0, 1.0),
        "risk_direction": "up",
        "source": "ethics audits",
        "falsification_flags": {"jump_mad_mult": 10.0, "range_violation_rate": 0.005},
    },
}


def required_proxies() -> List[str]:
    return sorted(PROXY_PROTOCOL.keys())
