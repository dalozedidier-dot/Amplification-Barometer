"""amplification_barometer

Implémentation modulaire et auditable d'un baromètre d'amplification.

Le package expose:
- composites P(t), O(t), E(t), R(t), G(t)
- signatures @(t) et Δd(t)
- opérateur de limite: L_cap et L_act, plus tests de performance L(t)
- stress tests, audit de stabilité, et suite anti-gaming (manipulabilité)
- modèle ODE de démonstration (non jumeau numérique)
"""

from .composites import (
    WEIGHTS_VERSION,
    compute_at,
    compute_delta_d,
    compute_e,
    compute_e_level,
    compute_e_stock,
    compute_de_dt,
    compute_e_irreversibility,
    compute_g,
    compute_o,
    compute_p,
    compute_r,
)
from .l_operator import (
    assess_maturity,
    compute_l_act,
    compute_l_cap,
    evaluate_l_performance,
)
from .audit_tools import audit_score_stability, run_stress_suite, run_stress_test
from .audit_report import build_audit_report, write_audit_report
from .calibration import discriminate_regimes, derive_thresholds, risk_signature
from .manipulability import run_manipulability_suite, validate_proxy_ranges
from .ode_model import simulate_barometer_ode, simulate_endogenous_g, simulate_minimal_po
from .proxy_protocol import PROXY_PROTOCOL, required_proxies

__version__ = "0.4.11"

__all__ = [
    "WEIGHTS_VERSION",
    "compute_p",
    "compute_o",
    "compute_e",
    "compute_e_level",
    "compute_e_stock",
    "compute_de_dt",
    "compute_e_irreversibility",
    "compute_r",
    "compute_g",
    "compute_at",
    "compute_delta_d",
    "compute_l_cap",
    "compute_l_act",
    "assess_maturity",
    "evaluate_l_performance",
    "audit_score_stability",
    "run_stress_test",
    "run_stress_suite",
    "build_audit_report",
    "write_audit_report",
    "risk_signature",
    "derive_thresholds",
    "discriminate_regimes",
    "validate_proxy_ranges",
    "run_manipulability_suite",
    "PROXY_PROTOCOL",
    "required_proxies",
    "simulate_minimal_po",
    "simulate_barometer_ode",
]
