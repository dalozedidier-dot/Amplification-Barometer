"""amplification_barometer

Implémentation modulaire et auditable d'un baromètre d'amplification.

Le package expose:
- composites P(t), O(t), E(t), R(t), G(t)
- signatures @(t) et Δd(t)
- opérateur de limite en deux composantes L_cap et L_act
- stress tests et audit de stabilité du score
- modèle ODE de démonstration (non jumeau numérique)
"""

from .composites import (
    WEIGHTS_VERSION,
    compute_at,
    compute_delta_d,
    compute_e,
    compute_g,
    compute_o,
    compute_p,
    compute_r,
)
from .l_operator import assess_maturity, compute_l_act, compute_l_cap
from .audit_tools import audit_score_stability, run_stress_suite, run_stress_test
from .audit_report import build_audit_report, write_audit_report
from .ode_model import simulate_barometer_ode, simulate_minimal_po

__version__ = "0.2.1"

__all__ = [
    "WEIGHTS_VERSION",
    "compute_p",
    "compute_o",
    "compute_e",
    "compute_r",
    "compute_g",
    "compute_at",
    "compute_delta_d",
    "compute_l_cap",
    "compute_l_act",
    "assess_maturity",
    "audit_score_stability",
    "run_stress_test",
    "run_stress_suite",
    "build_audit_report",
    "write_audit_report",
    "simulate_minimal_po",
    "simulate_barometer_ode",
]
