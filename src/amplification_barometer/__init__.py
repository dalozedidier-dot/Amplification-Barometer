"""amplification_barometer

Implémentation modulaire et auditable du cadre Baromètre d’amplification.

Le package expose des fonctions utilitaires de calcul de composites (P, O, E, R, G),
des signatures @(t) et Δd(t), un modèle ODE de démonstration, et des outils d'audit.
"""

from .composites import (
    WEIGHTS_VERSION,
    compute_p,
    compute_o,
    compute_e,
    compute_r,
    compute_g,
    compute_at,
    compute_delta_d,
)
from .ode_model import simulate_minimal_po, simulate_barometer_ode
from .audit_tools import run_stress_test, audit_score_stability

__version__ = "0.1.0"

__all__ = [
    "WEIGHTS_VERSION",
    "compute_p",
    "compute_o",
    "compute_e",
    "compute_r",
    "compute_g",
    "compute_at",
    "compute_delta_d",
    "simulate_minimal_po",
    "simulate_barometer_ode",
    "run_stress_test",
    "audit_score_stability",
]
