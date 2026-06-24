from __future__ import annotations

import math

import pandas as pd
import pytest

from amplification_barometer.composites import compute_g_level, compute_rho
from amplification_barometer.ode_model import (
    BarometerParams,
    assess_hurwitz_local_stability,
    compute_active_equilibrium,
    hurwitz_coefficients,
)


def _minimal_proxy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scale_proxy": [1.0, 1.2, 1.4],
            "speed_proxy": [1.0, 1.2, 1.4],
            "leverage_proxy": [1.0, 1.2, 1.4],
            "autonomy_proxy": [1.0, 1.2, 1.4],
            "replicability_proxy": [1.0, 1.2, 1.4],
            "stop_proxy": [1.0, 0.8, 0.0],
            "threshold_proxy": [1.0, 0.8, 0.0],
            "decision_proxy": [1.0, 0.8, 0.0],
            "execution_proxy": [1.0, 0.8, 0.0],
            "coherence_proxy": [1.0, 0.8, 0.0],
            "exemption_rate": [0.05, 0.10, 0.30],
            "sanction_delay": [5.0, 30.0, 120.0],
            "control_turnover": [0.02, 0.05, 0.20],
            "conflict_interest_proxy": [0.03, 0.10, 0.40],
            "rule_execution_gap": [0.02, 0.05, 0.30],
        }
    )


def test_rho_uses_positive_floor_but_can_be_strict() -> None:
    df = _minimal_proxy_frame()
    rho = compute_rho(df, eps=1e-6)
    assert rho.name == "RHO"
    assert rho.iloc[-1] > rho.iloc[0]
    assert math.isfinite(float(rho.iloc[-1]))

    with pytest.raises(ValueError):
        compute_rho(df, eps=1e-6, validate_positive=True)


def test_g_level_is_bounded_and_decreases_when_governance_risk_rises() -> None:
    df = _minimal_proxy_frame()
    g = compute_g_level(df)
    assert g.name == "G_LEVEL"
    assert float(g.min()) >= 0.0
    assert float(g.max()) <= 1.0
    assert float(g.iloc[0]) > float(g.iloc[-1])


def test_hurwitz_diagnostics_are_local_and_report_conditions() -> None:
    params = BarometerParams(
        a=0.8,
        b=0.4,
        c=0.8,
        u=0.6,
        v=0.05,
        n=0.8,
        m=0.02,
        alpha=0.2,
        beta=0.1,
        lam=0.8,
        delta=0.4,
        gamma=0.1,
        xi=0.3,
    )
    eq = compute_active_equilibrium(params)
    assert eq.feasible

    coeffs = hurwitz_coefficients(params, eq)
    assert set(coeffs) == {"A1", "A2", "A3", "A1A2_minus_A3"}

    diag = assess_hurwitz_local_stability(params, eq)
    assert diag.status in {"STABLE_LOCAL", "UNSTABLE_LOCAL"}
    assert "voisinage" in diag.scope_note
    assert set(diag.conditions) == {"A1_positive", "A2_positive", "A3_positive", "A1A2_gt_A3"}
