from pathlib import Path

import pandas as pd

from amplification_barometer.audit_tools import audit_score_stability, run_stress_test


def test_audit_tools_on_synthetic():
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "data" / "synthetic" / "stable_regime.csv"
    df = pd.read_csv(csv_path)

    res = run_stress_test(df, shock_magnitude=1.5)
    assert res.status in {"Résilient", "Instable sous stress"}

    stab = audit_score_stability(df, windows=(3, 5, 8))
    assert 0.0 <= stab["rank_consistency_at"] <= 1.0
    assert 0.0 <= stab["rank_consistency_delta_d"] <= 1.0
