from pathlib import Path

import pandas as pd

from amplification_barometer.calibration import discriminate_regimes


def test_calibration_flags_bifurcation_as_more_critical_than_stable():
    repo_root = Path(__file__).resolve().parents[1]
    sdir = repo_root / "data" / "synthetic"
    dfs = {
        "stable": pd.read_csv(sdir / "stable_regime.csv", parse_dates=["date"]).set_index("date"),
        "oscillating": pd.read_csv(sdir / "oscillating_regime.csv", parse_dates=["date"]).set_index("date"),
        "bifurcation": pd.read_csv(sdir / "bifurcation_regime.csv", parse_dates=["date"]).set_index("date"),
    }
    rep = discriminate_regimes(dfs, stable_name="stable", window=5)

    metrics = rep["datasets"]
    assert 0.03 <= metrics["stable"]["frac_risk_above_thr"] <= 0.07
    assert metrics["bifurcation"]["frac_risk_above_thr"] > metrics["stable"]["frac_risk_above_thr"]
    assert metrics["bifurcation"]["longest_run_above_thr"] >= metrics["stable"]["longest_run_above_thr"]

    ordering = rep["ordering_by_severity"]
    assert ordering[-1] == "bifurcation"
