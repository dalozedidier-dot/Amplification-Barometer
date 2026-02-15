from pathlib import Path

import pandas as pd

from amplification_barometer.l_operator import evaluate_l_performance


def test_l_performance_smoke_on_synthetic():
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "data" / "synthetic" / "bifurcation_regime.csv"
    df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date")
    perf = evaluate_l_performance(df, window=5, topk_frac=0.10, intensity=1.0)
    assert "prevented_exceedance" in perf
    assert 0.0 <= float(perf["prevented_exceedance"]) <= 1.0
    assert perf["verdict"] in {"Mature", "Immature", "Dissonant"}
