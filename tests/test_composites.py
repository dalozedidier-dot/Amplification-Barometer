import numpy as np
import pandas as pd

from amplification_barometer.composites import compute_at, compute_delta_d, compute_e, compute_g, compute_o, compute_p, compute_r


def _make_df(n: int = 50, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = [
        "scale_proxy","speed_proxy","leverage_proxy","autonomy_proxy","replicability_proxy",
        "stop_proxy","threshold_proxy","decision_proxy","execution_proxy","coherence_proxy",
        "impact_proxy","propagation_proxy","hysteresis_proxy",
        "margin_proxy","redundancy_proxy","diversity_proxy","recovery_time_proxy",
        "exemption_rate","sanction_delay","control_turnover","conflict_interest_proxy","rule_execution_gap",
    ]
    df = pd.DataFrame(rng.normal(1.0, 0.1, size=(n, len(cols))), columns=cols)
    df["exemption_rate"] = np.clip(rng.normal(0.1, 0.02, size=n), 0.0, 1.0)
    df["sanction_delay"] = np.clip(rng.normal(30, 5, size=n), 0.0, 365.0)
    df["control_turnover"] = np.clip(rng.normal(0.1, 0.03, size=n), 0.0, 1.0)
    df["conflict_interest_proxy"] = np.clip(rng.normal(0.1, 0.03, size=n), 0.0, 1.0)
    df["rule_execution_gap"] = np.clip(rng.normal(0.04, 0.01, size=n), 0.0, 1.0)
    return df


def test_composites_shapes():
    df = _make_df()
    assert len(compute_p(df)) == len(df)
    assert len(compute_o(df)) == len(df)
    assert len(compute_e(df)) == len(df)
    assert len(compute_r(df)) == len(df)
    assert len(compute_g(df)) == len(df)
    assert len(compute_at(df)) == len(df)
    assert len(compute_delta_d(df, window=5)) == len(df)


def test_at_is_finite():
    df = _make_df()
    at = compute_at(df)
    assert np.isfinite(at.to_numpy()).all()
