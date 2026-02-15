import numpy as np
import pandas as pd

from amplification_barometer.l_operator import assess_maturity, compute_l_act, compute_l_cap


def _make_df(n: int = 50, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "stop_proxy": rng.normal(1.0, 0.1, size=n),
            "threshold_proxy": rng.normal(1.0, 0.1, size=n),
            "execution_proxy": rng.normal(1.0, 0.1, size=n),
            "coherence_proxy": rng.normal(1.0, 0.1, size=n),
            "exemption_rate": np.clip(rng.normal(0.1, 0.02, size=n), 0.0, 1.0),
            "sanction_delay": np.clip(rng.normal(30, 5, size=n), 0.0, 365.0),
            "control_turnover": np.clip(rng.normal(0.1, 0.03, size=n), 0.0, 1.0),
            "conflict_interest_proxy": np.clip(rng.normal(0.1, 0.03, size=n), 0.0, 1.0),
        }
    )
    return df


def test_l_series_shapes():
    df = _make_df()
    assert len(compute_l_cap(df)) == len(df)
    assert len(compute_l_act(df)) == len(df)


def test_maturity_assessment_has_label():
    df = _make_df()
    m = assess_maturity(df)
    assert m.label in {"Mature", "Immature", "Dissonant"}
