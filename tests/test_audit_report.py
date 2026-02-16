import numpy as np
import pandas as pd

from amplification_barometer.audit_report import build_audit_report


def _make_df(n: int = 80, seed: int = 7) -> pd.DataFrame:
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
    df["date"] = pd.date_range("2020-01-01", periods=n, freq="D")
    df = df.set_index("date")
    return df


def test_build_audit_report_has_expected_keys():
    df = _make_df()
    rep = build_audit_report(df, dataset_name="unit")
    assert rep.version.startswith("0.")
    assert "AT" in rep.summary
    assert "spearman_mean_risk" in rep.stability
    assert "Overload" in rep.stress_suite
    assert "label" in rep.maturity
    assert "prevented_exceedance_rel_target_min" in rep.targets
    assert rep.verdict["label"] in {"Mature", "Immature", "Dissonant"}
    assert "dimensions" in rep.verdict
    assert "stability" in rep.verdict["dimensions"]
    assert "anti_gaming" in rep.verdict["dimensions"]
