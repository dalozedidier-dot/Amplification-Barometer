import pandas as pd

from amplification_barometer.audit_report import build_audit_report


def _load_fixture(rel_path: str) -> pd.DataFrame:
    df = pd.read_csv(rel_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    return df


def test_real_data_smoke_crypto_btc():
    df = _load_fixture("tests/fixtures/real/btc_5min_proxies_sample.csv")
    rep = build_audit_report(df, dataset_name="btc_5min_real")
    assert "AT" in rep.summary
    assert "DELTA_D" in rep.summary
    assert "E_stock" in rep.summary
    assert "R_level" in rep.summary
    assert "label" in rep.maturity
    assert "dimensions" in rep.verdict


def test_real_data_smoke_algae_raceway():
    df = _load_fixture("tests/fixtures/real/algae_raceway0_proxies_sample.csv")
    rep = build_audit_report(df, dataset_name="algae_raceway_real")
    assert "AT" in rep.summary
    assert "DELTA_D" in rep.summary
    assert "E_stock" in rep.summary
    assert "R_level" in rep.summary
    assert "label" in rep.maturity
    assert "dimensions" in rep.verdict
