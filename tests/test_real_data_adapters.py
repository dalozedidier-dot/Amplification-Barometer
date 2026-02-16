from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from amplification_barometer.audit_report import build_audit_report
from amplification_barometer.real_data_adapters import (
    REQUIRED_COLUMNS,
    aiops_phase2_to_proxies,
    binance_aggtrades_to_proxies,
    binance_trades_to_proxies,
    borg_traces_to_proxies,
)

FIX = Path(__file__).parent / "fixtures" / "raw"


def _assert_required(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, f"Missing required columns: {missing}"
    arr = df.to_numpy(dtype=float)
    assert np.isfinite(arr).all()


def _assert_report_smoke(rep) -> None:
    # Be tolerant to report schema evolution: require invariants + at least one E/R/G variant.
    for k in ["P", "O", "AT", "DELTA_D", "RISK"]:
        assert k in rep.summary

    assert ("E" in rep.summary) or ("E_stock" in rep.summary) or ("E_level" in rep.summary)
    assert ("R" in rep.summary) or ("R_level" in rep.summary)
    assert ("G" in rep.summary) or ("G_level" in rep.summary)


def test_binance_aggtrades_adapter_smoke() -> None:
    raw = pd.read_csv(FIX / "ada_aggtrades_sample.csv", header=None)
    prox = binance_aggtrades_to_proxies(raw, bar_freq="1min")
    _assert_required(prox)
    rep = build_audit_report(prox, dataset_name="finance_binance_agg")
    _assert_report_smoke(rep)


def test_binance_trades_adapter_smoke() -> None:
    raw = pd.read_csv(FIX / "cat_trades_sample.csv", header=None)
    prox = binance_trades_to_proxies(raw, bar_freq="1min")
    _assert_required(prox)
    rep = build_audit_report(prox, dataset_name="finance_binance_trades")
    _assert_report_smoke(rep)


def test_borg_traces_adapter_smoke() -> None:
    raw = pd.read_csv(FIX / "borg_traces_sample.csv", header=0)
    prox = borg_traces_to_proxies(raw, bucket_seconds=60)
    _assert_required(prox)
    rep = build_audit_report(prox, dataset_name="ia_borg")
    _assert_report_smoke(rep)


def test_aiops_phase2_adapter_smoke() -> None:
    raw = pd.read_csv(FIX / "aiops_phase2_sample.csv", header=0)
    prox = aiops_phase2_to_proxies(raw)
    _assert_required(prox)
    rep = build_audit_report(prox, dataset_name="ia_aiops")
    _assert_report_smoke(rep)
