"""Exogenous Shock Detection Module

Addresses the framework's blind spot: bifurcations triggered by exogenous shocks
(discrete events, discrete changes, structural breaks) rather than endogenous stress buildup.

This module detects:
1. Volatility spikes (sudden variance increase)
2. Structural breaks (regime changes)
3. Anomalies (outliers in proxy distributions)
4. Coordinated multi-proxy shifts (simultaneous changes across uncorrelated proxies)
5. Policy/regulatory events (sudden changes in G-proxies)

These are NOT captured by the L_cap/L_act framework alone, which assumes continuous stress.
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def detect_volatility_spike(
    series: pd.Series,
    window: int = 20,
    threshold_std_multiple: float = 2.5,
    min_tail_points: int = 5,
) -> Dict[str, any]:
    """
    Detect sudden increase in volatility (variance spike).

    This catches sudden regime changes like:
    - Market flash crash
    - AI model suddenly behaving erratically
    - Infrastructure suddenly unstable

    Returns:
    - spike_detected: bool
    - spike_location: index where spike started
    - volatility_ratio: new_vol / baseline_vol
    - severity: how extreme the spike is
    """
    if len(series) < window + 5:
        return {
            "spike_detected": False,
            "spike_location": -1,
            "volatility_ratio": 1.0,
            "severity": 0.0,
        }

    vals = series.astype(float).to_numpy()
    vals_clean = vals[np.isfinite(vals)]

    if len(vals_clean) < window + 5:
        return {
            "spike_detected": False,
            "spike_location": -1,
            "volatility_ratio": 1.0,
            "severity": 0.0,
        }

    # Baseline volatility (first 2/3 of series)
    baseline_len = max(window, int(len(vals_clean) * 0.66))
    baseline_vol = float(np.std(vals_clean[:baseline_len]))

    # Tail volatility (last points)
    if len(vals_clean) > baseline_len + min_tail_points:
        tail_vol = float(np.std(vals_clean[-min_tail_points:]))
    else:
        tail_vol = float(np.std(vals_clean[baseline_len:]))

    # Detect spike
    vol_ratio = tail_vol / (baseline_vol + 1e-9)
    spike_threshold = threshold_std_multiple
    spike_detected = vol_ratio > spike_threshold

    # Find location of spike (first point where rolling vol exceeds baseline)
    spike_location = -1
    if spike_detected:
        rolling_vols = []
        for i in range(window, len(vals_clean)):
            rolling_vol = float(np.std(vals_clean[i - window : i]))
            rolling_vols.append(rolling_vol)
            if rolling_vol > baseline_vol * threshold_std_multiple and spike_location < 0:
                spike_location = i - window

    # Severity: how many std devs above baseline
    severity = float(np.clip((vol_ratio - 1.0) / 2.0, 0.0, 1.0))

    return {
        "spike_detected": bool(spike_detected),
        "spike_location": int(spike_location),
        "volatility_ratio": float(vol_ratio),
        "severity": severity,
    }


def detect_structural_break(
    series: pd.Series,
    window: int = 20,
) -> Dict[str, any]:
    """
    Detect structural break (Chow test).

    A structural break indicates a regime change:
    - Before shock: mean = μ₀
    - After shock: mean = μ₁ ≠ μ₀

    This catches events like:
    - Regulatory change that shifts baseline behavior
    - AI model update that changes output distribution
    - New competitor entering market
    """
    if len(series) < window * 2:
        return {
            "break_detected": False,
            "break_location": -1,
            "chow_statistic": 0.0,
            "p_value": 1.0,
        }

    vals = series.astype(float).to_numpy()
    vals_clean = vals[np.isfinite(vals)]

    if len(vals_clean) < window * 2:
        return {
            "break_detected": False,
            "break_location": -1,
            "chow_statistic": 0.0,
            "p_value": 1.0,
        }

    # Chow test at each possible breakpoint
    max_chow_stat = 0.0
    best_break_idx = 0

    for break_idx in range(window, len(vals_clean) - window):
        before = vals_clean[:break_idx]
        after = vals_clean[break_idx:]

        mean_before = np.mean(before)
        mean_after = np.mean(after)
        var_before = np.var(before, ddof=1) if len(before) > 1 else 1e-9
        var_after = np.var(after, ddof=1) if len(after) > 1 else 1e-9

        # Chow statistic (F-test for equality of means)
        n_before = len(before)
        n_after = len(after)
        pooled_var = ((n_before - 1) * var_before + (n_after - 1) * var_after) / (
            n_before + n_after - 2
        )
        pooled_var = max(pooled_var, 1e-9)

        se = np.sqrt(pooled_var * (1.0 / n_before + 1.0 / n_after))
        se = max(se, 1e-9)

        chow_stat = ((mean_after - mean_before) / se) ** 2

        if chow_stat > max_chow_stat:
            max_chow_stat = chow_stat
            best_break_idx = break_idx

    # Convert Chow stat to p-value (F-dist with df=1 and df=n-2)
    from scipy.stats import f

    df_numerator = 1
    df_denominator = len(vals_clean) - 2
    p_value = 1.0 - f.cdf(max_chow_stat, df_numerator, df_denominator)

    break_detected = p_value < 0.05

    return {
        "break_detected": bool(break_detected),
        "break_location": int(best_break_idx),
        "chow_statistic": float(max_chow_stat),
        "p_value": float(p_value),
    }


def detect_anomalies(
    series: pd.Series,
    method: str = "iqr",  # "iqr" or "zscore"
    threshold: float = 3.0,
) -> Dict[str, any]:
    """
    Detect anomalies in a series (outliers that might indicate exogenous shock).

    Methods:
    - IQR: Interquartile range (robust to outliers)
    - Z-score: Standard deviation method (sensitive to extremes)

    Returns:
    - anomalies_detected: bool
    - anomaly_count: number of anomalies
    - anomaly_indices: where they occur
    - max_severity: how extreme the worst one is
    """
    vals = series.astype(float).to_numpy()
    vals_clean = vals[np.isfinite(vals)]

    if len(vals_clean) < 10:
        return {
            "anomalies_detected": False,
            "anomaly_count": 0,
            "anomaly_indices": [],
            "max_severity": 0.0,
        }

    if method == "iqr":
        q1 = np.percentile(vals_clean, 25)
        q3 = np.percentile(vals_clean, 75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        is_anomaly = (vals_clean < lower) | (vals_clean > upper)

    elif method == "zscore":
        mean = np.mean(vals_clean)
        std = np.std(vals_clean)
        z_scores = np.abs((vals_clean - mean) / (std + 1e-9))
        is_anomaly = z_scores > threshold

    else:
        return {
            "anomalies_detected": False,
            "anomaly_count": 0,
            "anomaly_indices": [],
            "max_severity": 0.0,
        }

    anomaly_indices = list(np.where(is_anomaly)[0])
    anomalies_detected = len(anomaly_indices) > 0
    max_severity = float(np.max(np.abs(vals_clean[is_anomaly]))) if anomalies_detected else 0.0

    # Normalize severity to [0, 1]
    data_range = np.max(vals_clean) - np.min(vals_clean)
    data_range = max(data_range, 1e-9)
    max_severity = float(np.clip(max_severity / data_range, 0.0, 1.0))

    return {
        "anomalies_detected": bool(anomalies_detected),
        "anomaly_count": int(len(anomaly_indices)),
        "anomaly_indices": anomaly_indices,
        "max_severity": max_severity,
    }


def detect_coordinated_shift(
    df: pd.DataFrame,
    proxy_columns: list,
    window: int = 5,
    correlation_threshold: float = 0.3,
) -> Dict[str, any]:
    """
    Detect coordinated shift: uncorrelated proxies suddenly moving together.

    Example: In normal times, exemption_rate and recovery_time are uncorrelated.
    But if an exogenous shock hits (regulatory change), both might shift simultaneously.

    This is a sign of exogenous pressure, not endogenous feedback.
    """
    if len(df) < window * 2 or len(proxy_columns) < 2:
        return {
            "shift_detected": False,
            "shift_location": -1,
            "correlation_increase": 0.0,
            "severity": 0.0,
        }

    # Get values for the proxies
    data = df[proxy_columns].astype(float).to_numpy()

    # Baseline correlation (first period)
    baseline_len = int(len(data) * 0.5)
    baseline_corr = np.corrcoef(data[:baseline_len].T)
    baseline_avg_corr = float(np.mean(np.abs(baseline_corr[np.triu_indices_from(baseline_corr, k=1)])))

    # Tail correlation (last points)
    tail_corr = np.corrcoef(data[-window:].T) if len(data) > window else baseline_corr
    tail_avg_corr = float(np.mean(np.abs(tail_corr[np.triu_indices_from(tail_corr, k=1)])))

    # Detect shift
    correlation_increase = tail_avg_corr - baseline_avg_corr
    shift_detected = correlation_increase > correlation_threshold

    # Severity: how much did correlation increase
    severity = float(np.clip(correlation_increase / (1.0 - baseline_avg_corr + 0.1), 0.0, 1.0))

    # Find location of shift
    shift_location = -1
    if shift_detected:
        for i in range(baseline_len, len(data) - window):
            window_corr = np.corrcoef(data[i : i + window].T)
            window_avg_corr = float(
                np.mean(np.abs(window_corr[np.triu_indices_from(window_corr, k=1)]))
            )
            if window_avg_corr > baseline_avg_corr + correlation_threshold:
                shift_location = i
                break

    return {
        "shift_detected": bool(shift_detected),
        "shift_location": int(shift_location),
        "correlation_increase": float(correlation_increase),
        "severity": severity,
    }


def assess_exogenous_shock_risk(
    df: pd.DataFrame,
    proxy_columns: list = None,
    window: int = 5,
) -> Dict[str, any]:
    """
    Comprehensive exogenous shock assessment.

    Combines 4 detection methods:
    1. Volatility spike detection
    2. Structural break detection
    3. Anomaly detection
    4. Coordinated shift detection

    Returns:
    - shock_detected: bool (any method detected something)
    - shock_risk_score: 0.0–1.0
    - methods_triggered: which detection methods fired
    - latest_shock_index: where was the last detected anomaly
    - severity_factors: detailed breakdown
    """
    if proxy_columns is None:
        proxy_columns = [c for c in df.columns if "proxy" in c.lower()]

    shock_detected = False
    methods_triggered = []
    max_severity = 0.0
    latest_shock_index = -1
    severity_factors = {}

    # 1. Volatility spike detection
    if len(proxy_columns) > 0:
        first_proxy = proxy_columns[0]
        if first_proxy in df.columns:
            vol_result = detect_volatility_spike(df[first_proxy], window=window)
            severity_factors["volatility_spike"] = vol_result["severity"]
            if vol_result["spike_detected"]:
                methods_triggered.append("volatility_spike")
                shock_detected = True
                max_severity = max(max_severity, vol_result["severity"])
                if vol_result["spike_location"] > latest_shock_index:
                    latest_shock_index = vol_result["spike_location"]

    # 2. Structural break detection
    if len(proxy_columns) > 0:
        first_proxy = proxy_columns[0]
        if first_proxy in df.columns:
            break_result = detect_structural_break(df[first_proxy], window=window)
            severity_factors["structural_break"] = float(1.0 - break_result["p_value"])
            if break_result["break_detected"]:
                methods_triggered.append("structural_break")
                shock_detected = True
                severity = float(1.0 - break_result["p_value"])
                max_severity = max(max_severity, severity)
                if break_result["break_location"] > latest_shock_index:
                    latest_shock_index = break_result["break_location"]

    # 3. Anomaly detection
    anomaly_count = 0
    max_anomaly_severity = 0.0
    for col in proxy_columns:
        if col in df.columns:
            anom_result = detect_anomalies(df[col])
            if anom_result["anomalies_detected"]:
                methods_triggered.append(f"anomalies_in_{col}")
                shock_detected = True
                anomaly_count += anom_result["anomaly_count"]
                max_anomaly_severity = max(max_anomaly_severity, anom_result["max_severity"])
                if anom_result["anomaly_indices"]:
                    latest_idx = max(anom_result["anomaly_indices"])
                    if latest_idx > latest_shock_index:
                        latest_shock_index = latest_idx

    if anomaly_count > 0:
        severity_factors["anomalies"] = max_anomaly_severity
        max_severity = max(max_severity, max_anomaly_severity)

    # 4. Coordinated shift detection
    if len(proxy_columns) >= 2:
        shift_result = detect_coordinated_shift(df, proxy_columns, window=window)
        severity_factors["coordinated_shift"] = shift_result["severity"]
        if shift_result["shift_detected"]:
            methods_triggered.append("coordinated_shift")
            shock_detected = True
            max_severity = max(max_severity, shift_result["severity"])
            if shift_result["shift_location"] > latest_shock_index:
                latest_shock_index = shift_result["shift_location"]

    # Compute risk score (average of detected methods)
    if methods_triggered:
        shock_risk_score = max_severity
    else:
        shock_risk_score = 0.0

    # Classification
    if shock_risk_score > 0.7:
        shock_assessment = "HIGH_RISK"
    elif shock_risk_score > 0.4:
        shock_assessment = "MODERATE_RISK"
    elif shock_risk_score > 0.2:
        shock_assessment = "LOW_RISK"
    else:
        shock_assessment = "SAFE"

    return {
        "shock_detected": bool(shock_detected),
        "shock_risk_score": float(np.clip(shock_risk_score, 0.0, 1.0)),
        "shock_assessment": shock_assessment,
        "methods_triggered": methods_triggered,
        "methods_count": len(methods_triggered),
        "latest_shock_index": int(latest_shock_index),
        "severity_factors": severity_factors,
    }
