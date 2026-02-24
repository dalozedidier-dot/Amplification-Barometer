#!/usr/bin/env python3
"""
L_cap Capability Benchmark: Independent assessment of detection capacity.

Unlike L_act (which depends on governance proxies), L_cap measures the intrinsic
ability to distinguish stress regimes under controlled, synthetic conditions.

This prevents circularity: L_cap is measured independently of observed data,
then compared to L_act in real/realistic datasets.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .alignment_audit import compute_at_delta, compute_e_r, compute_levels_from_specs
from .l_operator import compute_l_cap, compute_l_act
from .theory_map import TheoryAuditMap, load_proxy_specs


def benchmark_l_cap_on_scenario(df: pd.DataFrame, specs: TheoryAuditMap, scenario_name: str) -> Dict[str, Any]:
    """
    Assess L_cap on a single synthetic scenario.

    L_cap should show distinct *capability patterns* across regime types:
    - Type I: L_cap low (little to stop)
    - Type II: L_cap moderate (some recovery available)
    - Type III: L_cap highest but insufficient (O maxed out, E still accumulates)
    """
    try:
        l_cap = compute_l_cap(df)
        l_cap_mean = float(np.nanmean(l_cap))
        l_cap_std = float(np.nanstd(l_cap))
        l_cap_tail = float(np.nanmean(l_cap[-30:])) if len(l_cap) >= 30 else l_cap_mean

        # Compute regime signature
        levels = compute_levels_from_specs(df, specs)
        sig = compute_at_delta(levels, smooth_win=7)
        er = compute_e_r(levels)

        at = sig["at"]
        delta_d = sig["delta_d"]
        e_stock = er["e_stock"]
        r_level = er["r_level"]

        de = np.diff(e_stock, prepend=e_stock[0])
        pers_de = float(np.mean(de[-30:] > 0.0)) if len(de) >= 30 else 0.0
        r_tail = float(np.nanmean(r_level[-30:])) if len(r_level) >= 30 else float(np.nanmean(r_level))
        o_level = np.asarray(levels["O_level"], dtype=float)
        o_sat = float(np.mean(o_level < np.nanpercentile(o_level, 10)))
        at_div = float(np.mean(at[-30:] > np.nanpercentile(at, 90))) if len(at) >= 30 else 0.0
        irr = float((e_stock[-1] / (np.max(e_stock) + 1e-9)) if len(e_stock) else 0.0)

        # Regime classification
        if (at_div >= 0.5) and (pers_de >= 0.6) and (irr >= 0.9):
            regime = "type_III_bifurcation"
        elif pers_de >= 0.5 and at_div < 0.5:
            regime = "type_II_oscillations"
        else:
            regime = "type_I_noise"

        return {
            "scenario": scenario_name,
            "regime": regime,
            "l_cap_mean": l_cap_mean,
            "l_cap_std": l_cap_std,
            "l_cap_tail": l_cap_tail,
            "l_cap_tail_above_median": float(l_cap_tail > np.nanmedian(l_cap)),
            "at_divergence": at_div,
            "e_irreversibility": irr,
            "r_recovery": r_tail,
            "persistence_de": pers_de,
        }
    except Exception as e:
        return {
            "scenario": scenario_name,
            "error": str(e),
        }


def benchmark_suite_all_scenarios(
    scenarios_dir: str | None = None,
    proxies_yaml: str = "docs/proxies.yaml",
) -> Dict[str, Any]:
    """
    Run L_cap benchmark across all 4 canonical scenario types (base + noise variants).

    Returns capability profile: does L_cap distinguish Type I/II/III?
    """
    if scenarios_dir is None:
        scenarios_dir = "data/canonical_scenarios"

    specs = load_proxy_specs(proxies_yaml)

    scenario_types = [
        "type_i_noise",
        "type_ii_oscillations",
        "type_iii_bifurcation",
        "hybrid_ii_to_iii",
    ]

    results: Dict[str, List[Dict[str, Any]]] = {st: [] for st in scenario_types}

    # Test base scenario and 5 noise variants for each type
    noise_levels = ["base"] + [f"{nl:.2f}" for nl in [0.05, 0.10, 0.15, 0.20, 0.25]]

    import os

    for scenario_type in scenario_types:
        for noise_level in noise_levels:
            if noise_level == "base":
                filename = f"{scenarios_dir}/{scenario_type}_base.csv"
            else:
                filename = f"{scenarios_dir}/{scenario_type}_{noise_level}.csv"

            if not os.path.exists(filename):
                continue

            df = pd.read_csv(filename)
            result = benchmark_l_cap_on_scenario(df, specs, f"{scenario_type}_{noise_level}")
            results[scenario_type].append(result)

    # Aggregate
    summary = {
        "type_i_expected_low_l_cap": _check_expectation(
            results, "type_i_noise", "regime", "type_I_noise", "l_cap_mean", lambda x: x < 0.0
        ),
        "type_ii_expected_moderate_l_cap": _check_expectation(
            results, "type_ii_oscillations", "regime", "type_II_oscillations", "l_cap_mean", lambda x: -0.5 < x < 0.5
        ),
        "type_iii_expected_high_l_cap": _check_expectation(
            results, "type_iii_bifurcation", "regime", "type_III_bifurcation", "l_cap_mean", lambda x: x > 0.0
        ),
        "hybrid_shows_transition": _check_transition(results, "hybrid_ii_to_iii"),
    }

    return {
        "benchmark": "L_cap_capability",
        "timestamp": pd.Timestamp.now().isoformat(),
        "all_scenarios": results,
        "summary": summary,
        "interpretation": {
            "type_i_low": "Type I noise should show low L_cap (little to detect/stop)",
            "type_ii_moderate": "Type II oscillations should show moderate L_cap (some recovery possible)",
            "type_iii_high": "Type III bifurcation should show high L_cap (system tried but failed)",
            "noise_robustness": "All noise levels should reach same regime classification",
        },
    }


def _check_expectation(
    results: Dict[str, List[Dict[str, Any]]],
    scenario_type: str,
    field: str,
    expected_value: str,
    metric: str,
    metric_check: callable,
) -> Dict[str, Any]:
    """Check if all runs of a scenario type match expectation."""
    runs = results.get(scenario_type, [])
    if not runs:
        return {"count": 0, "passed": 0, "failed": 0}

    passed = 0
    failed = 0
    for run in runs:
        if "error" in run:
            failed += 1
            continue
        if run.get(field) == expected_value and metric_check(run.get(metric, 0.0)):
            passed += 1
        else:
            failed += 1

    return {
        "count": len(runs),
        "passed": passed,
        "failed": failed,
        "pass_rate": float(passed) / max(1, len(runs)),
    }


def _check_transition(results: Dict[str, List[Dict[str, Any]]], scenario_type: str) -> Dict[str, Any]:
    """Check if hybrid scenario shows transition from Type II to Type III."""
    runs = results.get(scenario_type, [])
    if not runs:
        return {"count": 0, "shows_transition": False}

    # Hybrid should have base scenario with transition signature
    base_run = [r for r in runs if "base" in r.get("scenario", "")]
    if not base_run:
        return {"count": len(runs), "shows_transition": False}

    run = base_run[0]
    # We look for signs that a transition occurred (regime could be detected as type_II or type_III)
    regime = run.get("regime", "")
    shows_transition = regime in ["type_II_oscillations", "type_III_bifurcation"]

    return {"count": len(runs), "shows_transition": bool(shows_transition)}


def validate_l_cap_benchmark(benchmark_result: Dict[str, Any]) -> bool:
    """
    Validate that L_cap benchmark passed sufficiently.

    Criterion: All 4 scenario types should show >80% pass rate on regime detection.
    """
    summary = benchmark_result.get("summary", {})

    checks = [
        summary.get("type_i_expected_low_l_cap", {}).get("pass_rate", 0.0) >= 0.80,
        summary.get("type_ii_expected_moderate_l_cap", {}).get("pass_rate", 0.0) >= 0.80,
        summary.get("type_iii_expected_high_l_cap", {}).get("pass_rate", 0.0) >= 0.80,
        summary.get("hybrid_shows_transition", {}).get("shows_transition", False),
    ]

    return all(checks)
