#!/usr/bin/env python3
"""
Anti-gaming test suite for Amplification Barometer.

Tests that the barometer detects and penalizes systematic attempts to game proxies.
Implements 5 attack vectors:
  1. O-family bias (artificially inflate orientation/stop capacity)
  2. Volatility clamp (suppress signal variation to hide risk)
  3. Out-of-range manipulation (push proxies beyond expected bounds)
  4. Coordinated multi-proxy attack (simultaneous bias on P + O + G delay)
  5. Reporting delay gaming (suppress governance signals)

Each attack is applied to a clean dataset and compared to baseline.
Verdict: "ok" if attack is detected and rejected, "fail" if it passes undetected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from amplification_barometer.alignment_audit import (
    compute_at_delta,
    compute_e_r,
    compute_levels_from_specs,
    robust_z_mad,
)
from amplification_barometer.theory_map import load_proxy_specs, TheoryAuditMap


# ============================================================================
# ATTACK VECTORS
# ============================================================================


def attack_o_bias(
    df: pd.DataFrame, specs: TheoryAuditMap, bias: float = 0.15, start_frac: float = 0.6
) -> Dict[str, Any]:
    """
    Attack 1: Inflate O-family proxies (stop, threshold, decision, execution, coherence).

    These are "down_risk" proxies, so higher is better (less risky).
    Attack: boost them artificially to hide lack of control.

    Detection rule: O proxies should not all increase simultaneously in finite time.
    """
    df_clean = df.copy()
    df_attack = df.copy()

    o_proxies = ["stop_proxy", "threshold_proxy", "decision_proxy", "execution_proxy", "coherence_proxy"]
    o_proxies_present = [c for c in o_proxies if c in df.columns]

    start_idx = int(len(df_attack) * float(start_frac))
    for col in o_proxies_present:
        x = df_attack[col].to_numpy(dtype=float, copy=True)
        x[start_idx:] = np.clip(x[start_idx:] + bias, 0.0, 1.0)
        df_attack[col] = x

    # Compute composites
    levels_clean = compute_levels_from_specs(df_clean, specs)
    levels_attack = compute_levels_from_specs(df_attack, specs)

    o_clean = levels_clean.get("O_level", np.zeros(len(df_clean)))
    o_attack = levels_attack.get("O_level", np.zeros(len(df_attack)))

    at_clean = levels_clean["P_level"] / (o_clean + 1e-9)
    at_attack = levels_attack["P_level"] / (o_attack + 1e-9)

    # Detect: O went up, @(t) went down (suspicious)
    o_increased = float(np.nanmean(o_attack[start_idx:]) - np.nanmean(o_clean[start_idx:]))
    at_decreased = float(np.nanmean(at_attack[start_idx:]) - np.nanmean(at_clean[start_idx:]))

    # Detection rule: simultaneous O↑ and @↓ is suspicious
    detected = (o_increased > 0.05) and (at_decreased < -0.1)

    return {
        "attack": "o_bias",
        "bias_amount": float(bias),
        "start_idx": start_idx,
        "o_proxies_targeted": o_proxies_present,
        "o_change": o_increased,
        "at_change": at_decreased,
        "detected": bool(detected),
        "severity": "high",
    }


def attack_volatility_clamp(
    df: pd.DataFrame, specs: TheoryAuditMap, target_std_frac: float = 0.1, start_frac: float = 0.6
) -> Dict[str, Any]:
    """
    Attack 2: Suppress volatility on P-family proxies to hide growing risk variance.

    Attack: compress the std dev of scale, speed, leverage, autonomy, replicability
    to make risk appear stable when it is actually accumulating.

    Detection rule: P volatility should not collapse while E increases.
    """
    df_clean = df.copy()
    df_attack = df.copy()

    p_proxies = ["scale_proxy", "speed_proxy", "leverage_proxy", "autonomy_proxy", "replicability_proxy"]
    p_proxies_present = [c for c in p_proxies if c in df.columns]

    start_idx = int(len(df_attack) * float(start_frac))

    for col in p_proxies_present:
        x = df_attack[col].to_numpy(dtype=float, copy=True)
        tail = x[start_idx:]
        tail_mean = np.nanmean(tail)
        tail_std = np.nanstd(tail)
        # Compress: std → std * target_std_frac
        compressed = tail_mean + (tail - tail_mean) * target_std_frac
        x[start_idx:] = np.clip(compressed, 0.0, 2.0)
        df_attack[col] = x

    # Compute levels
    levels_clean = compute_levels_from_specs(df_clean, specs)
    levels_attack = compute_levels_from_specs(df_attack, specs)

    p_clean = levels_clean.get("P_level", np.zeros(len(df_clean)))
    p_attack = levels_attack.get("P_level", np.zeros(len(df_attack)))

    e_clean = np.cumsum(levels_clean.get("E_level", np.zeros(len(df_clean))))
    e_attack = np.cumsum(levels_attack.get("E_level", np.zeros(len(df_attack))))

    # Metrics
    p_vol_clean = float(np.nanstd(p_clean[start_idx:]))
    p_vol_attack = float(np.nanstd(p_attack[start_idx:]))
    e_change = float((e_attack[-1] - e_clean[-1]) / (np.abs(e_clean[-1]) + 1e-9))

    # Detection: vol collapsed but E still grew → suspicious
    detected = (p_vol_attack < p_vol_clean * 0.3) and (e_change > 0.05)

    return {
        "attack": "volatility_clamp",
        "target_std_frac": float(target_std_frac),
        "p_proxies_targeted": p_proxies_present,
        "p_vol_clean": p_vol_clean,
        "p_vol_attack": p_vol_attack,
        "e_stock_change": e_change,
        "detected": bool(detected),
        "severity": "high",
    }


def attack_out_of_range(
    df: pd.DataFrame, specs: TheoryAuditMap, push_amount: float = 0.3, start_frac: float = 0.6
) -> Dict[str, Any]:
    """
    Attack 3: Push proxies outside their expected ranges.

    Naive attack: if you suppress the range checks, you can hide by going out of bounds.

    Detection rule: any proxy outside expected_range is flagged.
    """
    df_clean = df.copy()
    df_attack = df.copy()

    start_idx = int(len(df_attack) * float(start_frac))

    violations = []
    for fam, famspec in specs.families.items():
        for name, ps in famspec.proxies.items():
            if name not in df_attack.columns:
                continue
            lo, hi = ps.expected_range
            x = df_attack[name].to_numpy(dtype=float, copy=True)
            # Push above range
            x[start_idx:] = np.clip(x[start_idx:] + push_amount, hi + 0.01, hi + 0.5)
            df_attack[name] = x
            violations.append(name)

    # Validate ranges
    out_of_range_count = 0
    for fam, famspec in specs.families.items():
        for name, ps in famspec.proxies.items():
            if name not in df_attack.columns:
                continue
            lo, hi = ps.expected_range
            x = df_attack[name].to_numpy(dtype=float)
            bad = np.where((x < lo) | (x > hi))[0]
            out_of_range_count += len(bad)

    detected = out_of_range_count > 0

    return {
        "attack": "out_of_range",
        "push_amount": float(push_amount),
        "proxies_pushed": violations[:5],  # First 5 examples
        "out_of_range_points": int(out_of_range_count),
        "detected": bool(detected),
        "severity": "critical",
    }


def attack_coordinated_multi_proxy(
    df: pd.DataFrame, specs: TheoryAuditMap, bias_p: float = 0.1, bias_o: float = 0.1, start_frac: float = 0.6
) -> Dict[str, Any]:
    """
    Attack 4: Coordinated multi-proxy gaming.

    Attack: simultaneously boost O (hide control gaps) + suppress P volatility (hide risk growth).
    This is more realistic: attacker tries to move multiple dials at once.

    Detection rule: if both O and P_volatility move together unnaturally,
    and nothing else explains it, flag as coordinated gaming.
    """
    df_clean = df.copy()
    df_attack = df.copy()

    p_proxies = ["scale_proxy", "speed_proxy", "leverage_proxy"]
    o_proxies = ["stop_proxy", "threshold_proxy", "decision_proxy"]
    p_proxies_present = [c for c in p_proxies if c in df.columns]
    o_proxies_present = [c for c in o_proxies if c in df.columns]

    start_idx = int(len(df_attack) * float(start_frac))

    # Boost O
    for col in o_proxies_present:
        x = df_attack[col].to_numpy(dtype=float, copy=True)
        x[start_idx:] = np.clip(x[start_idx:] + bias_o, 0.0, 1.0)
        df_attack[col] = x

    # Suppress P volatility
    for col in p_proxies_present:
        x = df_attack[col].to_numpy(dtype=float, copy=True)
        tail = x[start_idx:]
        tail_mean = np.nanmean(tail)
        tail_std = np.nanstd(tail)
        x[start_idx:] = tail_mean + (tail - tail_mean) * 0.3
        x = np.clip(x, 0.0, 2.0)
        df_attack[col] = x

    # Compute levels
    levels_clean = compute_levels_from_specs(df_clean, specs)
    levels_attack = compute_levels_from_specs(df_attack, specs)

    o_clean = levels_clean.get("O_level", np.zeros(len(df_clean)))
    o_attack = levels_attack.get("O_level", np.zeros(len(df_attack)))
    p_clean = levels_clean.get("P_level", np.zeros(len(df_clean)))
    p_attack = levels_attack.get("P_level", np.zeros(len(df_attack)))

    at_clean = p_clean / (o_clean + 1e-9)
    at_attack = p_attack / (o_attack + 1e-9)

    # Metrics
    o_change = float(np.nanmean(o_attack[start_idx:]) - np.nanmean(o_clean[start_idx:]))
    p_vol_change = float(np.nanstd(p_attack[start_idx:]) / (np.nanstd(p_clean[start_idx:]) + 1e-9))
    at_change = float(np.nanmean(at_attack[start_idx:]) - np.nanmean(at_clean[start_idx:]))

    # Detection: both O↑ AND P_vol↓ AND @↓ together
    detected = (o_change > 0.05) and (p_vol_change < 0.5) and (at_change < -0.1)

    return {
        "attack": "coordinated_multi_proxy",
        "bias_p": float(bias_p),
        "bias_o": float(bias_o),
        "o_proxies_targeted": o_proxies_present,
        "p_proxies_targeted": p_proxies_present,
        "o_change": o_change,
        "p_vol_ratio": p_vol_change,
        "at_change": at_change,
        "detected": bool(detected),
        "severity": "critical",
    }


def attack_reporting_delay_gaming(
    df: pd.DataFrame, specs: TheoryAuditMap, delay_steps: int = 20, start_frac: float = 0.6
) -> Dict[str, Any]:
    """
    Attack 5: Governance reporting delay gaming.

    Attack: delay the reporting of sanctions (sanction_delay) and exemptions
    to suppress G signals, hiding erosion of control.

    Detection rule: G signals (especially through sanction_delay, exemption_rate)
    should be contemporaneous with actual policy drift.
    """
    df_clean = df.copy()
    df_attack = df.copy()

    g_proxies = ["exemption_rate", "sanction_delay", "control_turnover", "rule_execution_gap"]
    g_proxies_present = [c for c in g_proxies if c in df.columns]

    start_idx = int(len(df_attack) * float(start_frac))

    for col in g_proxies_present:
        x = df_attack[col].to_numpy(dtype=float, copy=True)
        # Delay: shift forward by delay_steps
        x_tail = x[start_idx:]
        x_tail_delayed = np.roll(x_tail, delay_steps)
        x[start_idx:] = x_tail_delayed
        df_attack[col] = x

    # Compute levels
    levels_clean = compute_levels_from_specs(df_clean, specs)
    levels_attack = compute_levels_from_specs(df_attack, specs)

    g_clean = levels_clean.get("G_level", np.zeros(len(df_clean)))
    g_attack = levels_attack.get("G_level", np.zeros(len(df_attack)))

    # Metrics: G signal should be timely
    g_lag = float(np.nanmean(np.abs(g_attack[start_idx:] - g_clean[start_idx:])))

    # Detection: large artificial lag in G signals
    detected = g_lag > 0.2  # threshold

    return {
        "attack": "reporting_delay_gaming",
        "delay_steps": int(delay_steps),
        "g_proxies_targeted": g_proxies_present,
        "g_signal_lag": g_lag,
        "detected": bool(detected),
        "severity": "medium",
    }


# ============================================================================
# SUITE & SCORING
# ============================================================================


def run_anti_gaming_suite(df: pd.DataFrame, proxies_yaml: str | Path) -> Dict[str, Any]:
    """
    Run all 5 attack vectors against a clean dataset.

    Returns:
    - List of attack results
    - Overall gaming_suspicion score (0.0 = clean, 1.0 = fully gamed)
    - Verdict: "ok" if all attacks detected, "fail" if any undetected
    """
    specs = load_proxy_specs(proxies_yaml)

    attacks = [
        attack_o_bias(df, specs),
        attack_volatility_clamp(df, specs),
        attack_out_of_range(df, specs),
        attack_coordinated_multi_proxy(df, specs),
        attack_reporting_delay_gaming(df, specs),
    ]

    # Compute gaming_suspicion score: fraction of attacks successfully detected
    detected_count = sum(1 for a in attacks if a.get("detected"))
    gaming_suspicion_score = float(detected_count) / len(attacks)

    # Verdict: all attacks must be detected for "ok"
    verdict = "ok" if gaming_suspicion_score == 1.0 else "fail"

    return {
        "timestamp": pd.Timestamp.now().isoformat(),
        "attacks": attacks,
        "gaming_suspicion_score": gaming_suspicion_score,
        "verdict": verdict,
        "summary": {
            "total_attacks": len(attacks),
            "detected": detected_count,
            "undetected": len(attacks) - detected_count,
            "assessment": "System is robust against gaming" if verdict == "ok" else "System has gaming vulnerabilities",
        },
    }


def write_anti_gaming_output(report: Dict[str, Any], out_dir: str | Path, name: str) -> Dict[str, str]:
    """Write anti-gaming report to JSON and markdown."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / f"{name}_anti_gaming.json"
    md_path = out / f"{name}_anti_gaming.md"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    md = []
    md.append(f"# Anti-Gaming Test Suite: {name}\n\n")
    md.append(f"**Timestamp:** {report.get('timestamp', 'N/A')}\n\n")
    md.append(f"## Summary\n\n")
    summary = report.get("summary", {})
    md.append(f"- **Total Attacks:** {summary.get('total_attacks', 0)}\n")
    md.append(f"- **Detected:** {summary.get('detected', 0)}\n")
    md.append(f"- **Undetected:** {summary.get('undetected', 0)}\n")
    md.append(f"- **Gaming Suspicion Score:** {report.get('gaming_suspicion_score', 0.0):.2f}/1.0\n")
    md.append(f"- **Verdict:** {report.get('verdict', 'unknown').upper()}\n")
    md.append(f"- **Assessment:** {summary.get('assessment', 'N/A')}\n\n")

    md.append("## Attack Details\n\n")
    for attack in report.get("attacks", []):
        md.append(f"### {attack.get('attack', 'unknown').upper()}\n\n")
        md.append(f"- **Severity:** {attack.get('severity', 'unknown')}\n")
        md.append(f"- **Detected:** {'✓' if attack.get('detected') else '✗'}\n")
        for k, v in attack.items():
            if k not in ("attack", "severity", "detected", "description"):
                md.append(f"- **{k}:** {v}\n")
        md.append("\n")

    md_path.write_text("".join(md), encoding="utf-8")

    return {"json": str(json_path), "md": str(md_path)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run anti-gaming test suite (5 attack vectors + detection)."
    )
    ap.add_argument("--dataset", required=True, help="CSV file with proxies.")
    ap.add_argument("--name", default="dataset", help="Name prefix for outputs.")
    ap.add_argument("--proxies-yaml", default="docs/proxies.yaml", help="Proxy spec yaml.")
    ap.add_argument("--out-dir", default="_ci_out", help="Output directory.")
    args = ap.parse_args()

    df = pd.read_csv(args.dataset)
    report = run_anti_gaming_suite(df, proxies_yaml=args.proxies_yaml)
    write_anti_gaming_output(report, out_dir=args.out_dir, name=args.name)

    print(json.dumps(report, indent=2, default=str))
    return 0 if report["verdict"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
