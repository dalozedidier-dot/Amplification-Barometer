#!/usr/bin/env python3
"""
L_cap vs L_act Validation: Distinguish intrinsic capacity from observed activation.

Produces a 2x2 matrix across test datasets:
- X-axis: L_cap (intrinsic capability, from benchmark)
- Y-axis: L_act (observed activation, from data)

Result shows if system is:
1. Capable + Activated (SUCCESS)
2. Capable + Not-Activated (MISS: failure to detect)
3. Incapable + Activated (FALSE ALARM)
4. Incapable + Not-Activated (BENIGN: both good)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from amplification_barometer.alignment_audit import run_alignment_audit
from amplification_barometer.l_capability_benchmark import benchmark_suite_all_scenarios, validate_l_cap_benchmark
from amplification_barometer.l_operator import compute_l_act, compute_l_cap


def load_test_datasets(scenarios_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """Load all canonical scenario datasets."""
    scenarios_dir = Path(scenarios_dir)
    datasets = {}

    for csv_file in sorted(scenarios_dir.glob("*.csv")):
        name = csv_file.stem
        datasets[name] = pd.read_csv(csv_file)

    return datasets


def classify_l_cap_l_act(l_cap: float, l_act: float, thresholds: Dict[str, float]) -> Dict[str, Any]:
    """
    Classify a (L_cap, L_act) point into quadrant.

    Thresholds:
      - l_cap_threshold: 0.0 (neutral/dividing line)
      - l_act_threshold: 0.0 (neutral/dividing line)
    """
    cap_threshold = thresholds.get("l_cap_threshold", 0.0)
    act_threshold = thresholds.get("l_act_threshold", 0.0)

    capable = l_cap >= cap_threshold
    activated = l_act >= act_threshold

    if capable and activated:
        quadrant = "success"
        interpretation = "Capable AND Activated: System working as designed"
    elif capable and not activated:
        quadrant = "miss"
        interpretation = "Capable BUT NOT Activated: Failed to detect/respond"
    elif not capable and activated:
        quadrant = "false_alarm"
        interpretation = "Incapable BUT Activated: Spurious detection or lucky"
    else:
        quadrant = "benign"
        interpretation = "Incapable AND Not-Activated: Both good (no alarm needed)"

    return {
        "l_cap": float(l_cap),
        "l_act": float(l_act),
        "capable": bool(capable),
        "activated": bool(activated),
        "quadrant": quadrant,
        "interpretation": interpretation,
    }


def run_validation(scenarios_dir: str, proxies_yaml: str) -> Dict[str, Any]:
    """
    Full L_cap vs L_act validation across all test datasets.

    1. Benchmark L_cap on canonical scenarios (independent)
    2. Compute L_act on same datasets (observed)
    3. Produce 2x2 matrix and interpretation
    """
    # Step 1: L_cap benchmark
    print("Step 1: Benchmarking L_cap on canonical scenarios...")
    l_cap_benchmark = benchmark_suite_all_scenarios(scenarios_dir=scenarios_dir, proxies_yaml=proxies_yaml)
    l_cap_valid = validate_l_cap_benchmark(l_cap_benchmark)

    if not l_cap_valid:
        print("⚠️  L_cap benchmark did not pass all criteria. Continuing with caution...")

    # Step 2: Load datasets and compute L_cap, L_act
    print("Step 2: Computing L_cap and L_act on all scenarios...")
    datasets = load_test_datasets(scenarios_dir)

    dataset_metrics = []
    for dataset_name, df in datasets.items():
        try:
            l_cap = compute_l_cap(df)
            l_act = compute_l_act(df)

            l_cap_mean = float(np.nanmean(l_cap))
            l_act_mean = float(np.nanmean(l_act))

            classification = classify_l_cap_l_act(l_cap_mean, l_act_mean, {"l_cap_threshold": 0.0, "l_act_threshold": 0.0})

            dataset_metrics.append({
                "dataset": dataset_name,
                **classification,
                "l_cap_std": float(np.nanstd(l_cap)),
                "l_act_std": float(np.nanstd(l_act)),
            })
        except Exception as e:
            dataset_metrics.append({"dataset": dataset_name, "error": str(e)})

    # Step 3: Aggregate into 2x2 matrix
    print("Step 3: Aggregating into 2x2 matrix...")

    quadrant_counts = {"success": 0, "miss": 0, "false_alarm": 0, "benign": 0}
    quadrant_examples = {q: [] for q in quadrant_counts}

    for metric in dataset_metrics:
        if "error" in metric:
            continue
        q = metric.get("quadrant")
        if q:
            quadrant_counts[q] += 1
            quadrant_examples[q].append(metric["dataset"])

    # Compute rates
    total = sum(quadrant_counts.values())
    quadrant_rates = {q: float(c) / max(1, total) for q, c in quadrant_counts.items()}

    return {
        "timestamp": pd.Timestamp.now().isoformat(),
        "l_cap_benchmark": {
            "passed": bool(l_cap_valid),
            "summary": l_cap_benchmark.get("summary", {}),
        },
        "dataset_metrics": dataset_metrics,
        "matrix_2x2": {
            "success": {"count": quadrant_counts["success"], "rate": quadrant_rates["success"], "examples": quadrant_examples["success"][:3]},
            "miss": {"count": quadrant_counts["miss"], "rate": quadrant_rates["miss"], "examples": quadrant_examples["miss"][:3]},
            "false_alarm": {"count": quadrant_counts["false_alarm"], "rate": quadrant_rates["false_alarm"], "examples": quadrant_examples["false_alarm"][:3]},
            "benign": {"count": quadrant_counts["benign"], "rate": quadrant_rates["benign"], "examples": quadrant_examples["benign"][:3]},
        },
        "interpretation": {
            "success_rate": quadrant_rates.get("success", 0.0),
            "miss_rate": quadrant_rates.get("miss", 0.0),
            "false_alarm_rate": quadrant_rates.get("false_alarm", 0.0),
            "overall_verdict": "credible" if quadrant_rates.get("success", 0.0) >= 0.60 and quadrant_rates.get("miss", 0.0) <= 0.20 else "needs_work",
        },
    }


def write_validation_report(report: Dict[str, Any], out_dir: str | Path) -> Dict[str, str]:
    """Write L_cap vs L_act report to JSON and markdown."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "l_cap_vs_act_validation.json"
    md_path = out / "l_cap_vs_act_validation.md"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")

    # Markdown
    md = []
    md.append("# L_cap vs L_act Validation Report\n\n")
    md.append(f"**Timestamp:** {report.get('timestamp', 'N/A')}\n\n")

    md.append("## L_cap Benchmark\n\n")
    bench = report.get("l_cap_benchmark", {})
    md.append(f"- **Passed:** {bench.get('passed', False)}\n")
    md.append(f"- **Type I (low L_cap):** {bench.get('summary', {}).get('type_i_expected_low_l_cap', {}).get('pass_rate', 'N/A')}\n")
    md.append(f"- **Type II (moderate L_cap):** {bench.get('summary', {}).get('type_ii_expected_moderate_l_cap', {}).get('pass_rate', 'N/A')}\n")
    md.append(f"- **Type III (high L_cap):** {bench.get('summary', {}).get('type_iii_expected_high_l_cap', {}).get('pass_rate', 'N/A')}\n\n")

    md.append("## 2x2 Matrix: L_cap × L_act\n\n")
    matrix = report.get("matrix_2x2", {})

    md.append("```\n")
    md.append("                 L_act ≥ 0            L_act < 0\n")
    md.append(f"L_cap ≥ 0        SUCCESS ({matrix.get('success', {}).get('count', 0)})       MISS ({matrix.get('miss', {}).get('count', 0)})\n")
    md.append(f"L_cap < 0        FALSE_ALARM ({matrix.get('false_alarm', {}).get('count', 0)})  BENIGN ({matrix.get('benign', {}).get('count', 0)})\n")
    md.append("```\n\n")

    md.append("## Verdict\n\n")
    interpretation = report.get("interpretation", {})
    md.append(f"- **Success Rate:** {interpretation.get('success_rate', 0.0):.1%}\n")
    md.append(f"- **Miss Rate:** {interpretation.get('miss_rate', 0.0):.1%}\n")
    md.append(f"- **False Alarm Rate:** {interpretation.get('false_alarm_rate', 0.0):.1%}\n")
    md.append(f"- **Overall:** {interpretation.get('overall_verdict', 'unknown').upper()}\n\n")

    md.append("## Interpretation\n\n")
    md.append("- **SUCCESS:** System has capacity AND activates appropriately → working design\n")
    md.append("- **MISS:** System capable but fails to detect/respond → detection gap\n")
    md.append("- **FALSE_ALARM:** System activates without real capacity → spurious signals\n")
    md.append("- **BENIGN:** No alarm needed, system quiet → healthy\n\n")

    md.append("## Examples by Quadrant\n\n")
    for quad in ["success", "miss", "false_alarm", "benign"]:
        examples = matrix.get(quad, {}).get("examples", [])
        md.append(f"### {quad.upper()}\n\n")
        for ex in examples:
            md.append(f"- {ex}\n")
        md.append("\n")

    md_path.write_text("".join(md), encoding="utf-8")

    return {"json": str(json_path), "md": str(md_path)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate L_cap vs L_act: separate intrinsic capacity from observed activation.")
    ap.add_argument("--scenarios-dir", default="data/canonical_scenarios", help="Scenarios directory.")
    ap.add_argument("--proxies-yaml", default="docs/proxies.yaml", help="Proxy spec yaml.")
    ap.add_argument("--out-dir", default="reports/l_cap_vs_act", help="Output directory.")
    args = ap.parse_args()

    print(f"Running L_cap vs L_act validation...")
    print(f"  Scenarios: {args.scenarios_dir}")
    print(f"  Proxies: {args.proxies_yaml}")
    print()

    report = run_validation(args.scenarios_dir, args.proxies_yaml)
    paths = write_validation_report(report, args.out_dir)

    print(f"\n✓ Report written:")
    print(f"  JSON: {paths['json']}")
    print(f"  MD:   {paths['md']}")
    print()
    print(f"Verdict: {report.get('interpretation', {}).get('overall_verdict', 'unknown').upper()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
