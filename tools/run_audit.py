#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from amplification_barometer.audit_report import build_audit_report, write_audit_report  # noqa: E402
from amplification_barometer.calibration import discriminate_regimes  # noqa: E402
from amplification_barometer.composites import compute_at, compute_delta_d  # noqa: E402
from amplification_barometer.l_operator import compute_l_act, compute_l_cap  # noqa: E402


def _plot_series(df: pd.DataFrame, out_dir: Path, *, window: int = 5, prefix: str = "") -> None:
    at = compute_at(df)
    dd = compute_delta_d(df, window=window)
    lcap = compute_l_cap(df)
    lact = compute_l_act(df)

    pfx = f"{prefix}_" if prefix else ""

    fig1 = plt.figure()
    plt.plot(at.index, at.to_numpy())
    plt.title("@(t)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig1.savefig(out_dir / f"{pfx}at.png", dpi=150)
    plt.close(fig1)

    fig2 = plt.figure()
    plt.plot(dd.index, dd.to_numpy())
    plt.title("Δd(t)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig2.savefig(out_dir / f"{pfx}delta_d.png", dpi=150)
    plt.close(fig2)

    fig3 = plt.figure()
    plt.plot(lcap.index, lcap.to_numpy(), label="L_cap")
    plt.plot(lact.index, lact.to_numpy(), label="L_act")
    plt.title("L_cap / L_act")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig3.savefig(out_dir / f"{pfx}l_cap_l_act.png", dpi=150)
    plt.close(fig3)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")


def _write_md_summary(report, out_path: Path) -> None:
    # keep this markdown minimal and audit-friendly
    lines = [
        f"# Audit report: {report.dataset_name}",
        "",
        f"Version: {report.version}",
        f"Weights version: {report.weights_version}",
        "",
        "## Stability",
        f"Stable flag: {report.stability.get('stable_flag')}",
        f"Spearman worst (risk): {float(report.stability.get('spearman_worst_risk')):.3f}",
        f"TopK Jaccard worst (risk): {float(report.stability.get('topk_jaccard_worst_risk')):.3f}",
        "",
        "## Limit operator performance",
        f"Verdict: {report.l_performance.get('verdict')}",
        f"Prevented exceedance: {float(report.l_performance.get('prevented_exceedance')):.3f}",
        f"First activation delay (steps): {report.l_performance.get('first_activation_delay_steps')}",
        f"Risk drop around activation: {report.l_performance.get('risk_drop_around_activation')}",
        "",
        "## Maturity",
        f"Label: {report.maturity.get('label')}",
        f"L_cap raw mean: {float(report.maturity.get('cap_score_raw')):.3f}",
        f"L_cap enforced mean: {float(report.maturity.get('cap_score_enforced')):.3f}",
        f"Turnover mean: {float(report.maturity.get('notes', {}).get('turnover_mean', 0.0)):.3f}",
        f"Enforcement factor: {float(report.maturity.get('notes', {}).get('enforcement_factor', 1.0)):.3f}",
        f"L_act raw mean: {float(report.maturity.get('act_score_raw')):.3f}",
        f"L_act enforced mean: {float(report.maturity.get('act_score_enforced')):.3f}",
        f"L_act drop under stress: {float(report.maturity.get('act_drop_under_stress')):.3f}",
        "",
        "## Anti-gaming",
        f"Detection rate (suite): {float(report.manipulability.get('summary', {}).get('detected_rate', 0.0)):.3f}",
        f"Scenarios tested: {int(report.manipulability.get('summary', {}).get('n_scenarios', 0))}",
        "",
        "## Stress suite",
    ]
    for k, v in report.stress_suite.items():
        lines.append(f"- {k}: {v.get('status')} (degradation={float(v.get('degradation')):.3f})")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Amplification Barometer audit reports (demo).")
    ap.add_argument("--dataset", type=str, help="Path to CSV dataset (single run)")
    ap.add_argument("--name", type=str, default="dataset", help="Dataset name for the report")
    ap.add_argument("--synthetic-dir", type=str, default="data/synthetic", help="Directory containing synthetic regimes")
    ap.add_argument("--all-synthetic", action="store_true", help="Run on stable/oscillating/bifurcation and build calibration report")
    ap.add_argument("--out-dir", type=str, default="_ci_out", help="Output directory")
    ap.add_argument("--plot", action="store_true", help="Generate simple plots (PNG)")
    ap.add_argument("--window", type=int, default=5, help="Window for Δd(t)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all_synthetic:
        sdir = Path(args.synthetic_dir)
        paths = {
            "stable": sdir / "stable_regime.csv",
            "oscillating": sdir / "oscillating_regime.csv",
            "bifurcation": sdir / "bifurcation_regime.csv",
        }
        for name, p in paths.items():
            if not p.exists():
                raise SystemExit(f"Missing synthetic dataset: {p}")

        dfs: Dict[str, pd.DataFrame] = {k: _read_csv(p) for k, p in paths.items()}

        # per-dataset audit reports
        for name, df in dfs.items():
            rep = build_audit_report(df, dataset_name=name, delta_d_window=args.window)
            write_audit_report(rep, out_dir / f"audit_report_{name}.json")
            _write_md_summary(rep, out_dir / f"audit_report_{name}.md")
            if args.plot:
                _plot_series(df, out_dir, window=args.window, prefix=name)

        # calibration / discrimination report
        calib = discriminate_regimes(dfs, stable_name="stable", window=args.window)

        # expected qualitative ordering: stable lowest risk, bifurcation highest risk
        ordering = calib.get("ordering_by_severity", calib.get("ordering_by_mean_risk", []))
        ok = (ordering[-1] == "bifurcation") if ordering else False
        calib["expected_ordering_ok"] = bool(ok)

        (out_dir / "calibration_report.json").write_text(
            __import__("json").dumps(calib, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )

        portfolio_lines = [
            "# Portfolio calibration report",
            "",
            f"Expected ordering ok: {ok}",
            f"Ordering by severity: {ordering}",
            "",
            "Thresholds derived from stable:",
        ]
        thr = calib.get("thresholds", {})
        for k, v in thr.items():
            portfolio_lines.append(f"- {k}: {v}")
        portfolio_lines.append("")
        out_dir.joinpath("portfolio_summary.md").write_text("\n".join(portfolio_lines), encoding="utf-8")
        return 0

    # single dataset mode
    if not args.dataset:
        raise SystemExit("Either provide --dataset or use --all-synthetic")

    path = Path(args.dataset)
    if not path.exists():
        raise SystemExit(f"Dataset not found: {path}")

    df = _read_csv(path)
    report = build_audit_report(df, dataset_name=args.name, delta_d_window=args.window)
    write_audit_report(report, out_dir / "audit_report.json")
    _write_md_summary(report, out_dir / "audit_report.md")

    if args.plot:
        _plot_series(df, out_dir, window=args.window)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
