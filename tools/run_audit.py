#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from amplification_barometer.audit_report import build_audit_report, write_audit_report  # noqa: E402
from amplification_barometer.calibration import discriminate_regimes  # noqa: E402
from amplification_barometer.composites import compute_at, compute_delta_d  # noqa: E402
from amplification_barometer.l_operator import compute_l_act, compute_l_cap  # noqa: E402


def _plot_series_matplotlib(
    df: pd.DataFrame, out_dir: Path, *, window: int = 5, prefix: str = ""
) -> None:
    """Legacy PNG plots (matplotlib).

    Keep this as a fallback (CI-friendly). For modern interactive plots, use --plotly.
    """
    at = compute_at(df)
    dd = compute_delta_d(df, window=window)
    lcap = compute_l_cap(df)
    lact = compute_l_act(df)

    pfx = f"{prefix}_" if prefix else ""

    fig1 = plt.figure(figsize=(11, 5.5))
    plt.plot(at.index, at.to_numpy(), linewidth=2.0)
    plt.title("@(t)", fontsize=14, pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("@(t)", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xticks(rotation=45)
    plt.tight_layout(pad=1.5)
    fig1.savefig(out_dir / f"{pfx}at.png", dpi=150)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(11, 5.5))
    plt.plot(dd.index, dd.to_numpy(), linewidth=2.0)
    plt.title("Δd(t)", fontsize=14, pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Δd(t)", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xticks(rotation=45)
    plt.tight_layout(pad=1.5)
    fig2.savefig(out_dir / f"{pfx}delta_d.png", dpi=150)
    plt.close(fig2)

    fig3 = plt.figure(figsize=(11, 6.0))
    plt.plot(lcap.index, lcap.to_numpy(), label="L_cap", linewidth=2.0)
    plt.plot(lact.index, lact.to_numpy(), label="L_act", linewidth=2.0)
    plt.title("L_cap vs L_act", fontsize=14, pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout(pad=1.5)
    fig3.savefig(out_dir / f"{pfx}l_cap_l_act.png", dpi=150)
    plt.close(fig3)


def _plot_series_plotly(
    df: pd.DataFrame, out_dir: Path, *, window: int = 5, prefix: str = ""
) -> None:
    """Interactive HTML plots (Plotly)."""
    from amplification_barometer.plotly_viz import (
        build_dashboard,
        plot_exponential_or_bifurcation,
        plot_lcap_lact,
        plot_oscillating,
    )

    at = compute_at(df)
    dd = compute_delta_d(df, window=window)
    lcap = compute_l_cap(df)
    lact = compute_l_act(df)

    pfx = f"{prefix}_" if prefix else ""
    name = prefix or "series"

    # Heuristic: oscillating-like if name contains 'oscill' or if variability is small
    oscill_hint = ("oscill" in name.lower()) or (float(at.std()) < 0.8 and float(at.abs().max()) < 6.0)

    if oscill_hint:
        plot_oscillating(
            at.index,
            at.to_numpy(),
            title=f"@(t) – {name}",
            y_label="@(t)",
            out_html=out_dir / f"{pfx}at.html",
            baseline=1.0,
        )
    else:
        plot_exponential_or_bifurcation(
            at.index,
            at.to_numpy(),
            title=f"@(t) – {name}",
            y_label="@(t)",
            out_html=out_dir / f"{pfx}at.html",
        )

    # Δd(t): keep linear (no log), smoothing + rangeslider does the job
    plot_exponential_or_bifurcation(
        dd.index,
        dd.to_numpy(),
        title=f"Δd(t) – {name}",
        y_label="Δd(t)",
        out_html=out_dir / f"{pfx}delta_d.html",
        smooth_sigma=3.0,
        raw_opacity=0.5,
    )

    plot_lcap_lact(
        lcap.index,
        lcap.to_numpy(),
        lact.to_numpy(),
        title=f"L_cap vs L_act – {name}",
        out_html=out_dir / f"{pfx}l_cap_l_act.html",
    )

    build_dashboard(
        at.index,
        at.to_numpy(),
        dd.to_numpy(),
        lcap.to_numpy(),
        lact.to_numpy(),
        title=f"Audit dashboard – {name}",
        out_html=out_dir / f"{pfx}dashboard.html",
    )


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")


def _write_md_summary(report, out_path: Path) -> None:
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
        f"Prevented exceedance (rel): {float(report.l_performance.get('prevented_exceedance_rel', 0.0)):.3f}",
        f"First activation delay (steps): {report.l_performance.get('first_activation_delay_steps')}",
        f"Risk drop around activation: {report.l_performance.get('risk_drop_around_activation')}",
        "",
        "## Limit operator performance (proactive variant)",
        f"Verdict: {report.l_performance_proactive.get('verdict')}",
        f"Prevented exceedance (rel): {float(report.l_performance_proactive.get('prevented_exceedance_rel', 0.0)):.3f}",
        f"TopK frac: {float(report.targets.get('proactive_topk_frac', 0.0)):.3f}",
        "",
        "## Targets",
        f"rule_execution_gap mean: {float(report.targets.get('rule_execution_gap_mean', float('nan'))):.3f}",
        f"rule_execution_gap target max: {float(report.targets.get('rule_execution_gap_target_max', 0.05)):.3f}",
        f"rule_execution_gap meets target: {bool(report.targets.get('rule_execution_gap_meets_target', False))}",
        f"prevented_exceedance_rel target min: {float(report.targets.get('prevented_exceedance_rel_target_min', 0.10)):.3f}",
        f"prevented_exceedance_rel meets target: {bool(report.targets.get('prevented_exceedance_meets_target', False))}",
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
    ap = argparse.ArgumentParser(description="Build audit reports + optional visualizations.")
    ap.add_argument("--dataset", type=str, help="CSV dataset with a 'date' column.")
    ap.add_argument("--name", type=str, default="", help="Dataset name for outputs.")
    ap.add_argument("--out-dir", type=str, default="_ci_out", help="Output directory.")
    ap.add_argument("--window", type=int, default=5, help="Smoothing window for Δd(t).")
    ap.add_argument("--plot", action="store_true", help="Write PNG plots with matplotlib.")
    ap.add_argument("--plotly", action="store_true", help="Write interactive HTML plots with Plotly.")
    ap.add_argument("--all-synthetic", action="store_true", help="Run on all synthetic datasets.")
    ap.add_argument("--synthetic-dir", type=str, default="data/synthetic", help="Synthetic data directory.")

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = []
    if args.all_synthetic:
        syn = Path(args.synthetic_dir)
        datasets = [
            (syn / "stable_regime.csv", "stable"),
            (syn / "oscillating_regime.csv", "oscillating"),
            (syn / "bifurcation_regime.csv", "bifurcation"),
        ]
    else:
        if not args.dataset:
            raise SystemExit("--dataset is required unless --all-synthetic is set")
        datasets = [(Path(args.dataset), args.name or Path(args.dataset).stem)]


    # Calibration discrimination summary (synthetic only): derived once from stable regime
    if args.all_synthetic:
        syn = Path(args.synthetic_dir)
        disc = discriminate_regimes(
            {
                "stable": _read_csv(syn / "stable_regime.csv"),
                "oscillating": _read_csv(syn / "oscillating_regime.csv"),
                "bifurcation": _read_csv(syn / "bifurcation_regime.csv"),
            },
            stable_name="stable",
            window=args.window,
        )
        (out_dir / "calibration_report.json").write_text(
            json.dumps(disc, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    for path, name in datasets:
        df = _read_csv(path)
        report = build_audit_report(df, dataset_name=name)
        write_audit_report(report, out_dir / f"audit_report_{name}.json")

        _write_md_summary(report, out_dir / f"audit_report_{name}.md")

        if args.plot:
            _plot_series_matplotlib(df, out_dir, window=args.window, prefix=name)
        if args.plotly:
            _plot_series_plotly(df, out_dir, window=args.window, prefix=name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
