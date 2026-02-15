#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from amplification_barometer.audit_report import build_audit_report, write_audit_report  # noqa: E402
from amplification_barometer.composites import compute_at, compute_delta_d  # noqa: E402
from amplification_barometer.l_operator import compute_l_act, compute_l_cap  # noqa: E402


def _plot_series(df: pd.DataFrame, out_dir: Path, *, window: int = 5) -> None:
    at = compute_at(df)
    dd = compute_delta_d(df, window=window)
    lcap = compute_l_cap(df)
    lact = compute_l_act(df)

    fig1 = plt.figure()
    plt.plot(at.index, at.to_numpy())
    plt.title("@(t)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig1.savefig(out_dir / "at.png", dpi=150)
    plt.close(fig1)

    fig2 = plt.figure()
    plt.plot(dd.index, dd.to_numpy())
    plt.title("Δd(t)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig2.savefig(out_dir / "delta_d.png", dpi=150)
    plt.close(fig2)

    fig3 = plt.figure()
    plt.plot(lcap.index, lcap.to_numpy(), label="L_cap")
    plt.plot(lact.index, lact.to_numpy(), label="L_act")
    plt.title("L_cap / L_act")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig3.savefig(out_dir / "l_cap_l_act.png", dpi=150)
    plt.close(fig3)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Amplification Barometer audit report (demo).")
    ap.add_argument("--dataset", type=str, required=True, help="Path to CSV dataset")
    ap.add_argument("--name", type=str, default="dataset", help="Dataset name for the report")
    ap.add_argument("--out-dir", type=str, default="_ci_out", help="Output directory")
    ap.add_argument("--plot", action="store_true", help="Generate simple plots (PNG)")
    ap.add_argument("--window", type=int, default=5, help="Window for Δd(t)")
    args = ap.parse_args()

    path = Path(args.dataset)
    if not path.exists():
        raise SystemExit(f"Dataset not found: {path}")

    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = build_audit_report(df, dataset_name=args.name, delta_d_window=args.window)
    write_audit_report(report, out_dir / "audit_report.json")

    summary_md = out_dir / "audit_report.md"
    summary_md.write_text(
        "\n".join(
            [
                f"# Audit report: {args.name}",
                "",
                f"Version: {report.version}",
                f"Weights version: {report.weights_version}",
                "",
                "## Stability",
                f"Stable flag: {report.stability.get('stable_flag')}",
                f"Spearman worst (risk): {report.stability.get('spearman_worst_risk'):.3f}",
                f"TopK Jaccard worst (risk): {report.stability.get('topk_jaccard_worst_risk'):.3f}",
                "",
                "## Maturity",
                f"Label: {report.maturity.get('label')}",
                f"L_cap (raw mean): {report.maturity.get('cap_score'):.3f}",
                f"L_act (raw mean): {report.maturity.get('act_score'):.3f}",
                f"L_act drop under stress: {report.maturity.get('act_drop_under_stress'):.3f}",
                "",
                "## Stress suite",
                *[
                    f"- {k}: {v.get('status')} (degradation={v.get('degradation'):.3f})"
                    for k, v in report.stress_suite.items()
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )

    if args.plot:
        _plot_series(df, out_dir, window=args.window)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
