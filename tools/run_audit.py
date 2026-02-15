
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from amplification_barometer.audit_report import build_audit_report, write_audit_report  # noqa: E402
from amplification_barometer.calibration import derive_thresholds, discriminate_regimes  # noqa: E402
from amplification_barometer.composites import compute_at, compute_delta_d  # noqa: E402
from amplification_barometer.l_operator import compute_l_act, compute_l_cap  # noqa: E402


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.set_index("date").sort_index()
    return df


def _write_md_summary(report, out_path: Path) -> None:
    s = report.summary
    m = report.maturity
    lp = report.l_performance
    out = []
    out.append(f"# Audit report: {report.dataset_name}\n")
    out.append(f"- Created UTC: {report.created_utc}")
    out.append(f"- Weights version: {report.weights_version}")
    out.append("")
    out.append("## Risk")
    out.append(f"- Risk threshold: {s.get('risk_threshold')}")
    out.append(f"- Mean Risk: {s.get('RISK', {}).get('mean')}")
    out.append("")
    out.append("## Maturity (anti-circularity)")
    out.append(f"- Label: {m.get('label')}")
    out.append(f"- L_cap_bench_score: {m.get('l_cap_bench_score')}")
    out.append(f"- L_act_mean: {m.get('l_act_mean')}")
    out.append("")
    out.append("## L performance (reactive)")
    out.append(f"- prevented_exceedance_rel: {lp.get('prevented_exceedance_rel')}")
    out.append(f"- prevented_topk_excess_rel: {lp.get('prevented_topk_excess_rel')}")
    out.append(f"- activation_delay_steps: {lp.get('activation_delay_steps')}")
    out.append(f"- e_reduction_rel: {lp.get('e_reduction_rel')}")
    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _plot_series_matplotlib(df: pd.DataFrame, out_dir: Path, *, window: int = 5, prefix: str = "") -> None:
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


def _plot_series_plotly(df: pd.DataFrame, out_dir: Path, *, window: int = 5, prefix: str = "") -> None:
    from amplification_barometer.plotly_viz import (  # noqa: WPS433
        build_dashboard,
        plot_exponential_or_bifurcation,
        plot_lcap_lact,
        plot_oscillating,
    )

    pfx = f"{prefix}_" if prefix else ""

    # Plotly helpers expect a date-like index. If the CSV has no 'date' column,
    # we synthesize a stable daily timeline for display purposes.
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        if "t" in df.columns:
            try:
                base = pd.Timestamp("2000-01-01")
                dates = base + pd.to_timedelta(pd.to_numeric(df["t"], errors="coerce").fillna(0).to_numpy(), unit="D")
                dates = pd.to_datetime(dates)
            except Exception:
                dates = pd.date_range("2000-01-01", periods=len(df), freq="D")
        else:
            dates = pd.date_range("2000-01-01", periods=len(df), freq="D")
    else:
        dates = idx

    at = compute_at(df)
    dd = compute_delta_d(df, window=window)
    lcap = compute_l_cap(df)
    lact = compute_l_act(df)

    plot_oscillating(
        dates,
        at.to_numpy(dtype=float),
        title="@ (oscillations)",
        y_label="@",
        out_html=out_dir / f"{pfx}at.html",
        baseline=float(at.median()),
    )
    plot_exponential_or_bifurcation(
        dates,
        dd.to_numpy(dtype=float),
        title="Δd (bifurcation?)",
        y_label="Δd",
        out_html=out_dir / f"{pfx}delta_d.html",
    )
    plot_lcap_lact(
        dates,
        lcap.to_numpy(dtype=float),
        lact.to_numpy(dtype=float),
        title="L_cap et L_act",
        out_html=out_dir / f"{pfx}l_cap_l_act.html",
    )

    build_dashboard(
        dates,
        at=at.to_numpy(dtype=float),
        dd=dd.to_numpy(dtype=float),
        l_cap=lcap.to_numpy(dtype=float),
        l_act=lact.to_numpy(dtype=float),
        out_html=out_dir / f"{pfx}dashboard.html",
        title=f"Dashboard {prefix or 'dataset'}",
    )
def _collect_synthetic(synth_dir: Path) -> List[Tuple[Path, str]]:
    csvs = sorted([p for p in synth_dir.glob("*.csv") if p.is_file()])
    out: List[Tuple[Path, str]] = []
    for p in csvs:
        # normalize names like stable_regime -> stable
        stem = p.stem
        name = stem.replace("_regime", "")
        out.append((p, name))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", default=[], help="Dataset CSV path (can be repeated)")
    ap.add_argument("--name", action="append", default=[], help="Name for each --csv (can be repeated)")
    ap.add_argument("--all-synthetic", action="store_true", help="Run on all CSVs found in --synthetic-dir (ignores --csv/--name)")
    ap.add_argument("--synthetic-dir", default="data/synthetic", help="Directory containing synthetic CSV datasets")
    ap.add_argument("--baseline-csv", default="", help="Stable baseline CSV for thresholds")
    ap.add_argument("--out-dir", default="_ci_out/audit", help="Output dir")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plotly", action="store_true")
    ap.add_argument("--calibrate", action="store_true", help="Run discriminate_regimes on synthetic set if present")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    synth_dir = Path(args.synthetic_dir)

    # Resolve datasets list
    datasets: List[Tuple[Path, str]] = []
    if args.all_synthetic:
        if synth_dir.exists():
            datasets = _collect_synthetic(synth_dir)
    else:
        for i, csv_path in enumerate(args.csv):
            p = Path(csv_path)
            name = args.name[i] if i < len(args.name) else p.stem
            datasets.append((p, name))

    if not datasets:
        # default synthetic set if present
        default_dir = synth_dir if synth_dir.exists() else Path("data/synthetic")
        for p in [
            default_dir / "stable_regime.csv",
            default_dir / "oscillating_regime.csv",
            default_dir / "bifurcation_regime.csv",
        ]:
            if p.exists():
                datasets.append((p, p.stem.replace("_regime", "")))

    # Baseline thresholds
    thresholds = None
    if args.baseline_csv:
        baseline_csv = Path(args.baseline_csv)
    else:
        # prefer stable_regime inside synthetic-dir when available
        candidate = synth_dir / "stable_regime.csv"
        baseline_csv = candidate if candidate.exists() else Path("data/synthetic/stable_regime.csv")

    if baseline_csv.exists():
        stable_df = _read_csv(baseline_csv)
        thresholds = derive_thresholds(stable_df, window=args.window)
        (out_dir / "baseline_thresholds.json").write_text(json.dumps(thresholds.__dict__, indent=2), encoding="utf-8")

    if args.calibrate:
        # Build synthetic set from synthetic dir if present, else from defaults
        synth: Dict[str, pd.DataFrame] = {}
        if synth_dir.exists():
            for p in _collect_synthetic(synth_dir):
                path, name = p
                synth[name] = _read_csv(path)
        else:
            for p in [
                Path("data/synthetic/stable_regime.csv"),
                Path("data/synthetic/oscillating_regime.csv"),
                Path("data/synthetic/bifurcation_regime.csv"),
            ]:
                if p.exists():
                    synth[p.stem.replace("_regime", "")] = _read_csv(p)

        if "stable" in synth and len(synth) >= 2:
            disc = discriminate_regimes(synth, stable_name="stable", window=args.window)
            (out_dir / "calibration_report.json").write_text(json.dumps(disc, indent=2), encoding="utf-8")

    for path, name in datasets:
        df = _read_csv(path)
        rep = build_audit_report(df, dataset_name=name, window=args.window, thresholds=thresholds)
        write_audit_report(rep, out_dir / f"audit_report_{name}.json")
        _write_md_summary(rep, out_dir / f"audit_report_{name}.md")
        if args.plot:
            _plot_series_matplotlib(df, out_dir, window=args.window, prefix=name)
        if args.plotly:
            _plot_series_plotly(df, out_dir, window=args.window, prefix=name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
