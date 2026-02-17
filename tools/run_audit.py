#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from amplification_barometer.audit_report import build_audit_report, write_audit_report  # noqa: E402
from amplification_barometer.calibration import Thresholds, derive_thresholds  # noqa: E402
from amplification_barometer.composites import compute_at, compute_delta_d  # noqa: E402
from amplification_barometer.l_operator import compute_l_cap, compute_l_act  # noqa: E402
from amplification_barometer.html_report import render_audit_html  # noqa: E402


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # common time columns
    for col in ("date", "datetime", "timestamp", "time", "open_time", "close_time"):
        if col in df.columns:
            if col in ("timestamp", "time", "open_time", "close_time"):
                # epoch ms or s
                s = df[col]
                if pd.api.types.is_numeric_dtype(s):
                    # heuristic: ms if large
                    unit = "ms" if float(s.dropna().iloc[0]) > 1e11 else "s"
                    dt = pd.to_datetime(s, unit=unit, utc=True)
                else:
                    dt = pd.to_datetime(s, utc=True, errors="coerce")
            else:
                dt = pd.to_datetime(df[col], utc=True, errors="coerce")
            df = df.drop(columns=[col])
            df.insert(0, "date", dt)
            df = df.dropna(subset=["date"]).set_index("date")
            return df.sort_index()

    # fallback: index is a datetime-like first column
    first = df.columns[0]
    try:
        dt = pd.to_datetime(df[first], utc=True, errors="coerce")
        if dt.notna().any():
            df = df.drop(columns=[first])
            df.insert(0, "date", dt)
            return df.dropna(subset=["date"]).set_index("date").sort_index()
    except Exception:
        pass

    raise SystemExit(f"Cannot find a date/timestamp column in {path}")


def _plot_series_matplotlib(df: pd.DataFrame, out_dir: Path, *, window: int = 5, prefix: str = "") -> None:
    at = compute_at(df)
    dd = compute_delta_d(df, window=window)

    pfx = f"{prefix}_" if prefix else ""

    fig1 = plt.figure(figsize=(11, 5.5))
    plt.plot(at.index, at.to_numpy(), linewidth=2.0)
    plt.title("@(t)", fontsize=14, pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("@(t)", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle="--")
    fig1.tight_layout()
    fig1.savefig(out_dir / f"{pfx}at.png", dpi=140)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(11, 5.5))
    plt.plot(dd.index, dd.to_numpy(), linewidth=2.0)
    plt.title("Δd(t)", fontsize=14, pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Δd(t)", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle="--")
    fig2.tight_layout()
    fig2.savefig(out_dir / f"{pfx}delta_d.png", dpi=140)
    plt.close(fig2)


def _load_stable_thresholds(*, window: int) -> Optional[Thresholds]:
    repo_root = Path(__file__).resolve().parents[1]
    stable_path = repo_root / "data" / "synthetic" / "stable_regime.csv"
    if not stable_path.exists():
        return None
    stable_df = _read_csv(stable_path)
    return derive_thresholds(stable_df, window=window)


def _write_md_summary(report_payload: dict, out_path: Path) -> None:
    ds = report_payload.get("dataset_name", "dataset")
    created = report_payload.get("created_utc", "")
    maturity = (report_payload.get("maturity") or {}).get("label", "unknown")
    risk_mean = ((report_payload.get("summary") or {}).get("RISK") or {}).get("mean", "nan")
    baseline_used = (report_payload.get("summary") or {}).get("baseline_used", False)
    anti = (report_payload.get("anti_gaming") or {}).get("o_bias", {})
    anti_flag = bool(anti.get("flag", False))

    lines = [
        f"# Audit report: {ds}",
        "",
        f"Date (UTC): {created}",
        "",
        f"Maturity: **{maturity}**",
        f"RISK mean: {risk_mean}",
        f"Baseline used: {baseline_used}",
        f"Anti-gaming o_bias: {'RED FLAG' if anti_flag else 'OK'}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _run_one(csv_path: Path, *, name: str, out_dir: Path, window: int, do_png: bool, do_plotly: bool, do_html_report: bool) -> None:
    df = _read_csv(csv_path)
    thresholds = _load_stable_thresholds(window=window)

    rep = build_audit_report(df, dataset_name=name, delta_d_window=window, thresholds=thresholds)

    # JSON report
    json_path = out_dir / f"audit_report_{name}.json"
    write_audit_report(rep, json_path)

    # MD summary
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    _write_md_summary(payload, out_dir / f"summary_{name}.md")

    # Plots
    if do_png:
        _plot_series_matplotlib(df, out_dir, window=window, prefix=name)

    if do_plotly:
        # Keep existing per-chart HTML for quick browsing
        from amplification_barometer.plotly_viz import plot_dashboard, plot_exponential_or_bifurcation, plot_lcap_lact, plot_oscillating  # noqa

        at = compute_at(df)
        dd = compute_delta_d(df, window=window)
        oscill_hint = ("oscill" in name.lower()) or (float(at.std()) < 0.8 and float(at.abs().max()) < 6.0)

        if oscill_hint:
            plot_oscillating(at.index, at.to_numpy(), title=f"@(t) - {name}", y_label="@(t)", out_html=out_dir / f"{name}_at.html", baseline=1.0)
        else:
            plot_exponential_or_bifurcation(at.index, at.to_numpy(), title=f"@(t) - {name}", y_label="@(t)", out_html=out_dir / f"{name}_at.html")
        plot_exponential_or_bifurcation(dd.index, dd.to_numpy(), title=f"Δd(t) - {name}", y_label="Δd(t)", out_html=out_dir / f"{name}_delta_d.html")
        l_cap = compute_l_cap(df).to_numpy()
        l_act = compute_l_act(df).to_numpy()
        plot_lcap_lact(df.index, l_cap, l_act, title=f"L_cap / L_act - {name}", out_html=out_dir / f"{name}_lcap_lact.html")
        plot_dashboard(
            df.index,
            at.to_numpy(),
            dd.to_numpy(),
            l_cap,
            l_act,
            title=f"Audit dashboard: {name}",
            out_html=out_dir / f"{name}_dashboard.html",
        )

    # Consolidated HTML (self-contained)
    if do_html_report:
        render_audit_html(
            report_dict=payload,
            df=df,
            out_html=out_dir / f"audit_report_{name}.html",
            thresholds=thresholds,
            delta_d_window=window,
            include_plotlyjs="inline",
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Build audit reports + optional visualizations.")
    ap.add_argument("--dataset", type=str, help="CSV dataset with a date/timestamp column.")
    ap.add_argument("--name", type=str, default="", help="Dataset name for outputs.")
    ap.add_argument("--out-dir", type=str, default="_ci_out", help="Output directory.")
    ap.add_argument("--window", type=int, default=5, help="Smoothing window for Δd(t).")
    ap.add_argument("--plot", action="store_true", help="Write PNG plots with matplotlib.")
    ap.add_argument("--plotly", action="store_true", help="Write interactive HTML plots with Plotly.")
    ap.add_argument("--html-report", action="store_true", help="Enable the consolidated self-contained HTML report (default on).")
    ap.add_argument("--no-html-report", action="store_true", help="Disable the consolidated self-contained HTML report.")
    ap.add_argument("--all-synthetic", action="store_true", help="Run on all synthetic datasets.")
    ap.add_argument("--synthetic-dir", type=str, default="data/synthetic", help="Synthetic data directory.")

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    do_html_report = not bool(args.no_html_report)

    if args.all_synthetic:
        sdir = Path(args.synthetic_dir)
        paths = sorted(sdir.glob("*.csv"))
        for p in paths:
            name = p.stem
            _run_one(p, name=name, out_dir=out_dir, window=int(args.window), do_png=bool(args.plot), do_plotly=bool(args.plotly), do_html_report=do_html_report)
        return 0

    if not args.dataset:
        raise SystemExit("Provide --dataset path or use --all-synthetic")

    csv_path = Path(args.dataset)
    if not csv_path.exists():
        raise SystemExit(f"Missing dataset: {csv_path}")

    name = args.name.strip() or csv_path.stem
    _run_one(csv_path, name=name, out_dir=out_dir, window=int(args.window), do_png=bool(args.plot), do_plotly=bool(args.plotly), do_html_report=do_html_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
