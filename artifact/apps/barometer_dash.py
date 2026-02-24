from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

import dash
from dash import dcc, html, Input, Output, dash_table

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from amplification_barometer.alignment_audit import run_alignment_audit
from amplification_barometer.theory_map import load_proxy_specs


def _safe_smooth(y: np.ndarray, sigma: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size < 5:
        return y
    y2 = y.copy()
    m = np.isfinite(y2)
    if not m.all():
        s = pd.Series(y2)
        y2 = s.ffill().bfill().to_numpy(dtype=float)
    return gaussian_filter1d(y2, sigma=float(sigma))


def build_fig_timeseries(df: pd.DataFrame, date_col: Optional[str], y_col: str, title: str, y_label: str) -> go.Figure:
    x = df[date_col] if (date_col and date_col in df.columns) else np.arange(len(df))
    y = df[y_col].to_numpy(dtype=float)
    y_sm = _safe_smooth(y, sigma=3.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="raw", opacity=0.35, line=dict(color="lightgray", width=1)))
    fig.add_trace(go.Scatter(x=x, y=y_sm, mode="lines", name="smoothed (σ=3)", line=dict(color="#1f77b4", width=2.6)))
    fig.update_layout(
        title=title,
        xaxis_title="Date" if (date_col and date_col in df.columns) else "Index",
        yaxis_title=y_label,
        template="plotly_white",
        hovermode="x unified",
        height=420,
        width=1050,
        xaxis_rangeslider_visible=True,
    )
    return fig


def build_fig_lcap_lact(df: pd.DataFrame, date_col: Optional[str]) -> go.Figure:
    x = df[date_col] if (date_col and date_col in df.columns) else np.arange(len(df))
    l_cap = df["l_cap"].to_numpy(dtype=float) if "l_cap" in df.columns else None
    l_act = df["l_act"].to_numpy(dtype=float) if "l_act" in df.columns else None

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if l_cap is None or l_act is None:
        fig.add_annotation(text="l_cap / l_act not provided in dataset", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_white", height=450, width=1050)
        return fig

    fig.add_trace(go.Scatter(x=x, y=l_cap, mode="lines", name="L_cap", line=dict(color="royalblue", width=2.2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=l_act, mode="lines", name="L_act", line=dict(color="darkorange", width=2.2)), secondary_y=True)

    fig.update_layout(
        title="L_cap vs L_act",
        template="plotly_white",
        height=520,
        width=1050,
        hovermode="x unified",
        xaxis_rangeslider_visible=True,
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
    )
    fig.update_yaxes(title_text="L_cap", title_font=dict(color="royalblue"), tickfont=dict(color="royalblue"), secondary_y=False)
    fig.update_yaxes(title_text="L_act", title_font=dict(color="darkorange"), tickfont=dict(color="darkorange"), secondary_y=True)
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(description="Interactive dashboard for the amplification barometer (demo).")
    ap.add_argument("--dataset", required=True, help="CSV file with proxies.")
    ap.add_argument("--proxies-yaml", default="docs/proxies.yaml", help="Proxy spec yaml.")
    ap.add_argument("--date-col", default="date", help="Optional date column name.")
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args()

    df = pd.read_csv(args.dataset)
    date_col = args.date_col if args.date_col in df.columns else None
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Compute alignment report and derived series using alignment audit internals
    specs = load_proxy_specs(args.proxies_yaml)
    # We recompute levels and series here for plotting convenience
    from amplification_barometer.alignment_audit import compute_levels_from_specs, compute_at_delta

    levels = compute_levels_from_specs(df, specs)
    sig = compute_at_delta(levels, smooth_win=7)
    df_plot = df.copy()
    df_plot["at"] = sig["at"]
    df_plot["delta_d"] = sig["delta_d"]

    report = run_alignment_audit(df, proxies_yaml=args.proxies_yaml)

    # Build a small metrics table
    dims = report.get("verdict", {}).get("dimensions", {})
    summary = report.get("summary", {}) or {}
    rows = []
    for k, v in summary.items():
        rows.append({"Metric": k, "Value": v, "Target": ""})
    for k, v in dims.items():
        rows.append({"Metric": f"verdict.{k}", "Value": v, "Target": ""})
    metrics_df = pd.DataFrame(rows)

    app = dash.Dash(__name__)
    app.title = "Amplification barometer"

    app.layout = html.Div(
        [
            html.H1("Amplification barometer dashboard", style={"textAlign": "center"}),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Alignment metrics"),
                            dash_table.DataTable(
                                data=metrics_df.to_dict("records"),
                                columns=[{"name": c, "id": c} for c in metrics_df.columns],
                                style_cell={"textAlign": "left", "minWidth": "130px", "whiteSpace": "normal"},
                                page_size=16,
                            ),
                        ],
                        style={"width": "32%", "display": "inline-block", "verticalAlign": "top"},
                    ),
                    html.Div(
                        [
                            dcc.Graph(
                                id="fig-at",
                                figure=build_fig_timeseries(
                                    df_plot,
                                    date_col,
                                    "at",
                                    title="@(t) with smoothing and rangeslider",
                                    y_label="@(t)",
                                ),
                            ),
                            dcc.Graph(
                                id="fig-dd",
                                figure=build_fig_timeseries(
                                    df_plot,
                                    date_col,
                                    "delta_d",
                                    title="Δd(t) with smoothing and rangeslider",
                                    y_label="Δd(t)",
                                ),
                            ),
                            dcc.Graph(id="fig-l", figure=build_fig_lcap_lact(df_plot, date_col)),
                        ],
                        style={"width": "66%", "display": "inline-block", "paddingLeft": "1%"},
                    ),
                ]
            ),
        ],
        style={"margin": "16px", "fontFamily": "Arial"},
    )

    app.run_server(debug=False, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
