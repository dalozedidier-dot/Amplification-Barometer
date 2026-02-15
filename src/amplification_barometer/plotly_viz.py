"""Plotly visualizations for audit-ready time-series.

This module is intentionally lightweight and purely optional: it produces
shareable HTML files (no sensitive data) to support empirical demonstrations.

Patterns implemented:
- Exponential/bifurcation: optional log scale + rangeslider + smoothing overlay
- Oscillating: raw + smoothed + baseline line + y-range zoom
- L_cap/L_act: secondary_y twin axis (interactive)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Plotly is required for amplification_barometer.plotly_viz. "
        "Install it with: pip install plotly"
    ) from e


@dataclass(frozen=True)
class PlotlyExport:
    at_html: Path
    dd_html: Path
    l_html: Path
    dashboard_html: Optional[Path] = None


def _maybe_log_y(y: np.ndarray, *, ratio_thr: float = 50.0) -> bool:
    """Use log scale only when it makes sense and is safe."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return False
    if np.any(y <= 0):
        return False
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if y_min <= 0:
        return False
    return (y_max / y_min) >= ratio_thr


def _safe_smooth(y: np.ndarray, sigma: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size < 5:
        return y
    # gaussian_filter1d expects finite values
    y2 = y.copy()
    m = np.isfinite(y2)
    if not m.all():
        # simple forward fill then back fill
        s = pd.Series(y2)
        y2 = s.ffill().bfill().to_numpy(dtype=float)
    return gaussian_filter1d(y2, sigma=float(sigma))


def plot_exponential_or_bifurcation(
    dates: pd.Series | pd.Index,
    y: np.ndarray,
    *,
    title: str,
    y_label: str,
    out_html: Path,
    color: str = "#1f77b4",
    smooth_sigma: float = 3.0,
    raw_opacity: float = 0.4,
    width: int = 1000,
    height: int = 500,
) -> Path:
    dates = pd.to_datetime(pd.Index(dates))
    y = np.asarray(y, dtype=float)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=y,
            mode="lines",
            line=dict(color=color, width=1.5),
            name="raw",
            opacity=raw_opacity,
        )
    )

    y_sm = _safe_smooth(y, smooth_sigma)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=y_sm,
            mode="lines",
            line=dict(color=color, width=2.8),
            name=f"smoothed (σ={smooth_sigma:g})",
        )
    )

    use_log = bool(allow_log) and _maybe_log_y(y_sm)
    fig.update_layout(
        title_text=title + (" (log scale)" if use_log else ""),
        xaxis_title="Date",
        yaxis_title=y_label,
        yaxis_type="log" if use_log else "linear",
        xaxis_rangeslider_visible=True,
        template="plotly_white",
        height=height,
        width=width,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")
    return out_html


def plot_oscillating(
    dates: pd.Series | pd.Index,
    y: np.ndarray,
    *,
    title: str,
    y_label: str,
    out_html: Path,
    baseline: Optional[float] = 1.0,
    smooth_sigma: float = 4.0,
    y_range: Optional[Tuple[float, float]] = None,
    width: int = 1000,
    height: int = 450,
) -> Path:
    dates = pd.to_datetime(pd.Index(dates))
    y = np.asarray(y, dtype=float)
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=y,
            mode="lines",
            line=dict(color="lightgray", width=1),
            name="raw",
            opacity=0.7,
        )
    )

    y_sm = _safe_smooth(y, smooth_sigma)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=y_sm,
            mode="lines",
            line=dict(color="#d62728", width=2.5),
            name=f"smoothed (σ={smooth_sigma:g})",
        )
    )

    if y_range is None:
        # zoom heuristics: baseline ± 3*std around smoothed, with safe fallback
        y0 = float(np.nanmedian(y_sm))
        s = float(np.nanstd(y_sm))
        if np.isfinite(s) and s > 0:
            y_range = (y0 - 3.0 * s, y0 + 3.0 * s)

    if baseline is not None and np.isfinite(baseline):
        fig.add_hline(
            y=float(baseline),
            line_dash="dash",
            line_color="gray",
            annotation_text=f"baseline ≈ {baseline:g}",
            annotation_position="top right",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        yaxis_range=list(y_range) if y_range is not None else None,
        xaxis_rangeslider_visible=True,
        template="plotly_white",
        height=height,
        width=width,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0, xanchor="left"),
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")
    return out_html


def plot_lcap_lact(
    dates: pd.Series | pd.Index,
    l_cap: np.ndarray,
    l_act: np.ndarray,
    *,
    title: str,
    out_html: Path,
    width: int = 1100,
    height: int = 550,
) -> Path:
    dates = pd.to_datetime(pd.Index(dates))
    l_cap = np.asarray(l_cap, dtype=float)
    l_act = np.asarray(l_act, dtype=float)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=l_cap,
            mode="lines",
            line=dict(color="royalblue", width=2.2),
            name="L_cap",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=l_act,
            mode="lines",
            line=dict(color="darkorange", width=2.2),
            name="L_act",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title_text=title,
        xaxis_title="Date",
        template="plotly_white",
        height=height,
        width=width,
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        hovermode="x unified",
        xaxis_rangeslider_visible=True,
    )

    fig.update_yaxes(
        title_text="L_cap",
        titlefont_color="royalblue",
        tickfont_color="royalblue",
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="L_act",
        titlefont_color="darkorange",
        tickfont_color="darkorange",
        secondary_y=True,
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")
    return out_html


def build_dashboard(
    dates: pd.Series | pd.Index,
    at: np.ndarray,
    dd: np.ndarray,
    l_cap: np.ndarray,
    l_act: np.ndarray,
    *,
    title: str,
    out_html: Path,
    at_log_ratio_thr: float = 50.0,
) -> Path:
    dates = pd.to_datetime(pd.Index(dates))
    at = np.asarray(at, dtype=float)
    dd = np.asarray(dd, dtype=float)
    l_cap = np.asarray(l_cap, dtype=float)
    l_act = np.asarray(l_act, dtype=float)

    specs = [
        [{"secondary_y": False}],
        [{"secondary_y": False}],
        [{"secondary_y": True}],
    ]
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        specs=specs,
        subplot_titles=("@(t)", "Δd(t)", "L_cap vs L_act"),
    )

    # @(t)
    at_sm = _safe_smooth(at, 3.0)
    fig.add_trace(
        go.Scatter(
            x=dates, y=at, mode="lines", line=dict(color="lightgray", width=1), name="@(t) raw", opacity=0.6
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=dates, y=at_sm, mode="lines", line=dict(color="#1f77b4", width=2.6), name="@(t) smoothed"
        ),
        row=1, col=1
    )

    use_log = _maybe_log_y(at_sm, ratio_thr=at_log_ratio_thr)
    fig.update_yaxes(type="log" if use_log else "linear", row=1, col=1)

    # Δd(t)
    dd_sm = _safe_smooth(dd, 3.0)
    fig.add_trace(
        go.Scatter(
            x=dates, y=dd, mode="lines", line=dict(color="lightgray", width=1), name="Δd raw", opacity=0.6
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=dates, y=dd_sm, mode="lines", line=dict(color="#1f77b4", width=2.2), name="Δd smoothed"
        ),
        row=2, col=1
    )

    # L_cap/L_act with secondary y
    fig.add_trace(
        go.Scatter(x=dates, y=l_cap, mode="lines", line=dict(color="royalblue", width=2.0), name="L_cap"),
        row=3, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=dates, y=l_act, mode="lines", line=dict(color="darkorange", width=2.0), name="L_act"),
        row=3, col=1, secondary_y=True
    )
    fig.update_yaxes(
        title_text="L_cap",
        titlefont_color="royalblue",
        tickfont_color="royalblue",
        row=3, col=1, secondary_y=False
    )
    fig.update_yaxes(
        title_text="L_act",
        titlefont_color="darkorange",
        tickfont_color="darkorange",
        row=3, col=1, secondary_y=True
    )

    fig.update_layout(
        title_text=title,
        template="plotly_white",
        height=900,
        width=1100,
        hovermode="x unified",
        xaxis_rangeslider_visible=True,
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")
    return out_html


def export_audit_viz(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    prefix: str = "",
    oscillating_hint: bool = False,
) -> PlotlyExport:
    """Produce the three core interactive HTML files + optional dashboard."""
    pfx = f"{prefix}_" if prefix else ""
    dates = df.index

    # Expect columns if already computed; otherwise caller computes series and passes them.
    at = df["at"].to_numpy(dtype=float) if "at" in df.columns else df["@(t)"].to_numpy(dtype=float)
    dd = df["delta_d"].to_numpy(dtype=float) if "delta_d" in df.columns else df["Δd(t)"].to_numpy(dtype=float)
    l_cap = df["l_cap"].to_numpy(dtype=float)
    l_act = df["l_act"].to_numpy(dtype=float)

    at_html = out_dir / f"{pfx}at.html"
    dd_html = out_dir / f"{pfx}delta_d.html"
    l_html = out_dir / f"{pfx}l_cap_l_act.html"
    dash_html = out_dir / f"{pfx}dashboard.html"

    if oscillating_hint:
        plot_oscillating(
            dates,
            at,
            title=f"@(t) – Oscillating view {prefix}".strip(),
            y_label="@(t)",
            out_html=at_html,
            baseline=1.0,
        )
    else:
        plot_exponential_or_bifurcation(
            dates,
            at,
            title=f"@(t) – {prefix}".strip(),
            y_label="@(t)",
            out_html=at_html,
        )

    # Δd: keep linear with smoothing; users can zoom with rangeslider
    plot_exponential_or_bifurcation(
        dates,
        dd,
        title=f"Δd(t) – {prefix}".strip(),
        y_label="Δd(t)",
        out_html=dd_html,
        smooth_sigma=3.0,
        raw_opacity=0.5,
    )

    plot_lcap_lact(
        dates,
        l_cap,
        l_act,
        title=f"L_cap vs L_act – {prefix}".strip(),
        out_html=l_html,
    )

    build_dashboard(
        dates,
        at,
        dd,
        l_cap,
        l_act,
        title=f"Audit dashboard – {prefix}".strip(),
        out_html=dash_html,
    )

    return PlotlyExport(at_html=at_html, dd_html=dd_html, l_html=l_html, dashboard_html=dash_html)
