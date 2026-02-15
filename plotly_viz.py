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

import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass(frozen=True)
class PlotlyOutputs:
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
    allow_log: bool = True,
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
    baseline: float = 1.0,
    y_range: Optional[Tuple[float, float]] = None,
    raw_color: str = "lightgray",
    smooth_color: str = "#d62728",
    smooth_sigma: float = 4.0,
    raw_opacity: float = 0.7,
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
            line=dict(color=raw_color, width=1.0),
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
            line=dict(color=smooth_color, width=2.5),
            name=f"smoothed (σ={smooth_sigma:g})",
        )
    )

    fig.add_hline(
        y=float(baseline),
        line_dash="dash",
        line_color="gray",
        annotation_text=f"baseline ~{baseline:g}",
        annotation_position="top right",
    )

    if y_range is None:
        # Zoom automatically based on robust spread
        y_fin = y[np.isfinite(y)]
        if y_fin.size:
            med = float(np.median(y_fin))
            mad = float(np.median(np.abs(y_fin - med))) or 1e-9
            spread = 6.0 * mad
            y_range = (med - spread, med + spread)

    fig.update_layout(
        title_text=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        yaxis_range=list(y_range) if y_range is not None else None,
        xaxis_rangeslider_visible=True,
        template="plotly_white",
        height=height,
        width=width,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
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
    cap_color: str = "royalblue",
    act_color: str = "darkorange",
    width: int = 1100,
    height: int = 550,
) -> Path:
    dates = pd.to_datetime(pd.Index(dates))
    l_cap = np.asarray(l_cap, dtype=float)
    l_act = np.asarray(l_act, dtype=float)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=dates, y=l_cap, mode="lines", line=dict(color=cap_color, width=2.2), name="L_cap"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=l_act, mode="lines", line=dict(color=act_color, width=2.2), name="L_act"),
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

    # NOTE: plotly uses 'title_font' and 'tickfont' objects (not titlefont_color/tickfont_color)
    fig.update_yaxes(
        title_text="L_cap",
        title_font=dict(color=cap_color),
        tickfont=dict(color=cap_color),
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="L_act",
        title_font=dict(color=act_color),
        tickfont=dict(color=act_color),
        secondary_y=True,
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")
    return out_html


def plot_dashboard(
    dates: pd.Series | pd.Index,
    at: np.ndarray,
    dd: np.ndarray,
    l_cap: np.ndarray,
    l_act: np.ndarray,
    *,
    title: str,
    out_html: Path,
    width: int = 1100,
    height: int = 900,
) -> Path:
    dates = pd.to_datetime(pd.Index(dates))
    at = np.asarray(at, dtype=float)
    dd = np.asarray(dd, dtype=float)
    l_cap = np.asarray(l_cap, dtype=float)
    l_act = np.asarray(l_act, dtype=float)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        specs=[[{}], [{}], [{"secondary_y": True}]],
        row_heights=[0.34, 0.33, 0.33],
    )

    # @(t)
    at_sm = _safe_smooth(at, 3.0)
    use_log = _maybe_log_y(at_sm)
    fig.add_trace(
        go.Scatter(x=dates, y=at, mode="lines", line=dict(color="lightgray", width=1), name="@(t) raw", opacity=0.6),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=at_sm, mode="lines", line=dict(color="#1f77b4", width=2.4), name="@(t) smoothed"),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="@(t)" + (" (log)" if use_log else ""), type="log" if use_log else "linear", row=1, col=1)

    # Δd(t)
    dd_sm = _safe_smooth(dd, 3.0)
    fig.add_trace(
        go.Scatter(x=dates, y=dd, mode="lines", line=dict(color="lightgray", width=1), name="Δd raw", opacity=0.6),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=dd_sm, mode="lines", line=dict(color="#1f77b4", width=2.4), name="Δd smoothed"),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Δd(t)", row=2, col=1)

    # L_cap / L_act
    fig.add_trace(
        go.Scatter(x=dates, y=l_cap, mode="lines", line=dict(color="royalblue", width=2.2), name="L_cap"),
        row=3,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=l_act, mode="lines", line=dict(color="darkorange", width=2.2), name="L_act"),
        row=3,
        col=1,
        secondary_y=True,
    )
    fig.update_yaxes(
        title_text="L_cap",
        title_font=dict(color="royalblue"),
        tickfont=dict(color="royalblue"),
        row=3,
        col=1,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="L_act",
        title_font=dict(color="darkorange"),
        tickfont=dict(color="darkorange"),
        row=3,
        col=1,
        secondary_y=True,
    )

    fig.update_layout(
        title_text=title,
        template="plotly_white",
        width=width,
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        xaxis_rangeslider_visible=True,
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")
    return out_html


def build_plotly_outputs(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    at_col: str = "at",
    dd_col: str = "delta_d",
    lcap_col: str = "l_cap",
    lact_col: str = "l_act",
    out_dir: Path,
    prefix: str,
    title_prefix: Optional[str] = None,
) -> PlotlyOutputs:
    title_prefix = title_prefix or prefix

    dates = pd.to_datetime(df[date_col])

    at_html = out_dir / f"{prefix}_at.html"
    dd_html = out_dir / f"{prefix}_delta_d.html"
    l_html = out_dir / f"{prefix}_l_cap_l_act.html"
    dash_html = out_dir / f"{prefix}_dashboard.html"

    plot_exponential_or_bifurcation(
        dates,
        df[at_col].to_numpy(dtype=float),
        title=f"{title_prefix} @(t)",
        y_label="@(t)",
        out_html=at_html,
        allow_log=True,
    )

    # Δd(t): keep linear scale, use oscillating style (raw + smoothed) for readability
    plot_oscillating(
        dates,
        df[dd_col].to_numpy(dtype=float),
        title=f"{title_prefix} Δd(t)",
        y_label="Δd(t)",
        out_html=dd_html,
    )

    plot_lcap_lact(
        dates,
        df[lcap_col].to_numpy(dtype=float),
        df[lact_col].to_numpy(dtype=float),
        title=f"{title_prefix} L_cap vs L_act",
        out_html=l_html,
    )

    plot_dashboard(
        dates,
        df[at_col].to_numpy(dtype=float),
        df[dd_col].to_numpy(dtype=float),
        df[lcap_col].to_numpy(dtype=float),
        df[lact_col].to_numpy(dtype=float),
        title=f"{title_prefix} dashboard",
        out_html=dash_html,
    )

    return PlotlyOutputs(at_html=at_html, dd_html=dd_html, l_html=l_html, dashboard_html=dash_html)


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# Some callers historically used "build_*" names. Keep these tiny wrappers so
# CI/demo scripts don't break when the public API evolves.
# ---------------------------------------------------------------------------

def build_exponential_or_bifurcation(*args, **kwargs):
    return plot_exponential_or_bifurcation(*args, **kwargs)

def build_oscillating(*args, **kwargs):
    return plot_oscillating(*args, **kwargs)

def build_lcap_lact(*args, **kwargs):
    return plot_lcap_lact(*args, **kwargs)

def build_dashboard(*args, **kwargs):
    return plot_dashboard(*args, **kwargs)
