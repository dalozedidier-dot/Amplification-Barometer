from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from .calibration import Thresholds, risk_signature
from .composites import compute_at, compute_delta_d, compute_o_level
from .l_operator import compute_l_act, compute_l_cap


def _fig_line(x, y, *, title: str, y_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=y_label, height=320, margin=dict(l=40, r=20, t=50, b=35))
    return fig


def render_audit_html(
    *,
    report_dict: Dict[str, Any],
    df: pd.DataFrame,
    out_html: Path,
    thresholds: Optional[Thresholds] = None,
    delta_d_window: int = 5,
    include_plotlyjs: str = "inline",
) -> None:
    """Render a single self-contained HTML report (no server required).

    - Header + executive summary
    - Key scores (RISK, maturity, anti-gaming)
    - Styled tables
    - Embedded Plotly figures (inline JS by default)
    - Conclusions & recommendations
    - Footer (version/date)
    """
    out_html.parent.mkdir(parents=True, exist_ok=True)

    # Series
    at = compute_at(df)
    dd = compute_delta_d(df, window=delta_d_window)
    lcap = compute_l_cap(df)
    lact = compute_l_act(df)

    if thresholds is not None:
        risk = risk_signature(df, thresholds=thresholds, window=delta_d_window)
        risk_thr = float(thresholds.risk_thr)
        baseline_used = True
    else:
        # Fallback: per-dataset robust z
        import numpy as np
        from .composites import robust_zscore

        risk = pd.Series(robust_zscore(at.to_numpy(dtype=float)) + robust_zscore(dd.to_numpy(dtype=float)), index=df.index, name="RISK")
        risk_thr = float(pd.Series(risk).quantile(0.95))
        baseline_used = False

    # Tables
    summary = report_dict.get("summary", {}) or {}
    maturity = report_dict.get("maturity", {}) or {}
    anti_gaming = report_dict.get("anti_gaming", {}) or {}

    # Compact executive section
    maturity_label = maturity.get("label", "unknown")
    risk_mean = float(summary.get("RISK", {}).get("mean", pd.Series(risk).mean()))
    risk_std = float(summary.get("RISK", {}).get("std", pd.Series(risk).std()))
    anti_flag = bool((anti_gaming.get("o_bias") or {}).get("flag", False))

    df_scores = pd.DataFrame(
        [
            ["RISK mean", risk_mean],
            ["RISK std", risk_std],
            ["RISK threshold", risk_thr],
            ["Maturity label", maturity_label],
            ["Baseline used", baseline_used],
            ["Anti-gaming (o_bias)", "RED FLAG" if anti_flag else "OK"],
        ],
        columns=["Metric", "Value"],
    )

    df_targets = pd.DataFrame.from_dict(report_dict.get("targets", {}) or {}, orient="index", columns=["value"]).reset_index().rename(columns={"index": "target"})

    # Figures
    figs = [
        _fig_line(at.index, at.to_numpy(dtype=float), title="@(t)", y_label="AT"),
        _fig_line(dd.index, dd.to_numpy(dtype=float), title="Δd(t)", y_label="DELTA_D"),
        _fig_line(risk.index, pd.Series(risk).to_numpy(dtype=float), title="RISK(t) (baseline-normalized)" if baseline_used else "RISK(t)", y_label="RISK"),
        _fig_line(lcap.index, lcap.to_numpy(dtype=float), title="L_cap (z)", y_label="L_cap"),
        _fig_line(lact.index, lact.to_numpy(dtype=float), title="L_act (z)", y_label="L_act"),
    ]
    fig_html_blocks = "\n".join([pio.to_html(fig, include_plotlyjs=include_plotlyjs if i == 0 else False, full_html=False) for i, fig in enumerate(figs)])

    # Recommendations (simple rule-based)
    recos = []
    if risk_mean > risk_thr:
        recos.append("RISK moyen supérieur au seuil de baseline. Priorité: réduire @(t) et Δd(t), renforcer O(t) et l'opérateur L.")
    if anti_flag:
        recos.append("Anti-gaming: signal o_bias. Vérifier manipulations de seuils, délais ou cohérence.")
    if maturity_label == "Dissonant":
        recos.append("Dissonant: capacité L_cap présente mais activation L_act faible ou gouvernance sous cibles. Corriger capture/exemptions.")
    if maturity_label == "Immature":
        recos.append("Immature: renforcer L_cap (bench) et/ou réduire les délais d'activation. Tester overload et lag-O.")
    if not recos:
        recos.append("Aucun drapeau majeur. Continuer stress tests Type I/II/III et surveiller dérive de gouvernance.")

    recos_html = "<ul>" + "".join([f"<li>{r}</li>" for r in recos]) + "</ul>"

    css = """
    body { font-family: Arial, sans-serif; margin: 28px; color: #111; }
    .header { background: #1a3c5e; color: #fff; padding: 18px 20px; border-radius: 10px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 18px; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 14px 16px; }
    table { border-collapse: collapse; width: 100%; margin: 10px 0; }
    th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
    th { background: #f4f4f4; }
    .alert { color: #b00020; font-weight: bold; }
    footer { margin-top: 26px; font-size: 12px; color: #666; }
    """

    title = f"Amplification-Barometer Audit - {report_dict.get('dataset_name','dataset')}"
    created = report_dict.get("created_utc", "")
    version = report_dict.get("version", "")
    weights_version = report_dict.get("weights_version", "")

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>{css}</style>
</head>
<body>
  <div class="header">
    <h1 style="margin:0;">Amplification-Barometer Audit Report</h1>
    <p style="margin:6px 0 0 0;">Dataset: <b>{report_dict.get('dataset_name','dataset')}</b> | Date: {created} | Version: {version} | Weights: {weights_version}</p>
  </div>

  <div class="grid">
    <div class="card">
      <h2 style="margin-top:0;">Résumé exécutif</h2>
      <p><b>Maturité:</b> {maturity_label}</p>
      <p><b>Risque moyen:</b> {risk_mean:.3f} (std: {risk_std:.3f})</p>
      <p class="{ 'alert' if anti_flag else '' }"><b>Anti-gaming:</b> {'RED FLAG' if anti_flag else 'OK'}</p>
      <p><b>Baseline:</b> {'appliquée' if baseline_used else 'non appliquée'}</p>
    </div>
    <div class="card">
      <h2 style="margin-top:0;">Scores clés</h2>
      {df_scores.to_html(index=False)}
    </div>
  </div>

  <div class="card">
    <h2 style="margin-top:0;">Targets et seuils</h2>
    {df_targets.to_html(index=False)}
  </div>

  <div class="card">
    <h2 style="margin-top:0;">Graphiques</h2>
    {fig_html_blocks}
  </div>

  <div class="card">
    <h2 style="margin-top:0;">Conclusions et recommandations</h2>
    {recos_html}
  </div>

  <footer>
    Généré par Amplification-Barometer | version {version} | {created} | GPT-5.2 Thinking
  </footer>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")
