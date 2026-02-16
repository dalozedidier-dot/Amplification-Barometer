
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from jinja2 import Template  # noqa: E402

from .composites import compute_at, compute_delta_d  # noqa: E402
from .l_operator import compute_l_act, compute_l_cap  # noqa: E402


_REPORT_TEMPLATE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Amplification-Barometer Audit - {{ dataset_name }}</title>
  <style>
    :root{
      --bg:#0b1220;
      --panel:#0f1a2e;
      --card:#121f36;
      --text:#e9eef7;
      --muted:#b8c3d6;
      --line:rgba(255,255,255,.10);
      --accent:#63b3ff;
      --ok:#46d39a;
      --warn:#ffcc66;
      --bad:#ff6b6b;
    }
    body{font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin:0; color:var(--text); background:linear-gradient(180deg,var(--bg),#070b14);}
    .wrap{max-width:1100px; margin:0 auto; padding:28px 20px 44px;}
    .header{background:linear-gradient(90deg,#0f2a4a,#112b55); border:1px solid var(--line); border-radius:18px; padding:22px 22px 18px; box-shadow:0 10px 30px rgba(0,0,0,.25);}
    .title{display:flex; flex-wrap:wrap; gap:10px; align-items:baseline; justify-content:space-between;}
    h1{margin:0; font-size:28px; letter-spacing:.2px;}
    .meta{color:var(--muted); font-size:13px;}
    .grid{display:grid; grid-template-columns:repeat(12,1fr); gap:14px; margin-top:14px;}
    .card{background:rgba(18,31,54,.92); border:1px solid var(--line); border-radius:16px; padding:14px 14px 12px; box-shadow:0 10px 24px rgba(0,0,0,.16);}
    .span4{grid-column:span 4;}
    .span6{grid-column:span 6;}
    .span8{grid-column:span 8;}
    .span12{grid-column:span 12;}
    .kpi{display:flex; flex-direction:column; gap:6px;}
    .kpi .label{color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em;}
    .kpi .value{font-size:22px; font-weight:700;}
    .badge{display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px; border:1px solid var(--line); background:rgba(255,255,255,.05); color:var(--text);}
    .badge.ok{border-color:rgba(70,211,154,.35); background:rgba(70,211,154,.12); color:var(--ok);}
    .badge.warn{border-color:rgba(255,204,102,.35); background:rgba(255,204,102,.12); color:var(--warn);}
    .badge.bad{border-color:rgba(255,107,107,.35); background:rgba(255,107,107,.12); color:var(--bad);}
    h2{margin:0 0 8px; font-size:18px;}
    p{margin:8px 0 0; color:var(--muted); line-height:1.45;}
    .table-wrap{overflow:auto; border-radius:14px; border:1px solid var(--line);}
    table{width:100%; border-collapse:collapse; background:rgba(10,18,32,.35);}
    th,td{padding:10px 12px; border-bottom:1px solid rgba(255,255,255,.08); text-align:left; font-size:13px;}
    th{color:var(--muted); font-weight:600; background:rgba(255,255,255,.04);}
    tr:hover td{background:rgba(255,255,255,.03);}
    .img{width:100%; border-radius:14px; border:1px solid var(--line); background:#060b14;}
    .cols{display:grid; grid-template-columns:1fr; gap:14px;}
    .reco li{margin:6px 0; color:var(--muted);}
    .footer{margin-top:18px; color:var(--muted); font-size:12px; text-align:center;}
    .small{font-size:12px; color:var(--muted);}
    @media (max-width: 860px){
      .span4,.span6,.span8{grid-column:span 12;}
      h1{font-size:24px;}
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <div class="title">
        <h1>Amplification-Barometer Audit Report</h1>
        <div class="meta">Dataset: <strong>{{ dataset_name }}</strong> | Date: {{ audit_date }} | Version: {{ version }}</div>
      </div>
      <p class="small">{{ tagline }}</p>

      <div class="grid">
        <div class="card span4">
          <div class="kpi">
            <div class="label">RISK (mean)</div>
            <div class="value">{{ risk_mean }}</div>
            <div class="small">std {{ risk_std }} | threshold {{ risk_thr }}</div>
          </div>
        </div>
        <div class="card span4">
          <div class="kpi">
            <div class="label">Maturité</div>
            <div class="value">{{ maturity_label }}</div>
            <div class="small">score {{ maturity_score }}</div>
          </div>
        </div>
        <div class="card span4">
          <div class="kpi">
            <div class="label">Anti-gaming</div>
            <div class="value">{{ anti_gaming_label }}</div>
            <div class="small">détection suite {{ anti_gaming_rate }}</div>
          </div>
        </div>

        <div class="card span12">
          <h2>Résumé exécutif</h2>
          <p>{{ executive_summary }}</p>
          <div style="margin-top:10px; display:flex; gap:10px; flex-wrap:wrap;">
            <span class="badge {{ stability_badge }}">{{ stability_label }}</span>
            <span class="badge {{ verdict_badge }}">{{ verdict_label }}</span>
            <span class="badge {{ governance_badge }}">{{ governance_label }}</span>
          </div>
        </div>
      </div>
    </div>

    <div class="grid" style="margin-top:16px;">
      <div class="card span6">
        <h2>Scores clés</h2>
        <div class="table-wrap">
          {{ key_scores_table | safe }}
        </div>
      </div>
      <div class="card span6">
        <h2>Dimensions du verdict</h2>
        <div class="table-wrap">
          {{ verdict_table | safe }}
        </div>
        <p class="small">Le score global est optionnel. Les dimensions restent séparées pour éviter les verdicts incohérents.</p>
      </div>

      <div class="card span12">
        <h2>Séries et signaux</h2>
        <div class="cols">
          <img class="img" src="data:image/png;base64,{{ at_plot_b64 }}" alt="@(t)">
          <img class="img" src="data:image/png;base64,{{ dd_plot_b64 }}" alt="Δd(t)">
          <img class="img" src="data:image/png;base64,{{ l_plot_b64 }}" alt="L_cap vs L_act">
          <img class="img" src="data:image/png;base64,{{ risk_plot_b64 }}" alt="RISK(t)">
        </div>
      </div>

      <div class="card span12">
        <h2>Conclusions et recommandations</h2>
        <p>{{ conclusions_text }}</p>
        <ul class="reco">
          {% for r in recommendations %}
          <li>{{ r }}</li>
          {% endfor %}
        </ul>
      </div>

      <div class="card span12">
        <h2>Détails auditables</h2>
        <div class="table-wrap">
          {{ summary_table | safe }}
        </div>
      </div>
    </div>

    <div class="footer">
      Généré le {{ generated_utc }} | Amplification-Barometer v{{ version }} | Assistant: {{ author }}
    </div>
  </div>
</body>
</html>
"""

@dataclass(frozen=True)
class HtmlReportOptions:
    window: int = 5
    author: str = "GPT-5.2 Thinking"
    tagline: str = "Rapport auto-contenu, tables et graphes embarqués, sans serveur."
    figsize: Tuple[float, float] = (11.0, 5.5)
    dpi: int = 150


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _plot_line(x, y, *, title: str, y_label: str, options: HtmlReportOptions) -> str:
    fig = plt.figure(figsize=options.figsize)
    plt.plot(x, np.asarray(y, dtype=float), linewidth=2.0)
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.grid(True, alpha=0.25, linestyle="--")
    plt.xticks(rotation=45)
    plt.tight_layout(pad=1.3)
    return _fig_to_b64(fig)


def _plot_lcap_lact(dates, lcap, lact, *, options: HtmlReportOptions) -> str:
    fig = plt.figure(figsize=(11.0, 6.0))
    plt.plot(dates, np.asarray(lcap, dtype=float), label="L_cap", linewidth=2.0)
    plt.plot(dates, np.asarray(lact, dtype=float), label="L_act", linewidth=2.0)
    plt.title("L_cap vs L_act", fontsize=14, pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.grid(True, alpha=0.25, linestyle="--")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout(pad=1.3)
    return _fig_to_b64(fig)


def _safe_get(d: Mapping[str, Any], path: Iterable[str], default: Any = None) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, Mapping) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _infer_regime(dataset_name: str, report: Mapping[str, Any]) -> str:
    name = (dataset_name or "").lower()
    if "bifur" in name:
        return "bifurcation"
    if "overload" in name:
        return "overload"
    if "oscill" in name:
        return "oscillating"
    if "stable" in name:
        return "stable"
    # Optional: if report includes discrimination
    r = str(_safe_get(report, ["regime"], "")).lower()
    if r:
        return r
    return "unknown"


def _recommendations_for(regime: str) -> List[str]:
    if regime == "bifurcation":
        return [
            "Renforcer O(t) avant le point de non retour. Surveiller la persistance de Δd(t) > 0 et le drift de RISK(t).",
            "Tester des chocs rares et mesurer l'irréversibilité de E(t). La stabilité apparente ne protège pas du tipping point.",
            "Réduire l'exposition au couplage non linéaire P×E et augmenter la redondance R(t).",
        ]
    if regime == "overload":
        return [
            "Augmenter L_cap et réduire le délai d'activation. Viser activation_delay_steps <= 5.",
            "Ajouter de la redondance et des marges. Monitorer R_mttr_proxy et R_level sous stress.",
            "Limiter la charge effective. Prioriser les actions qui réduisent E_stock et dE/dt.",
        ]
    if regime == "oscillating":
        return [
            "Mettre du damping: règles de stabilisation et cohérence d'activation. Vérifier la stabilité du classement (Spearman, Jaccard).",
            "Surveiller les cycles: superposer raw et smoothed, puis tester l'impact des fenêtres et normalisations.",
            "Éviter les ajustements opportunistes. Le système oscille et se prête au gaming si la gouvernance est faible.",
        ]
    if regime == "stable":
        return [
            "Ne pas confondre stable et sûr. Lancer un stress overload systématique et mesurer la dégradation.",
            "Maintenir une baseline stable. Interdire toute renormalisation par dataset si baseline fournie.",
            "Surveiller G(t) et les exemptions. Les faux signaux de maturité viennent souvent de la gouvernance.",
        ]
    return [
        "Documenter les proxys et vérifier les bornes, la normalisation et la comparabilité inter-datasets.",
        "Exécuter la stress suite Type I, II, III et comparer la signature RISK(t) sur baseline stable.",
        "Utiliser le verdict multidimensionnel plutôt qu'un label binaire.",
    ]


def _badge_for(value: float, *, good_hi: bool = True) -> str:
    if not np.isfinite(value):
        return "warn"
    if good_hi:
        if value >= 0.75:
            return "ok"
        if value >= 0.45:
            return "warn"
        return "bad"
    # good low
    if value <= 0.25:
        return "ok"
    if value <= 0.55:
        return "warn"
    return "bad"


def build_self_contained_html_report(
    df: pd.DataFrame,
    report: Mapping[str, Any],
    *,
    out_html: Path,
    options: Optional[HtmlReportOptions] = None,
) -> Path:
    """Create a single self-contained HTML report with embedded PNG plots.

    This is CI-friendly and does not require a server.
    """
    opt = options or HtmlReportOptions()
    out_html.parent.mkdir(parents=True, exist_ok=True)

    dataset_name = str(report.get("dataset_name", "dataset"))
    created = str(report.get("created_utc", datetime.now(timezone.utc).isoformat()))
    version = str(report.get("version", "0.0.0"))
    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Compute series for plots
    at = compute_at(df)
    dd = compute_delta_d(df, window=opt.window)
    lcap = compute_l_cap(df)
    lact = compute_l_act(df)

    # RISK series: try to reuse report if present, else robust z from series
    # We keep it simple here: risk = z(at)+z(dd) with robust median/MAD
    def _robust_z(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        m = np.isfinite(x)
        if not m.any():
            return np.zeros_like(x)
        med = float(np.median(x[m]))
        mad = float(np.median(np.abs(x[m] - med))) + 1e-9
        return (x - med) / mad

    risk = pd.Series(_robust_z(at.to_numpy()) + _robust_z(dd.to_numpy()), index=df.index, name="RISK")

    # Plots to base64
    at_b64 = _plot_line(at.index, at.to_numpy(), title="@(t)", y_label="@(t)", options=opt)
    dd_b64 = _plot_line(dd.index, dd.to_numpy(), title="Δd(t)", y_label="Δd(t)", options=opt)
    l_b64 = _plot_lcap_lact(df.index, lcap.to_numpy(), lact.to_numpy(), options=opt)
    risk_b64 = _plot_line(risk.index, risk.to_numpy(), title="RISK(t)", y_label="RISK(t)", options=opt)

    # Extract key metrics from report dict
    summary = report.get("summary", {}) if isinstance(report.get("summary"), Mapping) else {}
    # Prefer explicit RISK stats if present
    risk_stats = summary.get("RISK") if isinstance(summary.get("RISK"), Mapping) else {}
    risk_mean = float(risk_stats.get("mean")) if "mean" in risk_stats else float(np.nanmean(risk.to_numpy()))
    risk_std = float(risk_stats.get("std")) if "std" in risk_stats else float(np.nanstd(risk.to_numpy()))
    risk_thr = _safe_get(report, ["calibration", "thresholds", "risk_thr"], None)
    if risk_thr is None:
        # fallback to p95 style
        risk_thr = float(np.nanpercentile(risk.to_numpy(), 95))
    risk_thr = float(risk_thr)

    maturity = report.get("maturity", {}) if isinstance(report.get("maturity"), Mapping) else {}
    maturity_label = str(maturity.get("label", "Unknown"))
    maturity_score = maturity.get("score", maturity.get("score_0_100", None))
    if maturity_score is None:
        maturity_score = float(maturity.get("cap_score_enforced", 0.0)) * 100.0
    try:
        maturity_score = float(maturity_score)
    except Exception:
        maturity_score = float("nan")

    manip = report.get("manipulability", {}) if isinstance(report.get("manipulability"), Mapping) else {}
    det_rate = _safe_get(manip, ["summary", "detected_rate"], 0.0)
    try:
        det_rate = float(det_rate)
    except Exception:
        det_rate = 0.0
    anti_gaming_label = "RED FLAG" if det_rate >= 0.5 else "OK"
    anti_gaming_rate = f"{det_rate:.3f}"

    stability = report.get("stability", {}) if isinstance(report.get("stability"), Mapping) else {}
    stable_flag = bool(stability.get("stable_flag", False))
    stability_label = "Stabilité OK" if stable_flag else "Stabilité fragile"
    stability_badge = "ok" if stable_flag else "warn"

    verdict = report.get("verdict", {}) if isinstance(report.get("verdict"), Mapping) else {}
    verdict_label = str(verdict.get("label", maturity_label))
    verdict_badge = "warn" if verdict_label.lower() in {"immature", "dissonant"} else "ok"

    targets = report.get("targets", {}) if isinstance(report.get("targets"), Mapping) else {}
    gov_meets = bool(targets.get("rule_execution_gap_meets_target", False)) and bool(targets.get("control_turnover_meets_target", False))
    governance_label = "Gouvernance sous cibles" if gov_meets else "Gouvernance à risque"
    governance_badge = "ok" if gov_meets else "warn"

    regime = _infer_regime(dataset_name, report)
    recommendations = _recommendations_for(regime)

    executive_summary = (
        f"Régime détecté: {regime}. RISK moyen {risk_mean:.2f} (seuil {risk_thr:.2f}). "
        f"Maturité: {maturity_label} (score {maturity_score:.1f}). "
        f"Anti-gaming: {anti_gaming_label} (détection {det_rate:.3f})."
    )

    # Key scores table
    key_scores = [
        ("RISK mean", risk_mean),
        ("RISK std", risk_std),
        ("RISK threshold", risk_thr),
        ("Stabilité stable_flag", int(stable_flag)),
        ("Maturité score", maturity_score),
        ("Anti-gaming detected_rate", det_rate),
        ("Prevented exceedance rel", float(_safe_get(report, ["l_performance", "prevented_exceedance_rel"], 0.0) or 0.0)),
        ("Activation delay steps", _safe_get(report, ["l_performance", "first_activation_delay_steps"], None)),
        ("E irreversibility", float(_safe_get(summary, ["E_irreversibility", "mean"], float("nan")))),
        ("R level", float(_safe_get(summary, ["R_level", "mean"], float("nan")))),
        ("G level", float(_safe_get(summary, ["G", "mean"], float("nan")))),
    ]
    ks_df = pd.DataFrame(key_scores, columns=["Metric", "Value"])
    key_scores_table = ks_df.to_html(index=False, escape=True)

    # Verdict dimensions table
    dims = verdict.get("dimensions", {}) if isinstance(verdict.get("dimensions"), Mapping) else {}
    if dims:
        vd_df = pd.DataFrame([(k, float(v)) for k, v in dims.items()], columns=["Dimension", "Score"])
        vd_df["Score"] = vd_df["Score"].map(lambda x: round(float(x), 3))
    else:
        vd_df = pd.DataFrame([("maturity", maturity_score / 100.0)], columns=["Dimension", "Score"])
    verdict_table = vd_df.to_html(index=False, escape=True)

    # Summary table: flatten summary metrics
    rows = []
    for k, v in summary.items():
        if isinstance(v, Mapping) and "mean" in v:
            rows.append((k, v.get("mean"), v.get("std"), v.get("p95"), v.get("min"), v.get("max")))
        elif isinstance(v, (int, float, str, bool)):
            rows.append((k, v, "", "", "", ""))
    summ_df = pd.DataFrame(rows, columns=["Metric", "mean/value", "std", "p95", "min", "max"])
    summary_table = summ_df.to_html(index=False, escape=True)

    conclusions_text = (
        "Ce rapport isole les signaux d'amplification et la capacité de limitation sur une baseline robuste. "
        "Les recommandations ci-dessus sont centrées sur la réduction du risque, la stabilité du classement et la gouvernance mesurable."
    )

    tpl = Template(_REPORT_TEMPLATE)
    html = tpl.render(
        dataset_name=dataset_name,
        audit_date=created,
        version=version,
        author=opt.author,
        tagline=opt.tagline,
        generated_utc=generated_utc,
        risk_mean=f"{risk_mean:.3f}",
        risk_std=f"{risk_std:.3f}",
        risk_thr=f"{risk_thr:.3f}",
        maturity_label=maturity_label,
        maturity_score=f"{maturity_score:.1f}",
        anti_gaming_label=anti_gaming_label,
        anti_gaming_rate=anti_gaming_rate,
        executive_summary=executive_summary,
        stability_label=stability_label,
        stability_badge=stability_badge,
        verdict_label=verdict_label,
        verdict_badge=verdict_badge,
        governance_label=governance_label,
        governance_badge=governance_badge,
        key_scores_table=key_scores_table,
        verdict_table=verdict_table,
        summary_table=summary_table,
        at_plot_b64=at_b64,
        dd_plot_b64=dd_b64,
        l_plot_b64=l_b64,
        risk_plot_b64=risk_b64,
        conclusions_text=conclusions_text,
        recommendations=recommendations,
    )
    out_html.write_text(html, encoding="utf-8")
    return out_html


def build_reports_index(reports: List[Tuple[str, str]], *, out_html: Path, title: str = "Amplification-Barometer Reports") -> Path:
    """Build a simple index.html linking to generated reports.

    reports: list of (name, relative_href)
    """
    out_html.parent.mkdir(parents=True, exist_ok=True)
    links = "\n".join([f'<li><a href="{href}">{name}</a></li>' for name, href in reports])
    html = f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body{{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif; margin:32px;}}
h1{{margin:0 0 12px;}}
ul{{line-height:1.7;}}
.small{{color:#555; font-size:12px;}}
</style></head>
<body>
<h1>{title}</h1>
<p class="small">Index des rapports HTML auto-contenus.</p>
<ul>
{links}
</ul>
</body></html>"""
    out_html.write_text(html, encoding="utf-8")
    return out_html
