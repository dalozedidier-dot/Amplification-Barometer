from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from amplification_barometer.composites import E_SPEC, G_SPEC, O_SPEC, P_SPEC, CompositeSpec


def _spearman(x: pd.Series, y: pd.Series) -> float:
    s = pd.concat([x, y], axis=1).dropna()
    if len(s) < 8:
        return float("nan")
    return float(s.iloc[:, 0].corr(s.iloc[:, 1], method="spearman"))


def _normalize_nonneg(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.clip(scores, 0.0, None)
    if scores.sum() <= 0:
        return np.ones_like(scores) / len(scores)
    return scores / scores.sum()


def propose_weights_from_incidents(
    incidents: pd.DataFrame,
    *,
    specs: List[CompositeSpec],
    prior_strength: float = 0.60,
    exog_weight: float = 0.50,
    damage_col: str = "damage_weight",
    exog_col: str = "u_exog",
) -> Dict[str, Any]:
    """Compute an audit-friendly weight proposal from anonymized incident windows.

    Principle:
    - Use rank correlations (Spearman) to reduce sensitivity to scale choices.
    - Blend with prior weights (prior_strength) to avoid overfitting small datasets.
    - Optionally anchor on an exogenous shock intensity series (u_exog) when available.

    Required columns:
    - damage_weight (0..1 or any nonnegative scalar)

    Optional column:
    - u_exog (shock intensity) to reduce arbitrariness and align with u(t) calibration.
    """
    if damage_col not in incidents.columns:
        raise ValueError(f"Missing required column: {damage_col}")

    has_exog = exog_col in incidents.columns

    report: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "damage_col": damage_col,
        "exog_col": exog_col if has_exog else None,
        "prior_strength": float(prior_strength),
        "exog_weight": float(exog_weight) if has_exog else 0.0,
        "groups": {},
        "notes": [],
    }

    if not has_exog:
        report["notes"].append("u_exog column not present: weights use damage_weight only.")

    y_damage = incidents[damage_col].astype(float)

    for spec in specs:
        missing = [p for p in spec.proxies if p not in incidents.columns]
        if missing:
            report["groups"][spec.name] = {
                "status": "skipped_missing_proxies",
                "missing_proxies": missing,
            }
            continue

        cor_damage = []
        cor_exog = []
        for p in spec.proxies:
            cor_damage.append(_spearman(incidents[p].astype(float), y_damage))
            if has_exog:
                cor_exog.append(_spearman(incidents[p].astype(float), incidents[exog_col].astype(float)))

        cor_damage = np.asarray(cor_damage, dtype=float)
        cor_exog = np.asarray(cor_exog, dtype=float) if has_exog else np.zeros_like(cor_damage)

        # combined nonnegative importance score
        score = np.abs(cor_damage) + (float(exog_weight) * np.abs(cor_exog) if has_exog else 0.0)
        score_norm = _normalize_nonneg(score)

        prior = np.asarray(spec.weights, dtype=float)
        prior = prior / prior.sum()

        alpha = float(np.clip(prior_strength, 0.0, 1.0))
        proposed = alpha * prior + (1.0 - alpha) * score_norm
        proposed = proposed / proposed.sum()

        report["groups"][spec.name] = {
            "status": "ok",
            "spec": {
                "name": spec.name,
                "proxies": list(spec.proxies),
                "invert": bool(spec.invert),
            },
            "prior_weights": prior.tolist(),
            "corr_damage_spearman": [float(x) if np.isfinite(x) else None for x in cor_damage],
            "corr_exog_spearman": [float(x) if np.isfinite(x) else None for x in cor_exog] if has_exog else None,
            "score": score.tolist(),
            "score_norm": score_norm.tolist(),
            "proposed_weights": proposed.tolist(),
        }

    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Recalibrate composite weights from anonymized incidents (audit-friendly).")
    ap.add_argument("--incidents", type=str, default="data/incidents/anonymized_incidents_template.csv")
    ap.add_argument("--out", type=str, default="docs/weights_proposal.json")
    ap.add_argument("--prior-strength", type=float, default=0.60)
    ap.add_argument("--exog-weight", type=float, default=0.50)
    ap.add_argument("--damage-col", type=str, default="damage_weight")
    ap.add_argument("--exog-col", type=str, default="u_exog")
    args = ap.parse_args()

    inc = pd.read_csv(args.incidents)
    specs = [P_SPEC, O_SPEC, E_SPEC, G_SPEC]

    report = propose_weights_from_incidents(
        inc,
        specs=specs,
        prior_strength=float(args.prior_strength),
        exog_weight=float(args.exog_weight),
        damage_col=args.damage_col,
        exog_col=args.exog_col,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
