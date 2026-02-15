#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from amplification_barometer.composites import E_SPEC, G_SPEC, O_SPEC, P_SPEC


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank(method="average").to_numpy(dtype=float)
    rb = pd.Series(b).rank(method="average").to_numpy(dtype=float)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _normalize_nonneg(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, None)
    s = float(np.sum(x))
    if s <= eps:
        return np.ones_like(x) / float(len(x))
    return x / s


@dataclass(frozen=True)
class GroupWeights:
    proxies: List[str]
    weights: List[float]
    spearman_abs: List[float]
    n_used: int


def _compute_group(df: pd.DataFrame, target: np.ndarray, proxies: List[str], *, prior_strength: float = 0.50) -> GroupWeights:
    used = []
    corrs = []
    for p in proxies:
        if p not in df.columns:
            continue
        s = pd.to_numeric(df[p], errors="coerce")
        mask = np.isfinite(s.to_numpy()) & np.isfinite(target)
        if int(np.sum(mask)) < 4:
            continue
        rho = _spearman(s.to_numpy(dtype=float)[mask], target[mask])
        used.append(p)
        corrs.append(abs(float(rho)))

    if not used:
        # no data: equal weights on the declared proxies
        w = (np.ones(len(proxies)) / float(len(proxies))).tolist()
        return GroupWeights(proxies=list(proxies), weights=w, spearman_abs=[0.0] * len(proxies), n_used=0)

    corrs_arr = np.asarray(corrs, dtype=float)
    w_data = _normalize_nonneg(corrs_arr)

    # conservative blend: prior equal weights + data suggestion
    w_prior = np.ones(len(w_data), dtype=float) / float(len(w_data))
    w = (1.0 - float(prior_strength)) * w_prior + float(prior_strength) * w_data
    w = (w / float(np.sum(w))).tolist()

    return GroupWeights(
        proxies=list(used),
        weights=[float(x) for x in w],
        spearman_abs=[float(x) for x in corrs_arr.tolist()],
        n_used=int(len(target)),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Recalibrate proxy weights from anonymized incidents (audit-friendly).")
    ap.add_argument("--incidents", type=str, required=True, help="CSV file with damage_weight and optional proxies.")
    ap.add_argument("--out", type=str, default="docs/recalibrated_weights.json", help="Output JSON path.")
    ap.add_argument("--prior-strength", type=float, default=0.50, help="Blend between equal weights and data-driven weights.")
    args = ap.parse_args()

    inc_path = Path(args.incidents)
    df = pd.read_csv(inc_path)
    if "damage_weight" not in df.columns:
        raise SystemExit("Missing column: damage_weight")

    target = pd.to_numeric(df["damage_weight"], errors="coerce").to_numpy(dtype=float)

    groups: Dict[str, GroupWeights] = {}
    groups["P"] = _compute_group(df, target, list(P_SPEC.proxies), prior_strength=float(args.prior_strength))
    groups["O"] = _compute_group(df, target, list(O_SPEC.proxies), prior_strength=float(args.prior_strength))
    groups["E"] = _compute_group(df, target, list(E_SPEC.proxies), prior_strength=float(args.prior_strength))
    groups["G"] = _compute_group(df, target, list(G_SPEC.proxies), prior_strength=float(args.prior_strength))

    # R is a mixed-sign composite (recovery_time is inverted in code). For recalibration,
    # we keep equal weights by default and let real deployments handle sector specifics.
    groups["R"] = GroupWeights(
        proxies=["margin_proxy", "redundancy_proxy", "diversity_proxy", "recovery_time_proxy"],
        weights=[0.25, 0.25, 0.25, 0.25],
        spearman_abs=[0.0, 0.0, 0.0, 0.0],
        n_used=0,
    )

    payload = {
        "schema": "amplification-barometer.weights",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "spearman_abs_blend_with_equal_prior",
        "input_file": inc_path.name,
        "prior_strength": float(args.prior_strength),
        "groups": {k: asdict(v) for k, v in groups.items()},
        "notes": "Weights are suggestions. Review in PRs. Apply only after independent validation.",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
