from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .theory_map import TheoryAuditMap, load_proxy_specs


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def robust_z_mad(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = _mad(x) or eps
    return (x - med) / (1.4826 * mad + eps)


def composite_level(df: pd.DataFrame, cols: List[str], weights: Optional[np.ndarray] = None) -> np.ndarray:
    if weights is None:
        weights = np.ones(len(cols), dtype=float) / max(1, len(cols))
    x = df[cols].to_numpy(dtype=float)
    return np.average(x, axis=1, weights=weights)


def compute_levels_from_specs(df: pd.DataFrame, specs: TheoryAuditMap) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for fam, famspec in specs.families.items():
        cols = list(famspec.proxies.keys())
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing proxies for family {fam}: {missing}")
        out[f"{fam}_level"] = composite_level(df, cols)
    return out


def compute_at_delta(levels: Dict[str, np.ndarray], eps: float = 1e-9, smooth_win: int = 7) -> Dict[str, np.ndarray]:
    p = np.asarray(levels["P_level"], dtype=float)
    o = np.asarray(levels["O_level"], dtype=float)
    at = p / (o + eps)

    p_s = pd.Series(p).rolling(window=smooth_win, min_periods=1).mean().to_numpy(dtype=float)
    o_s = pd.Series(o).rolling(window=smooth_win, min_periods=1).mean().to_numpy(dtype=float)
    dp = np.diff(p_s, prepend=p_s[0])
    do = np.diff(o_s, prepend=o_s[0])
    delta_d = dp - do
    return {"at": at, "delta_d": delta_d}


def compute_e_r(levels: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    e_level = np.asarray(levels.get("E_level", np.zeros_like(levels["P_level"])), dtype=float)
    r_level = np.asarray(levels.get("R_level", np.zeros_like(levels["P_level"])), dtype=float)
    e_stock = np.cumsum(e_level)
    return {"e_stock": e_stock, "r_level": r_level}


def validate_proxy_ranges(df: pd.DataFrame, specs: TheoryAuditMap) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    for fam, famspec in specs.families.items():
        for name, ps in famspec.proxies.items():
            s = df[name].to_numpy(dtype=float)
            lo, hi = ps.expected_range
            bad = np.where((s < lo) | (s > hi))[0]
            if bad.size:
                issues.append(
                    {
                        "proxy": name,
                        "family": fam,
                        "expected_range": [lo, hi],
                        "n_out_of_range": int(bad.size),
                        "first_index": int(bad[0]),
                    }
                )
    return {"ok": len(issues) == 0, "issues": issues}


def anti_gaming_o_bias(df: pd.DataFrame, o_family_cols: List[str], bias: float = 0.15, start_frac: float = 0.6) -> Dict[str, Any]:
    df2 = df.copy()
    start = int(len(df2) * float(start_frac))
    # Bias on O proxies that are down_risk: make them artificially better by increasing them
    for c in o_family_cols:
        x = df2[c].to_numpy(dtype=float, copy=True)
        x[start:] = x[start:] + bias
        df2[c] = x
    return {"start_index": start, "bias": float(bias), "columns": list(o_family_cols)}


def stability_audit_rank(
    df: pd.DataFrame,
    specs: TheoryAuditMap,
    windows: List[int] = [5, 7, 9],
    topk: int = 20,
) -> Dict[str, Any]:
    levels = compute_levels_from_specs(df, specs)
    base = compute_at_delta(levels, smooth_win=7)["at"]
    base_score = robust_z_mad(base)
    base_rank = np.argsort(-base_score)[:topk]

    results: List[Dict[str, Any]] = []
    for w in windows:
        at_w = compute_at_delta(levels, smooth_win=int(w))["at"]
        score = robust_z_mad(at_w)
        rank = np.argsort(-score)[:topk]
        # Spearman on scores
        if np.std(score) > 0 and np.std(base_score) > 0:
            sp = float(pd.Series(score).corr(pd.Series(base_score), method="spearman"))
        else:
            sp = 0.0
        # Jaccard topk
        a = set(int(i) for i in base_rank)
        b = set(int(i) for i in rank)
        j = float(len(a & b) / max(1, len(a | b)))
        results.append({"window": int(w), "spearman": sp, "jaccard_topk": j})
    # Stability criterion
    spearman_min = min(r["spearman"] for r in results) if results else 0.0
    jaccard_min = min(r["jaccard_topk"] for r in results) if results else 0.0
    stable = (spearman_min >= 0.85) and (jaccard_min >= 0.80)
    return {"stable": bool(stable), "spearman_min": spearman_min, "jaccard_min": jaccard_min, "details": results}


def stress_signature_suite(df: pd.DataFrame, specs: TheoryAuditMap) -> Dict[str, Any]:
    levels = compute_levels_from_specs(df, specs)
    sig = compute_at_delta(levels, smooth_win=7)
    er = compute_e_r(levels)
    at = sig["at"]
    delta_d = sig["delta_d"]
    e_stock = er["e_stock"]
    r_level = er["r_level"]

    # Type signatures
    # Persistence of dE/dt: positive tail indicates accumulating externalities
    de = np.diff(e_stock, prepend=e_stock[0])
    pers_de = float(np.mean(de[-30:] > 0.0))
    # Recovery proxy: R_level should rebound after stress, so last segment mean should be high
    r_tail = float(np.nanmean(r_level[-30:])) if len(r_level) >= 30 else float(np.nanmean(r_level))
    # O saturation proxy: O_level low fraction
    o_level = np.asarray(levels["O_level"], dtype=float)
    o_sat = float(np.mean(o_level < np.nanpercentile(o_level, 10)))
    # Divergence proxy: at tail above its 90 percentile in baseline window
    at_div = float(np.mean(at[-30:] > np.nanpercentile(at, 90)))

    # Irreversibility proxy: E stock does not return, so last value close to max
    irr = float((e_stock[-1] / (np.max(e_stock) + 1e-9)) if len(e_stock) else 0.0)

    return {
        "persistence_dE_dt_tail_pos_frac": pers_de,
        "r_tail_mean": r_tail,
        "o_saturation_low_frac": o_sat,
        "at_divergence_tail_frac": at_div,
        "e_irreversibility_ratio": irr,
    }


def multidim_verdict(
    *,
    stability: Dict[str, Any],
    proxy_ranges: Dict[str, Any],
    stress: Dict[str, Any],
    turnover_target_ok: Optional[bool] = None,
    gap_target_ok: Optional[bool] = None,
) -> Dict[str, Any]:
    # Simple rule based scoring, conservative defaults
    dims: Dict[str, Any] = {}

    dims["stability"] = "ok" if stability.get("stable") else "fail"
    dims["proxy_ranges"] = "ok" if proxy_ranges.get("ok") else "fail"

    # Stress: classify by heuristic thresholds
    # High persistence and high irreversibility hint type III
    pers = float(stress.get("persistence_dE_dt_tail_pos_frac", 0.0))
    irr = float(stress.get("e_irreversibility_ratio", 0.0))
    at_div = float(stress.get("at_divergence_tail_frac", 0.0))

    if (at_div >= 0.5) and (pers >= 0.6) and (irr >= 0.9):
        regime = "type_III_bifurcation"
    elif pers >= 0.5 and at_div < 0.5:
        regime = "type_II_oscillations"
    else:
        regime = "type_I_noise"
    dims["regime_signature"] = regime

    if turnover_target_ok is not None:
        dims["governance_turnover_target"] = "ok" if turnover_target_ok else "fail"
    if gap_target_ok is not None:
        dims["governance_gap_target"] = "ok" if gap_target_ok else "fail"

    return {"dimensions": dims}


def run_alignment_audit(
    df: pd.DataFrame,
    *,
    proxies_yaml: str | Path,
    turnover_target: float = 0.05,
    gap_target: float = 0.05,
) -> Dict[str, Any]:
    specs = load_proxy_specs(proxies_yaml)

    proxy_ranges = validate_proxy_ranges(df, specs)
    stability = stability_audit_rank(df, specs)
    stress = stress_signature_suite(df, specs)

    # Governance targets if columns exist
    turnover_ok = None
    gap_ok = None
    if "control_turnover" in df.columns:
        turnover_ok = bool(float(np.nanmean(df["control_turnover"].to_numpy(dtype=float))) <= float(turnover_target))
    if "rule_execution_gap" in df.columns:
        gap_ok = bool(float(np.nanmean(df["rule_execution_gap"].to_numpy(dtype=float))) <= float(gap_target))

    verdict = multidim_verdict(
        stability=stability,
        proxy_ranges=proxy_ranges,
        stress=stress,
        turnover_target_ok=turnover_ok,
        gap_target_ok=gap_ok,
    )

    levels = compute_levels_from_specs(df, specs)
    sig = compute_at_delta(levels, smooth_win=7)
    er = compute_e_r(levels)

    summary = {
        "at_mean": float(np.nanmean(sig["at"])),
        "at_p95": float(np.nanpercentile(sig["at"], 95)),
        "delta_d_std": float(np.nanstd(sig["delta_d"])),
        "e_stock_end": float(er["e_stock"][-1]) if len(er["e_stock"]) else 0.0,
        "r_tail_mean": float(np.nanmean(er["r_level"][-30:])) if len(er["r_level"]) else 0.0,
    }

    return {
        "spec_version": specs.version,
        "summary": summary,
        "proxy_ranges": proxy_ranges,
        "stability": stability,
        "stress_signatures": stress,
        "verdict": verdict,
    }


def write_alignment_outputs(report: Dict[str, Any], out_dir: str | Path, name: str) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / f"{name}_alignment_audit.json"
    md_path = out / f"{name}_alignment_audit.md"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    dims = report.get("verdict", {}).get("dimensions", {})
    md = []
    md.append(f"# Alignment audit: {name}\n")
    md.append(f"Spec version: {report.get('spec_version','')}\n")
    md.append("## Summary\n")
    for k, v in (report.get("summary") or {}).items():
        md.append(f"- {k}: {v}\n")
    md.append("\n## Verdict dimensions\n")
    for k, v in dims.items():
        md.append(f"- {k}: {v}\n")
    md.append("\n## Stability\n")
    st = report.get("stability") or {}
    md.append(f"- stable: {st.get('stable')}\n")
    md.append(f"- spearman_min: {st.get('spearman_min')}\n")
    md.append(f"- jaccard_min: {st.get('jaccard_min')}\n")
    md_path.write_text("".join(md), encoding="utf-8")

    return {"json": str(json_path), "md": str(md_path)}
