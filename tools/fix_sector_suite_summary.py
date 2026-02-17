#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_reports(root: Path) -> List[Path]:
    return sorted(root.rglob("audit_report_*.json"))


def _get(d: Any, *keys: str, default: Any = None) -> Any:
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


@dataclass(frozen=True)
class Targets:
    rule_execution_gap_mean_max: float = 0.05
    proactive_topk_excess_rel_min: float = 0.10
    prevented_exceedance_rel_min: float = 0.10


def extract_row(rep: Dict[str, Any], *, targets: Targets) -> Dict[str, Any]:
    verdict = rep.get("verdict", {}) or {}
    notes = verdict.get("notes", {}) or {}

    lp = rep.get("l_performance", {}) or {}
    lpp = rep.get("l_performance_proactive", {}) or {}

    maturity = lpp.get("maturity") or lp.get("maturity") or {}
    gap_mean = maturity.get("rule_execution_gap_mean")
    risk_thr = lpp.get("risk_threshold", lp.get("risk_threshold"))

    prevented_ex_rel = lpp.get("prevented_exceedance_rel", lpp.get("prevented_exceedance"))
    topk_ex_rel = lpp.get("prevented_topk_excess_rel")

    # Fallback if an older report only has topk_overlap.
    if topk_ex_rel is None and "topk_overlap" in lpp:
        try:
            topk_ex_rel = float(1.0 - float(lpp["topk_overlap"]))
        except Exception:
            topk_ex_rel = None

    row = {
        "dataset": rep.get("dataset_name") or rep.get("name") or rep.get("dataset") or rep.get("id") or "unknown",
        "baseline_used": bool(notes.get("baseline_used")) if notes.get("baseline_used") is not None else None,
        "risk_threshold": float(risk_thr) if risk_thr is not None else None,
        "rule_execution_gap_mean": float(gap_mean) if gap_mean is not None else None,
        "gap_meets_target": None if gap_mean is None else (float(gap_mean) < float(targets.rule_execution_gap_mean_max)),
        "prevented_exceedance_rel": float(prevented_ex_rel) if prevented_ex_rel is not None else None,
        "prevented_meets_target": None
        if prevented_ex_rel is None
        else (float(prevented_ex_rel) >= float(targets.prevented_exceedance_rel_min)),
        "prevented_topk_excess_rel": float(topk_ex_rel) if topk_ex_rel is not None else None,
        "proactive_topk_excess_rel": float(topk_ex_rel) if topk_ex_rel is not None else None,
        "proactive_topk_frac": float(lpp.get("topk_frac")) if lpp.get("topk_frac") is not None else 0.2,
        "maturity_label": lpp.get("verdict") or lp.get("verdict") or verdict.get("label"),
        "l_verdict": lp.get("verdict"),
        "l_verdict_proactive": lpp.get("verdict"),
    }
    return row


def write_md_table(df: pd.DataFrame, out_md: Path, *, targets: Targets, baseline_info: Optional[Dict[str, Any]] = None) -> None:
    lines: List[str] = []
    lines.append("# Sector suite summary")
    lines.append("")
    if baseline_info:
        lines.append("Baseline:")
        for k, v in baseline_info.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    lines.append("Targets (demo):")
    lines.append(f"- rule_execution_gap_mean < {targets.rule_execution_gap_mean_max}")
    lines.append(f"- proactive_topk_excess_rel > {targets.proactive_topk_excess_rel_min}")
    lines.append(f"- prevented_exceedance_rel >= {targets.prevented_exceedance_rel_min}")
    lines.append("")
    lines.append("Results:")
    lines.append("")
    lines.append(df.to_markdown(index=False))
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Fix sector suite summary by reading l_performance_proactive keys.")
    p.add_argument("--reports-root", default="_ci_out", help="Directory containing audit_report_*.json (recursive)")
    p.add_argument("--out-csv", default="sector_suite_summary.csv", help="Output CSV")
    p.add_argument("--out-md", default="sector_suite_summary.md", help="Output Markdown")
    p.add_argument("--baseline-json", default=None, help="Optional baseline thresholds JSON to include in MD header")
    args = p.parse_args()

    root = Path(args.reports_root)
    reports = _find_reports(root)
    if not reports:
        raise SystemExit(f"Aucun audit_report_*.json trouvé sous: {root}")

    targets = Targets()

    rows: List[Dict[str, Any]] = []
    for rp in reports:
        rep = _load_json(rp)
        row = extract_row(rep, targets=targets)
        # prefer filename for dataset when missing or ambiguous
        if row["dataset"] in ("unknown", None, ""):
            row["dataset"] = rp.stem.replace("audit_report_", "")
        rows.append(row)

    df = pd.DataFrame(rows)

    # Only keep rows that look like suite outputs when present
    # If the directory contains many audits, we keep them all but sort by dataset.
    df = df.sort_values(["dataset"]).reset_index(drop=True)

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    baseline_info: Optional[Dict[str, Any]] = None
    if args.baseline_json:
        bj = Path(args.baseline_json)
        if bj.exists():
            payload = _load_json(bj)
            thr = payload.get("thresholds", payload)
            baseline_info = {"risk_thr": thr.get("risk_thr"), "at_p95_stable": thr.get("at_p95_stable"), "dd_p95_stable": thr.get("dd_p95_stable")}

    write_md_table(df, out_md, targets=targets, baseline_info=baseline_info)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
