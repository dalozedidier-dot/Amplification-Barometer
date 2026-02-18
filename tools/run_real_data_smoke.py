#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from amplification_barometer.audit_report import build_audit_report, write_audit_report
from amplification_barometer.calibration import Thresholds, derive_thresholds
from amplification_barometer.html_report import render_audit_html
from amplification_barometer.real_data_adapters import (
    ensure_datetime_index,
    finance_ohlcv_to_proxies,
    has_required_proxies,
)


def _load_thresholds(*, window: int) -> Optional[Thresholds]:
    repo_root = Path(__file__).resolve().parents[1]
    stable_path = repo_root / "data" / "synthetic" / "stable_regime.csv"
    if not stable_path.exists():
        return None
    stable_df = pd.read_csv(stable_path)
    stable_df["date"] = pd.to_datetime(stable_df["date"], utc=True, errors="coerce")
    stable_df = stable_df.dropna(subset=["date"]).set_index("date").sort_index()
    return derive_thresholds(stable_df, window=window)


def _discover_inputs(repo_root: Path, *, include_root_proxies: bool = True) -> List[Path]:
    candidates: List[Path] = []
    for sub in ("data/real", "data/real_fixtures"):
        p = repo_root / sub
        if p.exists():
            candidates += list(p.glob("*.csv"))
            candidates += list(p.glob("*.parquet"))
    return sorted(set(candidates))


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _to_proxies(df: pd.DataFrame) -> pd.DataFrame:
    # Already proxies
    if has_required_proxies(df):
        return ensure_datetime_index(df)

    # OHLCV style
    cols = {c.lower() for c in df.columns}
    if {"open", "high", "low", "close"}.issubset(cols):
        ren = {c: c.lower() for c in df.columns}
        df2 = df.rename(columns=ren)
        df2 = ensure_datetime_index(df2)
        vol_col = "volume" if "volume" in df2.columns else None
        return finance_ohlcv_to_proxies(df2, price_col="close", volume_col=vol_col)

    raise ValueError("Unsupported real-data format. Provide proxies columns or OHLCV columns.")


def _scenario_slices(
    df: pd.DataFrame,
    *,
    n: int,
    segment_len: int,
    seed: int,
    include_full: bool,
) -> List[Tuple[str, pd.DataFrame]]:
    df = df.sort_index()
    n = int(max(1, n))
    include_full = bool(include_full)

    out: List[Tuple[str, pd.DataFrame]] = []
    if include_full:
        out.append(("full", df))

    if n <= 1:
        return out

    rng = np.random.default_rng(int(seed))
    n_points = int(len(df))
    if n_points < 10:
        return out

    if segment_len <= 0:
        # Auto: a stable slice size that stays lightweight but meaningful.
        segment_len = int(np.clip(n_points // 3, 200, 2000))

    segment_len = int(min(segment_len, n_points))
    if segment_len < 10:
        return out

    # Pick start indices without replacement when possible.
    max_start = max(0, n_points - segment_len)
    if max_start == 0:
        # Only one possible slice
        out.append(("seg01", df.iloc[:segment_len]))
        return out

    starts = rng.integers(low=0, high=max_start + 1, size=int(n)).tolist()
    for i, s in enumerate(starts, start=1):
        seg = df.iloc[int(s) : int(s) + int(segment_len)]
        out.append((f"seg{i:02d}", seg))

    return out


def _report_row(report_dict: Dict[str, object], *, dataset: str, scenario: str, n_points: int) -> Dict[str, object]:
    targets = dict(report_dict.get("targets") or {})
    stability = dict(report_dict.get("stability") or {})
    verdict = dict(report_dict.get("verdict") or {})
    return {
        "dataset": dataset,
        "scenario": scenario,
        "n_points": int(n_points),
        "stability_state": (verdict.get("dimensions") or {}).get("stability", {}).get("state", ""),
        "stability_score": float(stability.get("spearman_mean_risk", float("nan"))),
        "rule_execution_gap_mean": float(targets.get("rule_execution_gap_mean", float("nan"))),
        "prevented_exceedance_rel": float(targets.get("prevented_exceedance_rel", float("nan"))),
        "prevented_topk_excess_rel": float(targets.get("prevented_topk_excess_rel", float("nan"))),
        "prevented_primary_metric": str(targets.get("prevented_primary_metric", "")),
        "prevented_primary_value": float(targets.get("prevented_primary_value", float("nan"))),
        "verdict_label": str(verdict.get("label", "")),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Real data smoke: convert + audit + HTML report.")
    ap.add_argument("--out-dir", type=str, default="_ci_out/real_data")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--scenarios", type=int, default=1, help="Number of additional scenario slices per dataset (full dataset is included by default).")
    ap.add_argument("--segment-len", type=int, default=0, help="Segment length for scenario slices. 0 = auto.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--inputs", nargs="*", default=None, help="Optional explicit list of CSV/Parquet inputs (proxies or OHLCV). If set, auto-discovery is skipped.")
    ap.add_argument("--include-root-proxies", action="store_true", help="Also include repo-root *_proxies.csv files when auto-discovering inputs.")
    ap.add_argument("--no-full", action="store_true", help="Do not include the full dataset report, only scenario slices.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    thresholds = _load_thresholds(window=int(args.window))

    if args.inputs:
        inputs = [Path(p) for p in args.inputs]
    else:
        inputs = _discover_inputs(repo_root, include_root_proxies=bool(args.include_root_proxies))
    if not inputs:
        note = {
            "note": "No real data files found under data/real, data/real_fixtures (and optionally repo-root *_proxies.csv).",
            "how_to": "Drop CSV/Parquet with proxies or OHLCV columns.",
            "hint": "To test multiple real scenarios, use --scenarios N and optionally --segment-len.",
        }
        (out / "real_data_smoke_note.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
        return 0

    rows: List[Dict[str, object]] = []
    for p in inputs:
        dataset = p.stem
        try:
            raw = _read_any(p)
            proxies = _to_proxies(raw)
        except Exception as e:
            (out / f"error_{dataset}.txt").write_text(str(e), encoding="utf-8")
            continue

        slices = _scenario_slices(
            proxies,
            n=int(args.scenarios),
            segment_len=int(args.segment_len),
            seed=int(args.seed),
            include_full=not bool(args.no_full),
        )

        for scenario, df_s in slices:
            name = f"{dataset}__{scenario}"
            rep = build_audit_report(df_s, dataset_name=name, delta_d_window=int(args.window), thresholds=thresholds)
            json_path = out / f"audit_report_{name}.json"
            write_audit_report(rep, json_path)

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            render_audit_html(
                report_dict=payload,
                df=df_s,
                out_html=out / f"audit_report_{name}.html",
                thresholds=thresholds,
                delta_d_window=int(args.window),
            )
            rows.append(_report_row(payload, dataset=dataset, scenario=scenario, n_points=len(df_s)))

    if rows:
        df_sum = pd.DataFrame(rows).sort_values(["dataset", "scenario"]).reset_index(drop=True)
        df_sum.to_csv(out / "real_data_smoke_summary.csv", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
