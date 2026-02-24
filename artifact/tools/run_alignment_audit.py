from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from amplification_barometer.alignment_audit import run_alignment_audit, write_alignment_outputs


def main() -> int:
    ap = argparse.ArgumentParser(description="Run theory-to-audit alignment checks (proxy ranges, stability, signatures).")
    ap.add_argument("--dataset", required=True, help="CSV file with proxies.")
    ap.add_argument("--name", default="dataset", help="Name prefix for outputs.")
    ap.add_argument("--proxies-yaml", default="docs/proxies.yaml", help="Proxy spec yaml.")
    ap.add_argument("--out-dir", default="_ci_out", help="Output directory.")
    ap.add_argument("--turnover-target", type=float, default=0.05, help="Turnover target for governance.")
    ap.add_argument("--gap-target", type=float, default=0.05, help="Rule-execution gap target for governance.")
    args = ap.parse_args()

    df = pd.read_csv(args.dataset)
    # If there is a date column, keep it but do not require it.
    report = run_alignment_audit(
        df,
        proxies_yaml=args.proxies_yaml,
        turnover_target=args.turnover_target,
        gap_target=args.gap_target,
    )
    write_alignment_outputs(report, out_dir=args.out_dir, name=args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
