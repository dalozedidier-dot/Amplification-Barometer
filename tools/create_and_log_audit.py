#!/usr/bin/env python3
"""
Create audit manifest and append to history log.

This tool:
1. Generates a manifest.json for an audit run (with hashes, versions, timestamps)
2. Appends a record to history.jsonl (append-only log)
3. Ensures auditability: every run is timestamped, hashed, and logged

Usage:
  python3 tools/create_and_log_audit.py \
    --dataset <csv> \
    --run-id <str> \
    --audit-json <json> \
    --verdict <verdict> \
    --status [published|failed]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def compute_sha256(file_path: str | Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_file_info(file_path: str | Path) -> Dict[str, Any]:
    """Get file metadata (size, hash, row/col count if CSV)."""
    path = Path(file_path)
    info = {
        "filename": path.name,
        "sha256": compute_sha256(file_path),
    }

    # If CSV, count rows/columns
    if path.suffix.lower() == ".csv":
        try:
            import pandas as pd

            df = pd.read_csv(file_path)
            info["rows"] = len(df)
            info["columns"] = len(df.columns)
        except Exception:
            pass

    return info


def create_manifest(
    dataset_path: str,
    run_id: str,
    audit_json: str,
    proxies_yaml: str = "docs/proxies.yaml",
    turnover_target: float = 0.05,
    gap_target: float = 0.05,
) -> Dict[str, Any]:
    """Create a manifest for an audit run."""
    timestamp = datetime.now(timezone.utc).isoformat()
    git_commit = get_git_commit()

    manifest = {
        "version": "v1.0",
        "run_id": run_id,
        "timestamp": timestamp,
        "barometer_version": "0.1.0",
        "git_commit": git_commit,
        "dataset": get_file_info(dataset_path),
        "spec": get_file_info(proxies_yaml),
        "audit_parameters": {
            "turnover_target": turnover_target,
            "gap_target": gap_target,
            "window_sizes": [5, 7, 9],
        },
        "output": get_file_info(audit_json),
    }

    return manifest


def append_to_history(
    manifest: Dict[str, Any],
    verdict: str,
    status: str = "published",
    note: Optional[str] = None,
) -> None:
    """Append a run record to history.jsonl (append-only log)."""
    history_path = Path("history.jsonl")

    record = {
        "timestamp": manifest["timestamp"],
        "run_id": manifest["run_id"],
        "verdict": verdict,
        "status": status,
    }

    if note:
        record["note"] = note

    # Append (never overwrite)
    with open(history_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Create audit manifest and log to history.")
    ap.add_argument("--dataset", required=True, help="Dataset CSV file.")
    ap.add_argument("--run-id", required=True, help="Unique run identifier.")
    ap.add_argument("--audit-json", required=True, help="Audit output JSON file.")
    ap.add_argument("--proxies-yaml", default="docs/proxies.yaml", help="Proxy spec YAML.")
    ap.add_argument("--verdict", required=True, help="Verdict string (e.g., type_I_noise, type_II_oscillations).")
    ap.add_argument("--status", default="published", choices=["published", "failed"], help="Run status.")
    ap.add_argument("--note", default=None, help="Optional note for history log.")
    ap.add_argument("--out-manifest", default=None, help="Output manifest path (default: run_id/manifest.json).")

    args = ap.parse_args()

    # Create manifest
    print(f"Creating manifest for run: {args.run_id}...")
    manifest = create_manifest(
        args.dataset,
        args.run_id,
        args.audit_json,
        proxies_yaml=args.proxies_yaml,
    )

    # Write manifest
    manifest_path = Path(args.out_manifest or f"reports/audits/{args.run_id}/manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"✓ Manifest written: {manifest_path}")

    # Append to history
    print(f"Appending to history log...")
    append_to_history(manifest, verdict=args.verdict, status=args.status, note=args.note)
    print(f"✓ History log updated: history.jsonl")

    # Summary
    print(f"\n✓ Audit logged:")
    print(f"  Run ID: {args.run_id}")
    print(f"  Timestamp: {manifest['timestamp']}")
    print(f"  Verdict: {args.verdict}")
    print(f"  Status: {args.status}")
    print(f"  Git Commit: {manifest['git_commit']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
