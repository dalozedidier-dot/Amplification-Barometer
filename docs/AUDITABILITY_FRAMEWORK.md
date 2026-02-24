# Auditability Framework: Making Results Immutable

**Version:** v0.1.0
**Date:** 2026-02-24

---

## Purpose

Once you've run an audit, how do you prove it wasn't rewritten, recomputed, or cherry-picked later?

This framework makes audit runs **immutable and timestamped**:
- Every run gets a manifest with version, hash, and date
- All results go into an append-only log (never overwritten)
- External parties can independently verify authenticity

**Goal:** Transform "trust our last audit" into "here's the notarized record."

---

## Architecture

### Per-Run Manifest

**File:** `reports/audits/{run_id}/manifest.json`

```json
{
  "run_id": "20260224_type_iii_bifurcation_base",
  "timestamp": "2026-02-24T15:47:23.123Z",
  "barometer_version": "0.1.0",
  "git_commit": "73676cc",
  "dataset": {
    "filename": "data/canonical_scenarios/type_iii_bifurcation_base.csv",
    "sha256": "a1b2c3d4e5f6...",
    "rows": 200,
    "columns": 23
  },
  "spec": {
    "proxies_yaml": "docs/proxies.yaml",
    "version": "v1.0",
    "sha256": "f6e5d4c3b2a1..."
  },
  "audit_parameters": {
    "turnover_target": 0.05,
    "gap_target": 0.05,
    "window_sizes": [5, 7, 9]
  },
  "output": {
    "json": "alignment_audit.json",
    "json_sha256": "x9y8z7w6v5u4...",
    "md": "alignment_audit.md",
    "md_sha256": "r3q2p1o0n9m8..."
  }
}
```

**Components:**
- **run_id:** Unique identifier (timestamp + scenario name)
- **timestamp:** ISO-8601, UTC (can't be faked after the fact)
- **barometer_version:** Code version running
- **git_commit:** Exact code commit hash
- **dataset.sha256:** Hash of input data (proves no tampering)
- **spec.sha256:** Hash of proxy spec used
- **output.*_sha256:** Hash of results (proves no post-hoc modification)

### Global Append-Only Log

**File:** `history.jsonl` (in repo root)

```jsonl
{"timestamp": "2026-02-24T15:00:00Z", "run_id": "20260224_type_i_noise_base", "verdict": "type_I_noise", "stability": "ok", "status": "published"}
{"timestamp": "2026-02-24T15:30:00Z", "run_id": "20260224_type_ii_oscillations_base", "verdict": "type_II_oscillations", "stability": "ok", "status": "published"}
{"timestamp": "2026-02-24T15:47:23Z", "run_id": "20260224_type_iii_bifurcation_base", "verdict": "type_III_bifurcation", "stability": "ok", "status": "published"}
```

**Rules:**
- **Append-only:** New lines added to end, never modified
- **No deletions:** Failed runs stay (marked with `status: failed`)
- **Reverse chronological:** Newest first when reading (but appended last)
- **Immutable once published:** A published run cannot be unpublished

---

## Workflow: Audit → Manifest → Log

### Step 1: Run Audit

```bash
python3 tools/run_alignment_audit.py \
  --dataset data/canonical_scenarios/type_iii_bifurcation_base.csv \
  --name 20260224_type_iii_bifurcation_base \
  --out-dir reports/audits/
```

**Output:**
- `alignment_audit.json`
- `alignment_audit.md`

### Step 2: Generate Manifest

```bash
python3 tools/create_audit_manifest.py \
  --dataset data/canonical_scenarios/type_iii_bifurcation_base.csv \
  --run-id 20260224_type_iii_bifurcation_base \
  --audit-dir reports/audits/20260224_type_iii_bifurcation_base \
  --out-dir reports/audits/20260224_type_iii_bifurcation_base
```

**Output:**
- `manifest.json` (with all hashes, timestamp, versions)

### Step 3: Append to History Log

```bash
python3 tools/append_to_history.py \
  --manifest reports/audits/20260224_type_iii_bifurcation_base/manifest.json \
  --verdict type_III_bifurcation \
  --status published
```

**Output:**
- `history.jsonl` updated (new line appended, never edited)

### Step 4: Publish

Once a run is in the log, it's public record.

---

## Verification: How to Check Authenticity

### Check 1: Manifest Integrity

```bash
# Verify dataset hash
sha256sum data/canonical_scenarios/type_iii_bifurcation_base.csv
# Compare to manifest.json dataset.sha256

# Verify output hash
sha256sum reports/audits/.../alignment_audit.json
# Compare to manifest.json output.json_sha256
```

### Check 2: Code Version

```bash
# Verify git commit
git log --oneline | grep 73676cc
# (Should match barometer_version.git_commit in manifest)

# Verify proxy spec version
git show 73676cc:docs/proxies.yaml | sha256sum
# (Should match spec.sha256 in manifest)
```

### Check 3: Log Integrity

```bash
# Check that line for run_id is in history.jsonl
grep "20260224_type_iii_bifurcation_base" history.jsonl

# Verify append-only (count lines, should be monotonically increasing)
wc -l history.jsonl  # Check in multiple snapshots
```

### Check 4: Cryptographic Notarization (Optional, Future)

If you want cryptographic proof that audit occurred at a specific time:
- Sign manifest.json with GPG key
- Timestamp via external service (e.g., timestamping authority)
- Include signature in manifest

```json
{
  "manifest": {...},
  "signature": {
    "gpg_key_id": "...",
    "signed_at": "2026-02-24T15:47:23Z",
    "timestamp_authority": "https://rfc3161.example.com/",
    "timestamp_proof": "..."
  }
}
```

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Audit & Manifest

on: [push]

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run audit
        run: python3 tools/run_alignment_audit.py ...

      - name: Create manifest
        run: python3 tools/create_audit_manifest.py ...

      - name: Append to history
        run: python3 tools/append_to_history.py ...

      - name: Commit manifest + history
        run: |
          git add reports/audits/ history.jsonl
          git commit -m "Audit run: ..."
          git push
```

**Result:** Every push triggers audit, manifest creation, and history append.
History becomes part of the repository record.

---

## Falsifiability & Break Conditions

### False Claim 1: "We never ran this test"
**Proof:** Show line in history.jsonl with timestamp

### False Claim 2: "Results changed after publication"
**Proof:** Output hash in manifest matches file checksum

### False Claim 3: "We're using different code than we say"
**Proof:** Git commit hash corresponds to code version

### False Claim 4: "We cherry-picked good runs and hid bad ones"
**Proof:** All runs (passed and failed) in history.jsonl with timestamps

---

## Tools Reference

### `tools/create_audit_manifest.py` (IMPLEMENT)
```bash
usage: create_audit_manifest.py
  --dataset <csv>
  --run-id <str>
  --audit-dir <dir>
  --out-dir <dir>
```
Creates manifest.json with hashes and metadata.

### `tools/append_to_history.py` (IMPLEMENT)
```bash
usage: append_to_history.py
  --manifest <json>
  --verdict <str>
  --status [published|failed|review]
```
Appends single line to history.jsonl (append-only).

### `tools/verify_audit.py` (IMPLEMENT)
```bash
usage: verify_audit.py
  --manifest <json>
```
Checks manifest integrity: hashes, versions, timestamps.

---

## Deployment Checklist

- [ ] Manifest JSON schema defined
- [ ] CLI tools created (create_audit_manifest, append_to_history, verify_audit)
- [ ] GitHub Actions workflow added
- [ ] history.jsonl initialized in repo
- [ ] Documentation (this file) completed
- [ ] First audit run goes through full pipeline
- [ ] history.jsonl and manifests committed to repo
- [ ] External verification script tested

---

## Version History

| Version | Date | Notes |
|---------|------|-------|
| v0.1.0 | 2026-02-24 | Initial framework. Manifest + append-only log. |

---

## Related Documents

- `docs/DEFINITION_OF_DONE.md` – Truth contract
- `docs/calibration_protocol.md` – Audit parameters
- `docs/PUBLIC_TEST_MATRIX.md` – Test checklist

