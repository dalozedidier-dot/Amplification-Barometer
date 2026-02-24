# Public Test Matrix: How to Know the Barometer Says True

**Version:** v0.1.0
**Published:** 2026-02-24
**Scope:** Minimal but sufficient falsifiability

---

## Purpose

This document publishes a **public checklist** of tests that, when passed, allow the Amplification Barometer to claim "it tells the truth."

This is **not** an internal spec. This is our commitment to the outside world: here are the specific tests we run; here's what falsifies our claim.

---

## Test Matrix: Minimal Sufficient Set

| # | Test | Input | Pass Criterion | Fail If | Reference |
|---|------|-------|---|---|---|
| **S1** | **Stability under variation** | Synthetic noise scenario | Spearman ≥ 0.85, Jaccard topk ≥ 0.80 across windows [5,7,9] | Either metric < threshold | `calibration_protocol.md` |
| **S2** | **Proxy ranges enforced** | Any real or synthetic dataset | Zero out-of-range proxy values (except documented exemptions) | Any proxy outside [expected_lo, expected_hi] | `proxies.yaml` + `proxy_specs/` |
| **S3** | **Composites are finite** | Any dataset | P, O, E, R, G, @(t), Δd(t) all finite (no NaN, inf) | Any NaN or inf in outputs | `alignment_audit.py` |
| **S4** | **E is monotone-increasing** | Any dataset | E_stock cumsum never decreases | Negative dE in any window | `alignment_audit.py` |
| **S5** | **Type I: Noise contained** | Synthetic pure noise (200 steps) | @(t) mean < 2.0, E_stock final ≈ initial, regime="type_I_noise" | E drifts or regime ≠ type_I | `canonical_scenarios/type_i_noise_base.csv` |
| **S6** | **Type II: Oscillation signature** | Synthetic oscillations (200 steps) | Δd(t) sign alternates ≥60% tail, R_tail mean > 0.40, regime="type_II_oscillations" | Any signature metric fails | `canonical_scenarios/type_ii_oscillations_base.csv` |
| **S7** | **Type III: Bifurcation signature** | Synthetic bifurcation (200 steps) | @(t) divergence ≥50%, E_stock irreversibility ≥0.90, R_tail < 0.30, regime="type_III_bifurcation" | Any metric below threshold | `canonical_scenarios/type_iii_bifurcation_base.csv` |
| **S8** | **Noise robustness** | All 4 scenario types × 5 noise levels (20 total) | Regime verdict same across all noise variants | Regime changes with noise level | `canonical_scenarios/` (all files) |
| **AG1** | **O-family bias detection** | Dataset with artificial O boost (+15%) from t=150 | Gaming verdict="fail" | Gaming verdict="ok" | `run_alignment_audit.py` (anti_gaming dimension) |
| **AG2** | **Volatility clamp detection** | Dataset with P std suppressed by 50% from t=150 | Gaming verdict="fail" | Gaming verdict="ok" | `run_alignment_audit.py` (anti_gaming dimension) |
| **AG3** | **Out-of-range gaming detection** | Dataset with proxies pushed >expected_hi from t=150 | Gaming verdict="fail" | Gaming verdict="ok" | `run_alignment_audit.py` (anti_gaming dimension) |
| **AG4** | **Multi-proxy coordinated attack** | Dataset with simultaneous O boost + P vol clamp from t=150 | Gaming verdict="fail" | Gaming verdict="ok" | `run_alignment_audit.py` (anti_gaming dimension) |
| **AG5** | **Reporting delay detection** | Dataset with G signals artificially delayed 20 steps from t=150 | Gaming verdict="fail" | Gaming verdict="ok" | `run_alignment_audit.py` (anti_gaming dimension) |

---

## Execution Checklist

To claim "dit vrai" for a given dataset or system:

### Pre-Audit
- [ ] Proxy spec selected (baseline or sector: finance/ai/critical_infrastructure)
- [ ] All 22 proxies present in dataset
- [ ] Data frequency matches spec (daily/weekly/monthly as applicable)
- [ ] Date column parseable (ISO-8601)

### Run Audit
```bash
python3 tools/run_alignment_audit.py \
  --dataset <data.csv> \
  --name <run_name> \
  --out-dir reports/
```

### Check Verdict
```json
{
  "verdict": {
    "dimensions": {
      "stability": "ok" ✓ S1
      "proxy_ranges": "ok" ✓ S2
      "regime_signature": "type_I_noise" or "type_II_oscillations" or "type_III_bifurcation" ✓ S5/S6/S7
      "anti_gaming": "ok" ✓ AG1-AG5
    }
  }
}
```

### Publish Claim

**You can publicly claim "dit vrai" if all of:**
- `stability: ok` ✓
- `proxy_ranges: ok` ✓
- `regime_signature ∈ [type_I_noise, type_II_oscillations, type_III_bifurcation]` ✓
- `anti_gaming: ok` ✓

**You must claim "suspect" if:**
- `regime_signature` detected but `anti_gaming: fail` (may be gamed)
- Any optional governance targets missed (not mandatory in Phase 1)

**You must claim "not ready" if:**
- `stability: fail` (core composite unstable)
- `proxy_ranges: fail` (data quality issue)
- Any finite/monotonicity check fails

---

## Falsification Examples (Proof We're Scientific)

### Falsification 1: Construct Validity
**Scenario:** We claim stability ≥ 0.85, but run test **S1** on pure noise.
**Expected:** Pass (Spearman ≥ 0.85).
**Falsified if:** Spearman < 0.85 → **We admit the composite is unstable and fix it.**

### Falsification 2: Stress Signature Failure
**Scenario:** We claim Type III bifurcation has @(t) tail divergence ≥ 50%.
**Expected:** Test **S7** on synthetic bifurcation passes.
**Falsified if:** @(t) divergence < 30% → **We acknowledge signal is too weak and recalibrate.**

### Falsification 3: Anti-Gaming Vulnerability
**Scenario:** We claim we detect O-bias attacks.
**Expected:** Test **AG1** on biased dataset fails gaming verdict.
**Falsified if:** Gaming verdict="ok" despite O boost → **We've been gamed and must fix detection.**

### Falsification 4: Noise Robustness Failure
**Scenario:** We claim robustness to noise.
**Expected:** Test **S8** on all 20 noise variants: all show same regime.
**Falsified if:** Regime changes (e.g., type_I → type_II) with noise level → **System is sensitive to noise; not reliable.**

---

## What This Matrix DOES Cover

✅ Construct validity (composites stable, finite, well-formed)
✅ Canonical stress signatures (Type I, II, III discriminable)
✅ Anti-gaming robustness (5 attack vectors detected)
✅ Noise robustness (regime verdict consistent)

## What This Matrix DOES NOT (Yet) Cover

❌ Real-world validation on timestamped events (Phase 5)
❌ L_cap vs L_act separation (Phase 4)
❌ Sector-specific performance (Phase 4)
❌ Audit auditability & manifest integrity (Phase 7)

Phase 1-2 complete these. Phases 3-4 expand scope.

---

## How to Use This Document

### For Internal Teams
Copy this checklist into your CI/CD pipeline:
```python
tests = [
    "stability_audit_rank()",      # S1
    "validate_proxy_ranges()",      # S2
    "check_finitude()",             # S3
    "check_e_monotonic()",          # S4
    "run_canonical_type_i()",       # S5
    "run_canonical_type_ii()",      # S6
    "run_canonical_type_iii()",     # S7
    "run_noise_variants()",         # S8
    "run_anti_gaming_suite()",      # AG1-AG5
]

for test in tests:
    result = test()
    if not result.passes:
        print(f"FAIL: {test} -- claim 'not_ready'")
        exit(1)

print("SUCCESS: All tests pass -- claim 'dit vrai' for this dataset")
```

### For External Auditors
- Download dataset and run: `python3 tools/run_alignment_audit.py --dataset <your_data.csv>`
- Check JSON output for all dimensions = "ok"
- If any dimension ≠ "ok", the "dit vrai" claim is false
- Report independently to stakeholders

### For Researchers
- Tests S1-S8 are **falsifiable hypotheses** about barometer behavior
- If any test fails, the barometer's core claim (construct validity) is invalidated
- Publish counter-examples to challenge the claim
- We commit to either:
  1. Explaining why the test is wrong, or
  2. Fixing the barometer

---

## Version History & Regression Testing

**Current version:** v0.1.0 (locked with Phase 1-2 implementation)

If barometer code changes:
- Re-run all S1-S8 + AG1-AG5
- If any test **fails**, regression detected → version bump required
- If all pass, increment patch (e.g., v0.1.0 → v0.1.1)

**No releases without passing full matrix.**

---

## Related Documents

- `docs/DEFINITION_OF_DONE.md` – Detailed 3D truth contract
- `docs/calibration_protocol.md` – Scenario specs & thresholds
- `data/canonical_scenarios/` – Test data (24 CSV files)
- `docs/proxy_specs/` – Sector-specific ranges
- `tools/run_alignment_audit.py` – Main audit entrypoint
- `tools/run_anti_gaming_suite.py` – Attack vector testing

---

## Call to Action

**To stakeholders:** If you see a dataset where this barometer produces a verdict, you now have the tools to verify or refute that claim independently. Here are the exact tests. Run them. Report findings.

**To researchers:** These tests are your hypothesis set. Falsify them. We will update the barometer accordingly.

**To implementers:** This is your SLA. Every deployment must pass this matrix. Every code change must show no regression.

