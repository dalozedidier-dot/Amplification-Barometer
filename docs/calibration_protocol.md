# Calibration Protocol v0.1.0

**Effective:** 2026-02-24
**Status:** Published & Locked

---

## Purpose

This document locks down the calibration of the Amplification Barometer. It specifies:

1. **Canonical stress scenarios** used for testing
2. **Fixed thresholds & tolerances** for regime detection
3. **Expected signatures** (Type I, II, III) and their empirical benchmarks
4. **Reproducibility criteria** to ensure audits are repeatable

Once published, this protocol applies to all audits. Changes require version bump + justification.

---

## Canonical Stress Scenarios

### Scenario 1: Type I (Noise)

**Intent:** Pure stochastic variation, no systematic drift.

**Generation:**
- Duration: 200 steps
- All proxies: Normal(μ=baseline, σ=0.1 × baseline)
- No trends, no shocks

**Expected signatures:**
- `@(t)` mean < 2.0, std < 0.5
- `E_stock` final ≈ E_stock initial (no cumulative drift)
- `regime_signature` = "type_I_noise"
- `Δd(t)` sign alternates (no persistent trend)

**Pass threshold:**
- Spearman stability ≥ 0.85
- Jaccard topk ≥ 0.80

**Test:** Run audit on this scenario. If `regime_signature != type_I_noise`, FAIL.

---

### Scenario 2: Type II (Oscillations)

**Intent:** Cyclical stress with recovery; system oscillates around equilibrium.

**Generation:**
- Duration: 200 steps
- Inject P oscillations: `scale_proxy += 0.3 * sin(2π * t / period)` with period=40 steps
- O responds with lag: `stop_proxy -= 0.15 * sin(2π * (t-10) / period)` (10-step lag)
- E grows slowly: `impact_proxy += 0.02 * t` (linear)
- R fluctuates: `recovery_time_proxy` oscillates with same period

**Expected signatures:**
- `Δd(t)` sign **alternates** ≥60% of tail window (last 30 steps)
- `R_tail_mean` (last 30 steps) > 0.40 (recovery partial but present)
- `O_saturation_low_frac` < 0.5 (O is not saturated at bottom)
- `at_divergence_tail_frac` < 0.5 (@ does not persistently diverge)
- `regime_signature` = "type_II_oscillations"

**Pass threshold:**
- At least 1 full cycle visible
- Alternating Δd pattern clearly visible

**Test:** Run audit. If `regime_signature != type_II_oscillations` OR `Δd` sign-changes < 60%, FAIL.

---

### Scenario 3: Type III (Bifurcation)

**Intent:** System enters irreversible stress; composites diverge, E accumulates, R degrades.

**Generation:**
- Duration: 200 steps
- Phase 1 (t=0-80): Baseline oscillation like Type II
- Phase 2 (t=80-200): **Structural shift**
  - `scale_proxy` jumps +0.5 and stays high (step function)
  - `speed_proxy` drifts upward continuously
  - `stop_proxy` collapses (steps down gradually to 0.2)
  - `E_level` jumps then accumulates (non-reversible)
  - `R_level` degrades and doesn't recover

**Expected signatures:**
- `at_divergence_tail_frac` ≥ 0.50 (@ consistently above 90th percentile in tail)
- `persistence_dE_dt_tail_pos_frac` ≥ 0.60 (dE/dt positive in tail, accumulating)
- `e_irreversibility_ratio` ≥ 0.90 (E_stock[-1] / max(E_stock) close to 1)
- `r_tail_mean` < 0.30 (R depressed in tail)
- `regime_signature` = "type_III_bifurcation"

**Pass threshold:**
- All 4 signature metrics above thresholds
- Visual inspection confirms divergence + accumulation

**Test:** Run audit. If any metric below threshold OR `regime_signature != type_III_bifurcation`, FAIL.

---

### Scenario 4: Hybrid (Type II → Type III transition)

**Intent:** System starts oscillating, then bifurcates; tests transition detection.

**Generation:**
- Duration: 300 steps
- Phase 1 (t=0-150): Type II oscillations
- Phase 2 (t=150-300): Gradually morph to Type III
  - P amplitude increases
  - O recovery weakens
  - E accumulation accelerates
  - R structural degradation begins

**Expected signatures:**
- Phase 1 (t=0-150): `regime_signature` = "type_II_oscillations"
- Phase 2 (t=200-300): `regime_signature` = "type_III_bifurcation"
- Transition visible: transition point (regime change) occurs around t=150±30

**Pass threshold:**
- Regime changes between phases
- No false oscillation in Phase 2 tail

**Test:** Run audit on full window. If regime doesn't change between phases, FAIL.

---

## Scenario Variants (Noise Robustness)

For each of the 4 canonical scenarios, generate 5 noise variants:

**Noise levels:** σ_noise ∈ {0.05, 0.10, 0.15, 0.20, 0.25} × baseline proxy std

**Rule:** Verdict should be same across all noise levels (robust to noise).

**Test:** Run all 20 variants (4 × 5). Verdicts must agree on regime_signature.

---

## Fixed Thresholds for Regime Detection

These thresholds are **locked** and must not change without version bump.

### Type I Detection

```python
regime = "type_I_noise" if (
    at_divergence_tail_frac < 0.30 and
    persistence_dE_dt_tail_pos_frac < 0.50 and
    e_irreversibility_ratio < 0.70
)
```

### Type II Detection

```python
regime = "type_II_oscillations" if (
    at_divergence_tail_frac < 0.50 and
    persistence_dE_dt_tail_pos_frac >= 0.50 and
    e_irreversibility_ratio < 0.80
)
```

### Type III Detection

```python
regime = "type_III_bifurcation" if (
    at_divergence_tail_frac >= 0.50 and
    persistence_dE_dt_tail_pos_frac >= 0.60 and
    e_irreversibility_ratio >= 0.90
)
```

Default: "type_I_noise"

---

## Anti-Gaming Threshold

**Gaming detection:** Gaming suspicion score = (detected attacks) / (total attacks)

- **Pass:** gaming_suspicion_score ≥ 1.0 (all attacks detected)
- **Fail:** gaming_suspicion_score < 1.0 (any attack undetected)

---

## Stability Thresholds

**Pass:** Spearman min ≥ 0.85 AND Jaccard topk ≥ 0.80

(These are conservative: 85% rank correlation + 80% topk overlap across window variations [5, 7, 9])

---

## Governance Targets

**Maturity thresholds** (optional in Phase 1, mandatory in Phase 2):

- `rule_execution_gap` ≤ 0.05 (5%)
- `control_turnover` ≤ 0.05 (5%)

Systems exceeding these flags as governance risk.

---

## CI Artifact: Expected vs Observed

For each canonical scenario run:

```json
{
  "scenario": "type_ii_oscillations_base",
  "seed": 42,
  "noise_level": 0.0,
  "expected": {
    "regime_signature": "type_II_oscillations",
    "stability": "ok",
    "anti_gaming": "ok"
  },
  "observed": {
    "regime_signature": "type_II_oscillations",
    "stability": "ok",
    "anti_gaming": "ok"
  },
  "match": true,
  "timestamp": "2026-02-24T...",
  "audit_json": "reports/calibration/scenario_output.json"
}
```

Publish a CSV table with all 20 runs (4 scenarios × 5 noise levels).

---

## Reproducibility & Versioning

**Calibration version:** v0.1.0 (locked with this document)

**Code version pinned:** src/amplification_barometer/ commit hash in manifest

**Scenario seeds:** Fixed seeds for deterministic generation (see data/canonical_scenarios/README.md)

**Any change** to thresholds, scenarios, or logic requires:
1. New calibration version (e.g., v0.2.0)
2. Re-run all 20 variants
3. Publish new expected vs observed table
4. Justify in changelog

---

## Related Documents

- `docs/DEFINITION_OF_DONE.md` – Truth contract
- `data/canonical_scenarios/README.md` – Scenario generation code
- `docs/proxy_specs/` – Sector-specific specs
- `tools/run_alignment_audit.py` – Main audit entrypoint

