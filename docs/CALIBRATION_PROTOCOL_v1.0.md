# Calibration Protocol: Frozen Thresholds

**Version:** v1.0.0
**Status:** FROZEN (2026-02-24)
**Git Commit:** 73676cc

---

## Purpose

This document freezes the calibration thresholds **before analyzing any real cases**.

If thresholds change after looking at results → **tuning bias**.
If thresholds are pre-committed → **falsifiable and defensible**.

---

## Canonical Scenario Definitions

### Type I: Noise (Baseline)
**Definition:** No true stress. System oscillates around stable equilibrium.

**Ground Truth:**
- P proxies: constant or very low drift
- O proxies: stable, reactive only to noise
- E stock: bounded, no accumulation
- R proxies: maintained
- G proxies: normal operation

**Expected Barometer Signature:**
- **@ mean:** 0.5 ± 0.3
- **E irreversibility:** < 0.30
- **Persistence(dE/dt > 0):** < 0.40
- **Regime classification:** Type I (regime_divergence_score < 0.30)
- **Stability (Spearman min):** ≥ 0.85
- **Anti-gaming:** all passes

**Tolerance:**
- ±5% noise perturbation → same regime classification
- ±10% proxy range change → same regime classification

**Falsification Criterion:**
If Type I scenario shows @ mean > 1.0 or E irreversibility > 0.40 → framework fails

---

### Type II: Oscillations (Damped Recovery)
**Definition:** Stress builds, O responds, system recovers, repeat.

**Ground Truth:**
- P proxies: cyclical elevation (peaks then decay)
- O proxies: respond to peaks (gap-filling)
- E stock: increases then stabilizes (bounded)
- R proxies: degrade during peaks, recover
- G proxies: reactive, control frequency > 0

**Expected Barometer Signature:**
- **@ mean:** 0.8 ± 0.4
- **E irreversibility:** 0.30–0.65
- **Persistence(dE/dt > 0):** 0.40–0.70
- **Delta-D std:** > 0.10 (oscillations visible)
- **Regime classification:** Type II (regime_divergence_score 0.30–0.65)
- **Stability (Spearman min):** ≥ 0.85
- **Anti-gaming:** all passes

**Tolerance:**
- ±5% noise → persists as Type II (may shift to Type I or III, but not jump to extreme)
- Oscillation period 5–20 days acceptable

**Falsification Criterion:**
If Type II shows no oscillations in delta_d or E irreversibility outside [0.30, 0.65] → framework fails

---

### Type III: Bifurcation (Divergence to Collapse)
**Definition:** O capabilities saturate, E accumulates monotonically, system tips.

**Ground Truth:**
- P proxies: continuous growth (no reversion)
- O proxies: initially respond but asymptote (can't keep up)
- E stock: monotone increase, no reversion (irreversible)
- R proxies: degrade, stay depressed
- G proxies: governance gaps emerge (exemptions, delays)

**Expected Barometer Signature:**
- **@ mean:** 1.2 ± 0.5
- **E irreversibility:** ≥ 0.75
- **Persistence(dE/dt > 0):** ≥ 0.70
- **@_tail divergence (last 30%):** ≥ 0.50 (@ stays high at end)
- **R_tail_mean:** < 0.60 (recovery exhausted)
- **O_saturation:** L_cap detects asymptote
- **Regime classification:** Type III (regime_divergence_score > 0.65)
- **Stability (Spearman min):** ≥ 0.85
- **Anti-gaming:** all passes

**Tolerance:**
- ±5% noise → persists as Type III (may shift from II to III, but not to I)
- E may oscillate early, but must trend monotone overall (last 40% of series)

**Falsification Criterion:**
If Type III shows E irreversibility < 0.70 or @ doesn't stay elevated → framework fails

---

### Type IV: Hybrid II→III (Transition)
**Definition:** Starts Type II (oscillating, recovering), then transitions to Type III (bifurcation).

**Ground Truth:**
- Phase 1 (days 1-40): Oscillations with recovery
- Phase 2 (days 40-80): Recovery weakens, divergence emerges
- Phase 3 (days 80-120): Full bifurcation

**Expected Barometer Signature:**
- **@ first half:** 0.8–1.0 (oscillatory)
- **@ second half:** 1.2–1.5 (divergent)
- **E irreversibility:** 0.40 early → 0.85 late
- **Delta-D:** large early, diminishes as divergence dominates
- **Regime classification:** Initially Type II, then Type III (tipping point detectable around day 40)
- **Stability (Spearman min):** ≥ 0.85

**Tolerance:**
- Tipping point can occur ±5 days (day 35–45)

**Falsification Criterion:**
If Hybrid scenario doesn't show transition or shows both simultaneously → framework fails

---

## Proxy Range Specifications

### O-Family Proxies (Detection Capacity)

| Proxy | Min | Normal | Max | Unit | Source | Normalisation |
|-------|-----|--------|-----|------|--------|---------------|
| **stop_proxy** | 0.0 | 0.7 | 1.0 | ratio | Position limits / leverage | (limit − current) / limit |
| **threshold_proxy** | 0.0 | 0.8 | 1.0 | ratio | Alert threshold headroom | (threshold − current) / threshold |
| **execution_proxy** | 0.0 | 0.6 | 1.0 | ratio | Speed to execute controls | 1 − (latency / max_latency) |
| **coherence_proxy** | 0.0 | 0.75 | 1.0 | ratio | Consistency of decision rules | agreement_rate among decision makers |

**Gate:** If any O proxy < 0.4 on Type III scenario → anti-gaming FAIL

### G-Family Proxies (Governance Activation)

| Proxy | Min | Normal | Max | Unit | Source | Normalisation | Gate |
|-------|-----|--------|-----|------|--------|---------------|------|
| **exemption_rate** | 0.0 | 0.02 | 0.30 | ratio | Policy exceptions granted | count_exemptions / count_requests | > 0.10 → WARN |
| **sanction_delay** | 1 | 15 | 365 | days | Time to enforce policy violation | min(time_to_sanction) | > 90 days → WARN |
| **control_turnover** | 0.0 | 0.05 | 0.50 | ratio/year | Annual staff turnover in controls | fired_or_resigned / total | > 0.20 → WARN |
| **conflict_interest_proxy** | 0.0 | 0.10 | 0.80 | ratio | Fraction with undisclosed conflicts | count_conflicts / count_staff | > 0.25 → WARN |
| **rule_execution_gap** | 0.0 | 0.05 | 0.40 | ratio | Fraction of violations not caught | undetected_violations / total_violations | > 0.15 → WARN |

**Gate:** If any G proxy at Type III scenario crosses WARN threshold → verdict dims to "Partial" or "Caution"

**Important:** These ranges are **FROZEN as of 2026-02-24, git 73676cc**.
To change: create new version v1.1.0, document reason, update git, publish amendment.

---

## Calibration Procedure

### Step 1: Generate Canonical Scenarios (Pre-commitment Phase)
1. Create 4 scenario types × 6 noise variants = 24 datasets
2. Fix random seeds (reproducible)
3. Compute expected signatures (Type I, II, III, IV)
4. **Freeze in git** (commit message: "Canonical scenarios v1.0, thresholds frozen")

### Step 2: Define Regime Decision Rules (Pre-commitment Phase)
```
regime_divergence_score = 0.50 * persistence_dE_dt
                        + 0.30 * e_irreversibility
                        + 0.20 * at_divergence_tail

Type I:   score < 0.30  → @ mean 0.5 ± 0.3, E_irr < 0.30
Type II:  score 0.30–0.65 → @ mean 0.8 ± 0.4, E_irr 0.30–0.65
Type III: score > 0.65  → @ mean 1.2 ± 0.5, E_irr > 0.75
```

**Freeze date:** 2026-02-24
**Cannot be changed without new protocol version**

### Step 3: Run on Canonical Scenarios (Validation Phase)
1. Compute @ and signatures for all 24 datasets
2. Check each Type I, II, III, IV scenario against expected signature table above
3. **Pass rate target:** ≥ 95% correct regime classification

### Step 4: Test Noise Robustness (Validation Phase)
1. Perturb each dataset by ±5% and ±10% random noise
2. Re-classify
3. **Pass rate target:** ≥ 90% keep same regime

### Step 5: Anti-gaming Checks (Validation Phase)
1. Run `tools/run_anti_gaming_suite.py` on all 24 datasets
2. **Pass target:** All 5 attack vectors fail to flip verdict

### Step 6: Freeze Calibration (Public Commitment Phase)
1. Publish `CALIBRATION_REPORT_v1.0.json` with results
2. Commit to git with tag `v1.0-calibration-frozen`
3. **After this:** Any changes require new version, dated amendment, and re-validation

---

## Multidimensional Verdict

Every audit run produces a **public verdict table**, not just "Mature/Immature":

```json
{
  "run_id": "20260224_case_001_finance",
  "timestamp": "2026-02-24T16:00:00Z",
  "verdict_table": {
    "stability": {
      "spearman_min": 0.89,
      "jaccard_min": 0.82,
      "threshold": 0.85,
      "status": "PASS"
    },
    "l_cap": {
      "value": 0.72,
      "expected_range": [0.50, 1.50],
      "regime_detected": "type_III_bifurcation",
      "status": "PASS"
    },
    "l_act": {
      "value": 0.64,
      "expected_range": [0.30, 1.00],
      "governance_alert": false,
      "status": "PASS"
    },
    "governance": {
      "exemption_rate": 0.08,
      "sanction_delay": 45,
      "control_turnover": 0.12,
      "conflict_interest": 0.09,
      "rule_execution_gap": 0.08,
      "warnings": [],
      "status": "PASS"
    },
    "anti_gaming": {
      "o_bias_attack": "FAIL_DETECTED",
      "vol_clamp_attack": "FAIL_DETECTED",
      "range_shift_attack": "FAIL_DETECTED",
      "coordinated_attack": "FAIL_DETECTED",
      "delay_attack": "FAIL_DETECTED",
      "status": "PASS"
    },
    "stress_signature": {
      "e_irreversibility": 0.88,
      "persistence_de": 0.62,
      "at_divergence": 0.51,
      "r_recovery": 0.52,
      "expected_signature": "type_III_bifurcation",
      "status": "PASS"
    }
  },
  "overall_verdict": "CREDIBLE",
  "reasoning": "All dimensions passed. Signal timing matches event date (0 days lag). Regime correct. Governance in normal bands. Anti-gaming passed."
}
```

---

## Falsification Thresholds (Hard Stops)

| Condition | Consequence |
|-----------|------------|
| Type I scenario: @ > 1.0 | **FRAMEWORK INVALID** |
| Type II scenario: E_irr outside [0.30, 0.65] | **FRAMEWORK INVALID** |
| Type III scenario: E_irr < 0.70 | **FRAMEWORK INVALID** |
| Hybrid: No transition detected | **FRAMEWORK INVALID** |
| Stability: Spearman min < 0.80 | **RUN INVALID** (retest) |
| Anti-gaming: Any attack succeeds | **RUN INVALID** (debug) |
| Real case: Signal lags event > 10 days | **SIGNAL QUESTIONED** (note in report) |

---

## Amendment Process

If someone finds a case that breaks the protocol:

1. **Document**: Create GitHub issue with case details
2. **Root cause analysis**: Why did it break?
3. **Decision**:
   - If framework is wrong → v2.0.0 with new thresholds
   - If implementation is wrong → v1.0.1 (patch)
   - If data quality issue → note in case report, framework stands
4. **Publish**: Amended protocol with date and reason

---

## Version History

| Version | Date | Changes | Status |
|---------|------|---------|--------|
| v1.0.0 | 2026-02-24 | Initial freeze: 4 scenario types, proxy ranges, verdicts | FROZEN |

