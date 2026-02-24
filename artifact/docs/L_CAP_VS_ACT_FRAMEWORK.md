# L_cap vs L_act Framework

**Version:** v0.1.0
**Date:** 2026-02-24

---

## Purpose

This document separates two concepts that can become circular:

1. **L_cap** – Intrinsic detection **capacity** (what the barometer *can* do in principle)
2. **L_act** – Observed **activation** (what the barometer *does* do in practice)

Without this separation, you risk circular reasoning: "We detected it because we were capable of detecting it, and we were capable because we detected it."

This framework breaks that circle.

---

## Definitions

### L_cap: Intrinsic Capacity

**Question:** "Under controlled conditions, can the barometer distinguish stress regimes?"

**Measurement:** Benchmark on **synthetic canonical scenarios** (Type I, II, III, Hybrid):
- No confounding from real-world governance noise
- Fixed, known ground truth (we injected the stress)
- Same stress detection rules applied consistently

**Source:** `compute_l_cap(df)` on `data/canonical_scenarios/` files

**Interpretation:**
- High L_cap → system *can* detect differences in regime
- Low L_cap → system lacks discriminative power
- L_cap depends on O-family proxies (stop, threshold, execution, coherence)

**Not circular because:**
- Computed independently from observed governance (L_act)
- Uses synthetic data where ground truth is known
- Benchmarked once, then frozen

### L_act: Observed Activation

**Question:** "In reality, does the system actually activate its detection/response capability?"

**Measurement:** Computed on **real or realistic datasets**:
- Actual governance signals (exemptions, sanctions, turnover, gaps)
- Real decision delays and execution gaps
- Depends on human/institutional factors

**Source:** `compute_l_act(df)` on real datasets or mixed datasets

**Interpretation:**
- High L_act → governance is alert and responsive
- Low L_act → governance is asleep, or policy gaps exist
- L_act depends on G-family proxies (exemption_rate, sanction_delay, control_turnover, conflict_interest, rule_execution_gap)

**Not circular because:**
- Computed from observable, external governance signals
- Independent of the barometer's technical capacity
- Measures human/institutional readiness

---

## 2×2 Matrix: Classification

Once you have L_cap and L_act for a system:

```
                    L_act ≥ 0 (Activated)        L_act < 0 (Quiet)
L_cap ≥ 0 (Capable)     SUCCESS                       MISS
L_cap < 0 (Limited)     FALSE_ALARM                   BENIGN
```

### Quadrant Interpretation

| Quadrant | Meaning | Implication |
|----------|---------|-------------|
| **SUCCESS** | High capacity + High activation | System working well. Can detect, does detect. ✓ |
| **MISS** | High capacity + Low activation | Detection ability exists but not triggered. Gap in governance or thresholds too high. ⚠️ |
| **FALSE_ALARM** | Low capacity + High activation | System alerts despite lacking real capacity. May be spurious. ✗ |
| **BENIGN** | Low capacity + Low activation | System quiet because either no threat OR no detection capability. Need context. ⓘ |

### Example Scenarios

#### Scenario A: Finance Trading System
- **L_cap:** High (system architecture supports position limits, rapid execution stops)
- **L_act:** High (exemption rate low, sanctions applied quickly, control teams stable)
- **Quadrant:** SUCCESS ✓
- **Implication:** Trading surveillance is working as designed.

#### Scenario B: AI Model Deployment
- **L_cap:** High (model can distinguish safe vs unsafe outputs, has emergency stops)
- **L_act:** Low (safety reviews often waived, incidents take weeks to escalate, team turnover high)
- **Quadrant:** MISS ⚠️
- **Implication:** We *could* catch issues, but governance is asleep. Need to fix processes, not code.

#### Scenario C: Power Grid Control
- **L_cap:** Low (automated controls struggle to distinguish cascading failure from normal oscillation)
- **L_act:** High (operators alert, policies strictly enforced, incidents responded to quickly)
- **Quadrant:** FALSE_ALARM ✗
- **Implication:** Governance trying hard, but system is fragile. Need technical improvements.

#### Scenario D: Infrastructure Without Monitoring
- **L_cap:** Low (no real-time monitoring, manual checks only)
- **L_act:** Low (informal, sporadic reporting)
- **Quadrant:** BENIGN ⓘ
- **Implication:** System is quiet because there's no automated detection, not because it's safe. Baseline risk is high.

---

## Methodology: How to Compute

### Step 1: Benchmark L_cap (Once)

```python
from amplification_barometer.l_capability_benchmark import benchmark_suite_all_scenarios

# Run on canonical scenarios (synthetic, ground truth known)
benchmark = benchmark_suite_all_scenarios(
    scenarios_dir="data/canonical_scenarios/",
    proxies_yaml="docs/proxies.yaml"
)

# Validate that L_cap can distinguish Type I, II, III
assert benchmark["summary"]["type_i_expected_low_l_cap"]["pass_rate"] >= 0.80
assert benchmark["summary"]["type_ii_expected_moderate_l_cap"]["pass_rate"] >= 0.80
assert benchmark["summary"]["type_iii_expected_high_l_cap"]["pass_rate"] >= 0.80
```

**Output:** Freezes L_cap capability profile. This is your baseline "what the system can do."

### Step 2: Compute L_cap on Real Data

```python
from amplification_barometer.l_operator import compute_l_cap

df_real = pd.read_csv("real_dataset.csv")
l_cap_real = compute_l_cap(df_real)  # Intrinsic capacity in this context

l_cap_mean = l_cap_real.mean()  # Should align with benchmark Type II/III if stressed
```

### Step 3: Compute L_act on Real Data

```python
from amplification_barometer.l_operator import compute_l_act

l_act_real = compute_l_act(df_real)  # Governance activation in this context
l_act_mean = l_act_real.mean()  # Is governance responsive?
```

### Step 4: Classify

```python
from tools.run_l_cap_vs_act_validation import classify_l_cap_l_act

classification = classify_l_cap_l_act(
    l_cap_mean, l_act_mean,
    thresholds={"l_cap_threshold": 0.0, "l_act_threshold": 0.0}
)

print(f"Quadrant: {classification['quadrant']}")
print(f"Interpretation: {classification['interpretation']}")
```

### Step 5: Validate via CLI

```bash
python3 tools/run_l_cap_vs_act_validation.py \
  --scenarios-dir data/canonical_scenarios \
  --proxies-yaml docs/proxies.yaml \
  --out-dir reports/l_cap_vs_act
```

**Output:** `l_cap_vs_act_validation.json` with full 2×2 matrix across all test datasets.

---

## Falsification Criteria

This framework is falsifiable:

### Falsification 1: L_cap Benchmark Fails
- **If:** Type III bifurcation scenario shows low L_cap (< 0.0)
- **Then:** System cannot distinguish bifurcation from noise
- **Consequence:** Claim "barometer detects Type III" is false → redesign needed

### Falsification 2: Quadrant Mismatch
- **If:** Finance system shows many MISS or FALSE_ALARM entries
- **Then:** Either capacity is not real, or governance is broken
- **Consequence:** Investigate and fix root cause

### Falsification 3: No Separation
- **If:** L_cap ≈ L_act always (perfectly correlated)
- **Then:** No independent information; L_cap is just a proxy for governance
- **Consequence:** Framework is circular; need to redesign L_cap measurement

---

## Related Components

- **L_cap computation:** `src/amplification_barometer/l_operator.py::compute_l_cap()`
- **L_act computation:** `src/amplification_barometer/l_operator.py::compute_l_act()`
- **Benchmark tool:** `src/amplification_barometer/l_capability_benchmark.py`
- **Validation CLI:** `tools/run_l_cap_vs_act_validation.py`
- **Canonical scenarios:** `data/canonical_scenarios/` (20 deterministic CSVs)

---

## Integration with DoD

This framework addresses **Phase 4** of the truth contract:

| Phase | Deliverable | Role |
|-------|-------------|------|
| 1-3 | Definition of Done, Calibration, Proxy Specs | Foundation |
| **4** | **L_cap vs L_act separation** | **Eliminate circular reasoning** |
| 5 | Real-world validation | Empirical proof |
| 7 | Auditability manifests | Trust infrastructure |

Without Phase 4, you risk intellectual vulnerability:
> "You detected it because you could detect it, and you could detect it because you detected it."

With Phase 4, you can answer:
> "We *can* detect Type III bifurcations (L_cap ≥ 0.5). We *did* detect it in case X (L_act ≥ 0.3). Here's the 2×2 matrix showing that SUCCESS cases are real."

---

## Version History

| Version | Date | Notes |
|---------|------|-------|
| v0.1.0 | 2026-02-24 | Initial framework. L_cap on canonical scenarios, L_act on governance proxies. |

