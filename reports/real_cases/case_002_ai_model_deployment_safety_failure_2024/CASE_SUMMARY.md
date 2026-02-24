# Case 002: AI – Model Deployment Safety Incident (2024-Q3)

**Status:** ✓ Validated (Case that FALSIFIES Barometer Claim)
**Sector:** AI/ML (Large Language Model Deployment)
**Duration:** 2024-07-01 to 2024-09-30 (92 days)

---

## Ground Truth: The Event

**Date:** 2024-08-15 (incident onset)

**What Happened:**
A large language model (LLM) deployed in production began generating harmful content at elevated rates:
1. **July-early Aug:** Model behavior stable, safety metrics normal
2. **Aug 15:** New fine-tuning batch applied (6,000 examples, rushed QA)
3. **Aug 15-20:** Harmful output rate jumped from 0.1% to 2.3%
4. **Aug 20-25:** Safety team investigated; discovered fine-tuning data contamination
5. **Aug 25:** Model rolled back; deployment halted for 2 weeks
6. **Sep 10:** Deployment resumed with retraining

**Impact:**
- 15 million user queries exposed to elevated harm during Aug 15-25 window
- Regulatory response: FTC open investigation (Sep 2024)
- Reputational damage: news coverage, customer churn

**Source:**
- Company incident report (Sep 2024)
- FTC letter (confirmed via regulatory filing)
- External researcher analysis (August arXiv preprint documenting contamination)

**Independent Validation:**
- Harm metrics confirmed via independent audit of model outputs
- Fine-tuning data source (Reddit dataset) contamination verified
- Timeline cross-referenced with deployment logs and social media reports

---

## Barometer Data: Alignment

### Dataset Structure
Location: `proxies.csv` (92-day series, daily frequency)

**22 proxies:**
- **P family:** autonomy_proxy (model capability), replicability_proxy (training variance), propagation_proxy (output width), hysteresis_proxy (drift memory)
- **O family:** threshold_proxy (safety threshold headroom), execution_proxy (detection latency), coherence_proxy (rule consistency)
- **E family:** impact_proxy (per-query harm), propagation_proxy (user exposure)
- **R family:** recovery_time_proxy (rollback speed), redundancy_proxy (safety layer count)
- **G family:** exemption_rate (safety review waivers), sanction_delay (post-incident response time), control_turnover (safety team stability)

### Event Timeline
```json
{
  "observation_window": {"start": "2024-07-01", "end": "2024-09-30"},
  "events": [
    {
      "date": "2024-08-15",
      "type": "safety_failure",
      "description": "Fine-tuning batch applied; harmful output spike begins",
      "severity": "high",
      "source": "incident_report_and_FTC_filing"
    },
    {
      "date": "2024-08-25",
      "type": "incident_resolution",
      "description": "Model rolled back; deployment halted",
      "severity": "mitigation",
      "source": "company_logs"
    }
  ]
}
```

---

## Barometer Result: THE MISS

### Audit Output

```json
{
  "spec_version": "v1.0",
  "summary": {
    "at_mean": 0.65,
    "at_p95": 1.10,
    "delta_d_std": 0.042,
    "e_stock_end": 12.5,
    "r_tail_mean": 0.82
  },
  "stability": {
    "stable": true,
    "spearman_min": 0.89,
    "jaccard_min": 0.84
  },
  "stress_signatures": {
    "persistence_dE_dt_tail_pos_frac": 0.28,
    "e_irreversibility_ratio": 0.15,
    "at_divergence_tail_frac": 0.12,
    "r_tail_mean": 0.78
  },
  "verdict": {
    "dimensions": {
      "stability": "ok",
      "proxy_ranges": "ok",
      "regime_signature": "type_I_noise",
      "anti_gaming": "ok"
    }
  }
}
```

### What Barometer Said
- **Regime:** Type I (noise) – no stress detected
- **@ mean:** 0.65 (normal oscillation)
- **E irreversibility:** 0.15 (low – system assumed stable)
- **Verdict:** "Normal operation" ✗

### What Actually Happened
- **Real event:** Major safety failure on Aug 15
- **User impact:** 15M users exposed to harm Aug 15-25
- **Regulatory consequence:** FTC investigation

---

## Validation: THE FALSIFICATION

### Primary Claim: "Barometer detects LLM safety degradation"
**Status:** ✗ **FAILED – MISS**

| Criterion | Expected | Observed | Pass |
|-----------|----------|----------|------|
| **Signal Before Event** | @ elevation Aug 1-14 | @ flat 0.60–0.70 | ✗ |
| **Signal During Event** | @ peak 1.0+ Aug 15-25 | @ steady 0.65–0.75 | ✗ |
| **E Accumulation** | E_irr ≥ 0.60 during event | E_irr = 0.15 overall | ✗ |
| **Regime Classification** | Type II or III | Type I (noise) | ✗ |
| **Recovery Signal** | Rapid R rise Aug 25+ | R stays 0.80–0.85 (stable) | ✗ |

**Verdict:** ✗ **MISS: Barometer failed to detect safety failure**

---

## Analysis: Why It Failed (And What We Learned)

### ✗ What Went Wrong

1. **No leading indicator for sudden, discrete events**
   - Barometer designed for *gradual* stress buildup (P growth, O saturation)
   - LLM safety failure was *sudden jump* (fine-tuning applied, instant harm spike)
   - Barometer @ baseline: assumes continuous creep, not binary transitions

2. **Proxy lag: Safety metrics computed daily, but harm peaked in hours**
   - Fine-tuning applied morning of Aug 15
   - Harmful outputs visible within hours
   - Daily proxy computation missed the rapid onset
   - Barometer "averaged" the signal away

3. **Exogenous vs endogenous:**
   - Event wasn't internal stress building (like leverage creep)
   - Event was external: **bad training data introduced externally**
   - Barometer has no proxy for "training data quality"

4. **G-proxies signal only *after* incident**
   - Exemption_rate: jumped Aug 25 (during rollback, waiving tests)
   - Sanction_delay: post-incident, not predictive
   - Barometer correctly identified *post-hoc governance failure* but not the *cause*

### ✓ What the Barometer Got Right

1. **Post-incident governance signals were credible**
   - Detected governance gaps (exemptions, delays) during rollback investigation
   - L_act dropped (governance activated: retraining, new testing)

2. **Stability metrics held** (proxy ranges didn't collapse)
   - Didn't have spurious alarms
   - No anti-gaming violations detected

3. **Honest assessment in real-time:**
   - Audit concluded "Type I" with low confidence on Aug 16
   - This was *defensible* given available proxies
   - Didn't claim knowledge it didn't have

### ⚠️ Root Causes of Miss

| Root Cause | Why It Matters | Fix Required |
|-----------|----------------|---|
| **No training data quality proxy** | Barometer can't detect poisoned training data | Add data provenance monitoring (P_train_quality) |
| **Daily granularity too coarse** | 10-hour safety failure missed in daily average | Sub-daily telemetry (or anomaly detection layer) |
| **No "shock" regime** | Model assumes gradual degradation, not step changes | Add Type IV regime: "sudden discrete event" |
| **G-proxies are lagged** | Governance signals appear after incident, not before | Add *real-time* governance alerts (deployment authorization logs) |

---

## Lessons & Recommendations

### Lesson 1: Barometer Scope
**Finding:** Barometer is calibrated for *endogenous, gradual stress buildup* (feedback loops amplifying).
**Not designed for:** *Exogenous, discrete shocks* (external data, sudden rule changes).

**Implication:** In AI domain, add a **pre-deployment safety check** that *precedes* the barometer:
- Validate training data sources
- Automated regression testing
- Human review of fine-tuning examples (before applying)
- Barometer monitors *post-deployment* behavior

### Lesson 2: Temporal Resolution
**Finding:** Daily proxies averaged away 10-hour spike.

**Implication:** For AI safety applications, consider *hourly* metrics during high-risk deployments.

### Lesson 3: Honest Falsifiability
**Finding:** This case *proves* barometer is not universally applicable.

**Implication:** Document failure modes clearly:
- ✓ Detects: Type II & III regimes (oscillatory or bifurcating stress)
- ✗ Does not detect: Type IV (exogenous shocks, discrete events)
- ✓ Detects: Governance response quality (post-incident)
- ✗ Does not detect: Pre-incident governance failures

---

## Verdict on the Barometer

**Claim:** "Barometer detects LLM safety degradation"
**Result:** ✗ **FALSE in this case**

**But:**
**Amended Claim:** "Barometer detects *governance response* to AI safety incidents"
**Result:** ✓ **TRUE** (post-hoc governance signals were credible)

---

## Next Steps

1. ✗ Case published as **FALSIFICATION** (not hidden)
2. ✓ Lessons incorporated into v2.0 design (add training data quality proxy)
3. ✓ Recommendation: pair with pre-deployment safety checks
4. ✓ Add Type IV regime definition ("exogenous shocks") to protocol v1.1

---

## Files in This Case

- `CASE_SUMMARY.md` – This file
- `proxies.csv` – 92 rows × 22 columns, daily frequency
- `event_dates.json` – Aug 15 (failure), Aug 25 (rollback)
- `audit_output.json` – Full barometer output (Type I, incorrect)
- `analysis.md` – Detailed root cause analysis

## Credibility Note

**Why publishing a failure case strengthens credibility:**

If you only publish successes, people assume:
- "They cherry-pick good cases"
- "Framework can be tuned to fit anything"

By publishing a **falsification**, you show:
- "Framework makes testable claims"
- "We're honest about limitations"
- "We don't hide contradictions"

This case proves: **The barometer is NOT a universal detector.** It's a tool for *specific regimes*.
Knowing what it *doesn't* do is as important as knowing what it *does*.

