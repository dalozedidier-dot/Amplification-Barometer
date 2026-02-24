# Case 003: Infrastructure – Power Grid Cascade Risk (2024-Q4)

**Status:** ◐ Partial (Correct Regime, No Independent Verification)
**Sector:** Critical Infrastructure (Power Grid)
**Duration:** 2024-10-01 to 2024-12-31 (92 days)

---

## Ground Truth: The Near-Miss

**Date:** 2024-10-18 (stress peak)

**What Happened:**
Regional power grid experienced elevated cascade risk during autumn storm event:
1. **Oct 1-17:** Seasonal demand rising (cooling → heating transition)
2. **Oct 18:** Line failure (weather-induced) in outer zone
3. **Oct 18-19:** Automated rerouting attempted; line loadings spiked
4. **Oct 19:** Grid operator detected instability; implemented controlled load shed (avoiding full cascade)
5. **Oct 20-Nov 5:** System operated in degraded state with high monitoring
6. **Nov 5:** Repaired line returned to service; normalization

**Impact:**
- 4 hours of controlled outages affecting 200K customers (Oct 19, 2pm–6pm)
- No uncontrolled cascade (event was successfully managed)
- Grid operator credited rapid detection with preventing larger failure

**Source:**
- Grid operator incident report (Nov 2024)
- Weather service records (storm timing and intensity)
- Utility post-event analysis: "Cascade was prevented by 2–3 minutes"
- Public utility commission filing (routine reporting)

**Independent Validation:**
- Event confirmed in utility commission database
- Weather records cross-referenced with line failure timing
- Load shedding volume matches grid model (internal consistency check)
- **Caveat:** No truly independent third-party verification (only utility's own report and regulator filing)

---

## Barometer Data: Alignment

### Dataset Structure
Location: `proxies.csv` (92-day series, 4-hourly frequency)

**22 proxies:**
- **P family:** scale_proxy (system load), speed_proxy (demand ramp), propagation_proxy (cascade potential)
- **O family:** threshold_proxy (line limit headroom), execution_proxy (protective relay speed), coherence_proxy (operator coordination)
- **E family:** impact_proxy (MW at risk), propagation_proxy (cascade spread)
- **R family:** redundancy_proxy (backup line availability), recovery_time_proxy (SCADA response)
- **G family:** exemption_rate (N-1 relaxations), sanction_delay (maintenance delays), control_turnover (operator experience)

### Event Timeline
```json
{
  "observation_window": {"start": "2024-10-01", "end": "2024-12-31"},
  "events": [
    {
      "date": "2024-10-18",
      "type": "line_failure",
      "description": "Weather-induced transmission line failure in outer zone",
      "severity": "high",
      "source": "grid_operator_incident_report"
    },
    {
      "date": "2024-10-19",
      "type": "cascade_prevention",
      "description": "Controlled load shedding initiated; cascade prevented",
      "severity": "mitigation",
      "source": "grid_operator_logs_and_PUC_filing"
    }
  ]
}
```

---

## Barometer Result: SUCCESS (With Caveats)

### Audit Output

```json
{
  "spec_version": "v1.0",
  "summary": {
    "at_mean": 1.15,
    "at_p95": 1.95,
    "delta_d_std": 0.12,
    "e_stock_end": 38.2,
    "r_tail_mean": 0.58
  },
  "stability": {
    "stable": true,
    "spearman_min": 0.87,
    "jaccard_min": 0.79
  },
  "stress_signatures": {
    "persistence_dE_dt_tail_pos_frac": 0.55,
    "e_irreversibility_ratio": 0.72,
    "at_divergence_tail_frac": 0.48,
    "r_tail_mean": 0.51
  },
  "verdict": {
    "dimensions": {
      "stability": "ok",
      "proxy_ranges": "ok",
      "regime_signature": "type_III_bifurcation",
      "anti_gaming": "ok"
    }
  }
}
```

### What Barometer Said
- **Regime:** Type III (bifurcation / cascade risk)
- **@ mean:** 1.15 (elevated, but recovering)
- **E irreversibility:** 0.72 (high – accumulation of cascade risk)
- **R_tail:** 0.51 (recovery capacity degraded)
- **Verdict:** "Cascade risk detected; system near limit"

### What Actually Happened
- **Real event:** Line failure on Oct 18; cascade risk on Oct 19
- **Grid response:** Operator implemented load shedding (forced outages to prevent bigger failure)
- **Outcome:** Controlled failure; cascade prevented

---

## Validation: PARTIAL SUCCESS

### Primary Claim: "Barometer detects power grid cascade risk"
**Status:** ◐ **PARTIAL** (Signal correct, but verification incomplete)

| Criterion | Expected | Observed | Pass |
|-----------|----------|----------|------|
| **Signal Timing** | 0–10 days before event | 1 day before (Oct 18) | ✓ |
| **Regime Classification** | Type III bifurcation | Type III bifurcation | ✓ |
| **E Accumulation** | ≥ 0.65 irreversibility | 0.72 | ✓ |
| **R Degradation** | recovery tail < 0.60 | 0.51 | ✓ |
| **Stability** | Spearman ≥ 0.85 | 0.87 | ✓ |
| **Anti-gaming** | All passes | All passes | ✓ |
| **Independent Verification** | Third-party confirmation | Only utility report + PUC filing | ⚠️ |

**Verdict:** ✓ **Signal detected correctly, but verification is "utility self-reporting"**

---

## Analysis: What Worked, What's Uncertain

### ✓ What the Barometer Got Right

1. **Regime Classification** – Type III bifurcation correct (not oscillatory, not noise)
2. **Temporal Detection** – Signal peaked 1 day *before* worst-case scenario (load shedding)
3. **Structural Signals** – E accumulation and R degradation matched theory predictions
4. **Recovery Pattern** – After Oct 19 mitigation, @ declined, R partially recovered (Nov 5 line repaired)
5. **Governance Signals** – G proxies remained stable (no control team failures, proper procedures followed)

### ⚠️ Verification Concerns

1. **No Independent Ground Truth**
   - Only source: utility's own incident report
   - PUC filing confirms event but doesn't verify barometer's correctness
   - No third-party expert auditor examining grid data

2. **Possibility of Post-Hoc Fitting**
   - Did barometer really predict cascade risk, or did we tune it after seeing the event?
   - Audit was published *after* Oct 19; could proxies have been selected to fit the narrative?

3. **No Counterfactual**
   - Did barometer trigger false alarms on Oct 10, 15, 22? (not reported here)
   - Only reporting the "hit"; not showing false positives

### ✓ Mitigating Factors

1. **Proxy ranges were sectorially appropriate** – Power grid specs from critical infrastructure baseline
2. **Governance signals clean** – No anti-gaming red flags; G proxies normal (strengthens credibility)
3. **Stability metrics held** – Results robust to noise perturbation (Spearman 0.87 > 0.85 threshold)

---

## Recommendations for Future Cases

### To strengthen this case:
1. **Get independent grid expert** to review barometer output alongside utility incident report
2. **Compare barometer output to existing grid monitoring** (RMS synchrophasors, PMU data)
3. **Test counterfactuals:** Run barometer on Oct 10, 15, 25; show false alarm rate or lack thereof

### For the barometer framework:
1. **Establish partnership with grid operator** for real-time data sharing
2. **Develop independent power flow simulator** to verify cascade risk independently
3. **Publish false positive/negative rates** alongside true positives

---

## Verdict on This Case

**Claim:** "Barometer detects power grid cascade risk"
**Result:** ✓ **LIKELY TRUE, but not fully independent**

**Assessment:**
- Signal appears genuine (regime type and timing match)
- But verification relies on utility self-reporting
- Elevated to "Partial" rather than "Full Success" until independent confirmation

---

## Next Steps

1. ✓ Case documented as "Partial Validation"
2. ✓ Flagged for: independent expert review, false positive check
3. → Recommend: partnering with grid operator for future shared data

---

## Files in This Case

- `CASE_SUMMARY.md` – This file
- `proxies.csv` – 92 rows × 22 columns, 4-hourly frequency
- `event_dates.json` – Oct 18 (line failure), Oct 19 (cascade risk peak)
- `audit_output.json` – Full barometer output (Type III, correct regime)
- `verification_status.txt` – "Partial: utility-reported, needs independent review"

## Credibility Reflection

**Why "Partial" is honest:**

A "published success" with uncertain verification is still progress, because:
1. It's better than hiding the case
2. It flags what needs independent confirmation
3. It shows we're tracking what we said we'd detect
4. It identifies path to higher confidence (independent review)

**But:**
- Don't claim victory until independent third party confirms
- Acknowledge this is utility self-reporting (they have incentive to show good management)
- Next step: bring in PUC auditors or independent grid experts

