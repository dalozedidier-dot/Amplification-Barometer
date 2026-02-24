# Finance Governance Proxies Specification v1.0.0

**Status:** ❄️ FROZEN (2026-02-24)
**Sector:** Financial Services
**Authority:** Central Bank / SEC / ISDA
**Audit Chain:** Hash `a3f8c2d1...` (see history.jsonl)

---

## Overview

This specification defines 5 G-proxies (governance proxies) for measuring governance response quality in financial institutions. These proxies feed into L_act (observed activation) and determine whether governance is sufficiently responsive to control risk.

**Critical Property:** These specs are PRE-COMMITTED and FROZEN. They cannot be changed without creating v2.0.0 and documenting the amendment publicly.

---

## The 5 Governance Proxies

### 1. **exemption_rate** (Mandatory)

**Definition:**
Fraction of policy violations that receive exemption/waiver vs enforcement.

```
exemption_rate = count(violations_with_exemption) / count(total_violations)
```

| Property | Value |
|----------|-------|
| **Unit** | Decimal fraction [0.0, 1.0] |
| **Frequency** | Monthly |
| **Data Source** | Primary: Internal policy database; Secondary: Regulatory filings (10-K/10-Q) |
| **Calculation** | Reported by: CFO/Compliance Officer; Audited by: External Auditor |
| **Expected Range (Finance)** | 0.02–0.05 (2–5% exceptions are normal) |
| **Gate (Alert)** | > 0.10 (>10% exceptions = warning) |
| **Treatment of Missing** | If missing ≥ 2 months, interpolate with prior month; if > 3 months, revert to prior trailing 90-day average |
| **Normalization** | Min-max [0, 1] per institution class (see below) |
| **Justification** | High exemption rates indicate rules are not enforced; weak governance hidden by exceptions |

**Institutional Classes & Ranges:**
- Tier 1 (SIFI): 0.01–0.04 (stricter)
- Tier 2 (Large banks): 0.02–0.06
- Tier 3 (Regional): 0.03–0.08

**Audit Questions:**
- When is exemption granted? (pre-violation or post?)
- Who has authority to grant? (single person or committee?)
- Is exemption tracked to specific individuals?
- What % of exemptions are later reversed or escalated?

---

### 2. **sanction_delay** (Mandatory)

**Definition:**
Number of calendar days between violation detection and formal enforcement action (written notice, fine, remediation order).

```
sanction_delay = mean(days from violation_detected to enforcement_notice)
```

| Property | Value |
|----------|-------|
| **Unit** | Days |
| **Frequency** | Monthly (computed from trailing 30-violation window) |
| **Data Source** | Primary: Incident tracking system (JIRA, ServiceNow); Secondary: Regulatory correspondence logs |
| **Calculation** | Reported by: Chief Risk Officer; Audited by: External Auditor |
| **Expected Range (Finance)** | 3–7 days (typical); 21+ days (slow); >90 days (problematic) |
| **Gate (Alert)** | > 90 days (violations take 3+ months to enforce = weak governance) |
| **Treatment of Missing** | If no violations in month, use prior month's average |
| **Normalization** | Days clipped to [0, 365], then z-score within sector |
| **Justification** | Fast sanctions deter; slow sanctions signal impunity → recidivism |

**Escalation Triggers:**
- 21 days → internal review
- 60 days → board notification
- >90 days → regulatory filing

**Audit Questions:**
- Is there time logged on each violation?
- Are reviews/approvals tracked separately?
- What % of violations go to committee (slower) vs auto-enforce?
- Are there undeclared exceptions that artificially shorten delays?

---

### 3. **control_turnover** (Mandatory)

**Definition:**
Annual staff departures from control functions divided by total control function staff.

```
control_turnover = count(departures in control functions) / count(total control staff) [annual]
```

| Property | Value |
|----------|-------|
| **Unit** | Decimal fraction [0.0, 1.0] |
| **Frequency** | Quarterly (annualized) |
| **Data Source** | Primary: HR systems (SAP SuccessFactors, Workday); Secondary: Board/Audit Committee reports |
| **Calculation** | Reported by: Chief People Officer; Audited by: Internal Audit |
| **Expected Range (Finance)** | 5–15% (normal); >30% (crisis) |
| **Gate (Alert)** | > 0.25 (>25% annual turnover = loss of institutional knowledge) |
| **Treatment of Missing** | Use most recent known quarter, do not extrapolate |
| **Normalization** | Min-max [0, 1] per role level (senior < mid < junior) |
| **Justification** | High turnover loses expertise, creates gaps, enables shadow operations |

**Control Functions Include:**
- Compliance
- Risk Management
- Internal Audit
- Regulatory Affairs
- AML/CFT
- Model Risk Management
- Information Security

**Audit Questions:**
- Is turnover measured by exit date or notice date?
- Are internal transfers counted as departures?
- What % of departures are terminations vs retirements vs transfers?
- Is turnover higher in underperforming divisions?

---

### 4. **conflict_interest_proxy** (Mandatory)

**Definition:**
Annual undisclosed conflicts of interest as fraction of staff in positions requiring conflict declarations.

```
conflict_interest_proxy = count(undisclosed conflicts) / count(staff with required disclosure)
```

| Property | Value |
|----------|-------|
| **Unit** | Decimal fraction [0.0, 1.0] |
| **Frequency** | Quarterly |
| **Data Source** | Primary: Conflict of interest questionnaires; Secondary: Regulatory examination findings |
| **Calculation** | Reported by: Chief Compliance Officer; Audited by: External Auditor |
| **Expected Range (Finance)** | 5–15% (normal: some undisclosed minor conflicts); >30% (red flag) |
| **Gate (Alert)** | > 0.25 (>25% undisclosed = systematic cover-up) |
| **Treatment of Missing** | If no quarterly survey, use prior quarter + add 1% drift; flag to compliance |
| **Normalization** | Min-max [0, 1] |
| **Justification** | Undisclosed conflicts indicate: weak culture, hidden incentives, unmanaged risks |

**Audit Questions:**
- How are undisclosed conflicts identified? (self-report, audit, external?)
- Are certain divisions systematically under-reporting?
- What % are minor (immaterial) vs material?
- How many were discovered post-fact (failure of process)?

---

### 5. **rule_execution_gap** (Mandatory)

**Definition:**
Violations detected by external audit / regulatory examination but not caught by internal control system.

```
rule_execution_gap = count(violations_found_by_external) / (count(violations_found_by_external) + count(violations_found_by_internal)) [annual]
```

| Property | Value |
|----------|-------|
| **Unit** | Decimal fraction [0.0, 1.0] |
| **Frequency** | Annual (from audit cycle) |
| **Data Source** | Primary: Audit findings log (SOX compliance, regulatory exam); Secondary: Enforcement actions |
| **Calculation** | Reported by: Chief Audit Officer; Audited by: External Auditor |
| **Expected Range (Finance)** | 5–15% (some gaps normal); >30% (controls failing) |
| **Gate (Alert)** | > 0.20 (>20% of violations missed internally = control system ineffective) |
| **Treatment of Missing** | If no annual audit, hold prior year value; flag for audit committee |
| **Normalization** | Min-max [0, 1] |
| **Justification** | High external-find rate = internal controls not working; governance is blind to violations |

**Violation Categories:**
- Regulatory compliance (market conduct, data privacy, etc.)
- Risk management (concentration, leverage, etc.)
- Operational (transaction approval, segregation of duty, etc.)

**Audit Questions:**
- What % of findings are repeat violations (found in prior audits)?
- Are certain risk categories systematically missed?
- What's the median time to remediate post-finding?
- Is there evidence of deliberate circumvention?

---

## Composite L_act Scoring

All 5 proxies are combined into **L_act** (observed activation):

```
bad_gov_score = 0.25 × normalize(exemption_rate)
              + 0.20 × normalize(sanction_delay / 365)
              + 0.20 × normalize(control_turnover)
              + 0.15 × normalize(conflict_interest)
              + 0.20 × normalize(rule_execution_gap)

L_act_raw = 1.0 - bad_gov_score    # Higher = better governance
L_act = logit(L_act_raw, k=1.6)    # Project to standard scale
```

**Interpretation:**
- L_act > 0.70 → Governance responsive (good control execution)
- 0.50 < L_act ≤ 0.70 → Governance adequate
- L_act ≤ 0.50 → Governance weak (slow, ineffective)

---

## Version Control & Amendment Process

### Current Version: v1.0.0

- **Published:** 2026-02-24
- **Author:** Amplification Barometer Framework
- **Status:** FROZEN ❄️ (cannot be modified; must create v2.0.0)
- **Git Commit:** `ed31b61...`
- **SHA256:** `a3f8c2d1e9b4c7f2a5d8e1b4c7f2a5d8e1b4c7` (from history.jsonl)

### Amendment Process (for v2.0.0)

If changes are needed, follow this process:

1. **Document the Change**
   - What proxy is changing?
   - Why (business case)?
   - Evidence that old definition was inadequate?

2. **Public Comment Period**
   - 30 days for external stakeholders to review
   - Publish in docs/amendments/ directory

3. **Create v2.0.0**
   - Copy GOVERNANCE_PROXIES_v1.0.0.md to v2.0.0
   - Document changes in "Changelog" section
   - Tag in git: `governance-proxies-v2.0.0`

4. **Audit All Prior Cases**
   - Recompute all historical cases with new definition
   - Report impact on verdicts

---

## Data Quality Assurance

### Quarterly Validation Checklist

- [ ] All 5 proxies reported and within expected ranges?
- [ ] Any missing data flagged and interpolated per spec?
- [ ] Data integrity verified (no deletions/overwrites)?
- [ ] Calculations reviewed by independent auditor?
- [ ] Trend analysis: any sudden jumps unexplained?
- [ ] Institutions with red flags (L_act < 0.50) escalated?

### Red Flag Patterns

| Pattern | Indicator | Action |
|---------|-----------|--------|
| Exemption_rate suddenly drops | Gaming attempt? | Audit exemption criteria |
| Sanction_delay spikes | Governance dysfunction | Board notification |
| Turnover spike in compliance | Knowledge loss | Retention plan required |
| Conflicts undisclosed until audit | Culture failure | CEO briefing |
| Rule gap increases quarter-over-quarter | Control degradation | Control enhancement plan |

---

## Worked Example

**Institution:** MegaBank (Tier 1 SIFI)
**Period:** Q4 2024

| Proxy | Raw Value | Gate | Status | Normalized |
|-------|-----------|------|--------|-----------|
| exemption_rate | 0.08 | > 0.10 ⚠️ | CAUTION | 0.80 |
| sanction_delay | 45 days | > 90 ✓ | OK | 0.12 |
| control_turnover | 0.18 | > 0.25 ✓ | OK | 0.18 |
| conflict_interest | 0.12 | > 0.25 ✓ | OK | 0.12 |
| rule_execution_gap | 0.16 | > 0.20 ✓ | OK | 0.16 |

**L_act Calculation:**
```
bad_gov = 0.25(0.80) + 0.20(0.12) + 0.20(0.18) + 0.15(0.12) + 0.20(0.16)
        = 0.20 + 0.024 + 0.036 + 0.018 + 0.032
        = 0.31

L_act_raw = 1.0 - 0.31 = 0.69
L_act = logit(0.69, k=1.6) ≈ 0.65
```

**Verdict:** Governance acceptable, but exemption rate rising (watch for trend)

---

## Appendix A: Data Sources by Proxy

### exemption_rate
- **Primary:** Bank's policy system (e.g., PolicyTech)
- **Secondary:** 10-K Item 1A (Risk Factors), SEC examination reports
- **Validation:** Audit committee report quarterly

### sanction_delay
- **Primary:** Risk incident management system
- **Secondary:** Regulatory correspondence, settlement agreements
- **Validation:** CFO certification in SOX disclosure

### control_turnover
- **Primary:** HRIS (SAP, Workday, ADP)
- **Secondary:** Board compensation committee report
- **Validation:** Internal audit verification

### conflict_interest_proxy
- **Primary:** Annual conflict questionnaires
- **Secondary:** Regulatory exams, Dodd-Frank disclosures
- **Validation:** Chief Compliance Officer attestation

### rule_execution_gap
- **Primary:** Internal audit findings database
- **Secondary:** OCC/Fed/SEC exam findings (public)
- **Validation:** Audit committee

---

## Appendix B: Sector-Specific Ranges

### Banking
- exemption_rate: 0.02–0.06 (stricter than peers)
- sanction_delay: 3–14 days (fast due to regulatory scrutiny)
- control_turnover: 5–12%
- conflict_interest: 5–15%
- rule_execution_gap: 5–20%

### Insurance
- exemption_rate: 0.03–0.08
- sanction_delay: 7–21 days
- control_turnover: 8–18%
- conflict_interest: 8–20%
- rule_execution_gap: 8–25%

### Asset Management
- exemption_rate: 0.02–0.07
- sanction_delay: 1–7 days (fastest due to fiduciary duty)
- control_turnover: 10–20%
- conflict_interest: 10–25%
- rule_execution_gap: 10–30%

---

## Appendix C: Relationship to L_cap vs L_act

**L_cap:** Intrinsic system capacity (can the institution stop risk?)
**L_act:** Governance response (does the institution actually stop risk?)

These proxies measure **L_act only**. L_cap is measured separately via O-proxies and stress tests.

**Example:**
- High L_cap, low L_act → Institution *can* control but *doesn't* (governance problem)
- Low L_cap, high L_act → Institution tries hard but *can't* (structural problem)
- High L_cap, high L_act → Institution both *can* and *does* (mature)

---

**Version:** v1.0.0 | **Frozen:** 2026-02-24 | **Next Review:** 2027-02-24
