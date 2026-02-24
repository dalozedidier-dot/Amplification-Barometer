# Finance Sector: Governance (G) Proxy Specifications

**Version:** v1.0.0
**Sector:** Finance (Algorithmic Trading & Risk Management)
**Date:** 2026-02-24
**Status:** FROZEN

---

## Overview

G-proxies measure whether governance infrastructure **activates** in response to detected stress.
Unlike O-proxies (what the system *can* do), G-proxies measure what governance *actually does*.

**Why These 5?**
1. **exemption_rate** – Are controls being bypassed?
2. **sanction_delay** – How fast do violations get punished?
3. **control_turnover** – Is the control team stable?
4. **conflict_interest_proxy** – Undisclosed conflicts weakening oversight?
5. **rule_execution_gap** – Are violations even detected?

Together, they measure: *Is governance alert and responsive?*

---

## G1: Exemption Rate

### Definition
**Fraction of policy requests that receive exemptions rather than compliance.**

Example:
- Risk policy: "Max leverage 5x"
- Desk requests exemption: "Need 7x for this trade"
- If approved → counts as 1 exemption

### Range

| Level | Value | Interpretation |
|-------|-------|-----------------|
| **Excellent** | 0.00–0.02 | Controls are strict; exemptions are rare (≤2 per 100) |
| **Good** | 0.02–0.05 | Occasional exemptions, normal variation |
| **Caution** | 0.05–0.10 | Exemptions frequent; controls loosening |
| **Warning** | 0.10–0.20 | Widespread exemptions; controls largely ceremonial |
| **Critical** | > 0.20 | Exemptions more common than compliance |

### Measurement

```
exemption_rate = count(exemptions approved) / count(exemption requests)

Where:
  - exemption = written request to deviate from policy
  - approved = exemption was granted (not rejected)
  - requests = all exemption requests in period (approved + rejected)
```

### Data Source

**Primary (Preferred):**
- Governance database: exemption log (date, policy, requester, approver, reason)
- Export: weekly CSV with columns [date, policy_name, requester, approved_Y_N]
- Frequency: weekly (more granular if available)

**Secondary (If primary unavailable):**
- Audit report: "Exceptions granted" section
- Compliance memo: policy deviations

**Missingness Protocol:**
- If no exemption requests in week → rate = 0.00 (not missing)
- If exemption log not available but audit says "no exemptions" → rate = 0.00 (credible)
- If exemption log missing and audit silent → rate = 0.05 (conservative assumption)

### Normalization

```
exemption_rate_normalized = exemption_rate (already in [0, 1])
```

### Finance-Specific Context

**Why it matters:**
- Algorithmic trading desks constantly push leverage, position size, and speed limits
- "Exemptions" are a pressure valve but also a danger signal
- Rate spike = traders have more power than risk managers
- Rate > 0.10 correlates with prior incidents (2008, 2014, 2020)

**Typical Values by Desk Type:**

| Desk Type | Normal Range | Warning Threshold |
|-----------|--------------|-------------------|
| Proprietary trading | 0.02–0.05 | > 0.10 |
| Market-making | 0.03–0.08 | > 0.12 |
| Prime brokerage | 0.01–0.03 | > 0.08 |

### Audit Questions

- How many exemptions were granted in the past 3 months?
- For each, what policy was exempted? Was it renewed or allowed to lapse?
- How many exemptions requests were *rejected*?
- Who approved exemptions? (same person each time = red flag)

---

## G2: Sanction Delay

### Definition
**Median (or max) number of days between violation detection and enforcement action.**

Examples:
- Trader exceeds leverage limit on Monday
- Violation detected Monday (automated or manual)
- Sanction applied: verbal warning (Tue), suspension (Wed), termination (Fri)
- **Sanction delay** = first enforcement action date − violation date

### Range

| Level | Days | Interpretation |
|-------|------|-----------------|
| **Excellent** | 1–3 | Violations caught and addressed same day / next day |
| **Good** | 3–7 | Quick turnaround; reasonable investigation time |
| **Caution** | 7–30 | Investigation taking weeks; delayed enforcement |
| **Warning** | 30–90 | Enforcement glacially slow; violators unpunished for months |
| **Critical** | > 90 | Violations effectively unpunished (forgotten) |

### Measurement

```
sanction_delay = median(days from violation_date to first_enforcement_date)

Where:
  - violation_date = date rule broken (automated detection or audit discovery)
  - first_enforcement_date = date of any action: warning, suspension, compensation, termination
```

### Data Source

**Primary:**
- Incident tracking system: violations log with [violation_date, detection_date, action_date, action_type]
- Export: weekly CSV

**Secondary:**
- Compliance memo: "Actions taken in Q1: suspension on date X for violation on date Y"
- HR records: termination dates matched to underlying violations

**Missingness Protocol:**
- If no violations detected in period → delay = 1 day (conservative: assume system is working fast)
- If violations exist but no enforcement recorded → delay = 365 days (worst case)
- If enforcement recorded but violation date unknown → estimate from investigation_start date

### Normalization

```
sanction_delay_normalized = min(sanction_delay, 365) / 365
  (scales to [0, 1], caps at 1 year)
```

### Finance-Specific Context

**Why it matters:**
- Fast enforcement deters violations; slow enforcement = impunity
- Delay > 30 days = traders know they can break rules and face consequences much later
- Regulatory view: "Delays of >90 days suggest controls are not real" (Basel, Dodd-Frank guidance)

**Typical Values by Institution:**

| Institution Type | Normal Range | Warning Threshold |
|------------------|--------------|-------------------|
| Systemically important bank | 3–7 days | > 30 days |
| Mid-size hedge fund | 5–15 days | > 60 days |
| Boutique trading firm | 7–21 days | > 90 days |

### Audit Questions

- What was the fastest time from violation to sanction?
- What was the slowest?
- What's the typical investigation duration before action?
- Are there violations with no action yet?

---

## G3: Control Turnover

### Definition
**Fraction of control staff who left (fired or resigned) in the past year.**

Examples:
- Risk department has 20 people on Jan 1
- 2 were fired for cause in Q1
- 1 resigned in Q2
- 1 reassigned to trading in Q3
- **Turnover** = 4 / 20 = 0.20 (20% annualized)

### Range

| Level | Ratio | Interpretation |
|-------|-------|-----------------|
| **Excellent** | 0.00–0.05 | Stable team (5% annual turnover is normal) |
| **Good** | 0.05–0.10 | Some natural attrition |
| **Caution** | 0.10–0.20 | Higher than normal; new people less experienced |
| **Warning** | 0.20–0.40 | Institutional knowledge leaving; team demoralized |
| **Critical** | > 0.40 | Wholesale exodus; no continuity |

### Measurement

```
control_turnover = count(terminations + resignations) / count(staff at year start)

Where:
  - terminations = fired for any reason (performance, compliance, etc.)
  - resignations = voluntary departures
  - excludes: retirements, transfers to other departments, unpaid leave
```

### Data Source

**Primary:**
- HR records: separation log [termination_date, reason, department]
- Include: both "for cause" and "voluntary"
- Export: annual or quarterly aggregation

**Secondary:**
- Finance team headcount report: "Q1 headcount 20, Q2 headcount 19 (one departure)"
- Org chart: compare Jan 1 roster to Dec 31 roster, identify turnover

**Missingness Protocol:**
- If HR data unavailable: estimate from LinkedIn (staff who left in year, if company profile public)
- If no data at all: assume 0.10 (10% annual) as conservative middle estimate

### Normalization

```
control_turnover_normalized = min(turnover, 1.0) (already in [0, 1])
```

### Finance-Specific Context

**Why it matters:**
- Risk managers who have been there 5+ years know where bodies are buried (past violations, weak spots)
- New risk managers can be pressured by traders ("everyone else tolerates this")
- High turnover = knowledge loss = repeat violations of same rules

**Typical Values:**

| Institution Type | Normal Range | Warning Threshold |
|------------------|--------------|-------------------|
| Bank (stable sector) | 0.05–0.15 | > 0.25 |
| Trading firm (high-pressure) | 0.10–0.25 | > 0.40 |
| Startup exchange | 0.15–0.40 | > 0.50 |

### Audit Questions

- Who was on the risk committee 2 years ago? Who is there now?
- How many risk managers left in the past year?
- Were any terminated for compliance reasons?
- What's the average tenure of risk staff?

---

## G4: Conflict of Interest Proxy

### Definition
**Fraction of governance staff with undisclosed (or poorly disclosed) conflicts of interest.**

Examples:
- Risk manager owns personal trading account (conflict: might approve risky trades to profit personally)
- Compliance officer's spouse works on trading desk (conflict: may soften enforcement for family member)
- Undisclosed conflict = didn't fill out annual conflict questionnaire, or questionnaire lacks detail

### Range

| Level | Ratio | Interpretation |
|-------|-------|-----------------|
| **Excellent** | 0.00–0.05 | Conflicts disclosed and managed; rare undisclosed |
| **Good** | 0.05–0.15 | Normal rate; conflicts disclosed and monitored |
| **Caution** | 0.15–0.25 | Significant fraction with conflicts; oversight questions |
| **Warning** | 0.25–0.50 | Most staff have undisclosed or poorly managed conflicts |
| **Critical** | > 0.50 | Majority compromised; no independent governance |

### Measurement

```
conflict_interest_proxy = count(staff with undisclosed conflicts) / count(total governance staff)

Where:
  - undisclosed = no conflict questionnaire on file, or questionnaire blank/evasive
  - governance staff = risk, compliance, audit, internal controls (not trading)
```

### Data Source

**Primary:**
- Conflict questionnaire archive (annual or onboarding)
- Flag: missing questionnaire, or filled with "None" despite working in related departments
- Review: spot-check 20% of staff by hand (ask: "Does this person trade on the side?")

**Secondary:**
- HR file: note any outside affiliations
- LinkedIn: personal trading accounts, external board memberships, consulting

**Missingness Protocol:**
- If no questionnaire on file → assume conflict = 1 (worst case, conservative)
- If questionnaire on file but appears incomplete → spot-check with manager ("Did X disclose all conflicts?")
- If data unavailable: proxy = 0.10 (assume 10% have undisclosed conflicts, middle estimate)

### Normalization

```
conflict_interest_proxy_normalized = conflict_count / total_staff (already in [0, 1])
```

### Finance-Specific Context

**Why it matters:**
- Risk manager who trades personally has incentive to approve risky trades
- Compliance officer with family on trading desk may soften enforcement
- Undisclosed conflicts = "sleeping conflict" (may explode if exposed)

**Typical Values:**

| Institution Type | Normal Range | Warning Threshold |
|------------------|--------------|-------------------|
| Regulated bank | 0.05–0.15 | > 0.25 |
| Hedge fund | 0.10–0.30 | > 0.40 |
| Private equity | 0.15–0.40 | > 0.50 |

### Audit Questions

- When was the last conflict questionnaire updated for each governance staffer?
- Are any governance staff also traders or commission-earning?
- Any family relationships between governance and trading staff?
- Any governance staff who have consulting/board roles elsewhere?

---

## G5: Rule Execution Gap

### Definition
**Fraction of detected violations that go undetected or unaddressed.**

This is tricky: we measure violations by looking at what *got through* without enforcement.

Example:
- Policy: "Max leverage 5x at end of day"
- System records 50 breaches in past quarter
- Enforcement acted on 42 breaches
- 8 breaches were never addressed
- **Gap** = 8 / 50 = 0.16 (16%)

### Range

| Level | Ratio | Interpretation |
|-------|-------|-----------------|
| **Excellent** | 0.00–0.05 | Almost all violations caught and handled |
| **Good** | 0.05–0.10 | High detection/enforcement rate |
| **Caution** | 0.10–0.20 | Meaningful fraction of violations slip through |
| **Warning** | 0.20–0.40 | Controls have significant blind spots |
| **Critical** | > 0.40 | Most violations undetected or unaddressed |

### Measurement

```
rule_execution_gap = count(violations detected but not enforced) / count(violations detected)

Where:
  - violations detected = incidents flagged by system or audit
  - not enforced = incident closed without action (e.g., "no action needed", or forgotten)
```

### Data Source

**Primary:**
- Incident tracking system: [violation_date, enforcement_status, action_type]
- Query: violations with status = "closed without action" or "no follow-up"

**Secondary:**
- Audit report: "20 violations found; enforcement action taken on 18"
- Compliance review: exception list

**Missingness Protocol:**
- If incident system is comprehensive: use it directly
- If incident system missing: estimate from audit findings = (violations found in audit − actions taken) / violations found
- If no audit data: assume gap = 0.10 (conservative estimate)

### Normalization

```
rule_execution_gap_normalized = gap (already in [0, 1])
```

### Finance-Specific Context

**Why it matters:**
- A violation that goes unaddressed = precedent for next violation
- Traders learn: "This breach was not enforced last time, so I'll do it again"
- Gap reflects both detection and enforcement capacity

**Typical Values:**

| Institution Type | Normal Range | Warning Threshold |
|------------------|--------------|-------------------|
| Regulated bank | 0.02–0.08 | > 0.15 |
| Trading firm | 0.05–0.15 | > 0.25 |
| Startup | 0.10–0.30 | > 0.40 |

### Audit Questions

- How many violations were detected in the past quarter?
- How many resulted in enforcement action?
- How many were "no action needed" — who decides that?
- Any violations that were discovered in audit but not in incident system?

---

## Integration: G-Proxies in Audit Verdict

### Gates (Alarm Thresholds)

When computing L_act verdict, if *any* G-proxy crosses its gate:

| Proxy | Gate | Action |
|-------|------|--------|
| exemption_rate | > 0.10 | Flag as "Governance Caution" in audit output |
| sanction_delay | > 90 days | Flag as "Slow Enforcement" in audit output |
| control_turnover | > 0.25 | Flag as "Instability Risk" in audit output |
| conflict_interest | > 0.25 | Flag as "Independence Risk" in audit output |
| rule_execution_gap | > 0.20 | Flag as "Detection Gap" in audit output |

**Consequence:**
- If 0 gates crossed → G status = "OK"
- If 1–2 gates crossed → G status = "CAUTION" (audit still credible, but note issues)
- If 3+ gates crossed → G status = "ALERT" (governance failing; verdict dims to "Partial Credibility")

---

## Version History

| Version | Date | Changes | Status |
|---------|------|---------|--------|
| v1.0.0 | 2026-02-24 | Initial G-proxies for finance: 5 measurable indicators with ranges, sources, and gates | FROZEN |

