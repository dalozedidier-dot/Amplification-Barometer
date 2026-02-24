# Real-World Case Studies

**Purpose:** Validate barometer against timestamped, independent events.

## Structure

Each case study is a self-contained directory:

```
real_cases/
├── README.md (this file)
├── case_001_finance_algo_trading_2024/
│   ├── CASE_SUMMARY.md
│   ├── timeline.csv (dates of events + barometer readings)
│   ├── proxies.csv (cleaned data)
│   ├── event_dates.json (independent event timestamps)
│   ├── audit_output.json (barometer verdict)
│   └── analysis.md (what worked, what didn't)
├── case_002_ai_model_deployment_2024/
│   └── (same structure)
├── case_003_power_grid_incident_2024/
│   └── (same structure)
└── VALIDATION_SUMMARY.md (aggregate across cases)
```

## Case Requirements

### Minimum Criteria
1. **Independent event dates** – From external source (incident reports, regulatory filings, news)
2. **Proxy data** – Aligned with barometer specs (22 proxies, consistent frequency)
3. **Temporal coherence** – Signal should not lag events by >30% of observation window
4. **No retuning** – Same proxy ranges and rules across all cases

### Optional (Nice to Have)
- Multi-day incident (shows accumulation of E, degradation of R)
- Multiple independent events in same dataset (tests regime transitions)
- Real governance signals (exemptions, sanctions, turnover)
- Post-incident postmortem (what actually failed)

## Template: Case Summary

```markdown
# Case: [Name] - [Date Range]

## Facts
- **Domain:** Finance / AI / Infrastructure
- **System:** [Name and brief description]
- **Duration:** [dates]
- **Event:** [What happened, in plain English]

## Ground Truth
- **Event Date(s):** ISO-8601 dates from independent source
- **Outcome:** [What was the impact: losses, downtime, incidents?]
- **Source:** [Regulatory filing, news report, incident tracking, postmortem]

## Barometer Result
- **Signal Detection:** [Did @(t) or E spike before/during/after event?]
- **Lead/Lag:** [Days relative to event date]
- **Regime Detected:** Type I / Type II / Type III / Missed
- **L_cap / L_act:** [Values]
- **Verdict:** Success / Miss / False Alarm / Benign

## Analysis
- **What it got right:** ...
- **What it missed:** ...
- **Why:** Root cause analysis
```

## Validation Rules

### Temporal Coherence
- Signal should peak **0 to +10 days** before independent event date (not after)
- Lag >30% of window = potential post-hoc artifact

### Regime Correctness
- Type II scenario should show oscillations, not monotone growth
- Type III should show E accumulation + O collapse, not just noise
- No regime should flip with ±5% noise perturbation

### Governance Quality
- If case shows Type III, G proxies should degrade before O collapse
- If case is MISS, governance signals should explain why (high exemptions, turnover, etc.)

## Case Status Indicators

| Status | Meaning | Action |
|--------|---------|--------|
| ✓ | Published & validated | Reference case |
| ◐ | Partial data or ambiguous | Pending more evidence |
| ✗ | Contradicts barometer claim | Post-mortem & redesign |
| ? | Not yet analyzed | Under investigation |

## Examples (To Be Filled)

### Case 001: [Finance] ✓
- Event: [Name]
- Result: [Success / Miss / etc.]
- Status: Published

### Case 002: [AI] ◐
- Event: [Name]
- Result: [Success / Miss / etc.]
- Status: Pending

### Case 003: [Infrastructure] ✓
- Event: [Name]
- Result: [Success / Miss / etc.]
- Status: Published

## Running Validation

```bash
python3 tools/validate_real_cases.py --cases-dir reports/real_cases
```

Output: `VALIDATION_SUMMARY.md` with:
- Total cases analyzed
- Success rate
- Common patterns (what barometer gets right/wrong)
- Identified weaknesses requiring redesign

## Adding a New Case

1. Create `reports/real_cases/case_NNN_[sector]_[event]_[year]/`
2. Place CASE_SUMMARY.md, proxies.csv, event_dates.json, audit_output.json
3. Run audit: `python3 tools/run_alignment_audit.py --dataset proxies.csv`
4. Fill analysis.md
5. Update this README with new case
6. Run validation: `python3 tools/validate_real_cases.py`

