# Case 006: Cloud Service Provider Regional Outage

**Date Range:** March 15 - April 5, 2024
**Sector:** Technology/Infrastructure
**Status:** ◐ **PARTIAL**

## Facts

**System:** CloudSync Infrastructure (top-3 cloud provider, US-WEST region)
**What Happened:**
- March 15: Network switch failure in US-WEST-2
- March 16-18: Cascading failures across 3 availability zones
- March 19: Failover to backup region (degraded performance)
- March 22: Root infrastructure replaced
- April 2: Full redundancy restored
- April 5: Performance returned to baseline

**Impact:** 2,400 customer applications affected, 87-hour regional outage, estimated $180M lost revenue

## Ground Truth

**Event Date:** March 15, 2024
**Regime:** Type III bifurcation (infrastructure + service quality collapse)
**Source:** Cloud provider incident report, customer SLA violation notices, tech news coverage

## Barometer Results

| Metric | Value | Status |
|--------|-------|--------|
| Signal Peak | March 15 | ✅ Same-day detection |
| Regime Detected | Type III ✓ | Correct |
| E Irreversibility | 0.71 | ≥0.70 threshold PASS |
| R Recovery | 0.43 | Slow recovery (degraded mode) |
| Anti-gaming | All 5 attacks FAIL | ✅ Robust |
| Temporal Lag | 0 days | Perfect timing |

## Analysis

**What Worked:**
- Immediate bifurcation detection
- Type III classification correct
- E accumulation tracked trust erosion
- Anti-gaming all vectors passed

**What Missed:**
- Recovery lag (R stayed depressed for 18 days post-incident)
- Governance response signals unclear (provider had redundancy but it failed)

**Why:** Cloud infrastructure failures have long recovery tails. Framework correctly identified bifurcation but predicted faster recovery than observed in practice.

## Verdict

**Status:** ◐ PARTIAL (Correct detection, Recovery lag unexplained)

Framework successfully detected cloud service bifurcation. Recovery curve slower than predicted - suggests need for sector-specific R calibration for infrastructure vs. finance.
