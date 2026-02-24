# Definition of Done: Amplification Barometer

**Version:** v0.1.0
**Date:** 2026-02-24
**Scope:** Truth contract and falsifiability claims

---

## Purpose

This document defines **precisely** what it means for the Amplification Barometer to claim "**dit vrai**" (tells the truth):
- What are the measurable criteria?
- What tests must pass?
- What would falsify the claim?
- What are optional vs. mandatory dimensions?

This is the foundation for auditability and scientific credibility.

---

## Three Mandatory Dimensions

### 1. **Construct Validity** (P, O, E, R, G, @(t), Δd(t))

**Claim:** The composites measure what they claim to measure, with stable behavior.

#### Required Tests (MUST PASS)

| Test | Criterion | Source | Falsified By |
|------|-----------|--------|--------------|
| **Stability under window variation** | Spearman rank correlation ≥ 0.85, Jaccard topk ≥ 0.80 across windows [5, 7, 9] | `stability_audit_rank()` | If either metric < threshold |
| **Proxy ranges valid** | No proxies outside expected ranges (see proxies.yaml) | `validate_proxy_ranges()` | If >0 out-of-range points without explicit exemption |
| **Composites are finite** | All composite values (P, O, E, R, G, @, Δd) are finite (no NaN, inf) | `compute_levels_from_specs()` output | Any NaN or inf in composite |
| **E(t) monotone-increasing** | Stock E = cumsum(E_level) is monotone non-decreasing | `compute_e_r()` | If E decreases (negative dE in tail) |
| **MAD-normalized reproducibility** | Results identical when using MAD z-score normalization | `robust_z_mad()` | Differing results with different seed/order |

#### Optional Tests (NICE TO HAVE)

- Alignment with external domain expertise (subject matter review)
- Temporal coherence: @(t) changes smoothly (no spurious jumps)
- Proxy correlation: families (P, O, E, R, G) internally consistent

**Falsification:** Claim invalidated if any MUST PASS test fails.

---

### 2. **Predictive / Event Validity** (Stress signatures & L_act)

**Claim:** When a system enters Type II stress or bifurcates to Type III, the barometer reflects it with bounded latency and does not produce false signals in stable regimes (Type I).

#### Required Tests (MUST PASS)

| Test | Criterion | Source | Falsified By |
|------|-----------|--------|--------------|
| **Type I noise containment** | In pure noise regime (synthetic), @(t) median < 2.0, no drift in E_stock over 200 steps | Synthetic "stable" scenario | If @(t) drifts or E_stock accumulates |
| **Type II oscillation signature** | In oscillating regime, Δd(t) sign alternates ≥60% of tail window, R_level recovers (mean in top 40%) | Synthetic "oscillation" scenario | If Δd monotone or R stays depressed |
| **Type III bifurcation signature** | In bifurcation regime: @(t) tail > 90th percentile ≥50%, E_stock/max(E) ≥0.85, R_tail mean <30% | Synthetic "bifurcation" scenario | If signatures absent or inverted |
| **L_act activation latency** | When system enters stress, detector activates within ≤ 20% of stress window | Real or synthetic event data | If latency > threshold or false positives >5% |
| **No systematic post-hoc signals** | Signal does not peak *after* documented event; median lead = 0 to +10 steps | Timestamped event validation | If signal systematically trails event |

#### Optional Tests (NICE TO HAVE)

- Sub-regime discriminability (Type II vs Type III clearly separable)
- Sensitivity analysis: varying window, noise level doesn't invert verdicts
- Temporal lead-lag validation on >3 real-world cases

**Falsification:** Claim invalidated if any MUST PASS test fails or if >10% of events are missed.

---

### 3. **Anti-Gaming / Manipulability Robustness**

**Claim:** The barometer detects and penalizes systematic attempts to game proxies; improvement via fraud scores lower than legitimate improvement.

#### Required Tests (MUST PASS)

| Test | Criterion | Source | Falsified By |
|------|-----------|--------|--------------|
| **O-family bias detection** | Artificially boost O proxies by +15%: verdict gaming_suspicion="fail", ΔP_illegit < 0.5 * ΔP_clean | `anti_gaming_o_bias()` | If detection fails or illegit gains approach clean gains |
| **Volatility clamp detection** | Suppress volatility (std →0) on P proxies: gaming_suspicion="fail" | `anti_gaming_volatility_clamp()` | If verdict doesn't flag manipulation |
| **Range out-of-bounds gaming** | Push proxies outside expected ranges: gaming_suspicion="fail" | `anti_gaming_out_of_range()` | If undetected excursions pass verdict |
| **Multi-proxy coordinated attack** | Simultaneous bias on P *and* O + delay on G: combined gaming_suspicion="fail" | `anti_gaming_coordinated()` | If multi-proxy triche undetected |
| **Reporting delay gaming** | Suppress G signals by delaying sanction_delay/exemption_rate: gaming_suspicion="fail" | `anti_gaming_reporting_delay()` | If governance manipulation undetected |
| **Gaming score integration** | Any verdict with gaming_suspicion="fail" → overall verdict includes "anti_gaming: FAIL" | `run_alignment_audit()` verdict | If gaming failures not propagated to verdict |

#### Optional Tests (NICE TO HAVE)

- Ensemble robustness: gaming detection consistent across 10 random perturbations
- Attacker model refinement: detect subtle gaming (δ<5%, e.g., micromanipulation)
- Cost-benefit analysis: cost of gaming vs. benefit capped

**Falsification:** Claim invalidated if any MUST PASS anti-gaming test fails.

---

## Verdicts

### Multidimensional Verdict Structure

Each run of `run_alignment_audit()` SHALL produce a JSON verdict with the following structure:

```json
{
  "spec_version": "v1.0",
  "timestamp": "ISO-8601",
  "dimensions": {
    "construct_validity": "ok" | "fail",
    "event_signatures": "ok" | "fail" | "unknown",
    "anti_gaming": "ok" | "fail",
    "overall": "credible" | "suspect" | "not_ready"
  },
  "details": {
    "construct": { "stability": {...}, "ranges": {...}, "finitude": {...} },
    "events": { "type_i": {...}, "type_ii": {...}, "type_iii": {...} },
    "anti_gaming": { "o_bias": {...}, "vol_clamp": {...}, "coordinated": {...}, "score": 0.0-1.0 },
  }
}
```

### Overall Credibility Verdict

**"credible"** := construct_validity="ok" AND (event_signatures="ok" OR "unknown") AND anti_gaming="ok"

**"suspect"** := construct_validity="ok" AND anti_gaming="ok" AND event_signatures="fail"
→ Signals observed but may be post-hoc; needs real-world validation.

**"not_ready"** := Any MUST PASS test fails in construct_validity or anti_gaming.

---

## Explicit Falsifiability Examples

To prove we are scientific, not narrative:

### Example 1: Construct Validity Failure
- Data: pure random noise, n=500
- Expected: @(t) bounded, no drift in E_stock, regime="type_I_noise"
- **Falsified if:** E_stock rises linearly, regime="type_III_bifurcation"
→ Shows we can detect broken composites.

### Example 2: Event Signature Failure
- Data: synthetic bifurcation injected at t=250
- Expected: @(t) spike at t=250±20, regime="type_III" by t=300
- **Falsified if:** Signal peaks at t=350, or regime stays "type_I"
→ Shows we can fail to detect real stress.

### Example 3: Anti-Gaming Failure
- Data: clean baseline, then O-family artificially boosted +15% from t=300
- Expected: gaming_suspicion="fail", verdict="suspect"
- **Falsified if:** gaming_suspicion="ok", verdict="credible"
→ Shows we can be gamed.

---

## Scope & Exclusions

### In Scope for "dit vrai"
- Construct validity of composites (P, O, E, R, G, @, Δd)
- Detection of canonical stress signatures (Type I/II/III)
- Resistance to systematic proxy manipulation
- Reproducibility of verdict under reasonable parameter variation

### Out of Scope (Next Phases)
- Sector-specific calibration (proxy ranges vary; handled in Phase 2)
- Real-world validation on 10+ timestamped cases (Phase 5)
- L_cap vs L_act separation (Phase 4)
- Auditability manifests & append-only history (Phase 5)

---

## Process: When to Invoke DoD

1. **Before release:** Run full DoD checklist; all MUST PASS must ✅
2. **On failure:** Document in postmortem; fix root cause before retry
3. **On ambiguity:** Err conservative; mark as "unknown" rather than guessing
4. **On sector variation:** Document exemptions explicitly (e.g., "IA sector proxy ranges differ per Phase 2 spec")

---

## Version History

| Version | Date | Change |
|---------|------|--------|
| v0.1.0 | 2026-02-24 | Initial definition. Construct + Event + Anti-Gaming. |

---

## Related Documents

- `docs/theory_audit_mapping.md` – Theoretical mapping
- `docs/proxies.yaml` – Current proxy specs (demo ranges)
- `src/amplification_barometer/alignment_audit.py` – Implementation
- `tools/run_alignment_audit.py` – CLI entrypoint
- `tools/run_anti_gaming_suite.py` – Anti-gaming suite (Phase 1)

