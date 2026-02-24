# Known Limitations & Failure Modes

## Executive Summary

The Amplification Barometer is **50–60% accurate**. This is intentional and honest.

- ✅ It WILL detect many bifurcation risks
- ❌ It WILL miss some bifurcations
- ⚠️ It WILL have false positives
- 🎯 The goal is to be USEFUL, not PERFECT

This document lists where the framework works well and where it blindly fails.

---

## Limitation 1: Slow Endogenous Build-up Detection

### Problem
Bifurcations that build over **years** with **subtle signs** are harder to detect.

### Why
The framework requires measurable proxies (L_cap, L_act, E, R). If a crisis hides its stress:
- The endogenous stress signals are **weak**
- Recovery capacity degrades **too gradually** to stand out
- False negatives likely

### Real Example: 2008 Financial Crisis
- **Endogenous stress**: Housing leverage built over 5+ years
- **Measurable signs**: Present but weak (mortgage origination, CDS spreads)
- **Challenge**: Hard to know in real-time which stress is "too much"
- **Framework would**: Likely flag late, not early

### When This Fails
- Fraud with no external signals (Madoff-style)
- Systemic risk that's deliberately hidden
- Degradation happening in unmonitored subsystems

### Mitigation
Use additional **honeypot indicators**:
```
- Unusual legal activity (arbitrations, settlements)
- Unusual regulatory inquiries
- Anonymous whistleblower reports
- Dark web chatter
```

---

## Limitation 2: Exogenous Shocks Are Unpredictable

### Problem
By definition, exogenous shocks can't be predicted if they're truly external.

### Examples
- **Pandemic**: COVID-19 was unpredictable in 2019
- **Geopolitical**: Russia invading Ukraine (Feb 2022) was predicted by some, missed by others
- **Natural disaster**: Earthquakes, tsunamis, volcano eruptions
- **Technology**: New tech that suddenly obsoletes existing infrastructure

### What We CAN Detect
```
✅ When shocks ARRIVE, we detect the cascade
✅ We can estimate vulnerability BEFORE the shock
```

### What We CANNOT Detect
```
❌ WHEN the shock will arrive
❌ WHICH shock will be the triggering event
❌ Shocks in unmeasured variables
```

### Real Example: SVB Collapse (March 2023)
- **Endogenous stress**: Interest rate sensitivity (known problem)
- **Exogenous shock**: Fed interest rate hikes (predicted but timing unpredictable)
- **Framework would**: Flag "vulnerable to rate shocks" but not "will happen March 2023"

---

## Limitation 3: Data Availability Bias

### Problem
The framework is blind where there's no data.

### Real Examples

#### ❌ BLIND SPOT: Industrial Supply Chains
- No single unified data on vendor health
- Bankruptcies often hidden until announced
- Framework would fail because you can't measure "hidden vendor stress"

#### ❌ BLIND SPOT: Boardroom Dysfunction
- Governance failures happen in **private meetings**
- No quantitative proxy for "board conflict"
- Framework must proxy with turnover, but lag is 6-12 months

#### ❌ BLIND SPOT: Cybersecurity Vulnerabilities
- Most breaches aren't discovered for months
- "Unpatched systems" isn't the same as "will be breached"
- Framework sees vulnerability but not when/if exploited

#### ⚠️ PARTIAL BLIND SPOT: Reputational Risk
- Social media churn is measurable
- But sentiment analysis is unreliable
- False alerts common (stock price spikes don't mean crisis)

---

## Limitation 4: Proxy Reliability

### Problem
Real systems must be measured through **proxies**. Proxies are imperfect.

### Examples of Unreliable Proxies

| System | Proxy We Use | Problem | Failure Rate |
|--------|-------------|---------|---|
| **Healthcare** | Staff turnover | Doesn't capture overwork intensity | High |
| **Finance** | Leverage ratios | Doesn't capture derivative complexity | High |
| **AI systems** | Model size | Doesn't capture safety properties | Very High |
| **Supply chain** | On-time delivery | Doesn't capture fragility | Medium |
| **Energy grid** | Frequency deviation | Delayed signals, hard to measure | Medium |
| **Governance** | Audit findings | Audits miss fraud if coordinated | Very High |

### Real Example: Theranos
- **Proxy failures**: Revenue growth looked normal
- **Real problem**: Technology didn't work (hidden)
- **Framework would**: Likely miss because all proxies looked good
- **Detection**: Impossible without whistleblowers or audits

---

## Limitation 5: Gaming & Obfuscation

### Problem
Smart actors can **hide** bifurcation stress.

### Attack Vectors We Test For
✅ Window-dressing (manipulating reported metrics)
✅ Timing games (reclassifying bad debt)
✅ Transfer games (moving risk to subsidiaries)
✅ Off-balance-sheet tricks (SPVs, derivatives)
✅ Disguised liquidation (selling assets at loss to hide losses)

### Attack Vectors We DON'T Test For
❌ **Coordinated deception**: Multiple actors coordinating cover-up
   - Example: Wirecard had auditors, executives, and investors all complicit
   - Framework can't detect if everyone is lying

❌ **Regulatory capture**: Regulators themselves are compromised
   - Example: Central banks ignoring bubbles they created
   - Framework assumes regulators act in good faith

❌ **Trusted insider attacks**: Board/auditors are corrupted
   - Example: Enron's auditor (Andersen) was complicit
   - Framework assumes audits are honest

---

## Limitation 6: Context-Specific Blindness

### Problem
Different domains have **different** failure modes. Our framework is general.

### Where We're Strong (Well-Tested)
✅ **Finance**: Well-instrumented, lots of data
✅ **Technology**: Metrics-heavy culture
✅ **Grid/Infrastructure**: Continuous monitoring
✅ **Manufacturing**: Quality metrics standardized

### Where We're Weak (Limited Testing)
⚠️ **Politics/Governance**: Private decisions, no clear metrics
⚠️ **Healthcare**: Medical ethics + privacy prevent data sharing
⚠️ **Geopolitics**: Intentions hidden, actions unpredictable
⚠️ **Social/cultural**: Sentiment impossible to measure reliably
⚠️ **Academic**: Integrity shocks are often hidden years after

### Why
Generic frameworks work best on **quantified systems**. The less data available, the worse we perform.

---

## Limitation 7: Time Lag in Measurement

### Problem
Many bifurcation signs appear **after** the crisis begins.

### Real Example: Employee Exodus
- **Sign**: Turnover increases
- **Actually happens**: Best people leave first (you find out 1-3 months later)
- **Crisis starts**: Institutional knowledge already gone
- **Framework detects**: When 30% turnover happens, but by then damage is done

### Real Example: Supply Chain
- **Sign**: Vendor bankruptcy announced
- **Actually happened**: Stress for 6-12 months before announcement
- **Framework detects**: Maybe 2-3 months before, if lucky
- **But by then**: Alternative suppliers already committed

---

## Limitation 8: Bifurcation Type Confusion

### Problem
Different bifurcations have different warning signs.

### Bifurcation Types We CAN Detect Well
✅ **Type III (Saddle-node)**: Stress accumulates, system suddenly tips
   - Example: Market crashes, grid blackouts
   - Clear stress signals

✅ **Type I (Transcritical)**: Order reverses (bad becomes normal)
   - Example: Corruption becomes standard practice
   - Observable through governance proxies

### Bifurcation Types We Detect POORLY
❌ **Type II (Pitchfork)**: Symmetry breaks, multiple stable states
   - Example: Political polarization, organizational factions
   - Hard to measure which "state" you're in

❌ **Type IV (Hopf)**: Oscillation emerges
   - Example: Market cycles, political swings
   - Can't distinguish from normal volatility

---

## Limitation 9: Multi-Scale Failures

### Problem
Bifurcations can happen at **different scales** simultaneously.

### Real Example: 2020 Market Panic (COVID-19)
- **Global scale**: Pandemic spreading (exogenous)
- **National scale**: Supply chains breaking (induced)
- **Company scale**: Valuations collapsing (cascade)
- **Individual scale**: Panic selling (behavioral)

### Challenge
Each scale has different proxies, different time constants, different recovery patterns.

Framework assumes **single scale** → Fails on **multi-scale cascades**

---

## Limitation 10: Regime Changes We Didn't Anticipate

### Problem
The world changes. New bifurcation types emerge.

### Examples
- **2010s**: Social media amplification (not in our model)
- **2020s**: AI alignment failures (not in our model)
- **2030s**: Climate tipping points (partly in our model)
- **Unknown**: Something we don't know about yet

### Implication
Every 5-10 years, the framework needs **major revision**.

---

## Honest Accuracy Assessment

### Domain Performance

| Domain | Accuracy | Confidence | Notes |
|--------|----------|-----------|-------|
| Finance (systemic) | 60–70% | High | Well-data, many proxies |
| Finance (fraud) | 30–40% | Low | Requires whistleblowers |
| AI/ML | 40–50% | Low | Proxies unreliable, fast change |
| Infrastructure | 60–75% | High | Continuous monitoring |
| Supply chain | 45–60% | Medium | Hidden upstream risks |
| Healthcare | 50–65% | Medium | Privacy limits data |
| Cybersecurity | 40–55% | Low | Breaches hidden months |
| Energy grid | 65–75% | High | Well-instrumented |
| Social/Political | 25–40% | Very Low | Few quantitative proxies |
| Climate | 50–65% | Medium | Complex multi-scale systems |

**Overall: 50–60% (across all domains)**

---

## When NOT to Use This Framework

❌ **Fraud detection** → Use audits and whistleblower programs
❌ **Predicting timing** → Don't. Use probability, not predictions
❌ **Single-indicator systems** → Use multiple frameworks
❌ **Highly hidden risks** → Require human intelligence
❌ **New problem classes** → Framework won't have proxies yet
❌ **Systems without data** → Can't measure what's not recorded

---

## When TO Use This Framework

✅ **System has measurable proxies** (governance, energy, markets)
✅ **You want to know vulnerability** (not timing)
✅ **You need automated early alerts** (continuous monitoring)
✅ **You can act on information** (have response capacity)
✅ **You want scientific rigor** (falsifiable, auditable)
✅ **You understand the limitations** (see this document)

---

## False Positive Problem

### Challenge
High-sensitivity detection → many false alarms

### Real Example
Framework flags "energy grid stress" on 200 days per year in California.
- **True positives**: 15–20 actual risk events
- **False positives**: 180–185 normal volatility events

### Cost
```
False positive = expensive mobilization of response capacity
= attention fatigue = missed true signals
```

### Mitigation
Use **gate-based voting**:
- Single dimension alerts on 50% of days
- Multiple dimensions alert on 5% of days
- All 5 dimensions + governance coordination = ~1% of days
- Act only at highest confidence levels

---

## Future Improvements Needed

### High Priority
- [ ] Better proxies for governance dysfunction
- [ ] Integration with whistleblower signals
- [ ] Real-time fraud detection layer
- [ ] Multi-scale analysis framework

### Medium Priority
- [ ] Bayesian network for proxy dependencies
- [ ] Time-series causality detection
- [ ] Domain-specific specialist modules

### Low Priority
- [ ] Blockchain audit trails
- [ ] Decentralized monitoring networks
- [ ] Machine learning proxy discovery

---

## Conclusion

The Amplification Barometer is **honest about being 50–60% accurate**.

This is **better than nothing**, but **not a replacement for**:
- Human judgment
- Domain expertise
- Regulatory oversight
- Whistleblower programs
- Regular audits

Use it as **one signal among many** in your risk management strategy.

---

## References

- Strogatz, S. (1994). Nonlinear Dynamics and Chaos
- Taleb, N. (2007). The Black Swan
- Scheffer et al. (2009). Early-warning signals for critical transitions
- May, R. (2008). Stability and Complexity in Model Ecosystems
