# Recommendations for passage to a real audit

This repository demonstrates partial auditability on synthetic data:
reproducible regime discrimination, stability checks, stress tests, anti-gaming heuristics, and L(t) performance metrics.

## Immediate improvements

1. Boost prevented_exceedance > 10% with proactive L activation
- Use a proactive trigger: activate L when risk is high OR when O(t) is degraded (low O_level)
- Track prevention using both:
  - prevented_exceedance_rel (count-based)
  - prevented_topk_excess_rel (tail severity reduction, more stable)
- Keep persistence and max_delay explicit to limit false positives

2. Recalibrate weights on anonymized incidents (and exogenous shocks when available)
- Collect incident windows with damage_weight and, if possible, u_exog (shock intensity)
- Run tools/recalibrate_weights.py to generate a versioned JSON proposal (audit-friendly)
- Review and pin any change in a pull request

3. Calibrate AI on real E(t) shocks (anonymized)
- For AI-like systems, externalities E(t) are often driven by network waves
- Provide a u_exog series derived from audit logs (rate-limits, incident bursts, propagation anomalies)
- Validate that E proxies (propagation/hysteresis/impact) respond in the expected direction and latency

## Extensions

- Dataset 2027+ and biotech instantiation
  - finance: shocks primarily on P(t)
  - AI: shocks primarily on E(t) network propagation
  - biotech: shocks on E(t) with stronger hysteresis and recovery costs
- Fully endogenize G(t)
  - governance proxies are derived from P/O/E pressure (rule_execution_gap target < 5% in mature regimes)
  - avoid injecting G as an independent synthetic time series

## Global validity statement (demo)

- This demo shows reproducible tests and audit-friendly artefacts, but it is not an empirical proof
- Main residual risks:
  - absence of real quantified cases
  - remaining arbitrariness in weights without enough incidents (mitigate via incident + u_exog anchoring)
- Next iteration should expand sector coverage and integrate real-anonymized incident windows
