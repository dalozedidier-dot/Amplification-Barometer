# Recommendations for passage to a real audit

This repository demonstrates partial auditability on synthetic data:
reproducible regime discrimination, stability checks, stress tests, and anti-gaming heuristics.

## Immediate improvements

1. Recalibrate weights on anonymized incidents
- Collect a small number of incident windows with a single damage_weight target
- Run tools/recalibrate_weights.py to generate a versioned JSON proposal
- Review and pin any change in a pull request

2. Add targeted anti-gaming tests on O(t)
- Simulate explicit bias injection on O proxies
- Require that at least one detector triggers when the risk reduction is significant

3. Validate L(t) empirically with enforcement constraints
- Use a clear rule: control_turnover mean < 5% is required for high L_cap credibility
- Report both cap_score_raw and cap_score_enforced, and track the enforcement factor

## Theoretical extensions

- Integrate non-linear feedback loops where G(t) is endogenous, not only a measured proxy
- Instantiate the barometer on real sectors (AI, finance) with anonymized 2026 data

## Global validity statement (demo)

- This demo shows reproducible tests and audit-friendly artefacts, but it is not an empirical proof
- Current scoring target for the demo narrative: ~62% overall (maturity ~70%, stability ~55%)
- Next iteration should focus on better synthetic diversity and real-incident anchoring
