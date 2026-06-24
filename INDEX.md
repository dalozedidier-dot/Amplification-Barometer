# Amplification Barometer

This index supersedes the older promotional summary previously stored in this repository.

The repository should now be read as a research prototype and audit framework, not as a production-ready validation package.

## Safe status

- The synthetic and unit tests are reproducible.
- The real case material is exploratory unless independently reviewed.
- The ODE model provides local stability diagnostics, not a digital twin.
- `rho(t)` is the public notation for the P/O ratio. `AT` remains only as a legacy internal alias.
- `G_LEVEL` is bounded in [0, 1]. `G_SCORE` is an unbounded robust variation score.

## Start here

1. Read `README.md`.
2. Read `docs/METHODOLOGICAL_SCOPE.md`.
3. Run `pytest -q`.
4. Run an audit on a known synthetic fixture.
5. Only then test real data with documented proxy provenance.

## Main files

- `README.md`: current overview and safe claim boundary.
- `docs/METHODOLOGICAL_SCOPE.md`: methodological scope and limits.
- `docs/KNOWN_LIMITATIONS_AND_FAILURE_MODES.md`: known failure modes.
- `docs/PUBLIC_TEST_MATRIX.md`: falsifiable test matrix.
- `docs/L_CAP_VS_ACT_FRAMEWORK.md`: separation of L_cap and L_act.
- `reports/real_cases/VALIDATION_SUMMARY.md`: real case status, with caution.

## Current safe claim

The Amplification Barometer detects auditable signatures of amplification, dissociation, externality accumulation and resilience erosion. It is useful for regime diagnostics and falsifiable stress testing. It does not predict exact events and does not replace domain expertise.
