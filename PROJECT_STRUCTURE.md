# Project structure

This file summarizes the repository after the methodological alignment patch.

## Main purpose

The repository implements a reproducible experimental audit framework for amplification regimes. It computes proxy-based indicators and stress signatures. It must not be presented as a universal prediction engine.

## Main directories

```text
.
├── src/amplification_barometer/     # Python package
├── tools/                           # CLI utilities and audit runners
├── tests/                           # Unit, smoke and methodology tests
├── data/                            # Synthetic and fixture datasets
├── docs/                            # Methodology and audit documentation
├── reports/real_cases/              # Exploratory real case material
├── apps/                            # Dashboard helper
└── .github/workflows/               # CI workflows
```

## Important modules

- `composites.py`: computes P, O, E, R, G_SCORE, G_LEVEL, rho and Δd.
- `ode_model.py`: ODE demonstrations, active equilibrium, Hurwitz local stability diagnostics.
- `audit_report.py`: builds structured audit reports.
- `l_operator.py`: L_cap, L_act and limiting-operator performance.
- `manipulability.py`: anti-gaming checks.
- `real_data_adapters.py`: adapters for selected data fixtures.

## Public notation

`rho(t)` is the public ratio notation. Internally, old reports may still contain `AT` as a compatibility alias.

## Methodological guardrails

- Local stability diagnostics are local, parameter-dependent and model-dependent.
- Real data conclusions depend on proxy quality and source documentation.
- G(t) must be distinguished between `G_SCORE` and bounded `G_LEVEL`.
- Real case studies are exploratory unless separately reviewed.

## Basic checks

```bash
pytest -q
python -m py_compile src/amplification_barometer/*.py
```

## Recommended reading order

1. `README.md`
2. `docs/METHODOLOGICAL_SCOPE.md`
3. `docs/KNOWN_LIMITATIONS_AND_FAILURE_MODES.md`
4. `docs/PUBLIC_TEST_MATRIX.md`
5. `reports/real_cases/VALIDATION_SUMMARY.md`
