# Improvements summary

This repository has been realigned with the corrected methodological document.

Main changes:

- Public notation changed from `@(t)` to `rho(t)`.
- Legacy `AT` output preserved for backward compatibility.
- `compute_rho()` now works on positive operational levels and floors near-zero orientation during stress tests to produce a finite warning signal.
- `G_LEVEL` is bounded in [0, 1]. `G_SCORE` remains the robust unbounded score.
- ODE local stability diagnostics now expose active equilibrium, Hurwitz coefficients and a scoped stability verdict.
- README, index and project structure were rewritten to avoid overclaiming.
- Real cases are explicitly marked as exploratory unless independently reviewed.
- Methodology tests were added.

Validation:

```bash
pytest -q
```

Current local result: 65 passed.
