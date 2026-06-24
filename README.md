# The Amplification Barometer

A research repository for auditing amplification regimes in complex systems.

The project is not a digital twin and does not claim to predict exact events. It provides a reproducible way to compute regime signatures from documented proxies: P(t), O(t), E(t), R(t), G(t), rho(t), Δd(t), L_cap and L_act.

## What this repository does

The repository implements an experimental audit pipeline for systems where capacity, speed, automation and coupling can grow faster than orientation, constraint, recovery or governance.

It can help detect:

- endogenous stress buildup;
- divergence between amplification and orientation;
- externality accumulation;
- erosion of resilience;
- weak or delayed activation of limiting mechanisms;
- local ODE stability loss under explicit model assumptions.

It does not, by itself, prove that a real system will collapse. Its outputs are regime diagnostics, not event predictions.

## Scope and limits

The ODE model is a formalization tool. It is useful for expressing couplings and local stability conditions, but it is not a full simulation of reality.

Main limits:

- Hurwitz diagnostics are local around an active equilibrium.
- The basin of attraction is not inferred automatically.
- Parameters are treated as fixed unless a specific run models drift.
- Network effects are only partially represented unless a separate network module is used.
- G(t) has two forms: a robust score for variation, and G_level in [0, 1] for bounded institutional interpretation.
- Real data quality dominates the validity of any conclusion.

See `docs/METHODOLOGICAL_SCOPE.md` for the formal scope statement.

## Core notation

| Concept | Meaning | Output |
|---|---|---|
| P(t) | Amplification capacity | `P`, `P_LEVEL` |
| O(t) | Orientation and operational constraint | `O`, `O_LEVEL` |
| rho(t) | Ratio `P_LEVEL / (O_LEVEL + eps)` | `RHO`, legacy `AT` |
| Δd(t) | Differential drift `dP/dt - dO/dt` | `DELTA_D` |
| E(t) | Externalities | `E`, `E_LEVEL`, `E_STOCK` |
| R(t) | Resilience | `R` |
| G(t) | Governance or legitimacy proxy | `G_SCORE`, `G_LEVEL` |
| L_cap | Tested limiting capacity | report fields |
| L_act | Observed activation of limit | report fields |

`AT` is kept as a legacy internal alias for compatibility. Public reports should prefer `rho(t)` or `RHO`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

Run an audit:

```bash
python tools/run_alignment_audit.py \
  --dataset data/synthetic/stable_regime.csv \
  --sector finance \
  --out-dir _ci_out/audit_demo
```

Run the public test matrix:

```bash
python tools/run_public_tests.py \
  --dataset data/synthetic/stable_regime.csv
```

## Local ODE stability diagnostics

The repository now exposes explicit Hurwitz diagnostics for the 4D ODE model.

```python
from amplification_barometer.ode_model import (
    BarometerParams,
    compute_active_equilibrium,
    assess_hurwitz_local_stability,
)

params = BarometerParams()
eq = compute_active_equilibrium(params)
diag = assess_hurwitz_local_stability(params, eq)
print(diag.status)
print(diag.conditions)
```

Interpretation:

- `STABLE_LOCAL` means the linearized system is locally asymptotically stable under the supplied parameters.
- `UNSTABLE_LOCAL` means at least one Hurwitz condition fails.
- `INFEASIBLE_EQUILIBRIUM` means the active equilibrium is not valid in the positive domain.

This is a local mathematical diagnostic. It does not estimate the full basin of attraction and does not replace empirical validation.

## Important documents

- `docs/METHODOLOGICAL_SCOPE.md`: scope, limits and safe interpretation.
- `docs/KNOWN_LIMITATIONS_AND_FAILURE_MODES.md`: known limits and failure modes.
- `docs/PUBLIC_TEST_MATRIX.md`: falsifiable tests.
- `docs/L_CAP_VS_ACT_FRAMEWORK.md`: separation between capacity and activation.
- `reports/real_cases/VALIDATION_SUMMARY.md`: real case status and caution notes.

## Validation status

The repository contains synthetic scenarios, public tests and real case studies. The real case material is exploratory unless independently reviewed and linked to full source data.

Current safe claim:

> The Amplification Barometer is a reproducible experimental framework for detecting regime signatures in systems with endogenous amplification. It supports falsification, audit trails and local stability diagnostics, but it does not provide universal prediction.

## Development checks

```bash
pytest -q
python -m py_compile src/amplification_barometer/*.py
```

The current test suite passes locally after the methodological alignment patch.

## License

MIT.
