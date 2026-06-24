# Methodological scope

This repository implements an experimental audit framework. It is designed to detect regime signatures, not to deliver exact predictions.

## Safe interpretation

The outputs should be read as signals of regime structure:

- rho(t) indicates the relative tension between amplification capacity and orientation capacity.
- Δd(t) indicates whether amplification is growing faster than orientation.
- E(t) and E_stock describe externality flow and accumulation.
- R(t) describes resilience proxies.
- G_SCORE tracks governance variation as a robust score.
- G_LEVEL maps governance risk proxies into a bounded [0, 1] level.
- L_cap and L_act separate tested limiting capacity from observed activation.

## What the ODE model proves

The 4D ODE model gives a local formal diagnostic. If an active equilibrium exists and the Hurwitz conditions fail, the linearized system is not locally asymptotically stable in the parameterized model.

That statement is mathematical but limited. It is valid only for:

- the specified ODE structure;
- the supplied parameters;
- the neighborhood of the active equilibrium;
- the chosen proxy mapping and normalization.

## What the ODE model does not prove

The model does not prove:

- the exact timing of a real event;
- the size of the basin of attraction;
- the impact of unmodeled network topology;
- the drift of parameters over time;
- the legitimacy or illegitimacy of a real institution;
- a universal law of collapse.

## Data caveat

The framework is only as strong as its proxy design and data provenance. Any real deployment must document:

- proxy definitions;
- source and frequency;
- missing data handling;
- normalization method;
- weights;
- thresholds;
- audit window;
- limitations and indeterminate points.

## Public claim boundary

The safest public claim is:

> This framework detects auditable signatures of amplification, dissociation, externality accumulation and resilience erosion. It is useful for regime diagnostics and falsifiable stress testing. It is not a digital twin and does not replace domain expertise.
