# Canonical Scenarios v0.1.0

Deterministic, reproducible stress scenarios for barometer calibration.

## Structure

```
canonical_scenarios/
├── README.md (this file)
├── type_i_noise_base.csv
├── type_i_noise_0.05.csv
├── type_i_noise_0.10.csv
├── type_i_noise_0.15.csv
├── type_i_noise_0.20.csv
├── type_i_noise_0.25.csv
├── type_ii_oscillations_base.csv
├── type_ii_oscillations_0.05.csv
├── ... (5 noise variants each for type_ii, type_iii, hybrid)
└── EXPECTED_VS_OBSERVED.csv (CI artifact)
```

## Generation

Run: `python3 tools/generate_canonical_scenarios.py`

All scenarios generated with fixed seeds for reproducibility.

## Scenario Specs

See `docs/calibration_protocol.md` for detailed expected signatures.

- **type_i_noise:** Pure stochastic variation, regime="type_I_noise"
- **type_ii_oscillations:** Cyclical stress with recovery, regime="type_II_oscillations"
- **type_iii_bifurcation:** Irreversible stress & accumulation, regime="type_III_bifurcation"
- **hybrid_ii_to_iii:** Transition from oscillations to bifurcation, regime changes mid-run

Each scenario has 5 noise-variant CSVs (noise_level ∈ {0.05, 0.10, 0.15, 0.20, 0.25}).

## Columns

All CSVs contain the 22 proxy columns (P, O, E, R, G families) as defined in `docs/proxies.yaml`:

```
date, scale_proxy, speed_proxy, leverage_proxy, autonomy_proxy, replicability_proxy,
stop_proxy, threshold_proxy, decision_proxy, execution_proxy, coherence_proxy,
impact_proxy, propagation_proxy, hysteresis_proxy,
margin_proxy, redundancy_proxy, diversity_proxy, recovery_time_proxy,
exemption_rate, sanction_delay, control_turnover, conflict_interest_proxy, rule_execution_gap
```

## Expected Verdicts

| Scenario | Base | Noise Variants | Regime | Stability | Anti-Gaming |
|----------|------|---|--------|-----------|-------------|
| type_i_noise | type_I_noise | All type_I_noise | type_I_noise | ok | ok |
| type_ii_oscillations | type_II_oscillations | All type_II_oscillations | type_II_oscillations | ok | ok |
| type_iii_bifurcation | type_III_bifurcation | All type_III_bifurcation | type_III_bifurcation | ok | ok |
| hybrid_ii_to_iii | Type II → III transition | All show transition | type_II→type_III | ok | ok |

## Usage

```bash
# Run audit on single scenario
python3 tools/run_alignment_audit.py \
  --dataset data/canonical_scenarios/type_ii_oscillations_base.csv \
  --name calibration_type_ii_base \
  --out-dir reports/calibration

# Validate all scenarios (generates EXPECTED_VS_OBSERVED.csv)
python3 tools/run_calibration_validation.py \
  --scenarios-dir data/canonical_scenarios \
  --out-dir reports/calibration
```

## CI Integration

GitHub Actions runs `run_calibration_validation.py` on every commit.

If any scenario produces unexpected regime or fails stability, CI fails.

This ensures no regression in core logic.

