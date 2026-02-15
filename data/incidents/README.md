# Anonymized incidents (template)

Goal: allow minimal recalibration of proxy weights using a small number of incidents,
without exposing sensitive operational data.

## Minimal contract

Each row represents an incident window (start_date to end_date) with a single target:

- damage_weight: float in [0, 1] or any strictly positive scale you choose
  (the script only uses ranks / Spearman, so monotonic transforms are acceptable)

You can optionally include per-incident proxy values. Recommended:
- aggregate the proxy time series over the incident window (mean or median)
- store only the aggregate, not the raw time series

Columns for proxies are provided in the CSV template, but you can leave proxies empty.
The recalibration tool will only use columns that are present and numeric.

## Anonymization rules

- No identifiers for systems, teams, products, clients, or locations.
- No raw timestamps beyond coarse dates (day-level is enough for the template).
- No free text that could re-identify a person or org. Keep notes generic.

## Output

The tool writes a JSON file containing:
- group-wise weights for P, O, E, R, G
- summary metrics (Spearman correlations, coverage)
- provenance metadata (timestamp, method, input file name)

This JSON can be versioned in docs/ and reviewed in PRs.
