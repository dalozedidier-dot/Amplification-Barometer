# Notes workflows

Le dépôt contient trois workflows principaux sous `.github/workflows/`.

1) `ci.yml`
- Déclenchements: push sur main, pull_request, workflow_dispatch.
- Étapes: installation, pytest, exécution demo audit, upload `_ci_out`.

2) `real_data_smoke.yml`
- Déclenchements: workflow_dispatch, schedule.
- Produit des rapports sur fixtures et proxys réels.

3) `real_data_finance_ia.yml`
- Déclenchements: workflow_dispatch, schedule.
- Smoke sur adaptateurs finance et IA.



Nouveaux workflows real data (fixtures):
- .github/workflows/real_data_univariate.yml
- .github/workflows/real_data_univariate_scenarios.yml
- .github/workflows/real_data_aiops_phase2.yml
- .github/workflows/real_data_aiops_phase2_scenarios.yml
- .github/workflows/real_data_creditcard_fraud_optional.yml
