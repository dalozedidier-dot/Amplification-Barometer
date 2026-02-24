# Audit report (template)

Ce dépôt fournit un rapport d'audit minimal qui sert de preuve de vie empirique sur données synthétiques.
Le but est de démontrer que les variables et signatures sont calculables et que les tests d'auditabilité sont exécutables.

## Sorties

Le script `tools/run_audit.py` génère dans un dossier de sortie:

1. `audit_report.json` (machine readable)
2. `audit_report.md` (résumé humain)
3. Optionnel: `at.png`, `delta_d.png`, `l_cap_l_act.png`

## Contenu du JSON

1. `summary` contient des statistiques simples sur P, O, E, R, G, @(t), Δd(t) et une signature de risque.
2. `stability` contient un audit de stabilité du score.
   Si de faibles variations de fenêtre ou de bruit inversent le classement des risques, le score est déclaré instable.
3. `stress_suite` contient une suite de scénarios standardisés.
4. `maturity` contient une séparation explicite entre L_cap et L_act, plus une typologie de démonstration.

## Exemple d'exécution

```bash
python tools/run_audit.py --dataset data/synthetic/bifurcation_regime.csv --name bifurcation --out-dir _ci_out --plot
```
