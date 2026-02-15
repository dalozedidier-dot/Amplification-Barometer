# Mapping théorie vers audit (résumé)

Ce document résume comment les exigences de mise en pratique et d'auditabilité sont matérialisées dans le code.

## 1. Preuve de vie empirique

1. Trois régimes synthétiques sont fournis dans `data/synthetic` (stable, oscillant, bifurcation).
2. Les notebooks et le script `tools/run_audit.py` montrent le calcul de @(t), Δd(t), E(t), R(t).

## 2. Audit de stabilité du score

1. Le module `amplification_barometer.audit_tools.audit_score_stability` réalise des variations de fenêtre et des perturbations faibles.
2. La stabilité est évaluée via corrélation de rangs et cohérence du Top K.
3. Si les inversions sont trop fortes, un indicateur est marqué instable.

## 3. G(t) opérationnalisé

1. `compute_g` agrège des proxys auditables (exemptions, délai de sanction, turnover contrôle, conflit d'intérêt).
2. Le sens de risque est explicite et inversé pour produire un score de stabilité narrative.

## 4. Séparation de la limite L_cap et L_act

1. `compute_l_cap` capture la capacité d'arrêt.
2. `compute_l_act` capture l'activation effective (exemptions, délais, capture).
3. `assess_maturity` produit une typologie de démonstration.

## 5. Stress tests standardisés

1. `run_stress_suite` exécute des scénarios Shock P, Automation, Coupling, Lag O, plus des scénarios adversariaux.
2. Les sorties sont intégrées au `audit_report.json`.
