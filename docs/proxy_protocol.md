# Protocole pour proxys (Audit Baromètre d’amplification)

Objectif: rendre P(t), O(t), E(t), R(t) et G(t) auditables même lorsque les proxys, fenêtres et pondérations comportent des choix.

## Fiche proxy minimale

Pour chaque proxy X_i, documenter:

1. Définition opérationnelle (mesure concrète, calcul, unité)
2. Plage attendue et unités (min, max)
3. Sens de risque (si X_i augmente, le risque augmente ou diminue)
4. Source (système, fréquence, latence, disponibilité)
5. Tests de manipulabilité (scénarios de falsification, conditions d'alerte)
6. Gestion de valeurs manquantes et dérives (imputation interdite par défaut, règles explicites si nécessaire)
7. Versioning (changement de définition, changement de source, changement de fenêtre)

## Exigence de stabilité du score

Un score composite n'est pas utilisable en audit s'il est instable. Exemple de critères:

- de faibles variations de fenêtre ou de normalisation changent fortement le classement de risque
- l'ajout d'un bruit faible sur les proxys provoque des inversions fréquentes
- un proxy unique domine la variance du score (risque de gaming)

Le module `amplification_barometer.audit_tools.audit_score_stability` fournit une démonstration simple de ces tests.
