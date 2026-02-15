# Protocole pour proxys (Audit baromètre d'amplification)

Objectif: rendre les composites P(t), O(t), E(t), R(t), G(t) et les opérateurs de limite auditable, même lorsque proxys, fenêtres et pondérations comportent des choix.

## 1. Fiche proxy minimale

Pour chaque proxy X_i, documenter:

1. Définition opérationnelle (mesure concrète, unité, calcul exact)
2. Plage attendue (min, max) et traitement des valeurs hors plage
3. Sens de risque (si X_i augmente, le risque augmente ou diminue)
4. Source (système, fréquence, latence, disponibilité)
5. Tests de manipulabilité (scénarios de falsification, conditions d'alerte)
6. Règles valeurs manquantes et dérives (imputation interdite par défaut)
7. Versioning (changement de définition, changement de source, changement de fenêtre)

## 2. Normalisation

Le dépôt applique une normalisation robuste proxy par proxy (médiane et MAD), puis agrège via pondérations versionnées.
Cela évite que des proxys à unités différentes dominent artificiellement le composite.

## 3. Exigence de stabilité du score

Un score composite n'est pas utilisable en audit s'il est instable. Exemples:

1. de faibles variations de fenêtre ou de normalisation changent fortement le classement des risques
2. l'ajout d'un bruit faible sur les proxys provoque des inversions fréquentes
3. un proxy unique domine la variance du score (risque de gaming)

Le module `amplification_barometer.audit_tools.audit_score_stability` fournit une démonstration simple de ces tests.

## 4. Séparation limite capacité et activation

Le dépôt sépare deux composantes:

1. `L_cap` pour la capacité d'arrêt testée (capacité organisationnelle)
2. `L_act` pour l'activation effective (enforcement réel, exemptions, délais, capture)

Voir `amplification_barometer.l_operator`.
