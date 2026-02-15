# Sector specs 2026 (finance, IA)

Objectif
- Décrire une instanciation minimale du baromètre par secteur.
- Rester compatible avec l'audit: proxies mesurables, définitions stables, tests reproductibles.

## Conventions
- Tous les proxys sont des séries temporelles alignées sur `date`.
- Plages attendues: voir `src/amplification_barometer/proxy_protocol.py`.
- Cible institutionnelle: `rule_execution_gap < 0.05` (5%).

## Finance 2026
Hypothèse audit
- Les chocs dominants sont des chocs exogènes sur P(t), typiquement sur `scale_proxy`, `speed_proxy`, `leverage_proxy`.
- Les externalités peuvent suivre via `impact_proxy` et `hysteresis_proxy`.

Mapping minimal
- `u_exog`: optionnel, encode un choc exogène (par exemple un indicateur d'évènement ou une intensité normalisée).

Stress tests recommandés
- Shock-P: +X sur `leverage_proxy` et `speed_proxy` sur un intervalle.
- Lag-O: retard sur les proxys O (exécution et seuils).
- Capture-light: augmentation de `exemption_rate` et `sanction_delay`.

Critère de performance L(t)
- Viser `prevented_exceedance_rel > 0.10` via une activation plus proactive (seuil plus bas, persistance plus courte).

## IA 2026
Hypothèse audit
- Les externalités dominantes sont de type réseau: `propagation_proxy` et `hysteresis_proxy` structurent E(t).
- L'autonomie et la réplicabilité poussent P(t) rapidement.

Stress tests recommandés
- Network wave: hausse persistante de `propagation_proxy` et `hysteresis_proxy`.
- Automation sprint: hausse de `autonomy_proxy` et `replicability_proxy`.
- Governance drift: dérive durable de `rule_execution_gap` au-dessus de 0.05, associée à une montée d'exemptions.

Critère de performance L(t)
- Viser `prevented_exceedance_rel > 0.10` et une réduction nette de `risk_drop_around_activation`.

## Datasets fournis
- `data/real_anonymized/*_template.csv`: gabarits à remplir.
- `data/sector_2026/*_synth.csv`: datasets synthétiques de démonstration (sans données sensibles).
