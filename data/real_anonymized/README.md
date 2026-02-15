# Real anonymized datasets (templates)

Ce dossier est prévu pour des séries réelles anonymisées, sans divulgation de données sensibles.

Principes:
- Pas de PII, pas de secrets industriels, pas d'identifiants internes.
- Les colonnes sont des proxys opérationnels du baromètre.
- Les unités doivent être stables dans le temps et documentées (même si elles sont arbitraires).

Deux gabarits sont fournis:
- `finance_2026_template.csv`: chocs principalement sur P(t) (vitesse, levier, scale) et retards d'orientation.
- `ia_2026_template.csv`: externalités réseau dominantes (propagation, hysteresis) et dérive institutionnelle possible.

Colonnes obligatoires (baromètre):
- date (ISO YYYY-MM-DD)
- 5 proxys P, 5 proxys O, 3 proxys E, 4 proxys R, 5 proxys G

Colonnes optionnelles:
- `u_exog`: signal exogène (choc) si disponible
- `sector_tag`: texte libre (`finance`, `ia`, etc.)

Note audit:
- Le proxy `rule_execution_gap` vise une cible < 0.05 (5%).
- Si tu utilises d'autres proxys, documente-les et ajoute-les dans `docs/sector_specs_2026.md`.
