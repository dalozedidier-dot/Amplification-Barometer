# Démonstration audit Baromètre d’amplification (proof of life)

Ce dépôt propose un minimum reproductible pour:

- calculer P(t), O(t), E(t), R(t) et G(t) à partir de proxys explicites
- produire @(t) et Δd(t)
- discriminer trois régimes synthétiques: stable, oscillant, bifurcation
- exécuter un stress test et un audit de stabilité des signatures

## Parcours recommandé

1. Installer
   - `pip install -r requirements.txt`
   - `pip install -e .`

2. Ouvrir le notebook `notebooks/demo_synthetic.ipynb`
3. Ouvrir le notebook `notebooks/stress_test.ipynb`
4. Lancer les tests
   - `pytest -q`

## Note sur le modèle ODE

Le module `amplification_barometer.ode_model` implémente:
- un système minimal P/O pour la démo
- un système 4D (P,O,E,R) inspiré du manuscrit, utilisé pour illustrer des régimes
Ces modèles ne sont pas des jumeaux numériques. Ils servent à rendre les signatures calculables et comparables.
