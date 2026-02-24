# Dashboard interactif

Le dashboard est optionnel. Il ne doit pas être une dépendance obligatoire de la librairie.

Installation
pip install -r requirements-dashboard.txt

Lancement
python apps/barometer_dash.py --dataset data/synthetic/stable_regime.csv

Ce dashboard lit un dataset de proxys, exécute un audit d alignement, puis affiche :
- séries @(t) et Δd(t)
- L_cap et L_act si colonnes présentes, sinon placeholders
- table de métriques d audit

Le but est démonstratif. Pour production, brancher des sources sectorielles versionnées et un export d artefacts.
