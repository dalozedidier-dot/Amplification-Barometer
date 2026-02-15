# Amplification Barometer

Audit-ready, modular barometer to measure amplification dynamics and validate limit-operator performance with reproducible synthetic stress tests.


Auditable barometer for amplification dynamics, with reproducible synthetic regimes, stability audits, anti-gaming checks, and L(t) performance tests.

Implémentation modulaire et auditable d'un baromètre d'amplification, avec une démonstration reproductible sur données synthétiques.

Le dépôt vise une preuve de vie empirique:

1. calcul des composites P(t), O(t), E(t), R(t), G(t)
2. signatures @(t) et Δd(t)
3. séparation explicite entre L_cap (capacité) et L_act (activation)
4. stress tests standardisés et tests adversariaux
5. audit de stabilité du score (cohérence des classements de risque)
6. tests anti-gaming (plages attendues, injections, détection)
7. validation de performance de L(t) sur séries synthétiques

## Installation

```bash
git clone https://github.com/votre-username/amplification-barometer.git
cd amplification-barometer
pip install -r requirements.txt
pip install -e .
```

## Démarrage rapide

Calculer @(t) et Δd(t) sur un dataset synthétique:

```python
import pandas as pd
from amplification_barometer import compute_at, compute_delta_d

df = pd.read_csv("data/synthetic/stable_regime.csv", parse_dates=["date"]).set_index("date")
at = compute_at(df)
dd = compute_delta_d(df)
print(at.head())
```

Générer un rapport d'audit:

```bash
python tools/run_audit.py --dataset data/synthetic/stable_regime.csv --name stable --out-dir _ci_out --plot
```

Exécuter l'ensemble des régimes synthétiques et produire un rapport de calibration:

```bash
python tools/run_audit.py --all-synthetic --synthetic-dir data/synthetic --out-dir _ci_out --plot
```

## Structure

- `src/amplification_barometer`: code (composites, opérateur de limite, audit, anti-gaming)
- `data/synthetic`: datasets de démonstration (stable, oscillant, bifurcation)
- `tools`: scripts CLI (génération synthétique, run d'audit)
- `docs`: protocoles et mapping théorie vers audit

## Licence

MIT.
