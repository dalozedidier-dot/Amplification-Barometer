# Amplification Barometer

Implémentation modulaire et auditable d’un baromètre d’amplification, avec une démonstration reproductible sur données synthétiques.

Ce dépôt est conçu pour montrer, de bout en bout, que les composites P(t), O(t), E(t), R(t) et G(t) sont:
- calculables
- versionnés
- testables sous stress
- évaluables en stabilité (résistance aux variations de fenêtre et à un bruit faible)

## Installation

```bash
git clone https://github.com/votre-username/amplification-barometer.git
cd amplification-barometer
pip install -r requirements.txt
pip install -e .
```

## Démarrage rapide

```python
import pandas as pd
from amplification_barometer.composites import compute_at

df = pd.read_csv("data/synthetic/stable_regime.csv", parse_dates=["date"]).set_index("date")
at = compute_at(df)
print(at.head())
```

## Démonstrations

- `notebooks/demo_synthetic.ipynb` : trois régimes synthétiques (stable, oscillant, bifurcation)
- `notebooks/stress_test.ipynb` : stress test et audit de stabilité

## Données synthétiques

Les CSV sont fournis dans `data/synthetic/`.
Vous pouvez les régénérer:

```bash
python tools/generate_synthetic.py --out-dir data/synthetic --n 120
```

## Auditabilité

- Version des pondérations: `v0.1.0` (voir `amplification_barometer.composites.WEIGHTS_VERSION`)
- Protocole des proxys: `docs/proxy_protocol.md`
- Proof of life: `docs/audit_demo.md`

## Licence

MIT
