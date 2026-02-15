# Amplification Barometer

Auditable barometer for amplification dynamics, with reproducible synthetic regimes and stability tests.

Implémentation modulaire et auditable d'un baromètre d'amplification, avec une démonstration reproductible sur données synthétiques.

Le dépôt vise une "preuve de vie" empirique:

1. calcul des composites P(t), O(t), E(t), R(t), G(t)
2. signatures @(t) et Δd(t)
3. séparation explicite entre L_cap (capacité) et L_act (activation)
4. stress tests standardisés et tests adversariaux
5. audit de stabilité du score (cohérence des classements de risque)

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
from amplification_barometer import compute_at

df = pd.read_csv("data/synthetic/stable_regime.csv", parse_dates=["date"]).set_index("date")
at = compute_at(df)
print(at.head())
```

## Rapport d'audit

```bash
python tools/run_audit.py --dataset data/synthetic/bifurcation_regime.csv --name bifurcation --out-dir _ci_out --plot
```

Sorties:

1. `_ci_out/audit_report.json`
2. `_ci_out/audit_report.md`
3. Optionnel: `_ci_out/at.png`, `_ci_out/delta_d.png`, `_ci_out/l_cap_l_act.png`

## Démonstrations

1. `notebooks/demo_synthetic.ipynb` (stable, oscillant, bifurcation)
2. `notebooks/stress_test.ipynb` (stress tests et audit)

## Données synthétiques

Les CSV sont fournis dans `data/synthetic/`.
Vous pouvez les régénérer:

```bash
python tools/generate_synthetic.py --out-dir data/synthetic --n 120
```

## Auditabilité

1. Version des pondérations: `v0.2.0` (voir `amplification_barometer.composites.WEIGHTS_VERSION`)
2. Protocole des proxys: `docs/proxy_protocol.md`
3. Proof of life: `docs/audit_demo.md`
4. Template de rapport: `docs/audit_report.md`

## Licence

MIT
