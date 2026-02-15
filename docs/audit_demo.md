# Proof of life (audit demo)

Objectif: démontrer que le baromètre est calculable et auditable sur trois régimes synthétiques.

## 1. Génération des données

```bash
python tools/generate_synthetic.py --out-dir data/synthetic --n 120
```

Les fichiers attendus:

1. `data/synthetic/stable_regime.csv`
2. `data/synthetic/oscillating_regime.csv`
3. `data/synthetic/bifurcation_regime.csv`

## 2. Exécution du rapport d'audit

Exemple sur le régime de bifurcation:

```bash
python tools/run_audit.py --dataset data/synthetic/bifurcation_regime.csv --name bifurcation --out-dir _ci_out --plot
```

Sorties:

1. `_ci_out/audit_report.json`
2. `_ci_out/audit_report.md`
3. Optionnel: `_ci_out/at.png`, `_ci_out/delta_d.png`, `_ci_out/l_cap_l_act.png`

## 3. Notebooks

1. `notebooks/demo_synthetic.ipynb` pour visualiser @(t), Δd(t), E(t), R(t), G(t)
2. `notebooks/stress_test.ipynb` pour exécuter une suite de stress tests

## 4. Auditabilité

1. Version des pondérations: `v0.2.0` (`amplification_barometer.composites.WEIGHTS_VERSION`)
2. Protocole des proxys: `docs/proxy_protocol.md`
3. Template de rapport: `docs/audit_report.md`
