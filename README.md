# Amplification-Barometer

Ce dépôt fournit un baromètre d’amplification basé sur des proxys observables et des tests auditables. Le but est de produire des rapports reproductibles sur données synthétiques et sur jeux réels via des adaptateurs de proxys.

## Installation rapide

Python 3.12 recommandé.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Vérifier:

```bash
PYTHONPATH=src pytest -q
```

## Exécuter un audit

### Démo synthétique

```bash
PYTHONPATH=src python tools/run_audit.py --all-synthetic --synthetic-dir data/synthetic --out-dir _ci_out/demo
```

### Sector suite 2026

```bash
PYTHONPATH=src python tools/generate_sector_2026.py --out-dir data/sector_2026 --n 365 --seed 7
PYTHONPATH=src python tools/run_sector_suite.py --sector-dir data/sector_2026 --out-dir _ci_out/sector_2026 --window 5
```

### Smoke real data

```bash
PYTHONPATH=src python tools/run_real_data_smoke.py --out-dir _ci_out/real_data --scenarios 4
```

## Sorties

Les scripts écrivent par défaut sous `_ci_out/`. Ce dossier ne doit pas être versionné.

## Workflows GitHub Actions

- `CI` : tests + démo synthétique
- `Real Data Smoke` : exécutions hebdomadaires, artefacts
- `Real Data Finance + IA` : adaptateurs, smoke hebdomadaire

