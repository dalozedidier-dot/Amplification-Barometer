# Installation

Python 3.12 recommandé.

## Environnement virtuel (pip)

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Vérification

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python tools/run_audit.py --all-synthetic --synthetic-dir data/synthetic --out-dir _ci_out/demo
```

## Remarques

- Les sorties vont dans `_ci_out/` par défaut.
- Ne pas committer `_ci_out/` ni les archives zip d’artefacts.
