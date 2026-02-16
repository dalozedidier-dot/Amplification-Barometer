# Rapport HTML auto-contenu

Objectif: produire un fichier HTML autonome (pas de serveur) avec en-tête, résumé exécutif, scores clés, tableaux stylés, graphes embarqués (PNG base64), conclusions et recommandations.

## Depuis run_audit.py

### Un dataset
```bash
python tools/run_audit.py --dataset data/synthetic/bifurcation_regime.csv --name bifurcation --out-dir _ci_out --html-report
```

### Tous les synthétiques + index
```bash
python tools/run_audit.py --all-synthetic --synthetic-dir data/synthetic --out-dir _ci_out --html-report --html-index
```

## Depuis un audit JSON existant

```bash
python tools/build_html_report.py --audit-json _ci_out/audit_report_bifurcation.json --dataset data/synthetic/bifurcation_regime.csv
```

## Personnaliser l'auteur
```bash
python tools/run_audit.py --dataset ... --html-report --author "@DDGraphisme"
```
