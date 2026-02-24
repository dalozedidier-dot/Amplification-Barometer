Amplification-Barometer – Patch v0.4.5 (full fixes)

Ce ZIP contient uniquement les fichiers ajoutés ou modifiés.

Correctifs principaux
- Baseline stable appliquable partout (thresholds) : RISK comparables entre datasets via data/synthetic/stable_regime.csv.
- L_cap v2 : supprime le plafond à ~0.50, ajoute gate gouvernance (turnover + rule_gap) et bonus recovery p90.
- L_performance : ajoute prevented_topk_excess_rel (et garde les clés legacy), corrige les NaN dans les résumés.
- run_audit.py : lecture CSV robuste (fallback timestamp), génère audit_report_*.html auto-contenu (Plotly inline).
- run_sector_suite.py : baseline stable + summary CSV/MD complet (inclut proactive_topk_excess_rel).
- Workflows : ajout Real Data Smoke (workflow séparé) + fixtures minimales.

Comment appliquer
1) Dézipper à la racine du repo (remplacement des fichiers aux mêmes chemins).
2) Lancer les tests:
   PYTHONPATH=src pytest -q
3) Smoke local:
   python tools/run_audit.py --all-synthetic --synthetic-dir data/synthetic --out-dir _ci_out --plot --plotly
   python tools/run_sector_suite.py --sector-dir data/sector_2026 --out-dir _ci_out

Real data (optionnel)
- Déposer des CSV/Parquet sous data/real/
  - soit déjà au format proxies (colonnes REQUIRED_PROXIES)
  - soit au format OHLCV (open/high/low/close[/volume] + date/timestamp)
- Lancer:
  python tools/run_real_data_smoke.py --out-dir _ci_out/real_data
- Conversion explicite:
  python tools/convert_real_data.py --infile <input.csv> --outfile data/real/<name>_proxies.csv
