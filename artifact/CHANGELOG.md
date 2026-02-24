# Changelog

## v0.4.13
- Fix adapters: clamp autonomy_proxy et replicability_proxy dans [0,1] (évite des proxies négatifs extrêmes).
- Stabilité: baseline alignée sur delta_d_window (fenêtre utilisée pour le risque).
- Diagnostics: Top-K Jaccard robuste aux égalités (Top-K défini par seuil et non argsort).
- Prévention: ajout des métriques de testabilité (baseline_exceedance_rate, baseline_topk_excess_mean) et état "n/a" si non testable.

## v0.4.12
- Nettoyage du dépôt pour ne conserver que Amplification-Barometer.
- CI simplifiée: pytest + démo audit + upload d’artefacts.
- Outils: audit synthétique, sector suite 2026, smoke real data.

## v0.4.6
- Décomposition E: E_level, E_stock, dE_dt, irréversibilité.
- Rapports enrichis et compatibilité conservée sur les clés historiques.

