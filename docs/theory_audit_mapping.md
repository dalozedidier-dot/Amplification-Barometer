# Mapping théorie vers audit

Ce document fournit un mapping formel entre les grandeurs du manuscrit et les sorties du framework.

Objectif
1. Rendre les composites isomorphes au texte théorique.
2. Rendre chaque proxy auditable, versionné et testable.
3. Rendre les verdicts multi dimensionnels, sans réduire à un seul mot.

## Variables principales

P(t)
Composite de puissance. Calculé depuis la famille de proxys P. Voir docs/proxies.yaml.

O(t)
Composite de capacité d arrêt et d orientation opérationnelle. Calculé depuis la famille O.

E(t)
Stock d externalités, dispersif, inertiel, potentiellement irréversible. Calculé depuis la famille E, puis cumulé.

R(t)
Résilience, marges et récupération. Calculé depuis la famille R. Une hausse de recovery_time_proxy dégrade R.

G(t)
Gouvernance mesurable. Famille G. La cible de maturité utilise rule_execution_gap < 0.05 et control_turnover < 0.05.

@(t)
Ratio de dissociation. Par défaut @(t) = P_level / (O_level + eps). P_level et O_level sont des composites non centrés, donc comparables entre régimes.

Δd(t)
Différence de dérivées. Par défaut Δd(t) = dP/dt - dO/dt sur séries lissées. Certains passages du manuscrit utilisent le terme Ad(t). Ici Δd(t) correspond à ce rôle.

L_cap
Capacité intrinsèque, testée sous stress. Mesurée comme performance sur scénarios standardisés, sans circularité avec L_act.

L_act
Activation effective, observée dans les séries. Elle dépend de la persistance, du délai d activation et des proxys G.

## Critères d audit

Stabilité du score
On mesure la stabilité de classement quand on varie faiblement la fenêtre, la normalisation et le bruit. Si des variations mineures inversent le top k, le score est instable.

Anti gaming
On simule des falsifications. Exemple, injection de biais sur les proxys de O. On mesure la chute artificielle du risque et la détection par règles proxy.

Stress tests alignés sur signatures
Type I, bruit. Le score doit rester borné, sans drift de E.
Type II, oscillations. Le score oscille mais R permet le retour. Δd(t) alterne de signe.
Type III, bifurcation. @(t) diverge, E s accumule, R se dégrade, L doit activer plus tôt.

Verdict multi dimensionnel
On produit un verdict par composante. Stabilité, L_cap, L_act, Résilience, Gouvernance, Anti gaming, Stress tests.

## Fichiers de référence

docs/proxies.yaml
Spécification des proxys. Définition, plage attendue, sens du risque, source, fréquence, test de manipulabilité.

tools/run_alignment_audit.py
Génère un alignment_audit.json et un alignment_audit.md à partir d un dataset, sans données sensibles obligatoires.
