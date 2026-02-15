from __future__ import annotations

from typing import Dict, Any

import pandas as pd


def validate_g_proxies(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Vérifie un protocole minimal pour les proxys de G(t).

    Retourne un dictionnaire de métadonnées. Cette fonction est volontairement descriptive:
    elle ne "corrige" pas les données, elle signale les écarts (audit).
    """
    protocol: Dict[str, Dict[str, Any]] = {
        "exemption_rate": {
            "definition": "Taux d'exemptions aux règles",
            "range": [0.0, 1.0],
            "risk_direction": "up",
            "source": "Logs internes",
            "falsification_test": "Audit manuel si dérive > 5% sur 30 jours",
        },
        "sanction_delay": {
            "definition": "Délai médian d'application des sanctions (jours)",
            "range": [0.0, 365.0],
            "risk_direction": "up",
            "source": "Base sanctions",
            "falsification_test": "Variance > 30j sur fenêtre glissante",
        },
        "control_turnover": {
            "definition": "Turnover équipes de contrôle (ratio)",
            "range": [0.0, 1.0],
            "risk_direction": "up",
            "source": "RH",
            "falsification_test": "Pics synchrones avec incidents majeurs",
        },
        "conflict_interest_proxy": {
            "definition": "Signal proxy de conflits d'intérêt déclarés",
            "range": [0.0, 1.0],
            "risk_direction": "up",
            "source": "Déclarations, audits",
            "falsification_test": "Décorrélation suspecte avec sanctions",
        },
    }

    for k, meta in protocol.items():
        if k not in df.columns:
            continue
        lo, hi = meta["range"]
        s = df[k].astype(float)
        if not ((s >= lo).all() and (s <= hi).all()):
            meta["range_alert"] = True
    return protocol
