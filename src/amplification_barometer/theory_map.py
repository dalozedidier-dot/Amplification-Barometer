from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass(frozen=True)
class ProxySpec:
    name: str
    definition: str
    expected_range: Tuple[float, float]
    risk_direction: str
    source: str
    frequency: str
    manipulability_test: str


@dataclass(frozen=True)
class FamilySpec:
    name: str
    description: str
    proxies: Dict[str, ProxySpec]


@dataclass(frozen=True)
class TheoryAuditMap:
    version: str
    families: Dict[str, FamilySpec]


def load_proxy_specs(path: str | Path) -> TheoryAuditMap:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    version = str(data.get("version", "v0"))
    families_raw = data.get("families", {}) or {}
    families: Dict[str, FamilySpec] = {}
    for fam_name, fam in families_raw.items():
        desc = str(fam.get("description", ""))
        proxies_raw = fam.get("proxies", {}) or {}
        proxies: Dict[str, ProxySpec] = {}
        for proxy_name, spec in proxies_raw.items():
            rng = spec.get("expected_range", [0.0, 1.0])
            proxies[proxy_name] = ProxySpec(
                name=proxy_name,
                definition=str(spec.get("definition", "")),
                expected_range=(float(rng[0]), float(rng[1])),
                risk_direction=str(spec.get("risk_direction", "up_risk")),
                source=str(spec.get("source", "")),
                frequency=str(spec.get("frequency", "")),
                manipulability_test=str(spec.get("manipulability_test", "")),
            )
        families[fam_name] = FamilySpec(name=fam_name, description=desc, proxies=proxies)
    return TheoryAuditMap(version=version, families=families)
