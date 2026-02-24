
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


EPS = 1e-12
SCALE_MAD = 1.4826  # consistent robust scale factor


@dataclass(frozen=True)
class RobustStats:
    median: float
    mad: float
    q05: float
    q50: float
    q95: float

    @property
    def scale(self) -> float:
        s = SCALE_MAD * float(self.mad)
        return float(s if np.isfinite(s) and s > EPS else EPS)


@dataclass(frozen=True)
class BaselineCalibration:
    """Baseline stable artefact for inter-dataset comparability.

    This object is intentionally simple:
    - per_proxy: robust stats for raw proxies (scale in original units)
    - derived: robust stats for derived series such as AT and DELTA_D
    """
    version: str
    created_utc: str
    stable_name: str
    per_proxy: Dict[str, RobustStats]
    derived: Dict[str, RobustStats]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created_utc": self.created_utc,
            "stable_name": self.stable_name,
            "per_proxy": {k: asdict(v) for k, v in self.per_proxy.items()},
            "derived": {k: asdict(v) for k, v in self.derived.items()},
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "BaselineCalibration":
        per_proxy = {k: RobustStats(**v) for k, v in (d.get("per_proxy") or {}).items()}
        derived = {k: RobustStats(**v) for k, v in (d.get("derived") or {}).items()}
        return BaselineCalibration(
            version=str(d.get("version", "v1.0")),
            created_utc=str(d.get("created_utc", "")),
            stable_name=str(d.get("stable_name", "stable")),
            per_proxy=per_proxy,
            derived=derived,
        )


def robust_stats(x: Sequence[float]) -> RobustStats:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return RobustStats(median=0.0, mad=1.0, q05=0.0, q50=0.0, q95=0.0)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)) + EPS)
    q05 = float(np.quantile(arr, 0.05))
    q50 = float(np.quantile(arr, 0.50))
    q95 = float(np.quantile(arr, 0.95))
    return RobustStats(median=med, mad=mad, q05=q05, q50=q50, q95=q95)


def baseline_mad_z(x: Sequence[float], stats: RobustStats) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return (arr - float(stats.median)) / float(stats.scale)


def derive_baseline(
    stable_df: pd.DataFrame,
    *,
    proxies: Sequence[str],
    derived_series: Optional[Mapping[str, Sequence[float]]] = None,
    stable_name: str = "stable",
    version: str = "v1.0",
) -> BaselineCalibration:
    """Derive baseline stats from a stable dataset.

    derived_series can include keys like "AT" and "DELTA_D".
    """
    per_proxy: Dict[str, RobustStats] = {}
    for p in proxies:
        if p in stable_df.columns:
            per_proxy[p] = robust_stats(pd.to_numeric(stable_df[p], errors="coerce").to_numpy(dtype=float))
    derived: Dict[str, RobustStats] = {}
    if derived_series:
        for k, v in derived_series.items():
            derived[k] = robust_stats(v)
    created_utc = datetime.now(timezone.utc).isoformat()
    return BaselineCalibration(
        version=version,
        created_utc=created_utc,
        stable_name=stable_name,
        per_proxy=per_proxy,
        derived=derived,
    )


def save_baseline(baseline: BaselineCalibration, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(baseline.to_dict(), f, indent=2, ensure_ascii=False)


def load_baseline(path: str | Path) -> BaselineCalibration:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        d = json.load(f)
    return BaselineCalibration.from_dict(d)
