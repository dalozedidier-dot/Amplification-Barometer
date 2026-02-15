
from dataclasses import dataclass
import numpy as np

@dataclass
class RobustStats:
    median: float
    mad: float

def robust_stats(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    return RobustStats(median=med, mad=mad)

def mad_z(x, stats: RobustStats):
    return (np.asarray(x) - stats.median) / stats.mad
