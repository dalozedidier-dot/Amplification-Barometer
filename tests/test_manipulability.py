import numpy as np
import pandas as pd

from amplification_barometer.manipulability import detect_falsification, inject_falsification


def test_detects_simple_shift_falsification():
    n = 120
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "exemption_rate": np.clip(0.10 + rng.normal(0.0, 0.01, size=n), 0.0, 1.0),
        }
    )
    df_f = inject_falsification(df, proxy="exemption_rate", kind="shift", magnitude=0.25, start_frac=0.5, seed=7)
    det = detect_falsification(df_f, proxy="exemption_rate")
    assert det.detected is True
