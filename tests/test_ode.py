
import numpy as np
from amplification_barometer.ode_model import simulate_barometer_ode


def test_ode_simulation_runs():
    t = np.linspace(0, 10, 50)
    df = simulate_barometer_ode(t=t)
    assert list(df.columns) == ["P", "O", "E", "R", "G"]
    assert len(df) == 50
    assert np.isfinite(df.to_numpy()).all()
