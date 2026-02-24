import numpy as np

from amplification_barometer.ode_model import BarometerParams, simulate_minimal_po, simulate_barometer_ode


def test_simulate_minimal_po():
    df = simulate_minimal_po(t=np.linspace(0, 5, 50))
    assert list(df.columns) == ["P", "O"]
    assert len(df) == 50


def test_simulate_barometer_ode_shapes():
    df = simulate_barometer_ode(t=np.linspace(0, 10, 100), params=BarometerParams())
    assert list(df.columns) == ["P", "O", "E", "R"]
    assert len(df) == 100
