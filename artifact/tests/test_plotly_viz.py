import pandas as pd
import numpy as np

from amplification_barometer.plotly_viz import plot_oscillating


def test_plotly_export_creates_html(tmp_path):
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    y = 1.0 + 0.02 * np.sin(np.linspace(0, 4 * np.pi, 30))
    out = tmp_path / "at.html"
    plot_oscillating(dates, y, title="test", y_label="@(t)", out_html=out, baseline=1.0)
    assert out.exists()
    # small sanity check
    txt = out.read_text(encoding="utf-8")
    assert "plotly" in txt.lower()
