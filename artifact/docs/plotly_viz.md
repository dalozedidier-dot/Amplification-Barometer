# Plotly visualizations (interactive)

This repo ships two visualization modes:

- **Matplotlib PNG**: `tools/run_audit.py --plot` (CI-friendly, static)
- **Plotly HTML**: `tools/run_audit.py --plotly` (interactive, shareable)

## Why Plotly here

Time series audits benefit from:
- range slider (fast zoom/pan on long series)
- unified hover (compare curves at the same timestamp)
- log scale where growth is exponential (bifurcation regimes)
- secondary axis for L_cap vs L_act (avoid clutter)

## Usage

Run a single dataset:

```bash
python tools/run_audit.py --dataset data/synthetic/bifurcation_regime.csv --name bifurcation --out-dir _ci_out --plotly
```

Run the full synthetic portfolio (stable / oscillating / bifurcation):

```bash
python tools/run_audit.py --all-synthetic --out-dir _ci_out --plotly
```

## Outputs

For each dataset `<name>`:

- `_ci_out/<name>_at.html`
- `_ci_out/<name>_delta_d.html`
- `_ci_out/<name>_l_cap_l_act.html`
- `_ci_out/<name>_dashboard.html`

The HTML files use PlotlyJS via CDN (small artifacts, easy sharing). If you need fully offline artifacts, change `include_plotlyjs="cdn"` to `include_plotlyjs=True` in `src/amplification_barometer/plotly_viz.py`.
