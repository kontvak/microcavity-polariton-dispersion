# Dispersion Curve

A numerical model of exciton-photon polaritons in a planar DBR microcavity.

- `microcavity_polaritons.ipynb` — original calculations, retained for comparison;
- `microcavity_polaritons_v2.ipynb` — revised calculations and all 15 figures;
- `microcavity.py` — numerical functions;
- `figures/` and `figures_v2/` — original and revised plots;
- `data_v2/` — numerical data behind the revised comparison figures;
- `latex/` — original and revised reports in Russian and English.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook microcavity_polaritons.ipynb
```

## Build the reports

```bash
latexmk -pdf -output-directory=latex latex/report.tex
latexmk -pdf -output-directory=latex latex/report_en.tex
```

## Reproduce the revised reports

Run from the repository root. The notebook calculation can take several minutes.

```bash
python build_v2_notebook.py
MPLBACKEND=Agg jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 microcavity_polaritons_v2.ipynb
python build_v2_reports.py
latexmk -pdf -output-directory=latex latex/report_v2.tex
latexmk -pdf -output-directory=latex latex/report_en_v2.tex
```

The revised notebook writes `figures_v2/` and `data_v2/`. The report builder
reads Figure 15's saved CSV to keep the numerical residual table in sync with
the calculation. The original notebook, figures, TeX files, and PDFs are left
available as control versions.

Figure 12 uses TE/TM transfer matrices and incident power fractions for a fixed
tangential electric field along x, collecting both output polarizations.
It plots actual transmission maxima on a 161 x 161 momentum grid, retains a
resolved TE/TM doublet, and connects continuing maxima across changes in peak
count. The separate `12_UP_spectral_diagnostic.png` shows the continuous UP
spectra versus momentum and azimuth; gaps in the lower 3D panels indicate
switches of the strongest peak.

Physical polarization and transmission checks:

```bash
python -m unittest -v test_polarization
```
