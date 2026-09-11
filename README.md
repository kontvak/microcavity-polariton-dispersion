# Dispersion Curve

A numerical model of exciton-photon polaritons in a planar DBR microcavity.

- `microcavity_polaritons.ipynb` — main calculations;
- `microcavity.py` — numerical functions;
- `figures/` — generated plots;
- `latex/` — reports in Russian and English.

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
