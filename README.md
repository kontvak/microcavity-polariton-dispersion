# Dispersion Curve

A numerical model of exciton–photon polaritons in a planar DBR microcavity,
with calculations and reports in Russian and English.

## Project files

- [Russian report](latex/report.pdf) · [LaTeX source](latex/report.tex)
- [English report](report_en.pdf) · [LaTeX source](latex/report_en.tex)
- `latex/` — both report sources and PDFs; the English PDF is also copied to the root.
- `microcavity_polaritons.ipynb` — main interface for calculations, plots and checks.
- `microcavity.py` — physical models, TMM solvers and plotting functions.
- `figures/` — report figures and spectral diagnostics.
- `data/` — saved numerical arrays, tables and calculation metadata.
- `test_polarization.py` — transmission and complex-mode checks.

## Setup and calculations

Run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook microcavity_polaritons.ipynb
```

To execute the whole notebook without opening Jupyter:

```bash
MPLBACKEND=Agg .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 microcavity_polaritons.ipynb
```

The notebook writes figures and numerical data to `figures/` and `data/`.
The reports contain 14 figures in 15 sections. Filenames retain the calculation
numbering: `15_complex_energy_comparison.png` is Figure 14 in the PDFs.
`08_RTA_fixed_angles.png` and `12_UP_spectral_diagnostic.png` are diagnostics
outside the reports.

## Final comparison: Sections 13–14

The same TE structure with 23/26 DBR pairs and one QW is compared at
241 real wave vectors, from 0 to 4 µm⁻¹:

1. **Non-Hermitian 2×2 eigenvalues: Re E** — an effective photon–exciton model.
2. **TMM reflectance minima** — minima of the real-energy reflection spectrum.
3. **TMM solutions of M₂₂ = 0: Re E** — real parts of full-TMM complex poles.

Figure 14 shows the energies and the differences E₁−E₃ and E₂−E₃ for each
LP/UP branch. The effective model includes diagonal losses and a constant
coupling calibrated to the original zero-k splitting. The geometry, TMM
radiative parameter and calculations in Sections 1–12 are retained.

RMS is the root mean square of the differences from method 3 over the wave-vector
grid. Energy RMS differences for the effective model are 0.767/1.578 µeV for
LP/UP; pole-linewidth RMS differences are 0.124/0.081 µeV. The improvement
in energy agreement involves both losses and coupling recalibration.
Adding losses alone at the original coupling increases the differences.

Saved comparison files:

- `data/complex_comparison.npz` — complex poles, effective-model variants,
  reflectance minima and validity masks.
- `data/complex_comparison.csv` — numerical values and signed differences.
- `data/complex_comparison.json` — exact configuration, calibration,
  statistics and numerical checks.

`data/fig15_three_methods_comparison.csv` and `data/fig13_pole_linewidths.csv`
retain the original full-TMM reference data.

To redraw the final figure from saved data without rerunning TMM:

```bash
MPLBACKEND=Agg .venv/bin/python -c "import microcavity as mc; mc.export_complex_comparison(mc.load_complex_comparison())"
```

Alternatively, set `LOAD_SAVED_COMPLEX_COMPARISON=True` in the notebook's
final comparison cell; it can then run independently of earlier cells.

## Build the reports

Install a TeX distribution with `latexmk`, `pdflatex`, and Russian language
support. From the repository root, run:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=latex latex/report.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=latex latex/report_en.tex
cp latex/report_en.pdf report_en.pdf
```

Both PDFs are built in `latex/`; the last command updates the English PDF copy
in the root. LaTeX auxiliary files are ignored by Git. To remove only those files:

```bash
latexmk -c -outdir=latex latex/report.tex latex/report_en.tex
```

## Physical and numerical checks

```bash
.venv/bin/python -m unittest -v test_polarization
```

The suite contains 11 tests covering polarized transmission, analytical limits,
damping conventions, coupling calibration and comparison validity masks.
Additional pole-refinement checks and residual statistics are saved with the data.
