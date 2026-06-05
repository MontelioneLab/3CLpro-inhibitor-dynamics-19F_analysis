# 2D 19F EXSY Kinetic Analysis (Nirmatrelvir Bound 3CLpro)

This directory contains the computational pipeline for modeling and fitting 
2D 19F Exchange Spectroscopy (EXSY) data to determine absolute rate 
constants ($k_{\mathrm{ex}}$) and bootstrap confidence intervals.

---

## 1. Directory Structure

* `raw_data/`: Native spectrometer data directories for EXSY and T1 tracks.
* `processed_data/`: Extracted peak volume matrices and 1D baseline constraints.
* `output/`: Optimized rate constants and statistical bootstrap files.
* `figures/`: High-resolution visual profile and regression plots.

---

## 2. Execution Sequence

Execute scripts in the following sequential order:

1. `python scripts/01_simulate_2D_EXSY_288K.py`
2. `python scripts/02_simulate_2D_EXSY_298K.py`
3. `python scripts/03_simulate_2D_EXSY_308K.py`
4. `python scripts/04_fit-exsy-ratios.py`
5. `python scripts/05_fit_exsy-bootstrap.py`
6. `python scripts/06_fit_exsy_BA-BB-ratio-extra.py`

---

## 3. Methodology Notes

* **Simulation Steps (01-03)**: Tracks matrix magnetization at discrete temps.
* **Ratio Engine (04)**: Computes direct cross-to-diagonal peak volume fits.
* **Bootstrap Wrapper (05)**: Resamples intensities to output 95% CI boundaries.
* **Constraints**: Integrates 1D populations and T1 relaxation baselines.

---

## 4. Primary Deliverables

| File | Description |
| :--- | :--- |
| `output/04_EXSY_fit_results.csv` | Optimized rate vectors and kex center values. |
| `output/05_EXSY_bootstrap_results.csv` | Master 95% CI error bounds for Eyring plots. |
| `processed_data/2D_EXSY_intensities.xlsx` | Raw integrated peak volumes from Sparky. |

