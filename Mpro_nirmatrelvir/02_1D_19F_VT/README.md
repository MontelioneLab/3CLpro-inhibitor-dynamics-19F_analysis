# 02_1D_19F_VT: 1D 19F Variable Temperature Analysis

This directory contains the computational pipeline for analyzing 1D 19F NMR line-shape data of the **3CLpro-Nirmatrelvir** complex. The workflow determines activation thermodynamics and conformational equilibrium between two bound states ($A \rightleftharpoons B$).

---

## 1. Directory Structure

* `scripts/`: Python analysis scripts (02-06).
* `processed_ascii/`: Input data in two-column (Frequency, Intensity) format.
* `output/`: Generated text reports and `.npz` data files.
* `figures/`: High-resolution (600 DPI) publication plots.

---

## 2. Execution Sequence

To ensure data consistency and proper error propagation, execute scripts in the following order:

1. **`python scripts/01_bruker_ascii_to_dat.py`**: convert Bruker Topspin exported spectra to dat file.
2. **`python scripts/02_global_fit.py`**: Performs simultaneous thermodynamic fit of all spectra.
3. **`python scripts/03_plot_VT_lineshapes.py`**: Produces a stacked visual report of experimental data vs. global fit.
4. **`python scripts/04_error_analysis.py`**: Calculates scatter-based errors and generates kinetic/thermo reports.
5. **`python scripts/05_vanthoff_analysis.py`**: Generates van't Hoff plot for conformational equilibrium.
6. **`python scripts/06_eyring_analysis.py`**: Generates Eyring plot with 95% CI bands.

---

## 3. Methodology & Terminology

* **Rigorous Uncertainties**: Center values are from the **Global Fit**. Uncertainties ($\pm$) are the **Sample Standard Deviation** ($|k_{600} - k_{800}| / \sqrt{2}$) of independent field measurements.
* **Intrinsic Linewidths**: Reported as $LW = R_2 - 10$ Hz to account for applied line broadening.
* **Validation Points**: Data at **283 K** is shown in figures with dashed lines to indicate it was a validation point excluded from thermodynamic regressions.

---

## 4. Primary Deliverables

| File | Description |
| :--- | :--- |
| `output/thermo_results.txt` | Final thermodynamic summary with scatter-based error bars. |
| `output/kex_vs_temperature.txt` | Table of exchange rates and population kinetics. |
| `figures/VT_lineshape_fits.png` | Stacked plot showing quality of line-shape fits across all temps. |

