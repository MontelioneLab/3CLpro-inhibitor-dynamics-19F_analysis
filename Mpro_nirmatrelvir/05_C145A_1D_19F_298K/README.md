# 1D 19F 298K Analysis of C145A Mutant (Nirmatrelvir Bound)

This directory contains the computational pipeline for modeling the 1D 19F
NMR lineshape of the catalytically inactive C145A 3CLpro mutant complexed 
with Nirmatrelvir at 298 K. This control isolations maps exchange 
kinetics ($k_{\mathrm{ex}}$) specifically at room temperature.

---

## 1. Directory Structure

* `processed_ascii/`: Frequency-converted text grids for the mutant complex.
* `raw_data/`: Native Bruker spectrometer directories and temporary files.
* `output/`: Optimized parameter matrices and rate scan data tables.
* `figures/`: High-resolution NRMSE profiles and lineshape overlay fits.

---

## 2. Execution Sequence

Execute scripts in the following sequential order:

1. `python scripts/01_convert_298K_ascii_to_hz.py`
2. `python scripts/02_fit_C145A_298K.py`
3. `python scripts/03_scan_kex_fits.py`

---

## 3. Methodology Notes

* **Grid Converter (01)**: Calibrates the 600 MHz 19F hertz axis framework.
* **Lineshape Fitter (02)**: Extracts chemical shift separations and amplitudes.
* **Rate Scanner (03)**: Maps the local NRMSE error surface across a $k_{\mathrm{ex}}$ grid.
* **Control Value**: Isolates exchange properties without enzymatic cleavage.

---

## 4. Primary Deliverables

| File | Description |
| :--- | :--- |
| `output/C145A_fit_params.txt` | Final optimized lineshape coordinates at 298 K. |
| `output/C145A_kex_scan_summary.csv` | NRMSE profile grid evaluated across rate space. |
| `figures/C145A_line-shape_fit.png` | Stacked overlay plot of experimental vs. simulated data. |

