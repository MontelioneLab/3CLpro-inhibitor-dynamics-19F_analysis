# 1D 19F Variable Temperature Analysis (Nirmatrelvir Bound 3CLpro)

This directory contains the computational pipeline for analyzing 1D 19F
NMR line-shape data of the 3CLpro-Nirmatrelvir complex.

---

## 1. Directory Structure

* `processed_ascii/`: Input raw text vectors from spectrometer.
* `output/`: Generated database matrices (.csv, .txt, .pkl).
* `figures/`: PDF and PNG publication visual assets.

---

## 2. Execution Sequence

Execute scripts in the following sequential order:

1. `python 01_convert_vt_ascii_to_hz.py`
2. `python 02_fit_lorentzians.py`
3. `python 03a_track_dnu_vs_temperature.py`
4. `python 03b_track_LW_trends.py`
5. `python 03c_track_pB_vs_temperature.py`
6. `python 04_vanthoff_regression.py`
7. `python 05_vanthoff_sensitivity_scan.py`
8. `python 06a_fit_hybrid_global.py`
9. `python 06b_plot_600_800.py`
10. `python 06c_make_exchange_LW_Table.py`
11. `python 07a_run_DGdd_grid_scan.py`
12. `python 07b_plot_activation_scan.py`
13. `python 08a_run_DGdd_scan_no_EXSY.py`
14. `python 08b_plot_activation_scan_no_EXSY.py`

---

## 3. Methodology Notes

* **Uncertainties**: Error boundaries represent 95% Bootstrap CI.
* **Intrinsic Linewidths**: Evaluated as LW = FWHM - 10.0 Hz applied LB.
* **Calibrated Hardwares**: Fits locked onto true SFO1 field coordinates.

---

## 4. Primary Deliverables

| File | Description |
| :--- | :--- |
| `output/02_lorentzian_summary.csv` | Extracted peak positions and raw areas. |
| `output/03b_linewidth_fit_equations.txt` | Low-T baseline linear equations. |
| `output/07b_DGscan_output_summary.txt` | Constrained activation parameter table. |
| `output/08b_DGscan_output_summary-noEXSY.txt` | Unconstrained control parameter table. |
| `figures/07b_Eyring_profile_ensemble.pdf` | Eyring spaghetti ensemble with EXSY. |

