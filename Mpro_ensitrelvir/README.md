# 3CLpro Inhibitor Dynamics: 19F-NMR Lineshape Analysis

This repository contains the processing, simulation, and thermodynamic analysis pipeline for investigating the conformational dynamics of the SARS-CoV-2 main protease (3CLpro/Mpro) complexed with the antiviral inhibitor Ensitrelvir using 19F Variable-Temperature (VT) NMR spectroscopy.

## 1. Repository Structure

The analysis is organized sequentially to ensure absolute auditability and reproducibility from raw instrument data to final publication figures:

3CLpro-inhibitor-dynamics-19F_analysis/
|-- Mpro_ensitrelvir/
|   `-- 01_1D_19F_VT/
|       |-- scripts/
|       |   |-- 01a_convert_vt_ascii_to_hz.py
|       |   |-- 01b_convert_high_sn_ascii_to_hz.py
|       |   |-- 02a_track_vt_coordinate_linearity.py
|       |   |-- 02b_global_thermodynamic_fit.py
|       |   |-- 03_plot_vt_lineshape_fits.py
|       |   |-- 04_propagate_equilibrium_errors.py
|       |   `-- 06b_validate_model_optimized_widths.py
|       |-- raw_bruker/       # Raw TopSpin ASCII exports (User-provided)
|       |-- processed_ascii/  # Generated matrix data (Hz centered)
|       |-- output/           # Regression logs and binary fit files
|       `-- figures/          # Publication-ready vector PDFs and PNGs
`-- README.md                 # Main project documentation

---

## 2. Pipeline Workflow & Script Descriptions

### Data Conversion & Preprocessing

* 01a_convert_vt_ascii_to_hz.py: Batch-processes raw Bruker TopSpin ASCII text files from the variable-temperature series. It trims datasets to a uniform size, converts the x-axis from PPM to centered Hertz (Hz) using the 19F Larmor frequency on a 600 MHz system, and applies initial baseline zeroing.

* 01b_convert_high_sn_ascii_to_hz.py: A dedicated script that handles the high signal-to-noise (S/N) long-duration 1D reference spectrum used for downstream validation, converting it under sharp line-broadening constraints (LB = 20 Hz).

### Lineshape Fitting & Thermodynamic Optimization

* 02a_track_vt_coordinate_linearity.py: Performs dynamic peak-picking on the splitting of the middle fluorine resonance across all temperatures. It maps peak coordinate trajectories and runs a linear regression to establish baseline frequency separation parameters, outputting 02a_dnu_slope_fits.txt.

* 02b_global_thermodynamic_fit.py: The core physics simulation engine. It applies a global multi-state symmetric Lorentzian optimization matrix across left, middle, and right fluorine resonances simultaneously. It fixes active site state populations (pB) across the temperature series and archives the global fit arrays into a binary matrix (vt_global_fit.npz).

* 03_plot_vt_lineshape_fits.py: Reads the archived global simulation arrays and renders a publication-quality 3-panel stacked figure, plotting the simulated multi-state fits directly over the raw experimental data points.


### Statistical Error Propagation

* 04_propagate_equilibrium_errors.py: Executes analytical derivative error propagation from the global covariance matrix down into populations (pB), equilibrium constants (Keq), and Gibbs free energies (dG). It calculates dG directly from the smoothed trendline parameters to isolate regional variance and logs 04_equilibrium_error_report.txt.


### Independent Cross-Validation

* 06b_validate_model_optimized_widths.py: An expanded 6-parameter validation script that projects the optimized 298 K active-site population model onto the high S/N dataset. It automatically applies a 30 Hz line-sharpening correction while permitting the bound-state linewidth (wB) to optimize freely, isolating localized exchange-broadening velocities.

---

## 3. Execution Instructions

To replicate the thermodynamic calculations and regenerate the manuscript figures locally, execute the pipeline sequentially from the terminal:

```bash
# Move to the workflow directory
cd Mpro_ensitrelvir/01_1D_19F_VT/scripts/

# Step 1: Preprocess raw spectral arrays
python 01a_convert_vt_ascii_to_hz.py
python 01b_convert_high_sn_ascii_to_hz.py

# Step 2: Establish coordinate boundaries and execute global simulations
python 02a_track_vt_coordinate_linearity.py
python 02b_global_thermodynamic_fit.py

# Step 3: Generate stacked lineshape figures
python 03_plot_vt_lineshape_fits.py

# Step 4: Perform statistical error analysis and data reporting
python 04_propagate_equilibrium_errors.py

# Step 5: Cross-validate with independent high S/N data
python 06b_validate_model_optimized_widths.py
