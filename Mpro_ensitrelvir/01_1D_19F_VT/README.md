# Ensitrelvir VT-NMR Analysis Pipeline

This directory contains the standardized analysis suite for the variable-temperature (VT) ¹⁹F NMR characterization of the Ensitrelvir protein-ligand complex. The analysis focuses on a three-peak fluorine system exhibiting significant local heterogeneity and a high minor-state population.

## Directory Structure
- `/processed_ascii/`: Input spectral data in standardized `.dat` format.
- `/output/`: Generated thermodynamic tables and `.npz` binary archives.
- `/figures/`: High-resolution PDF/PNG plots for manuscript preparation.

## Analysis Scripts
1. **01_bruker_ascii_to_dat.py** Converts exported Bruker Topsin  ascii NMR data into standardized frequency/intensity ASCII .dat files.
2. **02_global_fit.py** Primary line-shape engine. Performs multi-peak Lorentzian fitting. To prevent baseline artifacts, the minor state (State B) linewidth is constrained to a physically realistic 150–200 Hz.
3. **03_plot_VT_lineshapes.py** Generates publication-quality figures of experimental data overlaid with global fits.
4. **04_err_analysis.py** Calculates intrinsic linewidths ($R_2$) by subtracting the 50 Hz processing line broadening. Generates `ensitrelvir_intrinsic_linewidths.txt`.
5. **05_vanthoff_analysis.py** Performs thermodynamic regression to extract $\Delta H^\circ$, $\Delta S^\circ$, and $\Delta G_{298}$. Includes a 95% Confidence Interval (CI) shaded band and accounts for $\Delta H/\Delta S$ covariance.
6. **07_high_res_validation_298K.py** Validation script using fit parameters to verify the model against high-resolution, long-collection data.

## Key Results
- **Minor State Population ($p_B$):** ~40.5% at 298 K.
- **Intrinsic Linewidths ($R_2$):** ~100–150 Hz, significantly broader than nirmatrelvir.
