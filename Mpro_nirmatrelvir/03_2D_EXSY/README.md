# 03_2D_EXSY: 2D 19F-19F Exchange Spectroscopy

This directory contains experimental data and simulation scripts for 2D 19F-19F EXSY experiments performed on the SARS-CoV-2 Mpro:Nirmatrelvir complex.

## Purpose
The primary goal of this experiment was to validate the kinetic parameters derived from the global 1D lineshape analysis (see `../02_Eyring_Analysis`). Given the exchange rate determined from 1D analysis ($k_{ex} \approx 106 \text{ s}^{-1}$ at 298 K), this simulation predicts the expected cross-peak buildup in 2D spectra to confirm the exchange regime.

## Directory Structure

### `raw_data/`
Contains the raw Bruker NMR data for the 2D EXSY pseudo-3D experiment.
* **Dataset:** `Nirmat_Mpro_1:1_2D-19F_EXSY_20241112_600MHz`
* **Spectrometer:** 600 MHz (Bruker Avance III HD)
* **Experiment List:**
    * `1`: Initial 1D Reference
    * `2`: 2D NOESY/EXSY setup
    * `31-36`: 2D EXSY planes acquired with varying mixing times ($\tau_{mix}$).

### `scripts/`
* **`EXSY-simulation.py`**: A Python script that simulates 2D EXSY contour plots based on the consensus 1D kinetic parameters.
    * **Physics Engine:** Solves the Bloch-McConnell equations for two-site exchange using the matrix exponential method (`scipy.linalg.expm`).
    * **Inputs:**
        * $k_{ex} = 106.0 \text{ s}^{-1}$ (Fixed from 1D Global Fit)
        * $p_B = 44.0\%$ (Minor state population)
        * $R_1 = 1/1.6 \text{ s}^{-1}$ (Longitudinal relaxation)
    * **Peak Shape:** Cross-peaks are reconstructed as 2D Gaussians using experimental linewidths ($w_A \approx 33 \text{ Hz}, w_B \approx 77 \text{ Hz}$) plus $20 \text{ Hz}$ exponential line broadening. This accounts for the lower apparent height of broad peaks, even when volume is conserved.

### `output/`
* **`EXSY_simulation_summary.txt`**: Text report containing the predicted peak volumes and height ratios for the simulated mixing times.
* **`figures/Simulated_EXSY.png`**: Visual output showing simulated 2D spectra at 10, 30, 100, and 200 ms mixing times.

## Usage

To regenerate the simulations:

```bash
python scripts/EXSY-simulation.py
