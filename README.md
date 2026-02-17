
# 19F NMR Analysis of SARS-CoV-2 3CLpro Inhibitor Dynamics

This repository contains the data, scripts, and analysis workflows for characterizing the conformational dynamics of **SARS-CoV-2 3CLpro** (Main Protease) in complex with the inhibitors **Nirmatrelvir** and **Ensitrelvir**. 

By employing 19F NMR line-shape analysis and EXSY spectroscopy, we quantify the thermodynamic and kinetic parameters governing the equilibrium between distinct conformational states of the enzyme-inhibitor complexes.

## 📂 Repository Structure

The analysis is organized into a chronological experimental workflow. Each sub-directory contains its own scripts/, processed_ascii/, and figures/ folders.

### 💊 MPro + Nirmatrelvir
1. **01_1D_19F_titration**: Initial titration experiments to determine binding stoichiometry and equilibrium at 298 K.
2. **02_1D_19F_VT**: Variable-temperature line-shape analysis (283–318 K). Includes global fitting to Eyring and van't Hoff models to derive activation and equilibrium thermodynamic parameters.
3. **03_2D_EXSY**: 2D Exchange Spectroscopy data and simulations used to confirm the exchange topology between conformational states.
4. **04_C145A_1D_19F_298K**: Validation of the conformational equilibrium using the catalytically inactive C145A mutant.

### 💊 MPro + Ensitrelvir
* **01_1D_19F_VT**: Comparative VT-NMR analysis for the Ensitrelvir complex, following the same rigorous line-shape protocols.

## 🛠️ Analysis Workflow

All analysis is performed using Python scripts located in the respective scripts/ directories. These scripts are designed to:
* Load processed ASCII data from the processed_ascii/ folders.
* Perform non-linear least-squares fitting using SciPy.
* Automatically export publication-quality plots to figures/ and fit summaries to output/.

## 🧪 Experimental Notes
* **Nucleus**: 19F NMR.
* **Spectrometer**: 600 and 800 MHz Bruker NEO spectrometers with TCI probes.
* **Processing**: Data was processed in TopSpin and exported to ASCII for line-shape fitting.

---
