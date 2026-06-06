# NMR Dynamics Analysis of SARS-CoV-2 3CL<sup>pro</sup> Inhibitors

This repository contains the Python scripts and experimental 19F NMR data
associated with the paper:
**"Slowly-exchanging bound states of SARS-CoV-2 3CLpro-inhibitor complexes revealed by 19F NMR"**
*Authors: Anna De Falco, Ben Shurina, Rebecca Greene-Cramer, Theresa A. Ramelot, and Gaetano T. Montelione*
*Journal: [Journal Name], 2026*

---

## 🔬 Overview

We used 19F NMR lineshape analysis to characterize the conformational exchange
of SARS-CoV-2 3CLpro inhibitors. This codebase performs:
1. Global Lineshape Fitting: Fits variable-temperature (VT) spectra to a
   two-state Bloch-McConnell model.
2. Thermodynamic Analysis: Extracts ground-state equilibrium parameters 
   (dH, dS) from population data.
3. Kinetic Analysis: Extracts activation parameters (dH#, dS#) from rate 
   data via Eyring plots.
4. EXSY Validation: Models 2D Exchange Spectroscopy data to validate the
   kinetic parameters.

---

## 📂 Repository Structure

The project is structured by inhibitor target workflows:

### 🟦 Mpro_nirmatrelvir/
* 01_1D_19F_titration/     : Raw titration dataset tracks.
* 02_1D_1H_Zn_and_EDTA/    : Proton spectra check for metal effects.
* 03_1D_19F_VT/            : Variable-temperature lineshape analysis pipeline.
* 04_2D_EXSY/              : 2D magnetization transfer rate calculations.
* 05_C145A_1D_19F_298K/    : Control profiles using the inactive mutant protease.

### 🟩 Mpro_ensitrelvir/
* 01_1D_19F_VT/            : Variable-temperature lineshape analysis for Ensitrelvir.

---

## 🚀 Getting Started

To run the analysis, install dependencies via: pip install -r requirements.txt

Then navigate to a specific workflow folder and run the scripts in order, 
for example: cd Mpro_nirmatrelvir/03_1D_19F_VT && python scripts/01_convert_vt_ascii_to_hz.py

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
