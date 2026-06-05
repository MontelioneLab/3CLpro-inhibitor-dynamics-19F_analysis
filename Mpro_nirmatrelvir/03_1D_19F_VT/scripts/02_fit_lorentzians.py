#!/usr/bin/env python3
# ============================================================
# 02_fit_lorentzians.py
# Realignment-Safe Lorentzian Fitting Pipeline
# ============================================================

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.join(PROJECT_ROOT, "processed_ascii")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

DIRS = {
    600: "nirmat_VT_LB10_600",
    800: "nirmat_VT_LB10_800",
}

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "02_lorentzian_summary.csv")

# ============================================================
# CONFIGURATIONS
# ============================================================
SHOW_FITS = False
SAVE_FIT_PLOTS = False
LB = 10.0  

plt.rcParams.update({
    "axes.labelsize": 18, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 14, "axes.linewidth": 1.5,
    "xtick.major.width": 1.5, "ytick.major.width": 1.5,
    "xtick.major.size": 5, "ytick.major.size": 5,
    "pdf.fonttype": 42, "ps.fonttype": 42
})

def lorentzian(x, amp, x0, gamma):
    return amp * (gamma**2 / ((x - x0)**2 + gamma**2))

def two_lorentzian(x, a1, x1, g1, a2, x2, g2, offset):
    return lorentzian(x, a1, x1, g1) + lorentzian(x, a2, x2, g2) + offset

def parse_temperature(filename):
    m = re.search(r"spec_(\d+)K", filename)
    if not m:
        raise ValueError(f"Could not parse filename: {filename}")
    T = int(m.group(1))
    replicate = 1 if "_rep1" in filename else 0
    return T, replicate

# ============================================================
# FITTING ENGINE
# ============================================================
def fit_spectrum(filepath):
    data = np.loadtxt(filepath)
    x = data[:, 0]
    y = data[:, 1]

    # FIXED: Re-align primary maximum strictly to 0 Hz to match old centering parameters
    x = x - x[np.argmax(y)]
    x = -x

    mask = (x > -900) & (x < 900)
    x = x[mask]
    y = y[mask]

    baseline = np.median(y[:20])

    idx1 = np.argmax(y)
    x1_guess = x[idx1]
    a1_guess = y[idx1] - baseline

    mask2 = np.abs(x - x1_guess) > 80
    idx2_local = np.argmax(y[mask2])
    x2_guess = x[mask2][idx2_local]
    a2_guess = y[mask2][idx2_local] - baseline

    p0 = [a1_guess, x1_guess, 25.0, a2_guess, x2_guess, 25.0, baseline]
    bounds = ([0.0, -500.0, 1.0, 0.0, -500.0, 1.0, -np.inf],
              [np.inf, 500.0, 200.0, np.inf, 500.0, 200.0, np.inf])

    popt, pcov = curve_fit(two_lorentzian, x, y, p0=p0, bounds=bounds, maxfev=20000)
    a1, x1, g1, a2, x2, g2, offset = popt

    # Sort so that x1 is the lower frequency peak (left peak)
    if x1 > x2:
        a1, a2 = a2, a1
        x1, x2 = x2, x1
        g1, g2 = g2, g1

    # FIXED: True state assignment matching legacy calculation rules
    # Left Peak = State A, Right Peak = State B
    ampA, nuA, gA = a1, x1, g1
    ampB, nuB, gB = a2, x2, g2

    lwA, lwB = 2 * gA, 2 * gB
    areaA = np.pi * ampA * gA
    areaB = np.pi * ampB * gB

    return {
        "x": x, "y": y, "popt": popt,
        "nuA": nuA, "nuB": nuB, "dnu": abs(nuB - nuA),
        "lwA": lwA, "lwB": lwB,
        "lwA_intrinsic": lwA - LB, "lwB_intrinsic": lwB - LB,
        "areaA": areaA, "areaB": areaB, "pB": areaB / (areaA + areaB)
    }

# ============================================================
# MAIN LOOP
# ============================================================
rows = []
for field, dirname in DIRS.items():
    full_dir = os.path.join(BASE_DIR, dirname)
    files = sorted(glob.glob(os.path.join(full_dir, "spec_*K*.dat")))

    print("\n" + "="*60)
    print(f"Processing {field} MHz Spectrometer Pipeline")
    print("="*60)

    for filepath in files:
        fname = os.path.basename(filepath)
        T, replicate = parse_temperature(fname)

        try:
            fit = fit_spectrum(filepath)
            rows.append({
                "Field": field, "Temp": T, "Replicate": replicate,
                "dnu_Hz": fit["dnu"],
                "LW_A_Hz": fit["lwA"], "LW_B_Hz": fit["lwB"],
                "LW_A_intrinsic_Hz": fit["lwA_intrinsic"],
                "LW_B_intrinsic_Hz": fit["lwB_intrinsic"],
                "pB": fit["pB"],
                "nuA_Hz": fit["nuA"], "nuB_Hz": fit["nuB"],
                "AreaA": fit["areaA"], "AreaB": fit["areaB"]
            })
            print(f"  👉 Aligned Fit complete: {fname}")
        except Exception as e:
            print(f"  ❌ FAILED to fit spectrum {fname}: {e}")

df = pd.DataFrame(rows).sort_values(["Field", "Temp", "Replicate"])
df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "="*60)
print(f"🎉 SUCCESS: High-fidelity database written to:\n  👉 {OUTPUT_CSV}")
print("="*60 + "\n")
