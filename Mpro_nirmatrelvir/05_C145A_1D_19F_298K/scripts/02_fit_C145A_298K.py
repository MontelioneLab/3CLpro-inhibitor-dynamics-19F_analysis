#!/usr/bin/env python3

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. SETUP & DIRECTORY STRUCTURE
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "processed_ascii",
    "C145A_nirmat_298K_LB10_EDTA_600MHz"
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

FILE_NAME = os.path.join(DATA_DIR, "spec_298.dat")
TXT_OUTPUT = os.path.join(OUTPUT_DIR, "C145A_fit_params.txt")
FIG_OUTPUT = os.path.join(FIGURE_DIR, "C145A_line-shape_fit.png")

# ============================================================
# FIXED LINEWIDTHS
# ============================================================

LB = 10.0

# Intrinsic WT-like linewidths, before processing LB
LWA_intrinsic = 60.5
LWB_intrinsic = 60.0

# ============================================================
# 2. BLOCH-MCCONNELL MODEL
# ============================================================

def dnmr_2site(freq_hz, kex, center_offset, r2a, r2b, delta_nu, pB):
    pA = 1.0 - pB
    pi = np.pi

    v = freq_hz - center_offset

    vA = +0.5 * delta_nu
    vB = -0.5 * delta_nu

    w = 2 * pi * v
    wA = 2 * pi * vA
    wB = 2 * pi * vB

    R2A = r2a * pi
    R2B = r2b * pi

    # Population-weighted microscopic rates
    kAB = kex * pB
    kBA = kex * pA

    GA = R2A + 1j * (wA - w) + kAB
    GB = R2B + 1j * (wB - w) + kBA

    numerator = pA * (GB + kAB) + pB * (GA + kBA)
    denominator = GA * GB - kAB * kBA

    return np.real(numerator / denominator)

# ============================================================
# 3. LOAD DATA
# ============================================================

try:
    data = np.loadtxt(FILE_NAME)

    max_val = np.max(np.abs(data[:, 0]))

    if max_val < 0.5:
        data[:, 0] *= 1e6
    elif 0.5 < max_val < 500.0:
        data[:, 0] *= 1e3

    max_idx = np.argmax(data[:, 1])
    center_hz = data[max_idx, 0]
    data[:, 0] -= center_hz

    mask = np.abs(data[:, 0]) <= 600.0

    freq = data[mask, 0]
    y_exp = data[mask, 1]

    # Normalize experimental spectrum once only.
    y_exp = y_exp / np.max(y_exp)

    print(f"Loaded {FILE_NAME}")

except Exception as e:
    print(f"Error loading {FILE_NAME}: {e}")
    raise SystemExit

# ============================================================
# 4. RESIDUALS
# ============================================================

def residuals(params):
    # kex, pB, dNu, shift_offset, amp, base
    kex, pB, dNu, shift_offset, amp, base = params

    sim = dnmr_2site(
        freq,
        kex,
        shift_offset,
        LWA_intrinsic + LB,
        LWB_intrinsic + LB,
        dNu,
        pB
    )

    y_fit = amp * sim + base

    return y_fit - y_exp

# ============================================================
# 5. FIT
# ============================================================

# kex, pB, dNu, shift_offset, amp, base
p0 = [31.0, 0.20, 105.0, 0.0, 100.0, 0.0]

lower = [0.0,   0.05,  70.0, -50.0, 0.001, -0.05]
upper = [100.0, 0.40, 140.0,  50.0, 1e5,    0.05]

result = least_squares(
    residuals,
    p0,
    bounds=(lower, upper),
    loss="soft_l1"
)

kex_fit, pB_fit, dNu_fit, shift_fit, amp_fit, base_fit = result.x

# ============================================================
# 6. REPORT
# ============================================================

with open(TXT_OUTPUT, "w") as f:
    f.write("=== MPro C145A MUTANT FIT RESULTS ===\n")
    f.write(f"Source Data     : {FILE_NAME}\n")
    f.write(f"Assumed Width A : {LWA_intrinsic + LB:.1f} Hz\n")
    f.write(f"Assumed Width B : {LWB_intrinsic + LB:.1f} Hz\n")
    f.write("-" * 45 + "\n")
    f.write(f"k_ex            : {kex_fit:.2f} s^-1\n")
    f.write(f"Population B    : {pB_fit*100:.2f} %\n")
    f.write(f"Separation dNu  : {dNu_fit:.2f} Hz\n")
    f.write(f"Center Shift    : {shift_fit:.2f} Hz\n")
    f.write(f"Amplitude Scale : {amp_fit:.4g}\n")
    f.write(f"Baseline Floor  : {base_fit:.4f}\n")

print(f"Results written to {TXT_OUTPUT}")

print("\n=== FIT SUMMARY ===")
print(f"k_ex            : {kex_fit:.2f} s^-1")
print(f"Population B    : {pB_fit*100:.2f} %")
print(f"Separation dNu  : {dNu_fit:.2f} Hz")
print(f"Center Shift    : {shift_fit:.2f} Hz")
print(f"Amplitude Scale : {amp_fit:.4g}")
print(f"Baseline Floor  : {base_fit:.4f}")

# ============================================================
# 7. PLOT
# ============================================================

sim_best = dnmr_2site(
    freq,
    kex_fit,
    shift_fit,
    LWA_intrinsic + LB,
    LWB_intrinsic + LB,
    dNu_fit,
    pB_fit
)

y_best = amp_fit * sim_best + base_fit

plt.figure(figsize=(8, 6), dpi=150)

plt.plot(
    freq,
    y_exp,
    "ko",
    alpha=0.25,
    label="Data (C145A + NMV)"
)

plt.plot(
    freq,
    y_best,
    "r--",
    lw=2.5,
    label=f"Model (kex={kex_fit:.1f} s$^{{-1}}$)"
)

plt.xlabel("Frequency (Hz)", fontsize=12)
plt.ylabel("Normalized Intensity", fontsize=12)
plt.title(
    f"MPro C145A Line-Shape Fit\n"
    f"pB = {pB_fit*100:.1f}%, Δν = {dNu_fit:.1f} Hz",
    fontsize=14,
    fontweight="bold"
)

plt.legend(frameon=False, fontsize=11)
plt.gca().invert_xaxis()

plt.tight_layout()
plt.savefig(FIG_OUTPUT, dpi=300)

print(f"Figure saved to {FIG_OUTPUT}")

plt.show()
