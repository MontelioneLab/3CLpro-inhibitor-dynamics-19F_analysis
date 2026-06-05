#!/usr/bin/env python3

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os
import pandas as pd

# ============================================================
# 1. SETUP
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

CSV_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "C145A_kex_scan_summary.csv"
)

FIG_OUTPUT = os.path.join(
    FIGURE_DIR,
    "C145A_kex_scan_overlay.png"
)

FIG_NRMSE = os.path.join(
    FIGURE_DIR,
    "C145A_kex_scan_NRMSE.png"
)

# ============================================================
# 2. FIXED LINEWIDTHS
# ============================================================

LB = 10.0

LWA_intrinsic = 60.5
LWB_intrinsic = 60.0

LWA = LWA_intrinsic + LB
LWB = LWB_intrinsic + LB

# ============================================================
# 3. BLOCH-MCCONNELL MODEL
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

    kAB = kex * pB
    kBA = kex * pA

    GA = R2A + 1j * (wA - w) + kAB
    GB = R2B + 1j * (wB - w) + kBA

    numerator = pA * (GB + kAB) + pB * (GA + kBA)
    denominator = GA * GB - kAB * kBA

    return np.real(numerator / denominator)

# ============================================================
# 4. LOAD DATA
# ============================================================

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

y_exp = y_exp / np.max(y_exp)

print(f"Loaded {FILE_NAME}")

# ============================================================
# 5. FIT EACH FIXED kex
# ============================================================

kex_values = [0, 10, 20, 40, 60, 80, 100]

rows = []
fits = {}

for kex_fixed in kex_values:

    def residuals(params):
        # pB, dNu, shift_offset, amp, base
        pB, dNu, shift_offset, amp, base = params

        sim = dnmr_2site(
            freq,
            kex_fixed,
            shift_offset,
            LWA,
            LWB,
            dNu,
            pB
        )

        y_fit = amp * sim + base

        return y_fit - y_exp

    p0 = [0.18, 101.0, -45.0, 250.0, 0.0]

    lower = [0.05, 70.0, -80.0, 0.001, -0.05]
    upper = [0.40, 140.0, 80.0, 1e5, 0.05]

    result = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        loss="soft_l1"
    )

    pB_fit, dNu_fit, shift_fit, amp_fit, base_fit = result.x

    sim_best = dnmr_2site(
        freq,
        kex_fixed,
        shift_fit,
        LWA,
        LWB,
        dNu_fit,
        pB_fit
    )

    y_fit = amp_fit * sim_best + base_fit

    residual = y_fit - y_exp
    rmse = np.sqrt(np.mean(residual ** 2))
    nrmse = rmse / np.max(y_exp)

    rows.append({
        "kex": kex_fixed,
        "pB_percent": pB_fit * 100.0,
        "dNu_Hz": dNu_fit,
        "shift_Hz": shift_fit,
        "amp": amp_fit,
        "baseline": base_fit,
        "NRMSE": nrmse,
    })

    fits[kex_fixed] = y_fit

# ============================================================
# 6. OUTPUT TABLE
# ============================================================

df = pd.DataFrame(rows)
df["rel_NRMSE"] = df["NRMSE"] / df["NRMSE"].min()

df_round = df.copy()
for col in df_round.columns:
    if col == "kex":
        df_round[col] = df_round[col].astype(int)
    elif col in ["NRMSE", "rel_NRMSE"]:
        df_round[col] = df_round[col].round(4)
    else:
        df_round[col] = df_round[col].round(2)

df_round.to_csv(CSV_OUTPUT, index=False)

print("\nFixed-kex scan:")
print(df_round.to_string(index=False))
print(f"\nSaved CSV:\n  {CSV_OUTPUT}")

# ============================================================
# 7. OVERLAY PLOT
# ============================================================

plt.figure(figsize=(8, 6), dpi=150)

plt.plot(
    freq,
    y_exp,
    "k-",
    linewidth=2.2,
    label="Experimental"
)

for kex_fixed in kex_values:
    plt.plot(
        freq,
        fits[kex_fixed],
        linewidth=1.5,
        label=f"kex = {kex_fixed} s$^{{-1}}$"
    )

plt.xlabel("Frequency (Hz)", fontsize=12)
plt.ylabel("Normalized intensity", fontsize=12)
plt.title("C145A 298 K: fixed-kex sensitivity scan", fontsize=14, fontweight="bold")
plt.legend(frameon=False, fontsize=8)
plt.gca().invert_xaxis()
plt.tight_layout()

plt.savefig(FIG_OUTPUT, dpi=300)
plt.show()

print(f"Saved overlay figure:\n  {FIG_OUTPUT}")

# ============================================================
# 8. NRMSE PLOT
# ============================================================

plt.figure(figsize=(6, 4.5), dpi=150)

plt.plot(
    df["kex"],
    df["rel_NRMSE"],
    marker="o",
    linewidth=2
)

plt.xlabel("Fixed kex (s$^{-1}$)", fontsize=12)
plt.ylabel("Relative NRMSE", fontsize=12)
plt.title("C145A kex sensitivity", fontsize=14, fontweight="bold")
plt.axhline(1.0, linestyle=":", linewidth=1)
plt.tight_layout()

plt.savefig(FIG_NRMSE, dpi=300)
plt.show()

print(f"Saved NRMSE figure:\n  {FIG_NRMSE}")
