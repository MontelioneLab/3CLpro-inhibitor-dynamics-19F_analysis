#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import matplotlib.gridspec as gridspec
import pandas as pd
import os

# ============================================================
# 1. PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_DIR = os.path.join(PROJECT_ROOT, "processed_data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

CSV_FILE = os.path.join(INPUT_DIR, "1D_VT_fit_results.csv")

# ============================================================
# 2. USER SETTINGS
# ============================================================

TEMPERATURE = 288

# try many values by hand
k_ex = 50.0  # s^-1

# measured EXSY T1
T1 = 1.40

# EXSY mixing times (seconds)
mix_times = [
    0.005,
    0.010,
    0.020,
    0.030,
    0.045,
    0.060,
    0.080,
    0.100
]

# EXSY used LB=15
# 1D used LB=10
# therefore add +5 Hz
EXTRA_LB = 5.0

# ============================================================
# 3. LOAD 1D FIT PARAMETERS
# ============================================================

df = pd.read_csv(CSV_FILE)

df = df[
    (df["field"] == 600)
    &
    (df["T"] == TEMPERATURE)
]

# average replicates if present
pB = df["pB"].mean()

dnu = df["dnu"].mean()

lwA = df["lwA"].mean() + EXTRA_LB
lwB = df["lwB"].mean() + EXTRA_LB

print("\n=================================================")
print(f"EXSY SIMULATION : {TEMPERATURE} K")
print("=================================================\n")

print(f"kex     : {k_ex:.1f} s^-1")
print(f"T1      : {T1:.2f} s")
print(f"pB      : {pB:.3f}")
print(f"dnu     : {dnu:.2f} Hz")
print(f"lwA     : {lwA:.2f} Hz")
print(f"lwB     : {lwB:.2f} Hz")

# ============================================================
# 4. POPULATIONS / RATES
# ============================================================

pA = 1.0 - pB

kAB = k_ex * pB
kBA = k_ex * pA

# ============================================================
# 5. FREQUENCIES
# ============================================================

freq_A = +dnu/2
freq_B = -dnu/2

# ============================================================
# 6. EXCHANGE + RELAXATION
# ============================================================

def get_intensities(tm):

    R1 = 1.0 / T1

    K = np.array([
        [-(R1 + kAB),   +kBA],
        [ +kAB,       -(R1 + kBA)]
    ])

    E = scipy.linalg.expm(K * tm)

    # volumes
    V_AA = E[0,0] * pA
    V_BA = E[0,1] * pB

    V_AB = E[1,0] * pA
    V_BB = E[1,1] * pB

    return V_AA, V_BB, V_AB, V_BA

# ============================================================
# 7. GAUSSIAN HELPERS
# ============================================================

def gaussian_2d_height(volume, sig_x, sig_y):

    return volume / (2*np.pi*sig_x*sig_y)

def make_gaussian(X, Y, height, x0, y0, sig_x, sig_y):

    return height * np.exp(
        -(
            ((X-x0)**2)/(2*sig_x**2)
            +
            ((Y-y0)**2)/(2*sig_y**2)
        )
    )

# ============================================================
# 8. GRID
# ============================================================

x = np.linspace(-250, 250, 400)
y = np.linspace(-250, 250, 400)

X, Y = np.meshgrid(x, y)

sig_A = lwA / 2.355
sig_B = lwB / 2.355

# ============================================================
# 9. CONTOUR LEVELS
# ============================================================

V_AA0, _, _, _ = get_intensities(0.001)

max_ref = gaussian_2d_height(
    V_AA0,
    sig_A,
    sig_A
)

base_level = 0.090 * max_ref  # The 'floor' level for the whole series
multiplier = 1.2             # Each level is 1.3x the previous one
n_levels   = 50              # Number of rings to try and draw

global_levels = [
    base_level * (multiplier ** i)
    for i in range(n_levels)
]

# ============================================================
# 10. FIGURE
# ============================================================

fig = plt.figure(
    figsize=(14, 2.3),
    dpi=300
)

gs = gridspec.GridSpec(
    1,
    len(mix_times),
    wspace=0.00 # space between spectra
)

plt.rcParams.update({
    'font.sans-serif': 'Arial',
    'font.size': 10,
    'lines.linewidth': 1.0
})

# ============================================================
# 11. PLOT
# ============================================================

summary_lines = []

for i, tm in enumerate(mix_times):

    ax = plt.subplot(gs[i])

    V_AA, V_BB, V_AB, V_BA = get_intensities(tm)

    h_AA = gaussian_2d_height(
        V_AA,
        sig_A,
        sig_A
    )

    h_BB = gaussian_2d_height(
        V_BB,
        sig_B,
        sig_B
    )

    h_AB = gaussian_2d_height(
        V_AB,
        sig_B,
        sig_A
    )

    h_BA = gaussian_2d_height(
        V_BA,
        sig_A,
        sig_B
    )

    Z = np.zeros_like(X)

    # diagonal A
    Z += make_gaussian(
        X, Y,
        h_AA,
        freq_A, freq_A,
        sig_A, sig_A
    )

    # diagonal B
    Z += make_gaussian(
        X, Y,
        h_BB,
        freq_B, freq_B,
        sig_B, sig_B
    )

    # cross AB
    Z += make_gaussian(
        X, Y,
        h_AB,
        freq_B, freq_A,
        sig_B, sig_A
    )

    # cross BA
    Z += make_gaussian(
        X, Y,
        h_BA,
        freq_A, freq_B,
        sig_A, sig_B
    )

    # --------------------------------------------------------
    # contours
    # --------------------------------------------------------

    ax.contour(
        X,
        Y,
        Z,
        levels=global_levels,
        colors=['#003366'],
        linewidths=0.5
    )

    # --------------------------------------------------------
    # appearance
    # --------------------------------------------------------

    ax.set_aspect('equal')

# match experimental zoom
    ax.set_xlim(160, -160)
    ax.set_ylim(160, -160)

    ax.set_xticks([])
    ax.set_yticks([])

# spectral mixing times titles for each spectra
    ax.set_title(
        f"{tm*1000:.0f} ms",
        fontsize=11,
        fontweight='bold',
        pad=4
    )

    # --------------------------------------------------------
    # small text beneath panels
    # --------------------------------------------------------

    ratio_diag = h_BB / h_AA
    ratio_cross = h_AB / h_AA

    label = (
        f"Diag B/A: {ratio_diag:.2f}\n"
        f"Cross/A: {ratio_cross:.2f}"
    )

    ax.text(
        0.5,
        -0.035,
        label,
        ha='center',
        va='top',
        transform=ax.transAxes,
        fontsize=5
    )

    summary_lines.append(
        f"{tm*1000:6.0f} ms   "
        f"Diag={ratio_diag:.3f}   "
        f"Cross={ratio_cross:.3f}"
    )

# ============================================================
# 12. SAVE FIGURE
# ============================================================

fig.suptitle(
    f"{TEMPERATURE} K   kex = {k_ex:.1f} s$^{{-1}}$",
    fontsize=18,
    fontweight='bold',
    y=0.98
)

#plt.tight_layout()
plt.tight_layout(rect=[0,0,1,0.95])

FIG_OUT = os.path.join(
    FIGURE_DIR,
    f"EXSY_sim_{TEMPERATURE}K_kex{k_ex:.0f}.png"
)

plt.savefig(
    FIG_OUT,
    dpi=300,
    bbox_inches='tight'
)

print("\nSaved figure:")
print(FIG_OUT)

# ============================================================
# 13. SAVE TXT SUMMARY
# ============================================================

TXT_OUT = os.path.join(
    OUTPUT_DIR,
    f"EXSY_sim_{TEMPERATURE}K_kex{k_ex:.0f}.txt"
)

with open(TXT_OUT, "w") as f:

    f.write("=================================================\n")
    f.write("EXSY SIMULATION SUMMARY\n")
    f.write("=================================================\n\n")

    f.write(f"T           : {TEMPERATURE} K\n")
    f.write(f"kex         : {k_ex:.1f} s^-1\n")
    f.write(f"T1          : {T1:.2f} s\n")
    f.write(f"pB          : {pB:.3f}\n")
    f.write(f"dnu         : {dnu:.2f} Hz\n")
    f.write(f"lwA         : {lwA:.2f} Hz\n")
    f.write(f"lwB         : {lwB:.2f} Hz\n\n")

    for line in summary_lines:
        f.write(line + "\n")

print("\nSaved summary:")
print(TXT_OUT)

plt.show()
