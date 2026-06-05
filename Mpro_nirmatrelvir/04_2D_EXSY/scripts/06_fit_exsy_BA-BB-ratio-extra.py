#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm
import os

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_DIR = os.path.join(PROJECT_ROOT, "processed_data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")

os.makedirs(FIGURE_DIR, exist_ok=True)

# ============================================================
# INPUT FILES
# ============================================================

EXSY_FILE = os.path.join(INPUT_DIR, "exsy_peak_intensities.csv")

BOOT_FILE = os.path.join(
    OUTPUT_DIR,
    "06_EXSY_bootstrap_results.csv"
)

VT_FILE = os.path.join(
    INPUT_DIR,
    "1D_VT_fit_results.csv"
)

# ============================================================
# USER SETTINGS
# ============================================================

# highlight these curves
MAIN_KEX = {

    288: 20,

    298: 50,

    308: 125

}

KEX_VALUES = {

    288: [10, 20, 30, 40],

    298: [30, 50, 70, 80],

    308: [75, 125, 150, 175]

}
# measured T1
T1 = 1.45

# ------------------------------------------------------------
# Use Lorentzian linewidth scaling
# ------------------------------------------------------------

USE_HEIGHT_SCALING = True

# ============================================================
# READ FILES
# ============================================================

raw = pd.read_csv(EXSY_FILE)
boot = pd.read_csv(BOOT_FILE)
vt = pd.read_csv(VT_FILE)

raw.columns = raw.columns.str.strip()

# ============================================================
# MATRIX MODEL
# ============================================================

def simulate_exsy(tm, kex, pB, T1):

    pA = 1.0 - pB

    kAB = kex * pB
    kBA = kex * pA

    R1 = 1.0 / T1

    K = np.array([

        [-(R1 + kAB),  kBA],
        [ kAB,        -(R1 + kBA)]

    ])

    E = expm(K * tm)

    # --------------------------------------------------------
    # Initial magnetization
    # --------------------------------------------------------

    M_A0 = np.array([pA, 0.0])
    M_B0 = np.array([0.0, pB])

    M_A = E @ M_A0
    M_B = E @ M_B0

    AA = M_A[0]
    BA = M_A[1]

    AB = M_B[0]
    BB = M_B[1]

    return AA, BB, AB, BA

# ============================================================
# LOOP OVER TEMPERATURES
# ============================================================

for T in sorted(raw["T"].unique()):

    print("\n================================================")
    print(f"{T} K")
    print("================================================")

    # ========================================================
    # EXPERIMENTAL DATA
    # ========================================================

    df_exp = raw[
        raw["T"] == T
    ].copy()

    # --------------------------------------------------------
    # Experimental BA/BB
    # --------------------------------------------------------

    exp_ratio = (
        df_exp["h_BA"] /
        df_exp["h_BB"]
    )

    # ========================================================
    # BOOTSTRAP RESULTS
    # ========================================================

    row_boot = boot[
        boot["T"] == T
    ].iloc[0]

    median_kex = row_boot["median_kex"]

    print(f"Median kex = {median_kex:.1f}")

    # ========================================================
    # VT PARAMETERS
    # ========================================================

    row_vt = vt[
        (vt["field"] == 600)
        &
        (vt["T"] == T)
    ]

    if len(row_vt) == 0:

        print(f"No VT data for {T} K")
        continue

    pB = row_vt["pB"].mean()

    # --------------------------------------------------------
    # Lorentzian linewidths
    # --------------------------------------------------------

    lwA = row_vt["lwA"].mean()
    lwB = row_vt["lwB"].mean()

    print(f"pB = {pB:.3f}")
    print(f"LW A = {lwA:.1f} Hz")
    print(f"LW B = {lwB:.1f} Hz")

    # ========================================================
    # FIGURE
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8,6)
    )

    # --------------------------------------------------------
    # Experimental points
    # --------------------------------------------------------

    ax.scatter(

        df_exp["tmix_ms"],
        exp_ratio,

        s=120,
        color="black",

        label="Experimental",
        zorder=20

    )

    # ========================================================
    # SIMULATED CURVES
    # ========================================================

    tm_plot_ms = np.linspace(
        0,
        150,
        500
    )

    for kex_test in KEX_VALUES[T]:

        curve = []

        for tm_ms in tm_plot_ms:

            tm = tm_ms / 1000.0

            AA, BB, AB, BA = simulate_exsy(

                tm,
                kex_test,
                pB,
                T1

            )

            # =================================================
            # APPARENT PEAK HEIGHT RATIOS
            # =================================================

            if USE_HEIGHT_SCALING:

                # --------------------------------------------
                # Lorentzian peak height scaling:
                #
                # height ~ volume / LW^2
                # --------------------------------------------

                h_BA = BA / (lwA * lwB)

                h_BB = BB / (lwB * lwB)

                ratio = h_BA / h_BB

            else:

                ratio = BA / BB

            curve.append(ratio)

        # ====================================================
        # Styling
        # ====================================================

        if kex_test == MAIN_KEX[T]:

            lw = 4
            alpha = 1.0
            zorder = 10

        else:

            lw = 2
            alpha = 0.7
            zorder = 1

        ax.plot(

            tm_plot_ms,
            curve,

            linewidth=lw,
            alpha=alpha,
            zorder=zorder,

            label=f"kex = {kex_test:.0f} s$^{{-1}}$"

        )

    # ========================================================
    # FORMATTING
    # ========================================================

    ax.set_xlabel(
        "Mixing time (ms)",
        fontsize=18
    )

    ax.set_ylabel(
        "BA/BB apparent height ratio",
        fontsize=18
    )

    ax.set_title(

        f"{T} K   BA/BB comparison",

        fontsize=24,
        fontweight='bold'

    )

    ax.tick_params(
        axis='both',
        labelsize=14
    )

    ax.legend(
        fontsize=12
    )

    plt.tight_layout()

    # ========================================================
    # SAVE FIGURE
    # ========================================================

    FIG_OUT = os.path.join(

        FIGURE_DIR,
        f"BA_BB_height_ratio_{T}K.png"

    )

    plt.savefig(

        FIG_OUT,
        dpi=300,
        bbox_inches='tight'

    )

    print(f"Saved: {FIG_OUT}")

    plt.show(block=False)
    plt.pause(0.1)
    plt.close(fig)
