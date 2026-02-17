import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# Path Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")

# Load consolidated results from Script 03
stats = np.load(os.path.join(OUTPUT_DIR, "vt_stats_final.npz"))
glob_data = np.load(os.path.join(OUTPUT_DIR, "vt_global_fit.npz"))

xg = glob_data["xg"]
T_vals = stats["T_vals"]
k6, k8 = stats["k_600"], stats["k_800"]
kB = 1.3806e-23; h = 6.626e-34; R_kcal = 1.9872e-3; T0 = 298.15

def calc_band(T_arr):
    t_val = 2.57 # N=7, DOF=5
    var_G = (stats["ci_dG_act"] / t_val)**2
    var_H = (stats["ci_dH_act"] / t_val)**2
    cov_GH = -0.99 * np.sqrt(var_G * var_H)
    gG = -1.0 / (R_kcal * T0); gH = (1.0 / R_kcal) * (1.0/T0 - 1.0/T_arr)
    v = (gG**2 * var_G) + (gH**2 * var_H) + (2 * gG * gH * cov_GH)
    return t_val * np.sqrt(np.abs(v))

T_line = np.linspace(275, 325, 200); x_line = 1000.0/T_line
y_line = np.log(((kB*T_line/h) * np.exp(-(xg[1]*4184*(1-T_line/T0)+xg[0]*4184*(T_line/T0))/(8.314*T_line)))/T_line)
y_band = calc_band(T_line)

fig, ax = plt.subplots(figsize=(8.5, 6))
ax.fill_between(x_line, y_line-y_band, y_line+y_band, color='#dddddd', alpha=0.6, label='95% CI')
ax.plot(x_line, y_line, 'k-', lw=1.5, label='Global Fit')

# Data Points from Script 03
x_pts = 1000.0/T_vals
mask_inc = T_vals != 283; mask_exc = T_vals == 283

ax.plot(x_pts[mask_inc], np.log(k6[mask_inc]/T_vals[mask_inc]), 'o', color='navy', ms=8, label='600 MHz')
ax.plot(x_pts[mask_inc], np.log(k8[mask_inc]/T_vals[mask_inc]), 's', color='#D50032', ms=8, label='800 MHz')
ax.plot(x_pts[mask_exc], np.log(k6[mask_exc]/T_vals[mask_exc]), 'o', mfc='none', mec='navy', mew=1.5, ms=8, label='Excluded (600 MHz)')
ax.plot(x_pts[mask_exc], np.log(k8[mask_exc]/T_vals[mask_exc]), 's', mfc='none', mec='#D50032', mew=1.5, ms=8, label='Excluded (800 MHz)')

ax.set_xlabel("T$^{-1}$ (10$^3$ K$^{-1}$)", fontsize=18)
ax.set_ylabel("ln($k_{ex}$ / T)", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "eyring_plot.png"), dpi=600)
plt.show()
