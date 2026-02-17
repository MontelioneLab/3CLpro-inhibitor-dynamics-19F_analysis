import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from scipy.optimize import least_squares
from scipy.stats import t, linregress

# ============================================================
# 1. SETUP & PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DERIVED_DATA_DIR = os.path.join(PROJECT_ROOT, "output")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")

GLOBAL_NPZ = os.path.join(DERIVED_DATA_DIR, "vt_global_fit.npz")
DATA_ROOT = os.path.join(PROJECT_ROOT, "processed_ascii")
DIR_600 = os.path.join(DATA_ROOT, "nirmat_VT_LB10_600")
DIR_800 = os.path.join(DATA_ROOT, "nirmat_VT_LB10_800")

xg = np.load(GLOBAL_NPZ)["xg"]
TEMPS = [283, 288, 293, 298, 303, 308, 313, 318]
R_kcal = 1.9872e-3; T0 = 298.15

def get_water_viscosity_scale(T): return 10**(247.8/(T-140) - 247.8/(298.15-140))
def dnmr_2site_robust(freq, kex, shift, r2a, r2b, dnu, pB):
    pi = np.pi; pA = 1.0 - pB; v = freq - shift
    aA = r2a*pi - 1j*(2*pi*(0.5*dnu - v)); aB = r2b*pi - 1j*(2*pi*(-0.5*dnu - v))
    num = pA*aB + pB*aA + kex; den = aA*aB + kex*(aA+aB)
    return np.abs((num/den).real)

# ============================================================
# 2. STEP 1: INDEPENDENT EXTRACTION 
# ============================================================
print("--- STEP 1: Extracting Independent pB Points ---")
res_600, res_800 = [], []

for T in TEMPS:
    sc = get_water_viscosity_scale(T)
    k_g = (1.38e-23*T/6.62e-34) * np.exp(-(xg[1]*4184*(1-T/T0)+xg[0]*4184*(T/T0))/(8.314*T))
    
    def fit_single_pB(p_path, r2a, r2b, dnu, b):
        if not os.path.exists(p_path): return np.nan, np.nan
        d = np.loadtxt(p_path); d[:,0] -= d[np.argmax(d[:,1]), 0]
        f, y = d[np.abs(d[:,0])<600, 0], d[np.abs(d[:,0])<600, 1]/np.max(d[:,1])
        s = f[np.argmax(y)] - f[np.argmax(dnmr_2site_robust(f, k_g, 0, r2a*sc, r2b*sc, dnu, 0.4))]
        def resid(p):
            m = dnmr_2site_robust(f, k_g, s, r2a*sc, r2b*sc, dnu, p[0])
            return (m/np.max(m) + b) - y
        fit = least_squares(resid, [0.4], bounds=(0.01, 0.99))
        pB_val = fit.x[0]
        return np.log(pB_val/(1-pB_val)), 0.05 

    res_600.append(fit_single_pB(os.path.join(DIR_600, f"spec_{T}.dat"), xg[4], xg[5], xg[10], xg[8]))
    res_800.append(fit_single_pB(os.path.join(DIR_800, f"spec_{T}.dat"), xg[6], xg[7], xg[12], xg[9]))

# ============================================================
# 3. STEP 2: PLOTTING (With Updated Axis Limits)
# ============================================================
mask = np.array(TEMPS) != 283; T_fit = np.array(TEMPS)[mask]; x_reg = 1000.0/T_fit
y_reg = (np.array([r[0] for r in res_600])[mask] + np.array([r[0] for r in res_800])[mask]) / 2.0
lin = linregress(x_reg, y_reg)

def calc_vh_band(x_arr):
    t_val = 2.57 # N=7, DOF=5
    sigma_dH0 = lin.stderr * R_kcal * 1000.0
    Sxx = np.sum((x_reg - np.mean(x_reg))**2)
    se_pred = lin.stderr * np.sqrt(Sxx) * np.sqrt(1.0/7 + (1000.0/T0 - np.mean(x_reg))**2 / Sxx)
    sigma_dG0 = (R_kcal * T0) * se_pred
    vG, vH = (sigma_dG0/t_val)**2, (sigma_dH0/t_val)**2
    cvGH = -0.99 * np.sqrt(vG * vH)
    gG, gH = -1.0/(R_kcal*T0), (1.0/R_kcal)*(1.0/T0 - x_arr/1000.0)
    return t_val * np.sqrt(np.abs((gG**2 * vG) + (gH**2 * vH) + (2*gG*gH*cvGH)))

fig, ax = plt.subplots(figsize=(8.5, 6))
T_line = np.linspace(275, 325, 200); x_line = 1000.0/T_line
y_line = lin.intercept + lin.slope * x_line
y_band = calc_vh_band(x_line)

ax.fill_between(x_line, y_line-y_band, y_line+y_band, color='#dddddd', alpha=0.6, label='95% CI')
ax.plot(x_line, y_line, 'k-', lw=1.5, label='Global Fit')

# Data Points
x_all = 1000.0/np.array(TEMPS)
y6_all = np.array([r[0] for r in res_600]); y8_all = np.array([r[0] for r in res_800])

ax.plot(x_all[mask], y6_all[mask], 'o', color='navy', ms=8, label='600 MHz')
ax.plot(x_all[mask], y8_all[mask], 's', color='#D50032', ms=8, label='800 MHz')
ax.plot(x_all[~mask], y6_all[~mask], 'o', mfc='none', mec='navy', mew=1.5, ms=8, label='Excluded (600 MHz)')
ax.plot(x_all[~mask], y8_all[~mask], 's', mfc='none', mec='#D50032', mew=1.5, ms=8, label='Excluded (800 MHz)')

# Updated Styling for Tight Fit
ax.set_xlabel("T$^{-1}$ (10$^3$ K$^{-1}$)", fontsize=18)
ax.set_ylabel("ln $K_{eq}$", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=16)

# New Limits and Ticks
ax.set_ylim(-1.2, 0.4) 
ax.set_yticks([-1.0, -0.5, 0.0])
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "vanthoff_plot.png"), dpi=600)
plt.show()
