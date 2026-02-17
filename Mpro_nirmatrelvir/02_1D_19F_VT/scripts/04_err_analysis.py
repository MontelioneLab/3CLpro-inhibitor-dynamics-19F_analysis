import numpy as np
import os
import sys
from scipy.optimize import least_squares
from scipy.stats import t, linregress

# ============================================================
# 1. SETUP & PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DATA_ROOT = os.path.join(PROJECT_ROOT, "processed_ascii")

GLOBAL_NPZ = os.path.join(OUTPUT_DIR, "vt_global_fit.npz")
STATS_NPZ = os.path.join(OUTPUT_DIR, "vt_stats_final.npz")
REPORT_FILE = os.path.join(OUTPUT_DIR, "thermo_results.txt")
KEX_REPORT = os.path.join(OUTPUT_DIR, "kex_vs_temperature.txt")
EQ_REPORT = os.path.join(OUTPUT_DIR, "equilibrium_vs_temperature.txt")

if not os.path.exists(GLOBAL_NPZ):
    print(f"Error: Run Script 02 first."); sys.exit(1)

glob_data = np.load(GLOBAL_NPZ)
xg = glob_data['xg'] 
TEMPS = [283, 288, 293, 298, 303, 308, 313, 318]
kB = 1.3806e-23; h = 6.626e-34; R_kcal = 1.9872e-3; T0 = 298.15

def get_visc_scale(T): return 10**(247.8/(T-140) - 247.8/(T0-140))
def dnmr_2site(freq, kex, shift, r2a, r2b, dnu, pB):
    pi = np.pi; pA = 1.0 - pB; v = freq - shift
    aA, aB = r2a*pi - 1j*(2*pi*(0.5*dnu-v)), r2b*pi - 1j*(2*pi*(-0.5*dnu-v))
    return np.abs(((pA*aB + pB*aA + kex) / (aA*aB + kex*(aA+aB))).real)

# ============================================================
# 2. STEP 1: INDEPENDENT EXTRACTION (SCATTER)
# ============================================================
k_600, k_800, pB_600, pB_800 = [], [], [], []
DIR_600, DIR_800 = os.path.join(DATA_ROOT, "nirmat_VT_LB10_600"), os.path.join(DATA_ROOT, "nirmat_VT_LB10_800")

for T in TEMPS:
    p6, p8 = os.path.join(DIR_600, f"spec_{T}.dat"), os.path.join(DIR_800, f"spec_{T}.dat")
    if not (os.path.exists(p6) and os.path.exists(p8)):
        for l in [k_600, k_800, pB_600, pB_800]: l.append(np.nan)
        continue

    def fit_indep(p_path, r2a, r2b, dnu, b, mode='k'):
        d = np.loadtxt(p_path); d[:,0] -= d[np.argmax(d[:,1]),0]
        f, y = d[np.abs(d[:,0])<600,0], d[np.abs(d[:,0])<600,1]/np.max(d[:,1])
        sc = get_visc_scale(T)
        pb_g = 1/(1+np.exp((xg[3]*4184*(1-T/T0)+xg[2]*4184*(T/T0))/(8.314*T)))
        k_g = (kB*T/h) * np.exp(-(xg[1]*4184*(1-T/T0)+xg[0]*4184*(T/T0))/(8.314*T))
        s = f[np.argmax(y)] - f[np.argmax(dnmr_2site(f, k_g, 0, r2a*sc, r2b*sc, dnu, pb_g))]
        
        if mode == 'k': 
            res = lambda k: (dnmr_2site(f, k[0], s, r2a*sc, r2b*sc, dnu, pb_g)/np.max(dnmr_2site(f, k[0], s, r2a*sc, r2b*sc, dnu, pb_g)) + b) - y
            return least_squares(res, [k_g], bounds=(1, 50000)).x[0]
        else: 
            res = lambda p: (dnmr_2site(f, k_g, s, r2a*sc, r2b*sc, dnu, p[0])/np.max(dnmr_2site(f, k_g, s, r2a*sc, r2b*sc, dnu, p[0])) + b) - y
            p_val = least_squares(res, [pb_g], bounds=(0.01, 0.99)).x[0]
            return np.log(p_val/(1-p_val))

    k_600.append(fit_indep(p6, xg[4], xg[5], xg[10], xg[8], 'k'))
    k_800.append(fit_indep(p8, xg[6], xg[7], xg[12], xg[9], 'k'))
    pB_600.append(fit_indep(p6, xg[4], xg[5], xg[10], xg[8], 'p'))
    pB_800.append(fit_indep(p8, xg[6], xg[7], xg[12], xg[9], 'p'))

# ============================================================
# 3. STEP 2: STATISTICS & REPORTS
# ============================================================
mask = np.array(TEMPS) != 283; T_fit = np.array(TEMPS)[mask]; x_reg = 1000.0/T_fit
t_val = t.ppf(0.975, len(T_fit)-2); Sxx = np.sum((x_reg - np.mean(x_reg))**2)

def get_errors(y_data):
    reg = linregress(x_reg, y_data)
    se_pred = reg.stderr * np.sqrt(Sxx) * np.sqrt(1.0/len(T_fit) + (1000.0/T0 - np.mean(x_reg))**2 / Sxx)
    sigma_dG = se_pred * R_kcal * T0 * t_val
    sigma_dH = reg.stderr * R_kcal * 1000.0 * t_val
    return sigma_dG, sigma_dH

ci_dG_act, ci_dH_act = get_errors(np.log(((np.array(k_600)+np.array(k_800))/2)[mask]/T_fit))
ci_dG_eq, ci_dH_eq = get_errors(((np.array(pB_600)+np.array(pB_800))/2)[mask])

dS_act = (xg[1]-xg[0])*1000/T0
ci_dS_act = np.sqrt(ci_dG_act**2 + ci_dH_act**2) * 1000 / T0
dS_eq = (xg[3]-xg[2])*1000/T0
ci_dS_eq = np.sqrt(ci_dG_eq**2 + ci_dH_eq**2) * 1000 / T0

# --- FINAL CONSOLIDATED REPORT ---
with open(REPORT_FILE, "w") as f:
    f.write("3CLPRO-NIRMATRELVIR THERMODYNAMIC SUMMARY\n" + "="*45 + "\n")
    f.write(f"ACTIVATION (Transition State):\n")
    f.write(f"  DeltaG# (298 K) : {xg[0]:.2f} +/- {ci_dG_act:.2f} kcal/mol\n")
    f.write(f"  DeltaH#         : {xg[1]:.1f} +/- {ci_dH_act:.1f} kcal/mol\n")
    f.write(f"  DeltaS#         : {dS_act:.1f} +/- {ci_dS_act:.1f} cal/mol/K\n\n")
    f.write(f"CONFORMATIONAL EQUILIBRIUM (A <-> B):\n")
    f.write(f"  DeltaG0 (298 K) : {xg[2]:.2f} +/- {ci_dG_eq:.2f} kcal/mol\n")
    f.write(f"  DeltaH0         : {xg[3]:.1f} +/- {ci_dH_eq:.1f} kcal/mol\n")
    f.write(f"  DeltaS0         : {dS_eq:.1f} +/- {ci_dS_eq:.1f} cal/mol/K\n\n")
    f.write(f"Fit Linewidths (Hz):\n  R2A at 600 = {xg[4]:.1f}\n  R2B at 600 = {xg[5]:.1f}\n")
    f.write(f"  R2A at 800 = {xg[6]:.1f}\n  R2B at 800 = {xg[7]:.1f}\n\n")
    f.write(f"Intrinsic Linewidths (LB=10 Hz):\n  LWA at 600 = {xg[4]-10:.1f} Hz\n")
    f.write(f"  LWB at 600 = {xg[5]-10:.1f} Hz\n")
    f.write(f"  LWA at 800 = {xg[6]-10:.1f} Hz\n")
    f.write(f"  LWB at 800 = {xg[7]-10:.1f} Hz\n")

# --- KEX TABLE (CORRECTED INDEXING) ---
with open(KEX_REPORT, "w") as f:
    f.write("T (K)\tkex (s^-1)\t\tkAB (s^-1)\tkBA (s^-1)\tpB\n" + "-"*70 + "\n")
    for i, T in enumerate(TEMPS):
        if np.isnan(k_600[i]): continue
        k_glob = (kB*T/h) * np.exp(-(xg[1]*4184*(1-T/T0)+xg[0]*4184*(T/T0))/(8.314*T))
        k_err = np.abs(k_600[i] - k_800[i]) / np.sqrt(2.0)
        pb = 1/(1+np.exp((xg[3]*4184*(1-T/T0)+xg[2]*4184*(T/T0))/(8.314*T)))
        f.write(f"{T}\t{k_glob:5.1f} +/- {k_err:4.1f}\t{k_glob*pb:6.1f}\t{k_glob*(1-pb):6.1f}\t{pb:.3f}\n")

np.savez(STATS_NPZ, ci_dG_act=ci_dG_act, ci_dH_act=ci_dH_act, ci_dG_eq=ci_dG_eq, ci_dH_eq=ci_dH_eq, 
         k_600=k_600, k_800=k_800, pB_600=pB_600, pB_800=pB_800, T_vals=TEMPS)

# ============================================================
# 4. GENERATE EQUILIBRIUM VS TEMPERATURE REPORT
# ============================================================

with open(EQ_REPORT, "w") as f:
    f.write("CONFORMATIONAL EQUILIBRIUM CONSISTENCY CHECK\n")
    f.write("Fixed: Global kex and R2 | Floating: pB independently per field\n")
    f.write("-" * 80 + "\n")
    f.write("T (K)\tKeq\t\t+/- (std)\tpB (%)\t\t+/- (std %)\n")
    f.write("-" * 80 + "\n")

    for i, T in enumerate(TEMPS):
        if np.isnan(pB_600[i]): continue
        
        # Calculate lnK scatter (Sample Std Dev using sqrt(2))
        lnK_600, lnK_800 = pB_600[i], pB_800[i]
        lnK_err = np.abs(lnK_600 - lnK_800) / np.sqrt(2.0)
        
        # Center point from Global Fit (xg[2]=dG, xg[3]=dH)
        pb_g = 1/(1+np.exp((xg[3]*4184*(1-T/T0)+xg[2]*4184*(T/T0))/(8.314*T)))
        keq_g = pb_g / (1 - pb_g)
        
        # Propagate lnK scatter to Keq and pB (%)
        keq_err = keq_g * lnK_err
        pb_err = (keq_err / (1 + keq_g)**2) * 100 # Convert to percentage
        
        f.write(f"{T}\t{keq_g:6.3f} +/- {keq_err:6.3f}\t{pb_g*100:6.1f}% +/- {pb_err:6.2f}%\n")
