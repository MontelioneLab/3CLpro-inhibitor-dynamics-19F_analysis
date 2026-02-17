import numpy as np
from scipy.optimize import least_squares
import os

# ============================================================
# 1. SETUP & PATHS
# ============================================================
# Relative path logic for: Mpro_nirmatrelvir/02_1D_19F_VT/scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, "processed_ascii")

# Data directories for 600 MHz and 800 MHz datasets
DIR_600 = os.path.join(DATA_ROOT, "nirmat_VT_LB10_600")
DIR_800 = os.path.join(DATA_ROOT, "nirmat_VT_LB10_800")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CONCISE OUTPUT FILENAMES
NPZ_FILE = os.path.join(OUTPUT_DIR, "vt_global_fit.npz")

# EXPERIMENTAL CONSTANTS
FIT_WINDOW_WIDTH = 1200.0  # Hz
kB = 1.380649e-23; h = 6.62607015e-34; R = 8.314462618; T0 = 298.0
TEMPS = [283, 288, 293, 298, 303, 308, 313, 318]
EXCLUDED_TEMPS = [283] # Low temp excluded due to poor S/N or freezing issues

# ============================================================
# 2. PHYSICS MODELS
# ============================================================
def get_water_viscosity_scale(T): 
    """Scales R2 values based on temperature-dependent water viscosity."""
    return 10**(247.8/(T-140) - 247.8/(298-140))

def kex_eyring(T, dG, dH): 
    """Calculates kex using the Eyring equation anchored at 298 K."""
    return (kB*T/h) * np.exp(-(dH*4184*(1-T/T0) + dG*4184*(T/T0))/(R*T))

def pB_vanthoff(T, dG, dH): 
    """Calculates pB using the van't Hoff equation anchored at 298 K."""
    K = np.exp(-(dH*4184*(1-T/T0) + dG*4184*(T/T0))/(R*T))
    return K/(1+K)

def dnmr_2site_robust(freq, kex, shift, r2a, r2b, dnu, pB):
    """Core DNMR line-shape equation for two-site exchange."""
    pi = np.pi; pA = 1.0 - pB; v = freq - shift
    vA, vB = 0.5*dnu, -0.5*dnu
    w, wA, wB = 2*pi*v, 2*pi*vA, 2*pi*vB
    aA = r2a*pi - 1j*(wA-w); aB = r2b*pi - 1j*(wB-w)
    num = pA*aB + pB*aA + kex; den = aA*aB + kex*(aA+aB)
    return np.abs((num/den).real)

# ============================================================
# 3. DATA LOADING & FITTING
# ============================================================
if __name__ == "__main__":
    print("--- 1. LOADING VT DATA ---")
    temps_found, spectra_600, spectra_800 = [], [], []
    for T in TEMPS:
        if T in EXCLUDED_TEMPS: continue
        p6 = os.path.join(DIR_600, f"spec_{T}.dat")
        p8 = os.path.join(DIR_800, f"spec_{T}.dat")
        if os.path.exists(p6) and os.path.exists(p8):
            d6 = np.loadtxt(p6); d8 = np.loadtxt(p8)
            # Center on max peak and normalize
            d6[:,0] -= d6[np.argmax(d6[:,1]),0]; d8[:,0] -= d8[np.argmax(d8[:,1]),0]
            d6[:,1] /= d6[:,1].max(); d8[:,1] /= d8[:,1].max()
            
            m6 = np.abs(d6[:,0]) < FIT_WINDOW_WIDTH/2
            m8 = np.abs(d8[:,0]) < FIT_WINDOW_WIDTH/2
            spectra_600.append((d6[m6,0], d6[m6,1]))
            spectra_800.append((d8[m8,0], d8[m8,1]))
            temps_found.append(T)
            print(f"Loaded {T}K spectra (600 & 800 MHz)")

    print("\n--- 2. EXECUTING GLOBAL LEAST-SQUARES FIT ---")
    # Initial Guesses: dG_act, dH_act, dG_eq, dH_eq, R2s, baselines, deltas
    x0 = [14.0, 10.0, -1.5, -2.0, 30., 60., 40., 80., 0., 0., 500., 0., 660., 0.]
    x0 += [0.0] * len(temps_found) * 2 # Individual shifts per temp/field
    
    def residuals(x):
        res = []; dG, dH, dGe, dHe = x[:4]
        r2a6, r2b6, r2a8, r2b8 = x[4:8]; b6, b8 = x[8:10]; dn6, g6, dn8, g8 = x[10:14]
        shifts6 = x[14 : 14+len(temps_found)]
        shifts8 = x[14+len(temps_found) : ]
        for i, T in enumerate(temps_found):
            sc = get_water_viscosity_scale(T)
            k = kex_eyring(T, dG, dH)
            pb = pB_vanthoff(T, dGe, dHe)
            # 600 MHz residual
            s6 = dnmr_2site_robust(spectra_600[i][0], k, shifts6[i], r2a6*sc, r2b6*sc, dn6+g6*(T-298), pb)
            res.append((s6/s6.max() + b6) - spectra_600[i][1])
            # 800 MHz residual
            s8 = dnmr_2site_robust(spectra_800[i][0], k, shifts8[i], r2a8*sc, r2b8*sc, dn8+g8*(T-298), pb)
            res.append((s8/s8.max() + b8) - spectra_800[i][1])
        return np.concatenate(res)

    res = least_squares(residuals, x0, verbose=2)

    # ============================================================
    # 4. ERROR ANALYSIS & EXPORT
    # ============================================================
    J = res.jac
    residual = res.fun
    RSS = np.sum(residual**2)
    s_sq = RSS / (len(residual) - len(res.x))
    cov = s_sq * np.linalg.inv(J.T @ J)

    kex_vals = [kex_eyring(T, res.x[0], res.x[1]) for T in temps_found]

    # Save data for downstream plotting scripts (03, 04, 05)
    np.savez(
        NPZ_FILE,
        xg=res.x,
        jac=res.jac,
        cov=cov,
        s_sq=s_sq,
        T_vals=temps_found,
        kex_vals=kex_vals,
        excluded_temps=EXCLUDED_TEMPS
    )

    print(f"\nGlobal Fit Complete. Data saved to: {NPZ_FILE}")
