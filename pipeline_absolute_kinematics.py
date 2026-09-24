import os
import glob
import re
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import emcee
import corner

def compute_hdi(samples, credible_mass=0.6827):
    sorted_samples = np.sort(samples)
    window_size = int(np.ceil(credible_mass * len(sorted_samples)))
    if window_size == 0 or window_size > len(sorted_samples):
        return [sorted_samples[0], np.median(samples), sorted_samples[-1]]
    window_widths = sorted_samples[window_size-1:] - sorted_samples[:-window_size+1]
    min_idx = np.argmin(window_widths)
    hdi_lower = sorted_samples[min_idx]
    hdi_upper = sorted_samples[min_idx + window_size - 1]
    return [hdi_lower, np.median(samples), hdi_upper]

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# 1. LOAD AND PARSE DATA INTO ABSOLUTE OFFSETS FROM CATALOG
dir_path = '/media/kyle/kyle_phd/GRS1915/S-band-fits'
files = sorted(glob.glob(os.path.join(dir_path, '*2D_Fitting*.txt')))[:-1]

act_ra_deg = (19 + 15/60.0 + 11.558/3600.0) * 15.0
act_dec_deg = 10 + 56/60.0 + 44.905/3600.0
cos_dec = np.cos(np.radians(act_dec_deg))

core_data_raw = []
n_data = []
s_data = []

for f in files:
    filename = os.path.basename(f)
    m = re.search(r'img_(\d+)', filename)
    if not m: continue
    ts = int(m.group(1))
    mjd = ts / 86400.0 + 40587.0
    
    with open(f, 'r') as fp: lines = fp.readlines()
    
    comps = []
    curr_comp = None
    for line in lines:
        if line.startswith('Component #'):
            curr_comp = {'name': line.strip()}
            comps.append(curr_comp)
        elif 'Center X' in line and curr_comp is not None:
            m_ra = re.search(r'=\s*(\d+):(\d+):(\d+\.\d+)\s*±\s*(\S+)', line)
            if m_ra:
                h, m_time, s, err_s = float(m_ra.group(1)), float(m_ra.group(2)), float(m_ra.group(3)), float(m_ra.group(4))
                curr_comp['ra_deg'] = (h + m_time / 60.0 + s / 3600.0) * 15.0
                curr_comp['ra_err'] = err_s * 15.0 * cos_dec # arcsec
        elif 'Center Y' in line and curr_comp is not None:
            m_dec = re.search(r'=\s*([+-]?\d+):(\d+):(\d+\.\d+)\s*±\s*(\S+)', line)
            if m_dec:
                d, m_arc, s_arc, err_arcsec = float(m_dec.group(1)), float(m_dec.group(2)), float(m_dec.group(3)), float(m_dec.group(4))
                sign = -1 if '-' in m_dec.group(1) else 1
                curr_comp['dec_deg'] = sign * (abs(d) + m_arc / 60.0 + s_arc / 3600.0)
                curr_comp['dec_err'] = err_arcsec
        elif 'Integrated flux' in line and curr_comp is not None:
            m_flux = re.search(r'=\s*([\d\.eE+-]+)\s*±\s*([\d\.eE+-]+)', line)
            if m_flux:
                curr_comp['flux'] = float(m_flux.group(1))
                curr_comp['flux_err'] = float(m_flux.group(2))

    for c in comps:
        if 'ra_deg' not in c or 'dec_deg' not in c: continue
        
        dx = (c['ra_deg'] - act_ra_deg) * 3600.0 * cos_dec
        dy = (c['dec_deg'] - act_dec_deg) * 3600.0
        ex = c['ra_err']
        ey = c['dec_err']
        flux = c.get('flux', 0.0)
        flux_err = c.get('flux_err', 0.0)
        
        if 288.7979 < c['ra_deg'] < 288.7982:
            core_data_raw.append([mjd, dx, dy, ex, ey, flux, flux_err])
        elif dy > 0:
            n_data.append([mjd, dx, dy, ex, ey, flux, flux_err])
        else:
            s_data.append([mjd, dx, dy, ex, ey, flux, flux_err])

# Hard filter ONLY the farthest extreme outlier as requested for stability
core_data = []
for c in core_data_raw:
    if np.abs(c[1] - -0.317) < 0.01 and np.abs(c[2] - -0.601) < 0.01: continue
    core_data.append(c)

core_data = np.array(core_data)
n_data = np.array(n_data)
s_data = np.array(s_data)

n_data = n_data[np.argsort(n_data[:, 0])][2:]
s_data = s_data[np.argsort(s_data[:, 0])][2:]
num_obs_total = 2 * len(core_data) + 2 * len(n_data) + 2 * len(s_data)

print(f"Loaded {len(core_data)} core points, {len(n_data)} N points, {len(s_data)} S points")

# 2. UNIVERSAL LIKELIHOOD FUNCTION
def calc_log_likelihood(t_ej, v0_n, a_n, v0_s, a_s, pa_0_n, omega_n, pa_0_s, omega_s, theta_cone, x_c, y_c):
    tan2_cone = np.tan(np.radians(theta_cone))**2
    jitter2 = (1e-3)**2
    
    ll_total = 0.0
    
    # --- CORE LIKELIHOOD (Standard Gaussian, No Mixture) ---
    dx_c = core_data[:, 1] - x_c
    dy_c = core_data[:, 2] - y_c
    ex_c = core_data[:, 3]
    ey_c = core_data[:, 4]
    
    ll_in_x = -0.5 * (dx_c**2 / ex_c**2 + np.log(2 * np.pi * ex_c**2))
    ll_in_y = -0.5 * (dy_c**2 / ey_c**2 + np.log(2 * np.pi * ey_c**2))
    ll_total += np.sum(ll_in_x + ll_in_y)
        
    # --- NORTHERN LOBE LIKELIHOOD ---
    dt_n = n_data[:, 0] - t_ej
    dx_n = n_data[:, 1] - x_c
    dy_n = n_data[:, 2] - y_c
    ex_n = n_data[:, 3]
    ey_n = n_data[:, 4]
    
    theta_n = np.radians(pa_0_n + omega_n * dt_n)
    sin_n, cos_n = np.sin(theta_n), np.cos(theta_n)
    par_n = dx_n * sin_n + dy_n * cos_n
    perp_n = dx_n * cos_n - dy_n * sin_n
    
    model_par_n = v0_n * dt_n + 0.5 * a_n * dt_n**2
    model_par_n[dt_n < 0] = 0.0
    
    var_par_n = (ex_n * sin_n)**2 + (ey_n * cos_n)**2 + jitter2
    var_perp_n = (ex_n * cos_n)**2 + (ey_n * sin_n)**2 + model_par_n**2 * tan2_cone + jitter2
    
    ll_n = -0.5 * np.sum((par_n - model_par_n)**2 / var_par_n + np.log(2 * np.pi * var_par_n))
    ll_n += -0.5 * np.sum(perp_n**2 / var_perp_n + np.log(2 * np.pi * var_perp_n))
    ll_total += ll_n

    # --- SOUTHERN LOBE LIKELIHOOD ---
    dt_s = s_data[:, 0] - t_ej
    dx_s = s_data[:, 1] - x_c
    dy_s = s_data[:, 2] - y_c
    ex_s = s_data[:, 3]
    ey_s = s_data[:, 4]
    
    theta_s = np.radians(pa_0_s + omega_s * dt_s)
    sin_s, cos_s = np.sin(theta_s), np.cos(theta_s)
    par_s = dx_s * sin_s + dy_s * cos_s
    perp_s = dx_s * cos_s - dy_s * sin_s
    
    model_par_s = v0_s * dt_s + 0.5 * a_s * dt_s**2
    model_par_s[dt_s < 0] = 0.0
    
    var_par_s = (ex_s * sin_s)**2 + (ey_s * cos_s)**2 + jitter2
    var_perp_s = (ex_s * cos_s)**2 + (ey_s * sin_s)**2 + model_par_s**2 * tan2_cone + jitter2
    
    ll_s = -0.5 * np.sum((par_s - model_par_s)**2 / var_par_s + np.log(2 * np.pi * var_par_s))
    ll_s += -0.5 * np.sum(perp_s**2 / var_perp_s + np.log(2 * np.pi * var_perp_s))
    ll_total += ll_s
    
    return ll_total

def log_prior_universal(t_ej, v0_n, a_n, v0_s, a_s, pa_0_n, omega_n, pa_0_s, omega_s, theta_cone, x_c, y_c):
    if not (60000 < t_ej < 61000 and 1e-4 < v0_n < 0.5 and -1e-2 <= a_n <= 1e-2): return -np.inf
    if not (-0.5 < v0_s < -1e-4 and -1e-2 <= a_s <= 1e-2): return -np.inf
    if not (-180.0 < pa_0_n < 180.0 and -5.0 <= omega_n <= 5.0): return -np.inf
    if not (-180.0 < pa_0_s < 180.0 and -5.0 <= omega_s <= 5.0): return -np.inf
    if not (0.0 <= theta_cone <= 45.0): return -np.inf
    if not (-2.0 < x_c < 2.0 and -2.0 < y_c < 2.0): return -np.inf
    return 0.0

# 3. QUANTITATIVE MODEL SELECTION (BIC & AIC COMPARISONS)
print("\n" + "="*70)
print(f"{'QUANTITATIVE MODEL SELECTION (BIC / AIC SUITE)':^70}")
print("="*70)

# Model 1: 2nd-Order Polynomial + Precessing PA (12 params)
def nll_m1(p):
    lp = log_prior_universal(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11])
    if not np.isfinite(lp): return 1e12
    return -(lp + calc_log_likelihood(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11]))

res_m1 = minimize(nll_m1, [60370.0, 0.0055, 0.0, -0.0073, 0.0, -41.9, 0.0, -41.2, 0.0, 4.0, -0.14, -0.37], method='Nelder-Mead', options={'maxiter': 10000})
ll_m1 = calc_log_likelihood(*res_m1.x)
k_m1 = 12
bic_m1 = k_m1 * np.log(num_obs_total) - 2 * ll_m1

# Model 2: 1st-Order Linear Ballistic + Precessing PA (10 params: a_n = a_s = 0)
def nll_m2(p):
    lp = log_prior_universal(p[0], p[1], 0.0, p[2], 0.0, p[3], p[4], p[5], p[6], p[7], p[8], p[9])
    if not np.isfinite(lp): return 1e12
    return -(lp + calc_log_likelihood(p[0], p[1], 0.0, p[2], 0.0, p[3], p[4], p[5], p[6], p[7], p[8], p[9]))

res_m2 = minimize(nll_m2, [60370.0, 0.0055, -0.0073, -41.9, 0.0, -41.2, 0.0, 4.0, -0.14, -0.37], method='Nelder-Mead', options={'maxiter': 10000})
ll_m2 = calc_log_likelihood(res_m2.x[0], res_m2.x[1], 0.0, res_m2.x[2], 0.0, res_m2.x[3], res_m2.x[4], res_m2.x[5], res_m2.x[6], res_m2.x[7], res_m2.x[8], res_m2.x[9])
k_m2 = 10
bic_m2 = k_m2 * np.log(num_obs_total) - 2 * ll_m2

# Model 3: 2nd-Order Polynomial + Fixed PA (10 params: omega_n = omega_s = 0)
def nll_m3(p):
    lp = log_prior_universal(p[0], p[1], p[2], p[3], p[4], p[5], 0.0, p[6], 0.0, p[7], p[8], p[9])
    if not np.isfinite(lp): return 1e12
    return -(lp + calc_log_likelihood(p[0], p[1], p[2], p[3], p[4], p[5], 0.0, p[6], 0.0, p[7], p[8], p[9]))

res_m3 = minimize(nll_m3, [60370.0, 0.0055, 0.0, -0.0073, 0.0, -41.9, -41.2, 4.0, -0.14, -0.37], method='Nelder-Mead', options={'maxiter': 10000})
ll_m3 = calc_log_likelihood(res_m3.x[0], res_m3.x[1], res_m3.x[2], res_m3.x[3], res_m3.x[4], res_m3.x[5], 0.0, res_m3.x[6], 0.0, res_m3.x[7], res_m3.x[8], res_m3.x[9])
k_m3 = 10
bic_m3 = k_m3 * np.log(num_obs_total) - 2 * ll_m3

# Model 4: 1st-Order Linear Ballistic + Fixed PA (8 params: a_n = a_s = omega_n = omega_s = 0)
def nll_m4(p):
    lp = log_prior_universal(p[0], p[1], 0.0, p[2], 0.0, p[3], 0.0, p[4], 0.0, p[5], p[6], p[7])
    if not np.isfinite(lp): return 1e12
    return -(lp + calc_log_likelihood(p[0], p[1], 0.0, p[2], 0.0, p[3], 0.0, p[4], 0.0, p[5], p[6], p[7]))

res_m4 = minimize(nll_m4, [60370.0, 0.0055, -0.0073, -41.9, -41.2, 4.0, -0.14, -0.37], method='Nelder-Mead', options={'maxiter': 10000})
ll_m4 = calc_log_likelihood(res_m4.x[0], res_m4.x[1], 0.0, res_m4.x[2], 0.0, res_m4.x[3], 0.0, res_m4.x[4], 0.0, res_m4.x[5], res_m4.x[6], res_m4.x[7])
k_m4 = 8
bic_m4 = k_m4 * np.log(num_obs_total) - 2 * ll_m4

models = [
    {"id": 1, "name": "2nd-Order Polynomial + Precessing PA", "k": k_m1, "ll": ll_m1, "bic": bic_m1, "res": res_m1},
    {"id": 2, "name": "1st-Order Ballistic  + Precessing PA", "k": k_m2, "ll": ll_m2, "bic": bic_m2, "res": res_m2},
    {"id": 3, "name": "2nd-Order Polynomial + Fixed PA",      "k": k_m3, "ll": ll_m3, "bic": bic_m3, "res": res_m3},
    {"id": 4, "name": "1st-Order Ballistic  + Fixed PA",      "k": k_m4, "ll": ll_m4, "bic": bic_m4, "res": res_m4},
]

best_model = next(m for m in models if m["id"] == 4) # User forced fixed PA, no acceleration

print(f"{'Model Description':<40} {'k':<4} {'ln(L)':<10} {'BIC':<10} {'Delta BIC':<10}")
print("-" * 76)
for m in models:
    delta_bic = m["bic"] - best_model["bic"]
    print(f"{m['name']:<40} {m['k']:<4} {m['ll']:<10.2f} {m['bic']:<10.2f} {delta_bic:<10.2f}")
print("-" * 76)
print(f"--> ADOPTED BEST MODEL: Model {best_model['id']} ({best_model['name']})")
print("="*70 + "\n")


# 4. MCMC POSTERIOR SAMPLING FOR THE ADOPTED MODEL
if best_model["id"] == 4:
    ndim = 8
    param_labels = [r"$T_{\rm ej}$", r"$v_{0,n}$", r"$v_{0,s}$", r"${\rm PA}_n$", r"${\rm PA}_s$", r"$\theta_{\rm cone}$", r"$x_c$", r"$y_c$"]
    def log_prior(p): return log_prior_universal(p[0], p[1], 0.0, p[2], 0.0, p[3], 0.0, p[4], 0.0, p[5], p[6], p[7])
    def log_prob(p):
        lp = log_prior(p)
        if not np.isfinite(lp): return -np.inf
        return lp + calc_log_likelihood(p[0], p[1], 0.0, p[2], 0.0, p[3], 0.0, p[4], 0.0, p[5], p[6], p[7])
elif best_model["id"] == 2:
    ndim = 10
    param_labels = [r"$T_{\rm ej}$", r"$v_{0,n}$", r"$v_{0,s}$", r"${\rm PA}_{0,n}$", r"$\omega_n$", r"${\rm PA}_{0,s}$", r"$\omega_s$", r"$\theta_{\rm cone}$", r"$x_c$", r"$y_c$"]
    def log_prior(p): return log_prior_universal(p[0], p[1], 0.0, p[2], 0.0, p[3], p[4], p[5], p[6], p[7], p[8], p[9])
    def log_prob(p):
        lp = log_prior(p)
        if not np.isfinite(lp): return -np.inf
        return lp + calc_log_likelihood(p[0], p[1], 0.0, p[2], 0.0, p[3], p[4], p[5], p[6], p[7], p[8], p[9])
elif best_model["id"] == 3:
    ndim = 10
    param_labels = [r"$T_{\rm ej}$", r"$v_{0,n}$", r"$a_n$", r"$v_{0,s}$", r"$a_s$", r"${\rm PA}_n$", r"${\rm PA}_s$", r"$\theta_{\rm cone}$", r"$x_c$", r"$y_c$"]
    def log_prior(p): return log_prior_universal(p[0], p[1], p[2], p[3], p[4], p[5], 0.0, p[6], 0.0, p[7], p[8], p[9])
    def log_prob(p):
        lp = log_prior(p)
        if not np.isfinite(lp): return -np.inf
        return lp + calc_log_likelihood(p[0], p[1], p[2], p[3], p[4], p[5], 0.0, p[6], 0.0, p[7], p[8], p[9])
else:
    ndim = 12
    param_labels = [r"$T_{\rm ej}$", r"$v_{0,n}$", r"$a_n$", r"$v_{0,s}$", r"$a_s$", r"${\rm PA}_{0,n}$", r"$\omega_n$", r"${\rm PA}_{0,s}$", r"$\omega_s$", r"$\theta_{\rm cone}$", r"$x_c$", r"$y_c$"]
    def log_prior(p): return log_prior_universal(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11])
    def log_prob(p):
        lp = log_prior(p)
        if not np.isfinite(lp): return -np.inf
        return lp + calc_log_likelihood(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11])

print(f"Sampling posteriors via MCMC for Model {best_model['id']}...")
opt_init = best_model["res"].x
nwalkers = 48
n_steps = 15000
scales = np.abs(opt_init) * 0.05 + 1e-3
scales[0] = 2.0

pos_list = []
for _ in range(nwalkers):
    valid = False
    for _ in range(1000):
        trial = opt_init + scales * np.random.randn(ndim)
        if np.isfinite(log_prior(trial)):
            pos_list.append(trial)
            valid = True
            break
    if not valid: pos_list.append(opt_init)
pos = np.array(pos_list)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
sampler.run_mcmc(pos, n_steps, progress=False)

burnin = int(n_steps * 0.2)
thin = 10
flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

p_res = [compute_hdi(flat_samples[:, i]) for i in range(ndim)]

# 5. DERIVED KINEMATICS & OUTPUT
t_ej_chain = flat_samples[:, 0]
v0_n_chain = flat_samples[:, 1]
v0_s_idx = 2 if best_model["id"] in [2, 4] else 3
v0_s_chain = flat_samples[:, v0_s_idx]

pa_n_idx = param_labels.index(r"${\rm PA}_n$") if r"${\rm PA}_n$" in param_labels else param_labels.index(r"${\rm PA}_{0,n}$")
pa_s_idx = param_labels.index(r"${\rm PA}_s$") if r"${\rm PA}_s$" in param_labels else param_labels.index(r"${\rm PA}_{0,s}$")
xc_idx, yc_idx = ndim - 2, ndim - 1

mu_app_chain = np.abs(v0_s_chain)
mu_rec_chain = np.abs(v0_n_chain)

# 1. Distance-independent beta * cos(theta)
beta_cos_theta_chain = (mu_app_chain - mu_rec_chain) / (mu_app_chain + mu_rec_chain)
mu_true_chain = (2 * mu_app_chain * mu_rec_chain) / (mu_app_chain + mu_rec_chain)

# 2. Distance-dependent beta * sin(theta) (assumed D = 9.4 +/- 1.0 kpc)
D_kpc_chain = np.random.normal(9.4, 1.0, size=len(mu_app_chain))
D_m_chain = D_kpc_chain * 3.086e19
c = 2.99792458e8

mu_app_rad_s = mu_app_chain * (np.pi / (180.0 * 3600.0)) / 86400.0
beta_sin_theta_chain = mu_app_rad_s * (1.0 - beta_cos_theta_chain) * D_m_chain / c

# 3. Intrinsic Jet Speed beta, Lorentz factor gamma, and Inclination Angle theta
beta_chain = np.sqrt(beta_cos_theta_chain**2 + beta_sin_theta_chain**2)
beta_clamped = np.clip(beta_chain, 0.0, 0.999999)
gamma_chain = 1.0 / np.sqrt(1.0 - beta_clamped**2)
gamma_beta_chain = gamma_chain * beta_chain
theta_deg_chain = np.degrees(np.arctan2(beta_sin_theta_chain, beta_cos_theta_chain))

# Doppler factors for approaching and receding lobes
delta_app_chain = 1.0 / (gamma_chain * (1.0 - beta_cos_theta_chain))
delta_rec_chain = 1.0 / (gamma_chain * (1.0 + beta_cos_theta_chain))

# Maximum allowed distance
mu_rec_rad_s = mu_rec_chain * (np.pi / (180.0 * 3600.0)) / 86400.0
D_max_kpc_chain = (c / np.sqrt(mu_app_rad_s * mu_rec_rad_s)) / 3.086e19
delta_ratio_chain = ((1.0 + beta_cos_theta_chain) / (1.0 - beta_cos_theta_chain))

p_Tej = compute_hdi(t_ej_chain)
p_v0_n = compute_hdi(v0_n_chain)
p_v0_s = compute_hdi(v0_s_chain)
p_bct = compute_hdi(beta_cos_theta_chain)
p_beta = compute_hdi(beta_chain)
p_gamma = compute_hdi(gamma_chain)
p_gammabeta = compute_hdi(gamma_beta_chain)
p_theta = compute_hdi(theta_deg_chain)
p_delta_app = compute_hdi(delta_app_chain)
p_delta_rec = compute_hdi(delta_rec_chain)
p_mutrue = compute_hdi(mu_true_chain)
p_Dmax = compute_hdi(D_max_kpc_chain)
p_dratio = compute_hdi(delta_ratio_chain)
p_pa_n = p_res[pa_n_idx]
p_pa_s = p_res[pa_s_idx]

# Acceleration percentiles if present
if best_model["id"] in [1, 3]:
    p_a_n = p_res[2]
    p_a_s = p_res[4]

if best_model["id"] in [1, 2]:
    w_n_idx = param_labels.index(r"$\omega_n$")
    w_s_idx = param_labels.index(r"$\omega_s$")
    p_omega_n = p_res[w_n_idx]
    p_omega_s = p_res[w_s_idx]

print("\n" + "="*70)
print(f"{'GRS 1915+105 RELATIVISTIC KINEMATIC REPORT (ABSOLUTE)':^70}")
print("="*70)
print(f"Statistically Preferred Model : Model {best_model['id']} ({best_model['name']})")
print(f"Data Points Used (N_data)     : {num_obs_total}")
print(f"Optimized BIC                 : {best_model['bic']:.2f}")

print("\n[1] EJECTION PARAMETERS")
print("-" * 70)
print(f"Extrapolated Launch Time (T_ej): MJD {p_Tej[1]:.2f} + {p_Tej[2]-p_Tej[1]:.2f} - {p_Tej[1]-p_Tej[0]:.2f}")
print(f"Core Offset RA (x_c)           : {p_res[xc_idx][1]:.4f} + {p_res[xc_idx][2]-p_res[xc_idx][1]:.4f} - {p_res[xc_idx][1]-p_res[xc_idx][0]:.4f} arcsec")
print(f"Core Offset Dec (y_c)          : {p_res[yc_idx][1]:.4f} + {p_res[yc_idx][2]-p_res[yc_idx][1]:.4f} - {p_res[yc_idx][1]-p_res[yc_idx][0]:.4f} arcsec")

print("\n[2] APPARENT PROPER MOTIONS & JET AXIS")
print("-" * 70)
print(f"Receding Lobe (North) Speed (mu_r): {p_v0_n[1]:.4e} + {p_v0_n[2]-p_v0_n[1]:.4e} - {p_v0_n[1]-p_v0_n[0]:.4e} arcsec/day")
if best_model["id"] in [1, 3]:
    print(f"Receding Lobe Acceleration (a_n) : {p_a_n[1]:.4e} + {p_a_n[2]-p_a_n[1]:.4e} - {p_a_n[1]-p_a_n[0]:.4e} arcsec/day^2")
print(f"Approaching Lobe (South) Speed (mu_a): {np.abs(p_v0_s[1]):.4e} + {np.abs(p_v0_s[2]-p_v0_s[1]):.4e} - {np.abs(p_v0_s[1]-p_v0_s[0]):.4e} arcsec/day")
if best_model["id"] in [1, 3]:
    print(f"Approaching Lobe Acceleration (a_s) : {p_a_s[1]:.4e} + {p_a_s[2]-p_a_s[1]:.4e} - {p_a_s[1]-p_a_s[0]:.4e} arcsec/day^2")
print(f"Northern Position Angle (PA_n)    : {p_pa_n[1]:>8.2f} + {p_pa_n[2]-p_pa_n[1]:.2f} - {p_pa_n[1]-p_pa_n[0]:.2f} deg")
print(f"Standard Astronomical PA_n (EoN)  : {p_pa_n[1] + 180:>8.2f} deg")
if best_model["id"] in [1, 2]:
    print(f"Northern Precession Rate (Omega_n): {p_omega_n[1]:>8.3f} + {p_omega_n[2]-p_omega_n[1]:.3f} - {p_omega_n[1]-p_omega_n[0]:.3f} deg/day")
print(f"Southern Position Angle (PA_s)    : {p_pa_s[1]:>8.2f} + {p_pa_s[2]-p_pa_s[1]:.2f} - {p_pa_s[1]-p_pa_s[0]:.2f} deg")
print(f"Standard Astronomical PA_s (EoN)  : {p_pa_s[1] + 180:>8.2f} deg")
if best_model["id"] in [1, 2]:
    print(f"Southern Precession Rate (Omega_s): {p_omega_s[1]:>8.3f} + {p_omega_s[2]-p_omega_s[1]:.3f} - {p_omega_s[1]-p_omega_s[0]:.3f} deg/day")

print("\n[3] INTRINSIC RELATIVISTIC KINEMATICS")
print("-" * 70)
print(f"Assumed Distance (D)              : 9.4 +/- 1.0 kpc")
print(f"Derived beta * cos(theta)         : {p_bct[1]:>8.4f} + {p_bct[2]-p_bct[1]:.4f} - {p_bct[1]-p_bct[0]:.4f}")
print(f"Intrinsic Jet Speed (beta = v/c)  : {p_beta[1]:>8.4f} + {p_beta[2]-p_beta[1]:.4f} - {p_beta[1]-p_beta[0]:.4f} c")
print(f"Lorentz Factor (Gamma)            : {p_gamma[1]:>8.4f} + {p_gamma[2]-p_gamma[1]:.4f} - {p_gamma[1]-p_gamma[0]:.4f}")
print(f"Proper 4-Velocity (Gamma * Beta)  : {p_gammabeta[1]:>8.4f} + {p_gammabeta[2]-p_gammabeta[1]:.4f} - {p_gammabeta[1]-p_gammabeta[0]:.4f}")
print(f"Jet Inclination Angle (theta)     : {p_theta[1]:>8.2f} + {p_theta[2]-p_theta[1]:.2f} - {p_theta[1]-p_theta[0]:.2f} deg")
print(f"Approaching Doppler Factor (delta_a): {p_delta_app[1]:>6.4f} + {p_delta_app[2]-p_delta_app[1]:.4f} - {p_delta_app[1]-p_delta_app[0]:.4f}")
print(f"Receding Doppler Factor (delta_r) : {p_delta_rec[1]:>8.4f} + {p_delta_rec[2]-p_delta_rec[1]:.4f} - {p_delta_rec[1]-p_delta_rec[0]:.4f}")
print(f"Kinematic Doppler Factor Ratio    : {p_dratio[1]:>8.2f} + {p_dratio[2]-p_dratio[1]:.2f} - {p_dratio[1]-p_dratio[0]:.2f}")
print(f"True Intrinsic Angular Velocity   : {p_mutrue[1]:.4e} + {p_mutrue[2]-p_mutrue[1]:.4e} - {p_mutrue[1]-p_mutrue[0]:.4e} arcsec/day")
print(f"Maximum Physical Distance (D_max) : {p_Dmax[1]:>8.2f} + {p_Dmax[2]-p_Dmax[1]:.2f} - {p_Dmax[1]-p_Dmax[0]:.2f} kpc")
print("="*70 + "\n")

print("\nGenerating intrinsic kinematics corner plot...")
samples_intrinsic = np.vstack([D_m_chain / 3.086e19, theta_deg_chain, beta_chain]).T
fig_corner_kin = corner.corner(
    samples_intrinsic,
    labels=[r"Distance (kpc)", r"$\theta$ (deg)", r"$\beta$ ($v/c$)"],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12}
)
fig_corner_kin.savefig('/media/kyle/kyle_phd/GRS1915/figures/absolute_corner_plot_kinematics.png', dpi=300)
plt.close(fig_corner_kin)
print("Saved intrinsic corner plot to /media/kyle/kyle_phd/GRS1915/figures/absolute_corner_plot_kinematics.png")

# Generate plots
fig_corner = corner.corner(
    flat_samples, 
    labels=param_labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 10},
    title_fmt=".3f"
)
fig_corner.savefig('/media/kyle/kyle_phd/GRS1915/figures/absolute_corner_plot.png', dpi=300)
plt.close(fig_corner)

fig_core, ax = plt.subplots(figsize=(8,8))
ax.errorbar(core_data[:, 1], core_data[:, 2], xerr=core_data[:, 3], yerr=core_data[:, 4], fmt='o', alpha=0.5, label='Filtered Core Data')
ax.scatter(p_res[xc_idx][1], p_res[yc_idx][1], color='red', marker='*', s=300, label='Global Fitted Core Position', zorder=5)
ax.set_xlabel(r'$\Delta \alpha \cos \delta$ (arcsec)', fontsize=14)
ax.set_ylabel(r'$\Delta \delta$ (arcsec)', fontsize=14)
ax.axhline(0, color='grey', ls='--')
ax.axvline(0, color='grey', ls='--')
ax.legend()
plt.savefig('/media/kyle/kyle_phd/GRS1915/figures/core_mixture_fit.png', dpi=300)
plt.close(fig_core)

# -----------------------------------------------------------------------------
# 6. RA/DEC KINEMATICS PLOT (MERGED)
# -----------------------------------------------------------------------------
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MultipleLocator

print("\nGenerating RA/Dec Kinematics Plot...")

if best_model["id"] in [2, 4]:
    pa_n_deg = p_res[3][1]
    pa_s_deg = p_res[4][1] if best_model["id"] == 4 else p_res[5][1]
    theta_cone_deg = p_res[5][1] if best_model["id"] == 4 else p_res[7][1]
elif best_model["id"] == 3:
    pa_n_deg = p_res[5][1]
    pa_s_deg = p_res[6][1]
    theta_cone_deg = p_res[7][1]
else:
    pa_n_deg = p_res[5][1]
    pa_s_deg = p_res[7][1]
    theta_cone_deg = p_res[9][1]

weighted_mean_pa_deg = (pa_n_deg + pa_s_deg) / 2.0
weighted_std_deg = theta_cone_deg

avg_pa_rad = np.radians(weighted_mean_pa_deg)
std_pa_rad = np.radians(weighted_std_deg)

xc_fit = p_res[xc_idx][1]
yc_fit = p_res[yc_idx][1]

n_mjd = n_data[:, 0]
n_da = n_data[:, 1] - xc_fit
n_dd = n_data[:, 2] - yc_fit
n_da_err = n_data[:, 3]
n_dd_err = n_data[:, 4]

s_mjd = s_data[:, 0]
s_da = s_data[:, 1] - xc_fit
s_dd = s_data[:, 2] - yc_fit
s_da_err = s_data[:, 3]
s_dd_err = s_data[:, 4]

c_mjd = core_data[:, 0]
c_da = core_data[:, 1] - xc_fit
c_dd = core_data[:, 2] - yc_fit
c_da_err = core_data[:, 3]
c_dd_err = core_data[:, 4]

fig = plt.figure(figsize=(6, 9))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 0.06], height_ratios=[1, 1], wspace=0.08, hspace=0.15)

ax = fig.add_subplot(gs[:, 0])
cax_north = fig.add_subplot(gs[0, 1])
cax_south = fig.add_subplot(gs[1, 1])

mjd_min = min(np.min(n_mjd), np.min(s_mjd), np.min(c_mjd)) - 60000
mjd_max = max(np.max(n_mjd), np.max(s_mjd), np.max(c_mjd)) - 60000
norm = Normalize(vmin=mjd_min, vmax=mjd_max)

cmap_north = plt.cm.winter_r
cmap_south = plt.cm.autumn_r
cmap_core = plt.cm.viridis

L = 7.0
x_line = np.array([L * np.sin(avg_pa_rad), -L * np.sin(avg_pa_rad)])
y_line = np.array([L * np.cos(avg_pa_rad), -L * np.cos(avg_pa_rad)])
ax.plot(x_line, y_line, color='black', linestyle='--', lw=1.8, zorder=1.5)

pa_hi = avg_pa_rad + std_pa_rad
pa_lo = avg_pa_rad - std_pa_rad

ax.fill([0, L * np.sin(pa_lo), L * np.sin(pa_hi)], 
        [0, L * np.cos(pa_lo), L * np.cos(pa_hi)], 
        color='gray', alpha=0.22, zorder=1, edgecolor='none')
ax.fill([0, -L * np.sin(pa_lo), -L * np.sin(pa_hi)], 
        [0, -L * np.cos(pa_lo), -L * np.cos(pa_hi)], 
        color='gray', alpha=0.22, zorder=1, edgecolor='none')

ax.scatter(0, 0, marker='*', s=180, color='gold', edgecolors='black', linewidth=1.2, zorder=4)

ax.errorbar(n_da, n_dd, xerr=n_da_err, yerr=n_dd_err, fmt='none', ecolor='gray', capsize=0, zorder=2, alpha=0.6, elinewidth=1.2)
ax.errorbar(s_da, s_dd, xerr=s_da_err, yerr=s_dd_err, fmt='none', ecolor='gray', capsize=0, zorder=2, alpha=0.6, elinewidth=1.2)
ax.errorbar(c_da, c_dd, xerr=c_da_err, yerr=c_dd_err, fmt='none', ecolor='gray', capsize=0, zorder=2, alpha=0.6, elinewidth=1.2)

scatter_north = ax.scatter(n_da, n_dd, c=n_mjd - 60000, cmap=cmap_north, norm=norm, 
                           marker='^', edgecolors='black', s=60, zorder=3, label=r'$\Delta\theta^N$ Data')
scatter_south = ax.scatter(s_da, s_dd, c=s_mjd - 60000, cmap=cmap_south, norm=norm, 
                           marker='v', edgecolors='black', s=60, zorder=3, label=r'$\Delta\theta^S$ Data')
scatter_core = ax.scatter(c_da, c_dd, c=c_mjd - 60000, cmap=cmap_core, norm=norm,
                          marker='o', edgecolors='black', s=60, zorder=3, label=r'Core Data')

ax.set_xlim(-5.0, 5.0)
ax.set_ylim(-5.0, 5.0)
ax.set_xlabel(r'$\Delta\alpha \cos\delta$ ($^{\prime\prime}$)', fontsize=22)
ax.set_ylabel(r'$\Delta\delta$ ($^{\prime\prime}$)', fontsize=22)

ax.xaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(2))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
ax.minorticks_on()
ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, labelsize=16, length=7, width=1.4)
ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True, length=3.5, width=1.0)
ax.invert_xaxis()
ax.grid(True, linestyle=':', alpha=0.5)

cb_north = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap_north), cax=cax_north)
cb_north.set_label('MJD - 60000 (NW)', fontsize=14)
cb_north.ax.tick_params(labelsize=12)

cb_south = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap_south), cax=cax_south)
cb_south.set_label('MJD - 60000 (SE)', fontsize=14)
cb_south.ax.tick_params(labelsize=12)
cb_south.ax.invert_yaxis()

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    plt.tight_layout()
plt.savefig('/media/kyle/kyle_phd/GRS1915/figures/ra_dec_plot.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("Successfully generated and saved merged /media/kyle/kyle_phd/GRS1915/figures/ra_dec_plot.png")

# -----------------------------------------------------------------------------
# 6.5. SAVE DATA FOR STANDALONE SEPARATION PLOT
# -----------------------------------------------------------------------------
print("\nSaving data for standalone parallel/perpendicular separation plotting...")
os.makedirs('/media/kyle/kyle_phd/GRS1915/data', exist_ok=True)
np.savez('/media/kyle/kyle_phd/GRS1915/data/absolute_kinematics_plot_data.npz',
         flat_samples=flat_samples,
         best_model_id=best_model["id"],
         p_Tej=p_Tej,
         best_model_res_x=best_model["res"].x,
         n_mjd=n_mjd, n_da=n_da, n_dd=n_dd, n_da_err=n_da_err, n_dd_err=n_dd_err,
         s_mjd=s_mjd, s_da=s_da, s_dd=s_dd, s_da_err=s_da_err, s_dd_err=s_dd_err,
         c_mjd=c_mjd)
print("Saved plot data to /media/kyle/kyle_phd/GRS1915/data/absolute_kinematics_plot_data.npz")

# -----------------------------------------------------------------------------
# 7. FLUX EVOLUTION AND EXTRAPOLATION
# -----------------------------------------------------------------------------
print("\nGenerating Flux Evolution Plot...")
def exp_model(t, a, b):
    return a * np.exp(b * t)

def pl_model(t, a, b):
    # offset by 10 days to avoid singularity near t=0 if t is relative to outburst
    return a * (t + 10)**b

from scipy.optimize import curve_fit
import datetime

fig_flux, ax_flux = plt.subplots(figsize=(10, 6))

target_dates = ['2025-01-11 06:03:36', '2025-02-02 06:13:52', '2025-03-02 02:44:41', '2025-03-24 01:35:33']
target_ts = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S').timestamp() for d in target_dates]
t0_mjd = target_ts[0] / 86400.0 + 40587.0
target_mjd = np.array(target_ts) / 86400.0 + 40587.0

colors = {'Core': 'black', 'NW Lobe': 'blue', 'SE Lobe': 'red'}
markers = {'Core': 'o', 'NW Lobe': 's', 'SE Lobe': '^'}

for label, dataset in [('Core', core_data), ('NW Lobe', n_data), ('SE Lobe', s_data)]:
    if len(dataset) < 3: continue
    
    t_data = dataset[:, 0]
    f_data = dataset[:, 5]
    err_data = dataset[:, 6]
    
    valid = f_data > 0
    t_data = t_data[valid]
    f_data = f_data[valid]
    err_data = err_data[valid]
    
    t_days = t_data - t0_mjd
    target_days = target_mjd - t0_mjd
    
    try:
        if label == 'Core':
            popt, _ = curve_fit(pl_model, t_days, f_data, sigma=err_data, absolute_sigma=True, p0=[f_data[0] * 1000, -1.0])
            f_extrap = pl_model(target_days, *popt)
            t_fit = np.linspace(min(target_mjd) - 5, max(t_data) + 5, 200)
            f_fit = pl_model(t_fit - t0_mjd, *popt)
        else:
            popt, _ = curve_fit(exp_model, t_days, f_data, sigma=err_data, absolute_sigma=True, p0=[f_data[0], -0.01])
            f_extrap = exp_model(target_days, *popt)
            t_fit = np.linspace(min(target_mjd) - 5, max(t_data) + 5, 200)
            f_fit = exp_model(t_fit - t0_mjd, *popt)
            
        ax_flux.errorbar(t_data - 60000, f_data, yerr=err_data, fmt=markers[label], color=colors[label], label=f'{label} Data', capsize=3)
        ax_flux.plot(t_fit - 60000, f_fit, color=colors[label], linestyle='--', alpha=0.7)
        ax_flux.scatter(target_mjd - 60000, f_extrap, facecolors='none', edgecolors=colors[label], marker=markers[label], s=100, linewidth=2, label=f'{label} Extrapolated')
    except Exception as e:
        print(f"Failed to fit {label}: {e}")

ax_flux.set_yscale('log')
ax_flux.set_xlabel('Time (MJD - 60000)', fontsize=14)
ax_flux.set_ylabel('Flux (Jy)', fontsize=14)
ax_flux.set_title('Flux Evolution and Extrapolation', fontsize=16)
ax_flux.grid(True, which='both', linestyle='--', alpha=0.5)
ax_flux.legend(loc='best')
plt.tight_layout()
plt.savefig('/media/kyle/kyle_phd/GRS1915/figures/flux_evolution_with_extrap.png', dpi=300)
plt.close(fig_flux)
print("Successfully generated and saved /media/kyle/kyle_phd/GRS1915/figures/flux_evolution_with_extrap.png")
