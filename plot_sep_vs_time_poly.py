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
    # Computes 1D HDI
    sorted_samples = np.sort(samples)
    window_size = int(np.ceil(credible_mass * len(sorted_samples)))
    if window_size == 0 or window_size > len(sorted_samples):
        return [sorted_samples[0], np.median(samples), sorted_samples[-1]]
    window_widths = sorted_samples[window_size-1:] - sorted_samples[:-window_size+1]
    min_idx = np.argmin(window_widths)
    hdi_lower = sorted_samples[min_idx]
    hdi_upper = sorted_samples[min_idx + window_size - 1]
    return [hdi_lower, np.median(samples), hdi_upper]

def compute_hdi_2d(samples_2d, credible_mass=0.6827):
    # Computes 2D HDI along axis=0
    n_samples, n_points = samples_2d.shape
    window_size = int(np.ceil(credible_mass * n_samples))
    sorted_samples = np.sort(samples_2d, axis=0)
    window_widths = sorted_samples[window_size-1:, :] - sorted_samples[:-window_size+1, :]
    min_idx = np.argmin(window_widths, axis=0)
    point_indices = np.arange(n_points)
    hdi_lower = sorted_samples[min_idx, point_indices]
    hdi_upper = sorted_samples[min_idx + window_size - 1, point_indices]
    medians = np.median(samples_2d, axis=0)
    return np.array([hdi_lower, medians, hdi_upper])

# Set typography styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# -----------------------------------------------------------------------------
# 1. LOAD AND PARSE 2D FITTING DATA
# -----------------------------------------------------------------------------
dir_path = '/media/kyle/kyle_phd/GRS1915/S-band-fits'
files = glob.glob(os.path.join(dir_path, '*2D_Fitting*.txt'))

core_ras_deg = []
core_decs_deg = []
core_ra_errs_sec = []
core_dec_errs_arcsec = []
epochs_data = []

for f in sorted(files):
    filename = os.path.basename(f)
    m = re.search(r'img_(\d+)', filename)
    if not m:
        continue
    ts = int(m.group(1))
    mjd = ts / 86400.0 + 40587.0
    
    with open(f, 'r') as fp:
        lines = fp.readlines()
    
    components = []
    curr_comp = None
    for line in lines:
        if line.startswith('Component #'):
            curr_comp = {'name': line.strip()}
            components.append(curr_comp)
        elif 'Center X' in line and curr_comp is not None:
            m_ra = re.search(r'=\s*(\d+):(\d+):(\d+\.\d+)\s*±\s*(\d+\.\d+(?:[eE][+-]?\d+)?)', line)
            if m_ra:
                h, m_time, s, err_s = float(m_ra.group(1)), float(m_ra.group(2)), float(m_ra.group(3)), float(m_ra.group(4))
                curr_comp['ra_deg'] = (h + m_time / 60.0 + s / 3600.0) * 15.0
                curr_comp['ra_err'] = err_s
        elif 'Center Y' in line and curr_comp is not None:
            m_dec = re.search(r'=\s*([+-]?\d+):(\d+):(\d+\.\d+)\s*±\s*(\d+\.\d+(?:[eE][+-]?\d+)?)', line)
            if m_dec:
                d, m_arc, s_arc, err_arcsec = float(m_dec.group(1)), float(m_dec.group(2)), float(m_dec.group(3)), float(m_dec.group(4))
                sign = -1 if '-' in m_dec.group(1) else 1
                curr_comp['dec_deg'] = sign * (abs(d) + m_arc / 60.0 + s_arc / 3600.0)
                curr_comp['dec_err'] = err_arcsec
            
    epoch_comps = []
    epoch_core_ra = None
    epoch_core_dec = None
    epoch_core_ra_err = None
    epoch_core_dec_err = None
    
    for c in components:
        ra_deg = c.get('ra_deg')
        dec_deg = c.get('dec_deg')
        ra_err = c.get('ra_err')
        dec_err = c.get('dec_err')
        if ra_deg is not None and dec_deg is not None:
            epoch_comps.append({'ra_deg': ra_deg, 'dec_deg': dec_deg, 'ra_err': ra_err, 'dec_err': dec_err})
            if 288.7979 < ra_deg < 288.7982:
                epoch_core_ra = ra_deg
                epoch_core_dec = dec_deg
                epoch_core_ra_err = ra_err
                epoch_core_dec_err = dec_err
                core_ras_deg.append(ra_deg)
                core_decs_deg.append(dec_deg)
                core_ra_errs_sec.append(ra_err)
                core_dec_errs_arcsec.append(dec_err)
                
    epochs_data.append({
        'mjd': mjd, 
        'comps': epoch_comps, 
        'core_ra': epoch_core_ra, 
        'core_dec': epoch_core_dec,
        'core_ra_err': epoch_core_ra_err,
        'core_dec_err': epoch_core_dec_err
    })

mean_core_ra = np.mean(core_ras_deg)
mean_core_dec = np.mean(core_decs_deg)
mean_core_ra_err = np.sqrt(np.sum(np.array(core_ra_errs_sec)**2)) / len(core_ra_errs_sec)
mean_core_dec_err = np.sqrt(np.sum(np.array(core_dec_errs_arcsec)**2)) / len(core_dec_errs_arcsec)

n_mjd, n_da, n_dd, n_da_err, n_dd_err = [], [], [], [], []
s_mjd, s_da, s_dd, s_da_err, s_dd_err = [], [], [], [], []

for epoch in epochs_data:
    mjd = epoch['mjd']
    ref_ra = epoch['core_ra'] if epoch['core_ra'] is not None else mean_core_ra
    ref_dec = epoch['core_dec'] if epoch['core_dec'] is not None else mean_core_dec
    ref_ra_err = epoch['core_ra_err'] if epoch['core_ra_err'] is not None else mean_core_ra_err
    ref_dec_err = epoch['core_dec_err'] if epoch['core_dec_err'] is not None else mean_core_dec_err
    
    cos_dec = np.cos(np.radians(ref_dec))
    
    for c in epoch['comps']:
        ra_deg = c['ra_deg']
        dec_deg = c['dec_deg']
        ra_err = c['ra_err']
        dec_err = c['dec_err']
        
        if 288.7979 < ra_deg < 288.7982:
            continue 
            
        d_alpha = (ra_deg - ref_ra) * 3600.0 * cos_dec
        d_delta = (dec_deg - ref_dec) * 3600.0
        
        sigma_alpha_lobe = ra_err * 15.0 * cos_dec
        sigma_alpha_core = ref_ra_err * 15.0 * cos_dec
        sigma_d_alpha = np.sqrt(sigma_alpha_lobe**2 + sigma_alpha_core**2)
        
        sigma_delta_lobe = dec_err
        sigma_delta_core = ref_dec_err
        sigma_d_delta = np.sqrt(sigma_delta_lobe**2 + sigma_delta_core**2)
        
        if d_delta > 0:
            n_mjd.append(mjd)
            n_da.append(d_alpha)
            n_dd.append(d_delta)
            n_da_err.append(sigma_d_alpha)
            n_dd_err.append(sigma_d_delta)
        else:
            s_mjd.append(mjd)
            s_da.append(d_alpha)
            s_dd.append(d_delta)
            s_da_err.append(sigma_d_alpha)
            s_dd_err.append(sigma_d_delta)

# Sort arrays by time and slice off the initial two epochs
sort_n = np.argsort(n_mjd)
n_mjd = np.array(n_mjd)[sort_n][2:]
n_da = np.array(n_da)[sort_n][2:]
n_dd = np.array(n_dd)[sort_n][2:]
n_da_err = np.array(n_da_err)[sort_n][2:]
n_dd_err = np.array(n_dd_err)[sort_n][2:]

sort_s = np.argsort(s_mjd)
s_mjd = np.array(s_mjd)[sort_s][2:]
s_da = np.array(s_da)[sort_s][2:]
s_dd = np.array(s_dd)[sort_s][2:]
s_da_err = np.array(s_da_err)[sort_s][2:]
s_dd_err = np.array(s_dd_err)[sort_s][2:]

n_data = 2 * len(n_mjd) + 2 * len(s_mjd)

# -----------------------------------------------------------------------------
# 2. GENERAL LIKELIHOOD & PRIOR FUNCTIONS
# -----------------------------------------------------------------------------
def calc_log_likelihood(t_ej, v0_n, a_n, v0_s, a_s, pa_0_n, omega_n, pa_0_s, omega_s, theta_cone):
    tan2_cone = np.tan(np.radians(theta_cone))**2
    jitter2 = (1e-3)**2
    
    # Northern Lobe
    theta_n = np.radians(pa_0_n + omega_n * (n_mjd - t_ej))
    sin_n, cos_n = np.sin(theta_n), np.cos(theta_n)
    par_n = n_da * sin_n + n_dd * cos_n
    perp_n = n_da * cos_n - n_dd * sin_n
    
    dt_n = n_mjd - t_ej
    model_par_n = v0_n * dt_n + 0.5 * a_n * dt_n**2
    
    var_par_n = (n_da_err * sin_n)**2 + (n_dd_err * cos_n)**2 + jitter2
    var_perp_n = (n_da_err * cos_n)**2 + (n_dd_err * sin_n)**2 + model_par_n**2 * tan2_cone + jitter2
    
    ll_n = -0.5 * np.sum((par_n - model_par_n)**2 / var_par_n + np.log(2 * np.pi * var_par_n))
    ll_n += -0.5 * np.sum(perp_n**2 / var_perp_n + np.log(2 * np.pi * var_perp_n))
    
    # Southern Lobe
    theta_s = np.radians(pa_0_s + omega_s * (s_mjd - t_ej))
    sin_s, cos_s = np.sin(theta_s), np.cos(theta_s)
    par_s = s_da * sin_s + s_dd * cos_s
    perp_s = s_da * cos_s - s_dd * sin_s
    
    dt_s = s_mjd - t_ej
    model_par_s = v0_s * dt_s + 0.5 * a_s * dt_s**2
    
    var_par_s = (s_da_err * sin_s)**2 + (s_dd_err * cos_s)**2 + jitter2
    var_perp_s = (s_da_err * cos_s)**2 + (s_dd_err * sin_s)**2 + model_par_s**2 * tan2_cone + jitter2
    
    ll_s = -0.5 * np.sum((par_s - model_par_s)**2 / var_par_s + np.log(2 * np.pi * var_par_s))
    ll_s += -0.5 * np.sum(perp_s**2 / var_perp_s + np.log(2 * np.pi * var_perp_s))
    
    return ll_n + ll_s

def log_prior_universal(t_ej, v0_n, a_n, v0_s, a_s, pa_0_n, omega_n, pa_0_s, omega_s, theta_cone):
    if not (60000 < t_ej < 61000 and 1e-4 < v0_n < 0.5 and -1e-2 <= a_n <= 1e-2):
        return -np.inf
    if not (-0.5 < v0_s < -1e-4 and -1e-2 <= a_s <= 1e-2):
        return -np.inf
    if not (-180.0 < pa_0_n < 180.0 and -5.0 <= omega_n <= 5.0):
        return -np.inf
    if not (-180.0 < pa_0_s < 180.0 and -5.0 <= omega_s <= 5.0):
        return -np.inf
    if not (0.0 <= theta_cone <= 45.0):
        return -np.inf
    
    return 0.0

# -----------------------------------------------------------------------------
# 3. QUANTITATIVE MODEL SELECTION (BIC & AIC COMPARISONS)
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print(f"{'QUANTITATIVE MODEL SELECTION (BIC / AIC SUITE)':^70}")
print("="*70)

# Model 1: 2nd-Order Polynomial + Precessing PA (10 params)
def nll_m1(p):
    lp = log_prior_universal(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9])
    if not np.isfinite(lp): return 1e12
    return -(lp + calc_log_likelihood(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]))

res_m1 = minimize(nll_m1, [60440.0, 0.0065, 0.0, -0.0083, 0.0, -40.2, 0.0, -42.7, 0.0, 4.0], method='Nelder-Mead', options={'maxiter': 10000})
ll_m1 = calc_log_likelihood(*res_m1.x)
k_m1 = 10
bic_m1 = k_m1 * np.log(n_data) - 2 * ll_m1
aic_m1 = 2 * k_m1 - 2 * ll_m1

# Model 2: 1st-Order Linear Ballistic + Precessing PA (8 params: a_n = a_s = 0)
def nll_m2(p):
    lp = log_prior_universal(p[0], p[1], 0.0, p[2], 0.0, p[3], p[4], p[5], p[6], p[7])
    if not np.isfinite(lp): return 1e12
    return -(lp + calc_log_likelihood(p[0], p[1], 0.0, p[2], 0.0, p[3], p[4], p[5], p[6], p[7]))

res_m2 = minimize(nll_m2, [60440.0, 0.0065, -0.0083, -40.2, 0.0, -42.7, 0.0, 4.0], method='Nelder-Mead', options={'maxiter': 10000})
ll_m2 = calc_log_likelihood(res_m2.x[0], res_m2.x[1], 0.0, res_m2.x[2], 0.0, res_m2.x[3], res_m2.x[4], res_m2.x[5], res_m2.x[6], res_m2.x[7])
k_m2 = 8
bic_m2 = k_m2 * np.log(n_data) - 2 * ll_m2
aic_m2 = 2 * k_m2 - 2 * ll_m2

# Model 3: 2nd-Order Polynomial + Fixed PA (8 params: omega_n = omega_s = 0)
def nll_m3(p):
    lp = log_prior_universal(p[0], p[1], p[2], p[3], p[4], p[5], 0.0, p[6], 0.0, p[7])
    if not np.isfinite(lp): return 1e12
    return -(lp + calc_log_likelihood(p[0], p[1], p[2], p[3], p[4], p[5], 0.0, p[6], 0.0, p[7]))

res_m3 = minimize(nll_m3, [60440.0, 0.0065, 0.0, -0.0083, 0.0, -40.2, -42.7, 4.0], method='Nelder-Mead', options={'maxiter': 10000})
ll_m3 = calc_log_likelihood(res_m3.x[0], res_m3.x[1], res_m3.x[2], res_m3.x[3], res_m3.x[4], res_m3.x[5], 0.0, res_m3.x[6], 0.0, res_m3.x[7])
k_m3 = 8
bic_m3 = k_m3 * np.log(n_data) - 2 * ll_m3
aic_m3 = 2 * k_m3 - 2 * ll_m3

# Model 4: 1st-Order Linear Ballistic + Fixed PA (6 params: a_n = a_s = omega_n = omega_s = 0)
def nll_m4(p):
    lp = log_prior_universal(p[0], p[1], 0.0, p[2], 0.0, p[3], 0.0, p[4], 0.0, p[5])
    if not np.isfinite(lp): return 1e12
    return -(lp + calc_log_likelihood(p[0], p[1], 0.0, p[2], 0.0, p[3], 0.0, p[4], 0.0, p[5]))

res_m4 = minimize(nll_m4, [60440.0, 0.0065, -0.0083, -40.2, -42.7, 4.0], method='Nelder-Mead', options={'maxiter': 10000})
ll_m4 = calc_log_likelihood(res_m4.x[0], res_m4.x[1], 0.0, res_m4.x[2], 0.0, res_m4.x[3], 0.0, res_m4.x[4], 0.0, res_m4.x[5])
k_m4 = 6
bic_m4 = k_m4 * np.log(n_data) - 2 * ll_m4
aic_m4 = 2 * k_m4 - 2 * ll_m4

# Candidate models summary
models = [
    {"id": 1, "name": "2nd-Order Polynomial + Precessing PA", "k": k_m1, "ll": ll_m1, "bic": bic_m1, "aic": aic_m1, "res": res_m1},
    {"id": 2, "name": "1st-Order Ballistic  + Precessing PA", "k": k_m2, "ll": ll_m2, "bic": bic_m2, "aic": aic_m2, "res": res_m2},
    {"id": 3, "name": "2nd-Order Polynomial + Fixed PA",      "k": k_m3, "ll": ll_m3, "bic": bic_m3, "aic": aic_m3, "res": res_m3},
    {"id": 4, "name": "1st-Order Ballistic  + Fixed PA",      "k": k_m4, "ll": ll_m4, "bic": bic_m4, "aic": aic_m4, "res": res_m4},
]

# Adopt the conservative 1st-Order Ballistic + Fixed PA model (Model 4) as primary baseline
adopted_model_id = 4
best_model = [m for m in models if m["id"] == adopted_model_id][0]
stat_best = min(models, key=lambda m: m["bic"])

print(f"{'Model Description':<40} {'k':<4} {'ln(L)':<10} {'BIC':<10} {'Delta BIC':<10}")
print("-" * 76)
for m in models:
    delta_bic = m["bic"] - stat_best["bic"]
    print(f"{m['name']:<40} {m['k']:<4} {m['ll']:<10.2f} {m['bic']:<10.2f} {delta_bic:<10.2f}")
print("-" * 76)
print(f"--> ADOPTED BASELINE MODEL: Model {best_model['id']} ({best_model['name']}) [Conservative]")
print(f"    (Statistically best candidate: Model {stat_best['id']} with Delta BIC = {best_model['bic'] - stat_best['bic']:.2f})")
print("="*70 + "\n")

# -----------------------------------------------------------------------------
# 4. MCMC POSTERIOR SAMPLING FOR THE ADOPTED MODEL
# -----------------------------------------------------------------------------
print(f"Sampling posteriors for Model {best_model['id']} via MCMC...")

if best_model["id"] == 4:
    ndim = 6
    param_labels = [r"$T_{\rm ej}$", r"$v_{0,n}$", r"$v_{0,s}$", r"${\rm PA}_n$", r"${\rm PA}_s$", r"$\theta_{\rm cone}$"]
    
    def log_prior(params):
        return log_prior_universal(params[0], params[1], 0.0, params[2], 0.0, params[3], 0.0, params[4], 0.0, params[5])

    def log_prob(params):
        lp = log_prior(params)
        if not np.isfinite(lp): return -np.inf
        return lp + calc_log_likelihood(params[0], params[1], 0.0, params[2], 0.0, params[3], 0.0, params[4], 0.0, params[5])

    opt_init = res_m4.x

elif best_model["id"] == 2:
    ndim = 8
    param_labels = [r"$T_{\rm ej}$", r"$v_{0,n}$", r"$v_{0,s}$", r"${\rm PA}_{0,n}$", r"$\omega_n$", r"${\rm PA}_{0,s}$", r"$\omega_s$", r"$\theta_{\rm cone}$"]
    
    def log_prior(params):
        return log_prior_universal(params[0], params[1], 0.0, params[2], 0.0, params[3], params[4], params[5], params[6], params[7])

    def log_prob(params):
        lp = log_prior(params)
        if not np.isfinite(lp): return -np.inf
        return lp + calc_log_likelihood(params[0], params[1], 0.0, params[2], 0.0, params[3], params[4], params[5], params[6], params[7])

    opt_init = res_m2.x

elif best_model["id"] == 3:
    ndim = 8
    param_labels = [r"$T_{\rm ej}$", r"$v_{0,n}$", r"$a_n$", r"$v_{0,s}$", r"$a_s$", r"${\rm PA}_n$", r"${\rm PA}_s$", r"$\theta_{\rm cone}$"]
    
    def log_prior(params):
        return log_prior_universal(params[0], params[1], params[2], params[3], params[4], params[5], 0.0, params[6], 0.0, params[7])

    def log_prob(params):
        lp = log_prior(params)
        if not np.isfinite(lp): return -np.inf
        return lp + calc_log_likelihood(params[0], params[1], params[2], params[3], params[4], params[5], 0.0, params[6], 0.0, params[7])

    opt_init = res_m3.x

else:
    ndim = 10
    param_labels = [r"$T_{\rm ej}$", r"$v_{0,n}$", r"$a_n$", r"$v_{0,s}$", r"$a_s$", r"${\rm PA}_{0,n}$", r"$\omega_n$", r"${\rm PA}_{0,s}$", r"$\omega_s$", r"$\theta_{\rm cone}$"]
    
    def log_prior(params):
        return log_prior_universal(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9])

    def log_prob(params):
        lp = log_prior(params)
        if not np.isfinite(lp): return -np.inf
        return lp + calc_log_likelihood(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9])

    opt_init = res_m1.x


nwalkers = 48
n_steps = 15000
scales = np.abs(opt_init) * 0.1 + 1e-3  # Increased to 10% spread to ensure robust exploration
scales[0] = 2.0  # +/- 2 days initial spread for T_ej

# Safe initialization guaranteeing all walkers start in valid prior domain
pos_list = []
for _ in range(nwalkers):
    valid = False
    for _ in range(1000):
        trial = opt_init + scales * np.random.randn(ndim)
        if np.isfinite(log_prior(trial)):
            pos_list.append(trial)
            valid = True
            break
    if not valid:
        pos_list.append(opt_init)
pos = np.array(pos_list)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
sampler.run_mcmc(pos, n_steps, progress=False)

# Auto-calculate burn-in and thinning safely
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tau_ac = sampler.get_autocorr_time(quiet=True)
    if np.any(np.isnan(tau_ac)) or np.any(np.isinf(tau_ac)):
        raise ValueError("Invalid autocorrelation time")
    burnin = int(2 * np.max(tau_ac))
    thin = max(1, int(0.5 * np.min(tau_ac)))
    print(f"Auto-calculated burn-in: {burnin}, thin: {thin}")
except Exception:
    burnin = int(n_steps * 0.2)
    thin = 10
    print(f"Using robust default burn-in: {burnin}, thin: {thin}")

flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

# Helper to map any model's parameter vector to a standard 10-element representation
def unpack_params(sample, model_id):
    if model_id == 4:
        return sample[0], sample[1], 0.0, sample[2], 0.0, sample[3], 0.0, sample[4], 0.0, sample[5]
    elif model_id == 2:
        return sample[0], sample[1], 0.0, sample[2], 0.0, sample[3], sample[4], sample[5], sample[6], sample[7]
    elif model_id == 3:
        return sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], 0.0, sample[6], 0.0, sample[7]
    else:
        return sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], sample[7], sample[8], sample[9]

# -----------------------------------------------------------------------------
# 5. DIAGNOSTIC TRACE AND CORNER PLOTS
# -----------------------------------------------------------------------------
print("Generating MCMC trace plot...")
fig_trace, axes = plt.subplots(ndim, figsize=(10, 2 * ndim), sharex=True)
samples = sampler.get_chain()
for i in range(ndim):
    ax = axes[i] if ndim > 1 else axes
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(param_labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
if ndim > 1:
    axes[-1].set_xlabel("Step Number")
fig_trace.tight_layout()
fig_trace.savefig('/media/kyle/kyle_phd/GRS1915/figures/mcmc_trace.png', dpi=300)
plt.close(fig_trace)
print("Saved trace plot to /media/kyle/kyle_phd/GRS1915/mcmc_trace.png")

print("Generating corner plot...")
fig_corner = corner.corner(
    flat_samples, 
    labels=param_labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 10},
    title_fmt=".2e"
)
fig_corner.savefig('/media/kyle/kyle_phd/GRS1915/figures/corner_plot.png', dpi=300)
plt.close(fig_corner)
print("Saved corner plot to /media/kyle/kyle_phd/GRS1915/corner_plot.png")

# -----------------------------------------------------------------------------
# 6. EXTRACT PARAMETERS AND DERIVE RELATIVISTIC KINEMATICS
# -----------------------------------------------------------------------------
p_med = compute_hdi_2d(flat_samples)

t_ej_chain = flat_samples[:, 0]
v0_n_chain = flat_samples[:, 1]
v0_s_chain = flat_samples[:, 2] if best_model["id"] in [2, 4] else flat_samples[:, 3]

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

# Summarize the HDI
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

# Acceleration percentiles if present
if best_model["id"] in [1, 3]:
    a_n_chain = flat_samples[:, 2]
    a_s_chain = flat_samples[:, 4]
    p_a_n = compute_hdi(a_n_chain)
    p_a_s = compute_hdi(a_s_chain)

# Position angle parameter extraction
if best_model["id"] == 4:
    p_pa_n = compute_hdi(flat_samples[:, 3])
    p_pa_s = compute_hdi(flat_samples[:, 4])
elif best_model["id"] == 2:
    p_pa_n = compute_hdi(flat_samples[:, 3])
    p_omega_n = compute_hdi(flat_samples[:, 4])
    p_pa_s = compute_hdi(flat_samples[:, 5])
    p_omega_s = compute_hdi(flat_samples[:, 6])
elif best_model["id"] == 3:
    p_pa_n = compute_hdi(flat_samples[:, 5])
    p_pa_s = compute_hdi(flat_samples[:, 6])
else:
    p_pa_n = compute_hdi(flat_samples[:, 5])
    p_omega_n = compute_hdi(flat_samples[:, 6])
    p_pa_s = compute_hdi(flat_samples[:, 7])
    p_omega_s = compute_hdi(flat_samples[:, 8])


# Direct data-derived position angles
pa_n_ind = np.degrees(np.arctan2(n_da, n_dd))
pa_err_n_ind = np.degrees(np.sqrt((n_dd * n_da_err)**2 + (n_da * n_dd_err)**2) / (n_da**2 + n_dd**2))

pa_s_ind = np.degrees(np.arctan2(-s_da, -s_dd))
pa_err_s_ind = np.degrees(np.sqrt(((-s_dd) * s_da_err)**2 + ((-s_da) * s_dd_err)**2) / (s_da**2 + s_dd**2))

# -----------------------------------------------------------------------------
# 7. PRINT SCIENTIFIC & ASTROPHYSICAL SUMMARY REPORT
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print(f"{'GRS 1915+105 RELATIVISTIC KINEMATIC REPORT':^70}")
print("="*70)
print(f"Statistically Preferred Model : Model {best_model['id']} ({best_model['name']})")
print(f"Data Points Used (N_data)     : {n_data}")
print(f"Optimized BIC                 : {best_model['bic']:.2f}")

print("\n[1] EJECTION PARAMETERS")
print("-" * 70)
print(f"Extrapolated Launch Time (T_ej): MJD {p_Tej[1]:.2f} + {p_Tej[2]-p_Tej[1]:.2f} - {p_Tej[1]-p_Tej[0]:.2f}")

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

# -----------------------------------------------------------------------------
# 8. GENERATE PUBLICATION-QUALITY PLOTS
# -----------------------------------------------------------------------------
print("Generating publication plots for the preferred model...")

# --- Plot 1: Parallel and Perpendicular Separation vs Time ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.axhline(0, color='grey', linestyle=':', linewidth=1)

t_start = np.min(t_ej_chain) - 5
t_end = max(np.max(n_mjd), np.max(s_mjd)) + 10
t_smooth = np.linspace(t_start, t_end, 500)

models_n = []
models_s = []

# Draw sample paths from MCMC posterior to construct 68% HDI
inds = np.random.randint(len(flat_samples), size=5000)
for i in inds:
    t_ej_s, v0_n_s, a_n_s, v0_s_s, a_s_s, pa_0_n_s, omega_n_s, pa_0_s_s, omega_s_s, _ = unpack_params(flat_samples[i], best_model["id"])
    dt = t_smooth - t_ej_s
    
    m_n = v0_n_s * dt + 0.5 * a_n_s * dt**2
    m_n[dt < 0] = 0.0
    models_n.append(m_n)
    
    m_s = v0_s_s * dt + 0.5 * a_s_s * dt**2
    m_s[dt < 0] = 0.0
    models_s.append(m_s)

q_n = np.percentile(np.array(models_n), [16, 50, 84], axis=0)
q_s = np.percentile(np.array(models_s), [16, 50, 84], axis=0)

# Median lines
t_plot = t_smooth[t_smooth >= p_Tej[1]]
dt_plot = t_plot - p_Tej[1]
med_t_ej, med_v0_n, med_a_n, med_v0_s, med_a_s, med_pa_n, med_w_n, med_pa_s, med_w_s, _ = unpack_params(p_med[1], best_model["id"])

med_n = med_v0_n * dt_plot + 0.5 * med_a_n * dt_plot**2
med_s = med_v0_s * dt_plot + 0.5 * med_a_s * dt_plot**2

ax.plot(t_plot - 60000, med_n, color='orange', linestyle='-', linewidth=2, zorder=1.5, label='Model Posterior Median')
ax.fill_between(t_smooth - 60000, q_n[0], q_n[2], color='navajowhite', alpha=0.5, zorder=1.4, label='Model Posterior 68% HDI')

ax.plot(t_plot - 60000, med_s, color='orange', linestyle='-', linewidth=2, zorder=1.5)
ax.fill_between(t_smooth - 60000, q_s[0], q_s[2], color='navajowhite', alpha=0.5, zorder=1.4)

# Rotate data points using median fitted position angles
th_n_opt = np.radians(med_pa_n + med_w_n * (n_mjd - med_t_ej))
sin_n_opt, cos_n_opt = np.sin(th_n_opt), np.cos(th_n_opt)
n_par_opt = n_da * sin_n_opt + n_dd * cos_n_opt
n_perp_opt = n_da * cos_n_opt - n_dd * sin_n_opt
n_par_err_opt = np.sqrt((n_da_err * sin_n_opt)**2 + (n_dd_err * cos_n_opt)**2)
n_perp_err_opt = np.sqrt((n_da_err * cos_n_opt)**2 + (n_dd_err * sin_n_opt)**2)

th_s_opt = np.radians(med_pa_s + med_w_s * (s_mjd - med_t_ej))
sin_s_opt, cos_s_opt = np.sin(th_s_opt), np.cos(th_s_opt)
s_par_opt = s_da * sin_s_opt + s_dd * cos_s_opt
s_perp_opt = s_da * cos_s_opt - s_dd * sin_s_opt
s_par_err_opt = np.sqrt((s_da_err * sin_s_opt)**2 + (s_dd_err * cos_s_opt)**2)
s_perp_err_opt = np.sqrt((s_da_err * cos_s_opt)**2 + (s_dd_err * sin_s_opt)**2)

# Parallel separation data points
ax.errorbar(n_mjd - 60000, n_par_opt, yerr=n_par_err_opt, fmt='^', color='dodgerblue', markeredgecolor='black', markersize=12, zorder=2, ecolor='black', capsize=0, elinewidth=1.5, label=r'$\Delta\theta_\parallel^N$')
ax.errorbar(s_mjd - 60000, s_par_opt, yerr=s_par_err_opt, fmt='v', color='red', markeredgecolor='black', markersize=12, zorder=2, ecolor='black', capsize=0, elinewidth=1.5, label=r'$\Delta\theta_\parallel^S$')

# Perpendicular separation data points
# ax.errorbar(n_mjd - 60000, n_perp_opt, yerr=n_perp_err_opt, fmt='^-', color='lightgrey', markeredgecolor='lightgrey', markersize=12, zorder=1, ecolor='lightgrey', capsize=0, linewidth=1.5, label=r'$\Delta\theta_\perp^N$')
# ax.errorbar(s_mjd - 60000, s_perp_opt, yerr=s_perp_err_opt, fmt='v-', color='lightgrey', markeredgecolor='lightgrey', markersize=12, zorder=1, ecolor='lightgrey', capsize=0, linewidth=1.5, label=r'$\Delta\theta_\perp^S$')

ax.set_ylabel(r'Separation ($^{\prime\prime}$)', fontsize=24)
ax.set_xlabel('Time (MJD - 60000)', fontsize=24)
ax.set_xlim(left=200)

# Extract handles and labels to reorder them for the legend
handles, labels = ax.get_legend_handles_labels()
# Reorder to match reference: Par N, Par S, Median, HDI
order = [2, 3, 0, 1]
ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower left', ncol=2, fontsize=17)

ax.minorticks_on()
ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, labelsize=18, length=8, width=1.5)
ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True, length=4, width=1)

# Add kinematic results text box
avg_pa_val = (p_pa_n[1] + p_pa_s[1]) / 2.0 + 180.0
avg_pa_err = ((p_pa_n[2] - p_pa_n[0]) + (p_pa_s[2] - p_pa_s[0])) / 4.0

beta_err = (p_beta[2] - p_beta[0]) / 2.0
theta_err = (p_theta[2] - p_theta[0]) / 2.0

textstr = '\n'.join((
    r'$\beta = %.3f \pm %.3f\,c$' % (p_beta[1], beta_err),
    r'$\theta = %.1f^\circ \pm %.1f^\circ$' % (p_theta[1], theta_err),
    r'$\mathrm{PA} = %.1f^\circ \pm %.1f^\circ$' % (avg_pa_val, avg_pa_err)
))
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='darkgrey')
ax.text(0.02, 0.96, textstr, transform=ax.transAxes, fontsize=20,
        verticalalignment='top', bbox=props, zorder=3)

plt.savefig('/media/kyle/kyle_phd/GRS1915/figures/parallel_perp_fit_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('/media/kyle/kyle_phd/GRS1915/figures/parallel_perp_fit_plot_kinematics.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved displacement plot to /media/kyle/kyle_phd/GRS1915/parallel_perp_fit_plot.png and parallel_perp_fit_plot_kinematics.png")

# --- Plot 2: Position Angle vs Time ---
fig2, ax2 = plt.subplots(figsize=(10, 8), dpi=300)

models_pa_n = []
models_pa_s = []
for i in inds:
    t_ej_s, _, _, _, _, pa_0_n_s, omega_n_s, pa_0_s_s, omega_s_s, _ = unpack_params(flat_samples[i], best_model["id"])
    dt = t_smooth - t_ej_s
    models_pa_n.append(pa_0_n_s + omega_n_s * dt)
    models_pa_s.append(pa_0_s_s + omega_s_s * dt)

q_pa_n = np.percentile(np.array(models_pa_n), [16, 50, 84], axis=0)
q_pa_s = np.percentile(np.array(models_pa_s), [16, 50, 84], axis=0)

ax2.plot(t_smooth - 60000, q_pa_n[1], color='dodgerblue', linestyle='--', linewidth=2, label='North Model (Median)')
ax2.fill_between(t_smooth - 60000, q_pa_n[0], q_pa_n[2], color='dodgerblue', alpha=0.25)

ax2.plot(t_smooth - 60000, q_pa_s[1], color='darkorange', linestyle='--', linewidth=2, label='South Model (Median)')
ax2.fill_between(t_smooth - 60000, q_pa_s[0], q_pa_s[2], color='darkorange', alpha=0.25)

ax2.errorbar(n_mjd - 60000, pa_n_ind, yerr=pa_err_n_ind, fmt='^', color='dodgerblue', markeredgecolor='black', markersize=8, zorder=2, ecolor='gray', capsize=3, label='North PA Data')
ax2.errorbar(s_mjd - 60000, pa_s_ind, yerr=pa_err_s_ind, fmt='v', color='darkorange', markeredgecolor='black', markersize=8, zorder=2, ecolor='gray', capsize=3, label='South PA Data')

ax2.set_ylabel(r'Northern Lobe Position Angle ($^{\circ}$)', fontsize=24, color='dodgerblue')
ax2.set_xlabel('Time (MJD - 60000)', fontsize=24)
ax2.legend(loc='best', fontsize=12)
ax2.set_title('Jet Axis Position Angle Evolution', fontsize=24)
ax2.minorticks_on()
ax2.tick_params(axis='both', which='major', direction='in', top=True, labelsize=18, length=8, width=1.5)
ax2.tick_params(axis='both', which='minor', direction='in', top=True, length=4, width=1)
ax2.tick_params(axis='y', labelcolor='dodgerblue')

ax2_twin = ax2.twinx()
y1, y2 = ax2.get_ylim()
ax2_twin.set_ylim(y1 + 180, y2 + 180)
ax2_twin.set_ylabel(r'Southern Lobe Position Angle ($^{\circ}$)', fontsize=24, color='darkorange')
ax2_twin.minorticks_on()
ax2_twin.tick_params(axis='y', which='major', direction='in', right=True, labelsize=18, length=8, width=1.5)
ax2_twin.tick_params(axis='y', which='minor', direction='in', right=True, length=4, width=1)
ax2_twin.tick_params(axis='y', labelcolor='darkorange')

plt.tight_layout()
plt.savefig('/media/kyle/kyle_phd/GRS1915/figures/pa_evolution_plot.png', dpi=300)
plt.close(fig2)
print("Saved PA evolution plot to /media/kyle/kyle_phd/GRS1915/pa_evolution_plot.png")

# --- Export Clean PA Table ---
all_mjds = sorted(list(set(np.round(np.concatenate([n_mjd, s_mjd]), 4))))

with open('/media/kyle/kyle_phd/GRS1915/data/pa_evolution_table.txt', 'w') as f:
    f.write("MJD\tPA_North_deg\tPA_North_err\tPA_South_deg\tPA_South_err\tAverage_Axis_PA_deg\n")
    for m in all_mjds:
        idx_n = np.where(np.abs(n_mjd - m) < 1e-4)[0]
        if len(idx_n) > 0:
            pa_n = pa_n_ind[idx_n[0]]
            err_n = pa_err_n_ind[idx_n[0]]
        else:
            pa_n, err_n = np.nan, np.nan
            
        idx_s = np.where(np.abs(s_mjd - m) < 1e-4)[0]
        if len(idx_s) > 0:
            pa_s = pa_s_ind[idx_s[0]] + 180.0
            pa_s_shifted = pa_s_ind[idx_s[0]]
            err_s = pa_err_s_ind[idx_s[0]]
        else:
            pa_s, pa_s_shifted, err_s = np.nan, np.nan, np.nan
            
        if not np.isnan(pa_n) and not np.isnan(pa_s_shifted):
            avg_pa = (pa_n + pa_s_shifted) / 2.0
        elif not np.isnan(pa_n):
            avg_pa = pa_n
        elif not np.isnan(pa_s_shifted):
            avg_pa = pa_s_shifted
        else:
            avg_pa = np.nan
            
        f.write(f"{m:.4f}\t{pa_n:.2f}\t{err_n:.2f}\t{pa_s:.2f}\t{err_s:.2f}\t{avg_pa:.2f}\n")

print("Saved PA table to /media/kyle/kyle_phd/GRS1915/pa_evolution_table.txt\n")

# --- Export Best-Fit Parameters ---
# Extract median parameters for the best model
med_t_ej, med_v0_n, med_a_n, med_v0_s, med_a_s, med_pa_n, med_w_n, med_pa_s, med_w_s, med_theta_cone = unpack_params(p_med[1], best_model["id"])

with open('/media/kyle/kyle_phd/GRS1915/data/best_fit_params.txt', 'w') as f:
    f.write(f"theta_cone_deg\t{med_theta_cone}\n")
    f.write(f"pa_n_deg\t{med_pa_n}\n")
    f.write(f"pa_s_deg\t{med_pa_s}\n")
print("Saved best-fit parameters to /media/kyle/kyle_phd/GRS1915/best_fit_params.txt\n")

# -----------------------------------------------------------------------------
# 9. KINEMATIC POSTERIOR CORNER PLOT (D, theta, beta)
# -----------------------------------------------------------------------------
print("Generating kinematic corner plot (Distance, Theta, Beta)...")
kinematic_samples = np.vstack((D_kpc_chain, theta_deg_chain, beta_chain)).T
kinematic_labels = [r"Distance (kpc)", r"$\theta_{\rm VA}$ ($^\circ$)", r"$\beta$"]

fig_kin_corner = corner.corner(
    kinematic_samples,
    labels=kinematic_labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 10},
    title_fmt=".2f",
    color="mediumseagreen" # Matches the green theme from the paper
)

fig_kin_corner.savefig('/media/kyle/kyle_phd/GRS1915/figures/corner_plot_kinematics.png', dpi=300)
plt.close(fig_kin_corner)
print("Saved kinematic corner plot to /media/kyle/kyle_phd/GRS1915/corner_plot_kinematics.png\n")
