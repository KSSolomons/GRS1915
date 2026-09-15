import os
import glob
import re
import warnings
import numpy as np
from scipy.optimize import minimize
import emcee

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

def run_kinematics_test(ref_mode="per_epoch", est_ra_deg=None, est_dec_deg=None, est_err_zero=True):
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
        if ref_mode == "per_epoch":
            ref_ra = epoch['core_ra'] if epoch['core_ra'] is not None else mean_core_ra
            ref_dec = epoch['core_dec'] if epoch['core_dec'] is not None else mean_core_dec
            ref_ra_err = epoch['core_ra_err'] if epoch['core_ra_err'] is not None else mean_core_ra_err
            ref_dec_err = epoch['core_dec_err'] if epoch['core_dec_err'] is not None else mean_core_dec_err
        elif ref_mode == "established":
            ref_ra = est_ra_deg
            ref_dec = est_dec_deg
            if est_err_zero:
                ref_ra_err = 0.0
                ref_dec_err = 0.0
            else:
                ref_ra_err = mean_core_ra_err
                ref_dec_err = mean_core_dec_err
        elif ref_mode == "mean_fitted":
            ref_ra = mean_core_ra
            ref_dec = mean_core_dec
            ref_ra_err = mean_core_ra_err
            ref_dec_err = mean_core_dec_err
        
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

    # Model 4: 1st-Order Ballistic + Fixed PA (6 params)
    def nll_m4(p):
        lp = log_prior_universal(p[0], p[1], 0.0, p[2], 0.0, p[3], 0.0, p[4], 0.0, p[5])
        if not np.isfinite(lp): return 1e12
        return -(lp + calc_log_likelihood(p[0], p[1], 0.0, p[2], 0.0, p[3], 0.0, p[4], 0.0, p[5]))

    res_m4 = minimize(nll_m4, [60440.0, 0.0065, -0.0083, -40.2, -42.7, 4.0], method='Nelder-Mead', options={'maxiter': 10000})
    ll_m4 = calc_log_likelihood(res_m4.x[0], res_m4.x[1], 0.0, res_m4.x[2], 0.0, res_m4.x[3], 0.0, res_m4.x[4], 0.0, res_m4.x[5])
    bic_m4 = 6 * np.log(n_data) - 2 * ll_m4

    # Run MCMC
    ndim = 6
    def log_prior(params):
        return log_prior_universal(params[0], params[1], 0.0, params[2], 0.0, params[3], 0.0, params[4], 0.0, params[5])
    def log_prob(params):
        lp = log_prior(params)
        if not np.isfinite(lp): return -np.inf
        return lp + calc_log_likelihood(params[0], params[1], 0.0, params[2], 0.0, params[3], 0.0, params[4], 0.0, params[5])

    opt_init = res_m4.x
    nwalkers = 48
    n_steps = 10000
    scales = np.abs(opt_init) * 0.1 + 1e-3
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
        if not valid:
            pos_list.append(opt_init)
    pos = np.array(pos_list)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(pos, n_steps, progress=False)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tau_ac = sampler.get_autocorr_time(quiet=True)
        burnin = int(2 * np.max(tau_ac))
        thin = max(1, int(0.5 * np.min(tau_ac)))
    except Exception:
        burnin = int(n_steps * 0.2)
        thin = 10

    flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

    t_ej_chain = flat_samples[:, 0]
    v0_n_chain = flat_samples[:, 1]
    v0_s_chain = flat_samples[:, 2]

    mu_app_chain = np.abs(v0_s_chain)
    mu_rec_chain = np.abs(v0_n_chain)

    beta_cos_theta_chain = (mu_app_chain - mu_rec_chain) / (mu_app_chain + mu_rec_chain)
    np.random.seed(42)
    D_kpc_chain = np.random.normal(9.4, 1.0, size=len(mu_app_chain))
    D_m_chain = D_kpc_chain * 3.086e19
    c = 2.99792458e8

    mu_app_rad_s = mu_app_chain * (np.pi / (180.0 * 3600.0)) / 86400.0
    beta_sin_theta_chain = mu_app_rad_s * (1.0 - beta_cos_theta_chain) * D_m_chain / c

    beta_chain = np.sqrt(beta_cos_theta_chain**2 + beta_sin_theta_chain**2)
    beta_clamped = np.clip(beta_chain, 0.0, 0.999999)
    gamma_chain = 1.0 / np.sqrt(1.0 - beta_clamped**2)
    theta_deg_chain = np.degrees(np.arctan2(beta_sin_theta_chain, beta_cos_theta_chain))

    p_Tej = compute_hdi(t_ej_chain)
    p_v0_n = compute_hdi(v0_n_chain)
    p_v0_s = compute_hdi(v0_s_chain)
    p_beta = compute_hdi(beta_chain)
    p_gamma = compute_hdi(gamma_chain)
    p_theta = compute_hdi(theta_deg_chain)
    p_pa_n = compute_hdi(flat_samples[:, 3])
    p_pa_s = compute_hdi(flat_samples[:, 4])
    p_cone = compute_hdi(flat_samples[:, 5])

    res_dict = {
        "ref_mode": ref_mode,
        "ref_ra": ref_ra,
        "ref_dec": ref_dec,
        "bic": bic_m4,
        "ll": ll_m4,
        "T_ej": p_Tej,
        "mu_r": p_v0_n,
        "mu_a": p_v0_s,
        "beta": p_beta,
        "gamma": p_gamma,
        "theta": p_theta,
        "pa_n": p_pa_n,
        "pa_s": p_pa_s,
        "theta_cone": p_cone
    }
    return res_dict

if __name__ == "__main__":
    # Established coords from MeerKAT file header: 19h15m11.558s +10d56m44.905s
    h, m, s = 19, 15, 11.558
    d, dm, ds = 10, 56, 44.905
    est_ra = (h + m/60.0 + s/3600.0) * 15.0
    est_dec = d + dm/60.0 + ds/3600.0

    print("Running Test 1: Original (Per-Epoch Core Reference)...")
    r_orig = run_kinematics_test(ref_mode="per_epoch")

    print("\nRunning Test 2: Established GRS 1915+105 Core (19:15:11.558, +10:56:44.905)...")
    r_est = run_kinematics_test(ref_mode="established", est_ra_deg=est_ra, est_dec_deg=est_dec)

    print("\nRunning Test 3: Mean Fitted Core Position across all epochs...")
    r_mean = run_kinematics_test(ref_mode="mean_fitted")

    print("\n" + "="*80)
    print(f"{'COMPARATIVE SUMMARY TABLE':^80}")
    print("="*80)
    fmt = "{:<25} | {:<18} | {:<18} | {:<18}"
    print(fmt.format("Parameter", "1. Per-Epoch Core", "2. Established Cat.", "3. Mean Fitted Core"))
    print("-" * 80)
    print(fmt.format("Reference RA", "Individual epochs", f"{est_ra:.6f}°", f"{r_mean['ref_ra']:.6f}°"))
    print(fmt.format("Reference Dec", "Individual epochs", f"{est_dec:.6f}°", f"{r_mean['ref_dec']:.6f}°"))
    print(fmt.format("BIC", f"{r_orig['bic']:.2f}", f"{r_est['bic']:.2f}", f"{r_mean['bic']:.2f}"))
    print(fmt.format("log(Likelihood)", f"{r_orig['ll']:.2f}", f"{r_est['ll']:.2f}", f"{r_mean['ll']:.2f}"))
    
    def format_val(p, scale=1.0, unit=""):
        return f"{p[1]*scale:.3f} [{p[0]*scale:.3f}, {p[2]*scale:.3f}]"

    print(fmt.format("T_ej (MJD)", format_val(r_orig['T_ej']), format_val(r_est['T_ej']), format_val(r_mean['T_ej'])))
    print(fmt.format("mu_r (mas/day)", format_val(r_orig['mu_r'], 1e3), format_val(r_est['mu_r'], 1e3), format_val(r_mean['mu_r'], 1e3)))
    print(fmt.format("mu_a (mas/day)", format_val(r_orig['mu_a'], -1e3), format_val(r_est['mu_a'], -1e3), format_val(r_mean['mu_a'], -1e3)))
    print(fmt.format("beta (v/c)", format_val(r_orig['beta']), format_val(r_est['beta']), format_val(r_mean['beta'])))
    print(fmt.format("Lorentz factor Gamma", format_val(r_orig['gamma']), format_val(r_est['gamma']), format_val(r_mean['gamma'])))
    print(fmt.format("Inclination theta (°)", format_val(r_orig['theta']), format_val(r_est['theta']), format_val(r_mean['theta'])))
    print(fmt.format("PA_n (°)", format_val(r_orig['pa_n']), format_val(r_est['pa_n']), format_val(r_mean['pa_n'])))
    print(fmt.format("PA_s (°)", format_val(r_orig['pa_s']), format_val(r_est['pa_s']), format_val(r_mean['pa_s'])))
    print(fmt.format("theta_cone (°)", format_val(r_orig['theta_cone']), format_val(r_est['theta_cone']), format_val(r_mean['theta_cone'])))
    print("="*80)
