with open('/media/kyle/kyle_phd/GRS1915/plot_sep_vs_time_poly.py', 'r') as f:
    code = f.read()

# Just restore the old code by replacing the new blocks with the old blocks!
# 1. Update calc_log_likelihood and log_prior_universal
new_calc = """def calc_log_likelihood(t_ej, v0_n, a_n, v0_s, a_s, pa_0_n, omega_n, pa_0_s, omega_s, scatter_val, scatter_type='constant'):
    # Northern Lobe
    theta_n = np.radians(pa_0_n + omega_n * (n_mjd - t_ej))
    sin_n, cos_n = np.sin(theta_n), np.cos(theta_n)
    par_n = n_da * sin_n + n_dd * cos_n
    perp_n = n_da * cos_n - n_dd * sin_n
    
    dt_n = n_mjd - t_ej
    model_par_n = v0_n * dt_n + 0.5 * a_n * dt_n**2
    
    if scatter_type == 'constant':
        s2_n = (10**scatter_val)**2
    else:
        s2_n = (par_n * np.tan(np.radians(scatter_val)))**2
        
    var_par_n = (n_da_err * sin_n)**2 + (n_dd_err * cos_n)**2 + s2_n
    var_perp_n = (n_da_err * cos_n)**2 + (n_dd_err * sin_n)**2 + s2_n
    
    ll_n = -0.5 * np.sum((par_n - model_par_n)**2 / var_par_n + np.log(2 * np.pi * var_par_n))
    ll_n += -0.5 * np.sum(perp_n**2 / var_perp_n + np.log(2 * np.pi * var_perp_n))
    
    # Southern Lobe
    theta_s = np.radians(pa_0_s + omega_s * (s_mjd - t_ej))
    sin_s, cos_s = np.sin(theta_s), np.cos(theta_s)
    par_s = s_da * sin_s + s_dd * cos_s
    perp_s = s_da * cos_s - s_dd * sin_s
    
    dt_s = s_mjd - t_ej
    model_par_s = v0_s * dt_s + 0.5 * a_s * dt_s**2
    
    if scatter_type == 'constant':
        s2_s = (10**scatter_val)**2
    else:
        s2_s = (par_s * np.tan(np.radians(scatter_val)))**2
        
    var_par_s = (s_da_err * sin_s)**2 + (s_dd_err * cos_s)**2 + s2_s
    var_perp_s = (s_da_err * cos_s)**2 + (s_dd_err * sin_s)**2 + s2_s
    
    ll_s = -0.5 * np.sum((par_s - model_par_s)**2 / var_par_s + np.log(2 * np.pi * var_par_s))
    ll_s += -0.5 * np.sum(perp_s**2 / var_perp_s + np.log(2 * np.pi * var_perp_s))
    
    return ll_n + ll_s

def log_prior_universal(t_ej, v0_n, a_n, v0_s, a_s, pa_0_n, omega_n, pa_0_s, omega_s, scatter_val, scatter_type='constant'):
    if not (60000 < t_ej < 61000 and 1e-4 < v0_n < 0.5 and -1e-2 <= a_n <= 1e-2):
        return -np.inf
    if not (-0.5 < v0_s < -1e-4 and -1e-2 <= a_s <= 1e-2):
        return -np.inf
    if not (-90.0 < pa_0_n < 0.0 and -5.0 <= omega_n <= 5.0):
        return -np.inf
    if not (-90.0 < pa_0_s < 0.0 and -5.0 <= omega_s <= 5.0):
        return -np.inf
        
    if scatter_type == 'constant':
        if not (-5.0 < scatter_val < 1.0):
            return -np.inf
    elif scatter_type == 'expanding':
        if not (0.0 <= scatter_val < 90.0):
            return -np.inf
    
    mu_v0_n, sig_v0_n = 0.017, 0.005
    mu_v0_s, sig_v0_s = -0.009, 0.005
    return -0.5 * ((v0_n - mu_v0_n) / sig_v0_n)**2 - 0.5 * ((v0_s - mu_v0_s) / sig_v0_s)**2"""
old_calc = """def calc_log_likelihood(t_ej, v0_n, a_n, v0_s, a_s, pa_0_n, omega_n, pa_0_s, omega_s, log_s):
    s2 = (10**log_s)**2
    
    # Northern Lobe
    theta_n = np.radians(pa_0_n + omega_n * (n_mjd - t_ej))
    sin_n, cos_n = np.sin(theta_n), np.cos(theta_n)
    par_n = n_da * sin_n + n_dd * cos_n
    perp_n = n_da * cos_n - n_dd * sin_n
    var_par_n = (n_da_err * sin_n)**2 + (n_dd_err * cos_n)**2 + s2
    var_perp_n = (n_da_err * cos_n)**2 + (n_dd_err * sin_n)**2 + s2
    dt_n = n_mjd - t_ej
    model_par_n = v0_n * dt_n + 0.5 * a_n * dt_n**2
    ll_n = -0.5 * np.sum((par_n - model_par_n)**2 / var_par_n + np.log(2 * np.pi * var_par_n))
    ll_n += -0.5 * np.sum(perp_n**2 / var_perp_n + np.log(2 * np.pi * var_perp_n))
    
    # Southern Lobe
    theta_s = np.radians(pa_0_s + omega_s * (s_mjd - t_ej))
    sin_s, cos_s = np.sin(theta_s), np.cos(theta_s)
    par_s = s_da * sin_s + s_dd * cos_s
    perp_s = s_da * cos_s - s_dd * sin_s
    var_par_s = (s_da_err * sin_s)**2 + (s_dd_err * cos_s)**2 + s2
    var_perp_s = (s_da_err * cos_s)**2 + (s_dd_err * sin_s)**2 + s2
    dt_s = s_mjd - t_ej
    model_par_s = v0_s * dt_s + 0.5 * a_s * dt_s**2
    ll_s = -0.5 * np.sum((par_s - model_par_s)**2 / var_par_s + np.log(2 * np.pi * var_par_s))
    ll_s += -0.5 * np.sum(perp_s**2 / var_perp_s + np.log(2 * np.pi * var_perp_s))
    
    return ll_n + ll_s

def log_prior_universal(t_ej, v0_n, a_n, v0_s, a_s, pa_0_n, omega_n, pa_0_s, omega_s, log_s):
    if not (60000 < t_ej < 61000 and 1e-4 < v0_n < 0.5 and -1e-2 <= a_n <= 1e-2):
        return -np.inf
    if not (-0.5 < v0_s < -1e-4 and -1e-2 <= a_s <= 1e-2):
        return -np.inf
    if not (-90.0 < pa_0_n < 0.0 and -5.0 <= omega_n <= 5.0):
        return -np.inf
    if not (-90.0 < pa_0_s < 0.0 and -5.0 <= omega_s <= 5.0):
        return -np.inf
    if not (-5.0 < log_s < 1.0):
        return -np.inf
    
    mu_v0_n, sig_v0_n = 0.017, 0.005
    mu_v0_s, sig_v0_s = -0.009, 0.005
    return -0.5 * ((v0_n - mu_v0_n) / sig_v0_n)**2 - 0.5 * ((v0_s - mu_v0_s) / sig_v0_s)**2"""

if new_calc in code:
    code = code.replace(new_calc, old_calc)

import re

# We will just remove lines between # Model 5 and # Find best model
code = re.sub(r'# Model 5.*?# Candidate models summary\nmodels = \[.*?\]', r'# Candidate models summary\nmodels = [\n    {"id": 1, "name": "2nd-Order Polynomial + Precessing PA", "k": k_m1, "ll": ll_m1, "bic": bic_m1, "aic": aic_m1, "res": res_m1},\n    {"id": 2, "name": "1st-Order Ballistic  + Precessing PA", "k": k_m2, "ll": ll_m2, "bic": bic_m2, "aic": aic_m2, "res": res_m2},\n    {"id": 3, "name": "2nd-Order Polynomial + Fixed PA",      "k": k_m3, "ll": ll_m3, "bic": bic_m3, "aic": aic_m3, "res": res_m3},\n    {"id": 4, "name": "1st-Order Ballistic  + Fixed PA",      "k": k_m4, "ll": ll_m4, "bic": bic_m4, "aic": aic_m4, "res": res_m4},\n]', code, flags=re.DOTALL)

code = re.sub(r'if best_model\["id"\] in \[4, 8\]:.*?elif best_model\["id"\] in \[3, 7\]:.*?else:\n.*?opt_init = best_model\["res"\].x', r"""if best_model["id"] == 4:
    ndim = 6
    param_labels = [r"$T_{\rm ej}$", r"$v_{0,n}$", r"$v_{0,s}$", r"${\rm PA}_n$", r"${\rm PA}_s$", r"$\log s$"]
    
    def log_prior(params):
        return log_prior_universal(params[0], params[1], 0.0, params[2], 0.0, params[3], 0.0, params[4], 0.0, params[5])

    def log_prob(params):
        lp = log_prior(params)
        if not np.isfinite(lp): return -np.inf
        return lp + calc_log_likelihood(params[0], params[1], 0.0, params[2], 0.0, params[3], 0.0, params[4], 0.0, params[5])

    opt_init = res_m4.x

elif best_model["id"] == 2:
    ndim = 8
    param_labels = [r"$T_{\rm ej}$", r"$v_{0,n}$", r"$v_{0,s}$", r"${\rm PA}_{0,n}$", r"$\omega_n$", r"${\rm PA}_{0,s}$", r"$\omega_s$", r"$\log s$"]
    
    def log_prior(params):
        return log_prior_universal(params[0], params[1], 0.0, params[2], 0.0, params[3], params[4], params[5], params[6], params[7])

    def log_prob(params):
        lp = log_prior(params)
        if not np.isfinite(lp): return -np.inf
        return lp + calc_log_likelihood(params[0], params[1], 0.0, params[2], 0.0, params[3], params[4], params[5], params[6], params[7])

    opt_init = res_m2.x

elif best_model["id"] == 3:
    ndim = 8
    param_labels = [r"$T_{\rm ej}$", r"$v_{0,n}$", r"$a_n$", r"$v_{0,s}$", r"$a_s$", r"${\rm PA}_n$", r"${\rm PA}_s$", r"$\log s$"]
    
    def log_prior(params):
        return log_prior_universal(params[0], params[1], params[2], params[3], params[4], params[5], 0.0, params[6], 0.0, params[7])

    def log_prob(params):
        lp = log_prior(params)
        if not np.isfinite(lp): return -np.inf
        return lp + calc_log_likelihood(params[0], params[1], params[2], params[3], params[4], params[5], 0.0, params[6], 0.0, params[7])

    opt_init = res_m3.x

else:
    ndim = 10
    param_labels = [r"$T_{\rm ej}$", r"$v_{0,n}$", r"$a_n$", r"$v_{0,s}$", r"$a_s$", r"${\rm PA}_{0,n}$", r"$\omega_n$", r"${\rm PA}_{0,s}$", r"$\omega_s$", r"$\log s$"]
    
    def log_prior(params):
        return log_prior_universal(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9])

    def log_prob(params):
        lp = log_prior(params)
        if not np.isfinite(lp): return -np.inf
        return lp + calc_log_likelihood(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9])

    opt_init = res_m1.x""", code, flags=re.DOTALL)


code = re.sub(r'def unpack_params\(sample, model_id\):.*?else:\n\s+return sample\[0\], sample\[1\], sample\[2\], sample\[3\], sample\[4\], sample\[5\], sample\[6\], sample\[7\], sample\[8\], sample\[9\]', r"""def unpack_params(sample, model_id):
    if model_id == 4:
        return sample[0], sample[1], 0.0, sample[2], 0.0, sample[3], 0.0, sample[4], 0.0, sample[5]
    elif model_id == 2:
        return sample[0], sample[1], 0.0, sample[2], 0.0, sample[3], sample[4], sample[5], sample[6], sample[7]
    elif model_id == 3:
        return sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], 0.0, sample[6], 0.0, sample[7]
    else:
        return sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], sample[7], sample[8], sample[9]""", code, flags=re.DOTALL)


code = code.replace("if best_model[\"id\"] in [1, 3, 5, 7]:", "if best_model[\"id\"] in [1, 3]:")
code = code.replace("if best_model[\"id\"] in [1, 2, 5, 6]:", "if best_model[\"id\"] in [1, 2]:")
code = code.replace("v0_s_chain = flat_samples[:, 2] if best_model[\"id\"] in [2, 4, 6, 8] else flat_samples[:, 3]", "v0_s_chain = flat_samples[:, 2] if best_model[\"id\"] in [2, 4] else flat_samples[:, 3]")

code = re.sub(r'if best_model\["id"\] in \[4, 8\]:\n.*?p_omega_s = compute_hdi\(flat_samples\[:, 8\]\)', r"""if best_model["id"] == 4:
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
    p_omega_s = compute_hdi(flat_samples[:, 8])""", code, flags=re.DOTALL)

with open('/media/kyle/kyle_phd/GRS1915/plot_sep_vs_time_poly.py', 'w') as f:
    f.write(code)

