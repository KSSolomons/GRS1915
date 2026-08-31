import re

with open('/media/kyle/kyle_phd/GRS1915/plot_sep_vs_time_poly.py', 'r') as f:
    code = f.read()

# 1. Update calc_log_likelihood and log_prior_universal
old_calc = """def calc_log_likelihood(t_ej, v0_n, a_n, v0_s, a_s, pa_0_n, omega_n, pa_0_s, omega_s, log_s):
    s2 = 10**log_s
    
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

new_calc = """def calc_log_likelihood(t_ej, v0_n, a_n, v0_s, a_s, pa_0_n, omega_n, pa_0_s, omega_s, scatter_val, scatter_type='constant'):
    # Northern Lobe
    theta_n = np.radians(pa_0_n + omega_n * (n_mjd - t_ej))
    sin_n, cos_n = np.sin(theta_n), np.cos(theta_n)
    par_n = n_da * sin_n + n_dd * cos_n
    perp_n = n_da * cos_n - n_dd * sin_n
    
    dt_n = n_mjd - t_ej
    model_par_n = v0_n * dt_n + 0.5 * a_n * dt_n**2
    
    if scatter_type == 'constant':
        s2_n = 10**scatter_val
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
        s2_s = 10**scatter_val
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
code = code.replace(old_calc, new_calc)

# 2. Add Models 5-8
old_models_def = """# Candidate models summary
models = [
    {"id": 1, "name": "2nd-Order Polynomial + Precessing PA", "k": k_m1, "ll": ll_m1, "bic": bic_m1, "aic": aic_m1, "res": res_m1},
    {"id": 2, "name": "1st-Order Ballistic  + Precessing PA", "k": k_m2, "ll": ll_m2, "bic": bic_m2, "aic": aic_m2, "res": res_m2},
    {"id": 3, "name": "2nd-Order Polynomial + Fixed PA",      "k": k_m3, "ll": ll_m3, "bic": bic_m3, "aic": aic_m3, "res": res_m3},
    {"id": 4, "name": "1st-Order Ballistic  + Fixed PA",      "k": k_m4, "ll": ll_m4, "bic": bic_m4, "aic": aic_m4, "res": res_m4},
]"""

new_models_def = """# Model 5: 2nd-Order Polynomial + Precessing PA + Expanding Scatter (10 params)
def nll_m5(p):
    lp = log_prior_universal(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], scatter_type='expanding')
    if not np.isfinite(lp): return 1e12
    return -(lp + calc_log_likelihood(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], scatter_type='expanding'))

res_m5 = minimize(nll_m5, [60650.0, 0.017, 0.0, -0.009, 0.0, -36.6, 0.0, -36.6, 0.0, 5.0], method='Nelder-Mead', options={'maxiter': 10000})
ll_m5 = calc_log_likelihood(*res_m5.x, scatter_type='expanding')
k_m5 = 10
bic_m5 = k_m5 * np.log(n_data) - 2 * ll_m5
aic_m5 = 2 * k_m5 - 2 * ll_m5

# Model 6: 1st-Order Ballistic + Precessing PA + Expanding Scatter (8 params)
def nll_m6(p):
    lp = log_prior_universal(p[0], p[1], 0.0, p[2], 0.0, p[3], p[4], p[5], p[6], p[7], scatter_type='expanding')
    if not np.isfinite(lp): return 1e12
    return -(lp + calc_log_likelihood(p[0], p[1], 0.0, p[2], 0.0, p[3], p[4], p[5], p[6], p[7], scatter_type='expanding'))

res_m6 = minimize(nll_m6, [60650.0, 0.017, -0.009, -36.6, 0.0, -36.6, 0.0, 5.0], method='Nelder-Mead', options={'maxiter': 10000})
ll_m6 = calc_log_likelihood(res_m6.x[0], res_m6.x[1], 0.0, res_m6.x[2], 0.0, res_m6.x[3], res_m6.x[4], res_m6.x[5], res_m6.x[6], res_m6.x[7], scatter_type='expanding')
k_m6 = 8
bic_m6 = k_m6 * np.log(n_data) - 2 * ll_m6
aic_m6 = 2 * k_m6 - 2 * ll_m6

# Model 7: 2nd-Order Polynomial + Fixed PA + Expanding Scatter (8 params)
def nll_m7(p):
    lp = log_prior_universal(p[0], p[1], p[2], p[3], p[4], p[5], 0.0, p[6], 0.0, p[7], scatter_type='expanding')
    if not np.isfinite(lp): return 1e12
    return -(lp + calc_log_likelihood(p[0], p[1], p[2], p[3], p[4], p[5], 0.0, p[6], 0.0, p[7], scatter_type='expanding'))

res_m7 = minimize(nll_m7, [60650.0, 0.017, 0.0, -0.009, 0.0, -36.6, -36.6, 5.0], method='Nelder-Mead', options={'maxiter': 10000})
ll_m7 = calc_log_likelihood(res_m7.x[0], res_m7.x[1], res_m7.x[2], res_m7.x[3], res_m7.x[4], res_m7.x[5], 0.0, res_m7.x[6], 0.0, res_m7.x[7], scatter_type='expanding')
k_m7 = 8
bic_m7 = k_m7 * np.log(n_data) - 2 * ll_m7
aic_m7 = 2 * k_m7 - 2 * ll_m7

# Model 8: 1st-Order Ballistic + Fixed PA + Expanding Scatter (6 params)
def nll_m8(p):
    lp = log_prior_universal(p[0], p[1], 0.0, p[2], 0.0, p[3], 0.0, p[4], 0.0, p[5], scatter_type='expanding')
    if not np.isfinite(lp): return 1e12
    return -(lp + calc_log_likelihood(p[0], p[1], 0.0, p[2], 0.0, p[3], 0.0, p[4], 0.0, p[5], scatter_type='expanding'))

res_m8 = minimize(nll_m8, [60650.0, 0.017, -0.009, -36.6, -36.6, 5.0], method='Nelder-Mead', options={'maxiter': 10000})
ll_m8 = calc_log_likelihood(res_m8.x[0], res_m8.x[1], 0.0, res_m8.x[2], 0.0, res_m8.x[3], 0.0, res_m8.x[4], 0.0, res_m8.x[5], scatter_type='expanding')
k_m8 = 6
bic_m8 = k_m8 * np.log(n_data) - 2 * ll_m8
aic_m8 = 2 * k_m8 - 2 * ll_m8

# Candidate models summary
models = [
    {"id": 1, "name": "2nd-Order + Precessing + Constant Scatter", "k": k_m1, "ll": ll_m1, "bic": bic_m1, "aic": aic_m1, "res": res_m1, "scatter_type": "constant"},
    {"id": 2, "name": "1st-Order + Precessing + Constant Scatter", "k": k_m2, "ll": ll_m2, "bic": bic_m2, "aic": aic_m2, "res": res_m2, "scatter_type": "constant"},
    {"id": 3, "name": "2nd-Order + Fixed PA + Constant Scatter",   "k": k_m3, "ll": ll_m3, "bic": bic_m3, "aic": aic_m3, "res": res_m3, "scatter_type": "constant"},
    {"id": 4, "name": "1st-Order + Fixed PA + Constant Scatter",   "k": k_m4, "ll": ll_m4, "bic": bic_m4, "aic": aic_m4, "res": res_m4, "scatter_type": "constant"},
    {"id": 5, "name": "2nd-Order + Precessing + Expanding Scatter", "k": k_m5, "ll": ll_m5, "bic": bic_m5, "aic": aic_m5, "res": res_m5, "scatter_type": "expanding"},
    {"id": 6, "name": "1st-Order + Precessing + Expanding Scatter", "k": k_m6, "ll": ll_m6, "bic": bic_m6, "aic": aic_m6, "res": res_m6, "scatter_type": "expanding"},
    {"id": 7, "name": "2nd-Order + Fixed PA + Expanding Scatter",   "k": k_m7, "ll": ll_m7, "bic": bic_m7, "aic": aic_m7, "res": res_m7, "scatter_type": "expanding"},
    {"id": 8, "name": "1st-Order + Fixed PA + Expanding Scatter",   "k": k_m8, "ll": ll_m8, "bic": bic_m8, "aic": aic_m8, "res": res_m8, "scatter_type": "expanding"},
]"""
code = code.replace(old_models_def, new_models_def)

# 3. Update MCMC setup to handle models 1-8
old_mcmc = """if best_model["id"] == 4:
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

    opt_init = res_m1.x"""

new_mcmc = """if best_model["id"] in [4, 8]:
    ndim = 6
    scatter_label = r"$\theta_{\rm scatt}$" if best_model["scatter_type"] == 'expanding' else r"$\log s$"
    param_labels = [r"$T_{\rm ej}$", r"$v_{0,n}$", r"$v_{0,s}$", r"${\rm PA}_n$", r"${\rm PA}_s$", scatter_label]
    
    def log_prior(params):
        return log_prior_universal(params[0], params[1], 0.0, params[2], 0.0, params[3], 0.0, params[4], 0.0, params[5], scatter_type=best_model["scatter_type"])

    def log_prob(params):
        lp = log_prior(params)
        if not np.isfinite(lp): return -np.inf
        return lp + calc_log_likelihood(params[0], params[1], 0.0, params[2], 0.0, params[3], 0.0, params[4], 0.0, params[5], scatter_type=best_model["scatter_type"])

    opt_init = best_model["res"].x

elif best_model["id"] in [2, 6]:
    ndim = 8
    scatter_label = r"$\theta_{\rm scatt}$" if best_model["scatter_type"] == 'expanding' else r"$\log s$"
    param_labels = [r"$T_{\rm ej}$", r"$v_{0,n}$", r"$v_{0,s}$", r"${\rm PA}_{0,n}$", r"$\omega_n$", r"${\rm PA}_{0,s}$", r"$\omega_s$", scatter_label]
    
    def log_prior(params):
        return log_prior_universal(params[0], params[1], 0.0, params[2], 0.0, params[3], params[4], params[5], params[6], params[7], scatter_type=best_model["scatter_type"])

    def log_prob(params):
        lp = log_prior(params)
        if not np.isfinite(lp): return -np.inf
        return lp + calc_log_likelihood(params[0], params[1], 0.0, params[2], 0.0, params[3], params[4], params[5], params[6], params[7], scatter_type=best_model["scatter_type"])

    opt_init = best_model["res"].x

elif best_model["id"] in [3, 7]:
    ndim = 8
    scatter_label = r"$\theta_{\rm scatt}$" if best_model["scatter_type"] == 'expanding' else r"$\log s$"
    param_labels = [r"$T_{\rm ej}$", r"$v_{0,n}$", r"$a_n$", r"$v_{0,s}$", r"$a_s$", r"${\rm PA}_n$", r"${\rm PA}_s$", scatter_label]
    
    def log_prior(params):
        return log_prior_universal(params[0], params[1], params[2], params[3], params[4], params[5], 0.0, params[6], 0.0, params[7], scatter_type=best_model["scatter_type"])

    def log_prob(params):
        lp = log_prior(params)
        if not np.isfinite(lp): return -np.inf
        return lp + calc_log_likelihood(params[0], params[1], params[2], params[3], params[4], params[5], 0.0, params[6], 0.0, params[7], scatter_type=best_model["scatter_type"])

    opt_init = best_model["res"].x

else:
    ndim = 10
    scatter_label = r"$\theta_{\rm scatt}$" if best_model["scatter_type"] == 'expanding' else r"$\log s$"
    param_labels = [r"$T_{\rm ej}$", r"$v_{0,n}$", r"$a_n$", r"$v_{0,s}$", r"$a_s$", r"${\rm PA}_{0,n}$", r"$\omega_n$", r"${\rm PA}_{0,s}$", r"$\omega_s$", scatter_label]
    
    def log_prior(params):
        return log_prior_universal(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], scatter_type=best_model["scatter_type"])

    def log_prob(params):
        lp = log_prior(params)
        if not np.isfinite(lp): return -np.inf
        return lp + calc_log_likelihood(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], scatter_type=best_model["scatter_type"])

    opt_init = best_model["res"].x"""
code = code.replace(old_mcmc, new_mcmc)

# 4. Update unpack_params to match models 1-8
old_unpack = """def unpack_params(sample, model_id):
    if model_id == 4:
        return sample[0], sample[1], 0.0, sample[2], 0.0, sample[3], 0.0, sample[4], 0.0, sample[5]
    elif model_id == 2:
        return sample[0], sample[1], 0.0, sample[2], 0.0, sample[3], sample[4], sample[5], sample[6], sample[7]
    elif model_id == 3:
        return sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], 0.0, sample[6], 0.0, sample[7]
    else:
        return sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], sample[7], sample[8], sample[9]"""

new_unpack = """def unpack_params(sample, model_id):
    if model_id in [4, 8]:
        return sample[0], sample[1], 0.0, sample[2], 0.0, sample[3], 0.0, sample[4], 0.0, sample[5]
    elif model_id in [2, 6]:
        return sample[0], sample[1], 0.0, sample[2], 0.0, sample[3], sample[4], sample[5], sample[6], sample[7]
    elif model_id in [3, 7]:
        return sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], 0.0, sample[6], 0.0, sample[7]
    else:
        return sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], sample[7], sample[8], sample[9]"""
code = code.replace(old_unpack, new_unpack)

# 5. Fix plotting indices (models 1-4 vs 5-8 for acceleration and PA arrays)
old_accel = """if best_model["id"] in [1, 3]:
    a_n_chain = flat_samples[:, 2]
    a_s_chain = flat_samples[:, 4]
    p_a_n = compute_hdi(a_n_chain)
    p_a_s = compute_hdi(a_s_chain)"""
new_accel = """if best_model["id"] in [1, 3, 5, 7]:
    a_n_chain = flat_samples[:, 2]
    a_s_chain = flat_samples[:, 4]
    p_a_n = compute_hdi(a_n_chain)
    p_a_s = compute_hdi(a_s_chain)"""
code = code.replace(old_accel, new_accel)

old_pa_extract = """if best_model["id"] == 4:
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
    p_omega_s = compute_hdi(flat_samples[:, 8])"""
new_pa_extract = """if best_model["id"] in [4, 8]:
    p_pa_n = compute_hdi(flat_samples[:, 3])
    p_pa_s = compute_hdi(flat_samples[:, 4])
elif best_model["id"] in [2, 6]:
    p_pa_n = compute_hdi(flat_samples[:, 3])
    p_omega_n = compute_hdi(flat_samples[:, 4])
    p_pa_s = compute_hdi(flat_samples[:, 5])
    p_omega_s = compute_hdi(flat_samples[:, 6])
elif best_model["id"] in [3, 7]:
    p_pa_n = compute_hdi(flat_samples[:, 5])
    p_pa_s = compute_hdi(flat_samples[:, 6])
else:
    p_pa_n = compute_hdi(flat_samples[:, 5])
    p_omega_n = compute_hdi(flat_samples[:, 6])
    p_pa_s = compute_hdi(flat_samples[:, 7])
    p_omega_s = compute_hdi(flat_samples[:, 8])"""
code = code.replace(old_pa_extract, new_pa_extract)

old_print_accel1 = """if best_model["id"] in [1, 3]:
    print(f"Receding Lobe Acceleration (a_n) : {p_a_n[1]:.4e} + {p_a_n[2]-p_a_n[1]:.4e} - {p_a_n[1]-p_a_n[0]:.4e} arcsec/day^2")"""
new_print_accel1 = """if best_model["id"] in [1, 3, 5, 7]:
    print(f"Receding Lobe Acceleration (a_n) : {p_a_n[1]:.4e} + {p_a_n[2]-p_a_n[1]:.4e} - {p_a_n[1]-p_a_n[0]:.4e} arcsec/day^2")"""
code = code.replace(old_print_accel1, new_print_accel1)

old_print_accel2 = """if best_model["id"] in [1, 3]:
    print(f"Approaching Lobe Acceleration (a_s) : {p_a_s[1]:.4e} + {p_a_s[2]-p_a_s[1]:.4e} - {p_a_s[1]-p_a_s[0]:.4e} arcsec/day^2")"""
new_print_accel2 = """if best_model["id"] in [1, 3, 5, 7]:
    print(f"Approaching Lobe Acceleration (a_s) : {p_a_s[1]:.4e} + {p_a_s[2]-p_a_s[1]:.4e} - {p_a_s[1]-p_a_s[0]:.4e} arcsec/day^2")"""
code = code.replace(old_print_accel2, new_print_accel2)

old_print_pa1 = """if best_model["id"] in [1, 2]:
    print(f"Northern Precession Rate (Omega_n): {p_omega_n[1]:>8.3f} + {p_omega_n[2]-p_omega_n[1]:.3f} - {p_omega_n[1]-p_omega_n[0]:.3f} deg/day")"""
new_print_pa1 = """if best_model["id"] in [1, 2, 5, 6]:
    print(f"Northern Precession Rate (Omega_n): {p_omega_n[1]:>8.3f} + {p_omega_n[2]-p_omega_n[1]:.3f} - {p_omega_n[1]-p_omega_n[0]:.3f} deg/day")"""
code = code.replace(old_print_pa1, new_print_pa1)

old_print_pa2 = """if best_model["id"] in [1, 2]:
    print(f"Southern Precession Rate (Omega_s): {p_omega_s[1]:>8.3f} + {p_omega_s[2]-p_omega_s[1]:.3f} - {p_omega_s[1]-p_omega_s[0]:.3f} deg/day")"""
new_print_pa2 = """if best_model["id"] in [1, 2, 5, 6]:
    print(f"Southern Precession Rate (Omega_s): {p_omega_s[1]:>8.3f} + {p_omega_s[2]-p_omega_s[1]:.3f} - {p_omega_s[1]-p_omega_s[0]:.3f} deg/day")"""
code = code.replace(old_print_pa2, new_print_pa2)

old_v0s_extract = """v0_s_chain = flat_samples[:, 2] if best_model["id"] in [2, 4] else flat_samples[:, 3]"""
new_v0s_extract = """v0_s_chain = flat_samples[:, 2] if best_model["id"] in [2, 4, 6, 8] else flat_samples[:, 3]"""
code = code.replace(old_v0s_extract, new_v0s_extract)

with open('/media/kyle/kyle_phd/GRS1915/plot_sep_vs_time_poly.py', 'w') as f:
    f.write(code)

