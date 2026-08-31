with open('/media/kyle/kyle_phd/GRS1915/plot_sep_vs_time_poly.py', 'r') as f:
    code = f.read()

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

code = code.replace(old_calc, new_calc)

with open('/media/kyle/kyle_phd/GRS1915/plot_sep_vs_time_poly.py', 'w') as f:
    f.write(code)

