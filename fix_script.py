with open('/media/kyle/kyle_phd/GRS1915/plot_sep_vs_time_poly.py', 'r') as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if line.startswith('def calc_log_likelihood'):
        skip = True
        
        # INSERT THE ORIGINAL CODE BLOCK HERE
        original_code = """def calc_log_likelihood(t_ej, v0_n, a_n, v0_s, a_s, pa_0_n, omega_n, pa_0_s, omega_s, log_s):
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
    return -0.5 * ((v0_n - mu_v0_n) / sig_v0_n)**2 - 0.5 * ((v0_s - mu_v0_s) / sig_v0_s)**2

# -----------------------------------------------------------------------------
# 3. QUANTITATIVE MODEL SELECTION (BIC & AIC COMPARISONS)
# -----------------------------------------------------------------------------
print("\\n" + "="*70)
print(f"{'QUANTITATIVE MODEL SELECTION (BIC / AIC SUITE)':^70}")
print("="*70)

# Model 1: 2nd-Order Polynomial + Precessing PA (10 params)
def nll_m1(p):
    lp = log_prior_universal(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9])
    if not np.isfinite(lp): return 1e12
    return -(lp + calc_log_likelihood(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]))

res_m1 = minimize(nll_m1, [60440.0, 0.0065, 0.0, -0.0083, 0.0, -40.2, 0.0, -42.7, 0.0, -3.0], method='Nelder-Mead', options={'maxiter': 10000})
ll_m1 = calc_log_likelihood(*res_m1.x)
k_m1 = 10
bic_m1 = k_m1 * np.log(n_data) - 2 * ll_m1
aic_m1 = 2 * k_m1 - 2 * ll_m1

# Model 2: 1st-Order Linear Ballistic + Precessing PA (8 params: a_n = a_s = 0)
def nll_m2(p):
    lp = log_prior_universal(p[0], p[1], 0.0, p[2], 0.0, p[3], p[4], p[5], p[6], p[7])
    if not np.isfinite(lp): return 1e12
    return -(lp + calc_log_likelihood(p[0], p[1], 0.0, p[2], 0.0, p[3], p[4], p[5], p[6], p[7]))

res_m2 = minimize(nll_m2, [60440.0, 0.0065, -0.0083, -40.2, 0.0, -42.7, 0.0, -3.0], method='Nelder-Mead', options={'maxiter': 10000})
ll_m2 = calc_log_likelihood(res_m2.x[0], res_m2.x[1], 0.0, res_m2.x[2], 0.0, res_m2.x[3], res_m2.x[4], res_m2.x[5], res_m2.x[6], res_m2.x[7])
k_m2 = 8
bic_m2 = k_m2 * np.log(n_data) - 2 * ll_m2
aic_m2 = 2 * k_m2 - 2 * ll_m2

# Model 3: 2nd-Order Polynomial + Fixed PA (8 params: omega_n = omega_s = 0)
def nll_m3(p):
    lp = log_prior_universal(p[0], p[1], p[2], p[3], p[4], p[5], 0.0, p[6], 0.0, p[7])
    if not np.isfinite(lp): return 1e12
    return -(lp + calc_log_likelihood(p[0], p[1], p[2], p[3], p[4], p[5], 0.0, p[6], 0.0, p[7]))

res_m3 = minimize(nll_m3, [60440.0, 0.0065, 0.0, -0.0083, 0.0, -40.2, -42.7, -3.0], method='Nelder-Mead', options={'maxiter': 10000})
ll_m3 = calc_log_likelihood(res_m3.x[0], res_m3.x[1], res_m3.x[2], res_m3.x[3], res_m3.x[4], res_m3.x[5], 0.0, res_m3.x[6], 0.0, res_m3.x[7])
k_m3 = 8
bic_m3 = k_m3 * np.log(n_data) - 2 * ll_m3
aic_m3 = 2 * k_m3 - 2 * ll_m3

# Model 4: 1st-Order Linear Ballistic + Fixed PA (6 params: a_n = a_s = omega_n = omega_s = 0)
def nll_m4(p):
    lp = log_prior_universal(p[0], p[1], 0.0, p[2], 0.0, p[3], 0.0, p[4], 0.0, p[5])
    if not np.isfinite(lp): return 1e12
    return -(lp + calc_log_likelihood(p[0], p[1], 0.0, p[2], 0.0, p[3], 0.0, p[4], 0.0, p[5]))

res_m4 = minimize(nll_m4, [60440.0, 0.0065, -0.0083, -40.2, -42.7, -3.0], method='Nelder-Mead', options={'maxiter': 10000})
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

# Find best model according to BIC
best_model = min(models, key=lambda m: m["bic"])

print(f"{'Model Description':<40} {'k':<4} {'ln(L)':<10} {'BIC':<10} {'Delta BIC':<10}")
print("-" * 76)
for m in models:
    delta_bic = m["bic"] - best_model["bic"]
    print(f"{m['name']:<40} {m['k']:<4} {m['ll']:<10.2f} {m['bic']:<10.2f} {delta_bic:<10.2f}")
print("-" * 76)
print(f"--> BEST STATISTICAL MODEL SELECTED: Model {best_model['id']} ({best_model['name']})")
print("="*70 + "\\n")

# -----------------------------------------------------------------------------
# 4. MCMC POSTERIOR SAMPLING FOR THE BEST MODEL
# -----------------------------------------------------------------------------
print(f"Sampling posteriors for Model {best_model['id']} via MCMC...")

if best_model["id"] == 4:
    ndim = 6
    param_labels = [r"$T_{\\rm ej}$", r"$v_{0,n}$", r"$v_{0,s}$", r"${\\rm PA}_n$", r"${\\rm PA}_s$", r"$\log s$"]
    
    def log_prior(params):
        return log_prior_universal(params[0], params[1], 0.0, params[2], 0.0, params[3], 0.0, params[4], 0.0, params[5])

    def log_prob(params):
        lp = log_prior(params)
        if not np.isfinite(lp): return -np.inf
        return lp + calc_log_likelihood(params[0], params[1], 0.0, params[2], 0.0, params[3], 0.0, params[4], 0.0, params[5])

    opt_init = res_m4.x

elif best_model["id"] == 2:
    ndim = 8
    param_labels = [r"$T_{\\rm ej}$", r"$v_{0,n}$", r"$v_{0,s}$", r"${\\rm PA}_{0,n}$", r"$\omega_n$", r"${\\rm PA}_{0,s}$", r"$\omega_s$", r"$\log s$"]
    
    def log_prior(params):
        return log_prior_universal(params[0], params[1], 0.0, params[2], 0.0, params[3], params[4], params[5], params[6], params[7])

    def log_prob(params):
        lp = log_prior(params)
        if not np.isfinite(lp): return -np.inf
        return lp + calc_log_likelihood(params[0], params[1], 0.0, params[2], 0.0, params[3], params[4], params[5], params[6], params[7])

    opt_init = res_m2.x

elif best_model["id"] == 3:
    ndim = 8
    param_labels = [r"$T_{\\rm ej}$", r"$v_{0,n}$", r"$a_n$", r"$v_{0,s}$", r"$a_s$", r"${\\rm PA}_n$", r"${\\rm PA}_s$", r"$\log s$"]
    
    def log_prior(params):
        return log_prior_universal(params[0], params[1], params[2], params[3], params[4], params[5], 0.0, params[6], 0.0, params[7])

    def log_prob(params):
        lp = log_prior(params)
        if not np.isfinite(lp): return -np.inf
        return lp + calc_log_likelihood(params[0], params[1], params[2], params[3], params[4], params[5], 0.0, params[6], 0.0, params[7])

    opt_init = res_m3.x

else:
    ndim = 10
    param_labels = [r"$T_{\\rm ej}$", r"$v_{0,n}$", r"$a_n$", r"$v_{0,s}$", r"$a_s$", r"${\\rm PA}_{0,n}$", r"$\omega_n$", r"${\\rm PA}_{0,s}$", r"$\omega_s$", r"$\log s$"]
    
    def log_prior(params):
        return log_prior_universal(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9])

    def log_prob(params):
        lp = log_prior(params)
        if not np.isfinite(lp): return -np.inf
        return lp + calc_log_likelihood(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9])

    opt_init = res_m1.x

nwalkers = 48
n_steps = 15000
scales = np.abs(opt_init) * 1e-3 + 1e-4
scales[0] = 0.5

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
fig_trace.savefig('/media/kyle/kyle_phd/GRS1915/mcmc_trace.png', dpi=300)
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
fig_corner.savefig('/media/kyle/kyle_phd/GRS1915/corner_plot.png', dpi=300)
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
"""
        new_lines.append(original_code + "\n")
    elif line.startswith('# -----------------------------------------------------------------------------'):
        if skip:
            if '8. GENERATE PUBLICATION-QUALITY PLOTS' in lines[lines.index(line) + 1]:
                skip = False
    
    if not skip:
        new_lines.append(line)

with open('/media/kyle/kyle_phd/GRS1915/plot_sep_vs_time_poly.py', 'w') as f:
    f.write("".join(new_lines))

