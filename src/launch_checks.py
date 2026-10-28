"""
Two checks for the Discussion of docs/main.tex, from the apparent-motion Model 1 fit and the in-band lobe fluxes.

1. Model-independent latest launch, Eq. (tej_bound): for ejecta that never exceed c, r_a r_r <= (c dt / D)^2 at any
   inclination and speed history, so T_ej <= t_obs - (D/c) sqrt(r_a r_r). Monte Carlo over the lobe position errors
   (about the fitted core) and D = 9.4 +/- 0.7 kpc, per resolved epoch.
2. Flux ratio of the two lobes at equal intrinsic age t' for the ballistic geometry (posterior samples of T_ej and
   beta cos theta), from exponential fits to the in-band lobe fluxes, against the Doppler-boosting expectation
   (delta_a/delta_r)^(k - alpha) with delta_a/delta_r = (1 + beta cos theta)/(1 - beta cos theta).

  python src/launch_checks.py
"""

import numpy as np
from scipy.optimize import curve_fit

rng = np.random.default_rng(1915)
N_MC = 100_000
D_KPC, D_ERR = 9.4, 0.7
LD_PER_KPC = 3.0857e19 / (2.99792458e8 * 86400.0)  # light-days per kpc
RAD_PER_ARCSEC = np.pi / 180.0 / 3600.0
LOBE_ALPHA = -1.064  # alpha/inband_alpha_final.txt

z = np.load("data/absolute_kinematics/absolute_kinematics_plot_data_model1_without_limits.npz")
chain = z["flat_samples"]  # T_ej, v_n, v_s, PA_n, PA_s, theta_scatter, x_c, y_c


def separation(prefix, k):
    """Monte Carlo separations (arcsec) of lobe `prefix` ('n' or 's') at epoch index k, about the fitted core."""
    # The stored positions are already offsets from the fitted core (pipeline n_da = x - x_c)
    x = z[f"{prefix}_da"][k] + z[f"{prefix}_da_err"][k] * rng.standard_normal(N_MC)
    y = z[f"{prefix}_dd"][k] + z[f"{prefix}_dd_err"][k] * rng.standard_normal(N_MC)
    return np.hypot(x, y)


print("1. Latest launch, T_ej <= t_obs - (D/c) sqrt(r_a r_r):")
for i, t in enumerate(z["n_mjd"]):
    j = int(np.argmin(np.abs(z["s_mjd"] - t)))
    r_r, r_a = separation("n", i), separation("s", j)
    d = rng.normal(D_KPC, D_ERR, N_MC)
    bound = t - np.sqrt(r_a * r_r) * RAD_PER_ARCSEC * d * LD_PER_KPC
    q16, q50, q84, q977 = np.percentile(bound, [16, 50, 84, 97.7])
    print(f"  MJD {t:.1f}: r_a = {np.median(r_a):.2f}\", r_r = {np.median(r_r):.2f}\" -> "
          f"T_ej <= {q50:.0f} (+{q84 - q50:.0f}/-{q50 - q16:.0f}; {q977:.0f} at 97.7%)")

lobes = {}
for line in open("alpha/inband_alpha_final.txt"):
    f = line.split()
    if len(f) >= 9 and not line.startswith("#") and f[1] in ("N", "S"):
        lobes.setdefault(f[1], []).append((float(f[2]), float(f[5]), float(f[6])))
fade = {}
print("\n2. Exponential fade fits to the in-band lobe fluxes (5% flux-scale error per point):")
for lobe, v in lobes.items():
    t, s, e = np.array(v).T
    e = np.hypot(e, 0.05 * s)
    p, c = curve_fit(lambda t, ln_a, tau: ln_a - (t - 60800.0) / tau, t, np.log(s), p0=[0.0, 60.0],
                     sigma=e / s, absolute_sigma=True)
    fade[lobe] = (p, c)
    print(f"  {lobe}: tau_fade = {p[1]:.1f} +/- {np.sqrt(c[1, 1]):.1f} d")

n = 4000
idx = rng.choice(len(chain), n, replace=False)
t_ej, mu_r, mu_a = chain[idx, 0], chain[idx, 1], np.abs(chain[idx, 2])
bct = (mu_a - mu_r) / (mu_a + mu_r)
p_a = rng.multivariate_normal(*fade["S"], n)
p_r = rng.multivariate_normal(*fade["N"], n)
for tp in (450, 500, 550):
    t_a, t_r = t_ej + tp * (1 - bct), t_ej + tp * (1 + bct)
    ratio = np.exp((p_a[:, 0] - (t_a - 60800.0) / p_a[:, 1]) - (p_r[:, 0] - (t_r - 60800.0) / p_r[:, 1]))
    q16, q50, q84 = np.percentile(ratio, [16, 50, 84])
    print(f"  t' = {tp} d: S_a/S_r = {q50:.2f} ({q16:.2f}-{q84:.2f})")
dr = (1 + np.median(bct)) / (1 - np.median(bct))
print(f"  expected from boosting: k=2 {dr ** (2 - LOBE_ALPHA):.2f}, k=3 {dr ** (3 - LOBE_ALPHA):.2f}")
