"""
Test: does the linear model need a perpendicular offset of the jet axis from the core?

Both models of the paper assume that each lobe moves in a straight line from the core. The lobe centroids lie
systematically NE of that axis (Appendix "Position angle of the lobes"). Here the linear model (Model 1 of
src/pipeline_absolute_kinematics_with_limits.py, no limits) is refitted with the perpendicular model
r_perp = d instead of 0, i.e. an axis parallel to PA that misses the core by d (positive to the NE for both lobes,
as in the pipeline's sign convention):

  A  d = 0 (the paper's Model 1), k = 8
  B  one offset d for both lobes,  k = 9
  C  separate offsets d_n, d_s,    k = 10

Maximum-likelihood fits (Nelder-Mead from several starts) give lnL and BIC for each; B is then sampled with emcee to
see how the offset and the kinematics change. The data and likelihood are the pipeline's own (its loading code is
executed up to the model definitions), so A reproduces Model 1.

  python src/test_perp_offset.py [--n-steps 20000] [--report data/kinematics_perp_offset_test.txt]
"""

import argparse
import sys

import emcee
import numpy as np
from scipy.optimize import minimize

p = argparse.ArgumentParser()
p.add_argument("--n-steps", type=int, default=20000)
p.add_argument("--n-walkers", type=int, default=128)
p.add_argument("--report", default="data/kinematics_perp_offset_test.txt")
p.add_argument("--seed", type=int, default=1915)
args = p.parse_args()
rng = np.random.default_rng(args.seed)

# Pipeline data loading (argparse, frame offsets, positions), stopped before its model fits
src = open("src/pipeline_absolute_kinematics_with_limits.py").read()
argv, sys.argv = sys.argv, ["pipeline", "--no-include-limits", "--no-plot-separation"]
pipe = {"__name__": "pipeline_data", "__file__": "src/pipeline_absolute_kinematics_with_limits.py"}
exec(compile(src[: src.index("# 2. UNIVERSAL TRAJECTORY")], "pipeline_data", "exec"), pipe)
sys.argv = argv
core, n_data, s_data = pipe["core_data"], pipe["n_data"], pipe["s_data"]
N = pipe["num_obs_total"]


def lnlike(t_ej, vn, vs, pa_n, pa_s, scat, x_c, y_c, d_n, d_s):
    """Pipeline Model 1 likelihood (no limits) with perpendicular model d instead of 0."""
    ll = -0.5 * np.sum((core[:, 1] - x_c) ** 2 / core[:, 3] ** 2 + np.log(2 * np.pi * core[:, 3] ** 2))
    ll += -0.5 * np.sum((core[:, 2] - y_c) ** 2 / core[:, 4] ** 2 + np.log(2 * np.pi * core[:, 4] ** 2))
    tan2 = np.tan(np.radians(scat)) ** 2
    for lobe, v, pa, d in ((n_data, vn, pa_n, d_n), (s_data, vs, pa_s, d_s)):
        dx, dy, ex, ey = lobe[:, 1] - x_c, lobe[:, 2] - y_c, lobe[:, 3], lobe[:, 4]
        s, c = np.sin(np.radians(pa)), np.cos(np.radians(pa))
        model_par = v * np.clip(lobe[:, 0] - t_ej, 0.0, None)
        var_par = (ex * s) ** 2 + (ey * c) ** 2
        var_perp = (ex * c) ** 2 + (ey * s) ** 2 + model_par**2 * tan2
        ll += -0.5 * np.sum((dx * s + dy * c - model_par) ** 2 / var_par + np.log(2 * np.pi * var_par))
        ll += -0.5 * np.sum((dx * c - dy * s - d) ** 2 / var_perp + np.log(2 * np.pi * var_perp))
    return ll


def unpack(q, variant):
    base = list(q[:8])
    if variant == "A":
        return base + [0.0, 0.0]
    if variant == "B":
        return base + [q[8], q[8]]
    return base + [q[8], q[9]]


def lnprior(q, variant):
    t_ej, vn, vs, pa_n, pa_s, scat, x_c, y_c = q[:8]
    if not (60000 < t_ej < 60800 and 1e-4 < vn < 0.5 and -0.5 < vs < -1e-4 and abs(vs) > vn):
        return -np.inf
    if not (-180 < pa_n < 180 and -180 < pa_s < 180 and 0 <= scat <= 45 and abs(x_c) < 2 and abs(y_c) < 2):
        return -np.inf
    if variant != "A" and not np.all(np.abs(q[8:]) < 2.0):
        return -np.inf
    return 0.0


def lnprob(q, variant):
    lp = lnprior(q, variant)
    return lp + lnlike(*unpack(q, variant)) if np.isfinite(lp) else -np.inf


start = np.array([60372.0, 0.0055, -0.0072, -41.7, -42.5, 4.3, -0.198, -0.359])
extra = {"A": [], "B": [0.3], "C": [0.3, 0.3]}
fits = {}
for variant in "ABC":
    x0 = np.r_[start, extra[variant]]
    best = None
    for i in range(12):
        trial = x0 * (1 + (0.02 * rng.standard_normal(len(x0)) if i else 0))
        r = minimize(lambda q: -lnprob(q, variant), trial, method="Nelder-Mead",
                     options={"maxiter": 40000, "maxfev": 40000, "xatol": 1e-7, "fatol": 1e-9})
        if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
            best = r
    k = len(x0)
    fits[variant] = dict(x=best.x, lnL=-best.fun, k=k, bic=k * np.log(N) - 2 * (-best.fun))

lines = [f"Perpendicular-offset test, linear model, no limits, N_data = {N}",
         "A: axis through the core (paper Model 1); B: common offset d; C: separate d_n, d_s.",
         f"{'variant':<8}{'k':>3}{'lnL':>9}{'BIC':>9}{'dBIC(vs A)':>12}   d_n (\")  d_s (\")  theta_scatter  T_ej   PA_n   PA_s"]
for v, f in fits.items():
    q = unpack(f["x"], v)
    lines.append(f"{v:<8}{f['k']:>3}{f['lnL']:>9.2f}{f['bic']:>9.2f}{f['bic'] - fits['A']['bic']:>12.2f}   "
                 f"{q[8]:7.3f}  {q[9]:7.3f}  {q[5]:13.2f}  {q[0]:7.1f}  {q[3] % 360:6.1f}  {q[4] % 360:6.1f}")
print("\n".join(lines), flush=True)

# Posterior of B
ndim = 9
p0 = fits["B"]["x"] + 1e-4 * np.abs(fits["B"]["x"]) * rng.standard_normal((args.n_walkers, ndim))
sampler = emcee.EnsembleSampler(args.n_walkers, ndim, lnprob, args=("B",))
sampler.run_mcmc(p0, args.n_steps, progress=False)
chain = sampler.get_chain(discard=args.n_steps // 5, thin=10, flat=True)
tau_ac = sampler.get_autocorr_time(quiet=True)
names = ["T_ej", "mu_r (mas/d)", "mu_a (mas/d)", "PA_n (E of N)", "PA_s+180 (E of N)", "theta_scatter",
         "x_c", "y_c", "d (arcsec)"]
scale = np.array([1, 1e3, -1e3, 1, 1, 1, 1, 1, 1])
lines.append(f"\nMCMC, variant B ({args.n_steps} steps x {args.n_walkers} walkers, 20% burn-in; "
             f"max autocorr {np.nanmax(tau_ac):.0f} steps): median [16th, 84th]")
for i, name in enumerate(names):
    col = chain[:, i] * scale[i]
    if "PA" in name:
        col = col % 360
    q16, q50, q84 = np.percentile(col, [16, 50, 84])
    lines.append(f"  {name:<18} {q50:10.4f}  [{q16:10.4f}, {q84:10.4f}]")
mu_r, mu_a = chain[:, 1], -chain[:, 2]
bct = (mu_a - mu_r) / (mu_a + mu_r)
q16, q50, q84 = np.percentile(bct, [16, 50, 84])
lines.append(f"  {'beta cos theta':<18} {q50:10.4f}  [{q16:10.4f}, {q84:10.4f}]")
lines.append(f"  P(d <= 0) = {np.mean(chain[:, 8] <= 0):.2e}")
print("\n".join(lines[-12:]))
with open(args.report, "w") as fp:
    fp.write("\n".join(lines) + "\n")
np.savez(args.report.replace(".txt", ".npz"), samples=chain, names=names, autocorr=tau_ac,
         ml_A=fits["A"]["x"], ml_B=fits["B"]["x"], ml_C=fits["C"]["x"])
print(f"Saved {args.report}")
