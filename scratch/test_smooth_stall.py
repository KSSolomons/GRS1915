import os
import glob
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 1. LOAD DATA EXACTLY AS IN PIPELINE
dir_path = "S-band-fits"
files = sorted(glob.glob(os.path.join(dir_path, "*2D_Fitting*.txt")))

act_ra_deg = (19 + 15 / 60.0 + 11.558 / 3600.0) * 15.0
act_dec_deg = 10 + 56 / 60.0 + 44.905 / 3600.0
cos_dec = np.cos(np.radians(act_dec_deg))

core_data_raw, n_data_raw, s_data_raw = [], [], []

for f in files:
    filename = os.path.basename(f)
    m = re.search(r"img_(\d+)", filename)
    if not m:
        continue
    ts = int(m.group(1))
    mjd = ts / 86400.0 + 40587.0

    with open(f, "r") as fp:
        lines = fp.readlines()

    comps = []
    curr_comp = None
    for line in lines:
        if line.startswith("Component #"):
            curr_comp = {"name": line.strip()}
            comps.append(curr_comp)
        elif "Center X" in line and curr_comp is not None:
            m_ra = re.search(r"=\s*(\d+):(\d+):(\d+\.\d+)\s*±\s*(\S+)", line)
            if m_ra:
                h, m_time, s, err_s = float(m_ra.group(1)), float(m_ra.group(2)), float(m_ra.group(3)), float(m_ra.group(4))
                curr_comp["ra_deg"] = (h + m_time / 60.0 + s / 3600.0) * 15.0
                curr_comp["ra_err"] = err_s * 15.0 * cos_dec
        elif "Center Y" in line and curr_comp is not None:
            m_dec = re.search(r"=\s*([+-]?\d+):(\d+):(\d+\.\d+)\s*±\s*(\S+)", line)
            if m_dec:
                d, m_arc, s_arc, err_arcsec = float(m_dec.group(1)), float(m_dec.group(2)), float(m_dec.group(3)), float(m_dec.group(4))
                sign = -1 if "-" in m_dec.group(1) else 1
                curr_comp["dec_deg"] = sign * (abs(d) + m_arc / 60.0 + s_arc / 3600.0)
                curr_comp["dec_err"] = err_arcsec
        elif "Integrated flux" in line and curr_comp is not None:
            m_flux = re.search(r"=\s*([\d\.eE+-]+)\s*±\s*([\d\.eE+-]+)", line)
            if m_flux:
                curr_comp["flux"] = float(m_flux.group(1))
                curr_comp["flux_err"] = float(m_flux.group(2))

    for c in comps:
        if "ra_deg" not in c or "dec_deg" not in c:
            continue
        dx = (c["ra_deg"] - act_ra_deg) * 3600.0 * cos_dec
        dy = (c["dec_deg"] - act_dec_deg) * 3600.0
        ex = c["ra_err"]
        ey = c["dec_err"]
        flux = c.get("flux", 0.0)
        flux_err = c.get("flux_err", 0.0)

        if 288.7979 < c["ra_deg"] < 288.7982:
            core_data_raw.append([mjd, dx, dy, ex, ey, flux, flux_err])
        elif dy > 0:
            if ts != 1764664923:
                n_data_raw.append([mjd, dx, dy, ex, ey, flux, flux_err])
        else:
            if ts != 1764664923:
                s_data_raw.append([mjd, dx, dy, ex, ey, flux, flux_err])

temp_core = np.array(core_data_raw)
med_dx = np.median(temp_core[:, 1])
med_dy = np.median(temp_core[:, 2])
distances = np.sqrt((temp_core[:, 1] - med_dx)**2 + (temp_core[:, 2] - med_dy)**2)
mad = np.median(np.abs(distances - np.median(distances)))
threshold = max(5 * mad, 0.4)

core_data = [c for i, c in enumerate(core_data_raw) if distances[i] <= threshold]
core_data = np.array(core_data)
n_data = np.array(n_data_raw)
s_data = np.array(s_data_raw)

n_data = n_data[np.argsort(n_data[:, 0])][2:]
s_data = s_data[np.argsort(s_data[:, 0])][2:]
num_obs_total = 2 * len(core_data) + 2 * len(n_data) + 2 * len(s_data)

print(f"Loaded {len(core_data)} core points, {len(n_data)} N points, {len(s_data)} S points (N_obs = {num_obs_total})")

# 2. LOG LIKELIHOOD FUNCTION
ul_mjd = np.array([60736.11478, 60758.0668])
ul_r = np.array([2.423, 2.460])
ul_sigma = 0.05

def log_likelihood_general(r_n_func, r_s_func, pa_n, pa_s, theta_scatter, x_c, y_c):
    tan2_scatter = np.tan(np.radians(theta_scatter)) ** 2
    ll_total = 0.0

    # Core
    dx_c = core_data[:, 1] - x_c
    dy_c = core_data[:, 2] - y_c
    ex_c = core_data[:, 3]
    ey_c = core_data[:, 4]
    ll_total += np.sum(-0.5 * (dx_c**2 / ex_c**2 + np.log(2 * np.pi * ex_c**2)))
    ll_total += np.sum(-0.5 * (dy_c**2 / ey_c**2 + np.log(2 * np.pi * ey_c**2)))

    # Northern lobe
    dx_n = n_data[:, 1] - x_c
    dy_n = n_data[:, 2] - y_c
    ex_n = n_data[:, 3]
    ey_n = n_data[:, 4]

    th_n = np.radians(pa_n)
    sin_n, cos_n = np.sin(th_n), np.cos(th_n)
    par_n = dx_n * sin_n + dy_n * cos_n
    perp_n = dx_n * cos_n - dy_n * sin_n

    model_par_n = r_n_func(n_data[:, 0])
    var_par_n = (ex_n * sin_n)**2 + (ey_n * cos_n)**2
    var_perp_n = (ex_n * cos_n)**2 + (ey_n * sin_n)**2 + model_par_n**2 * tan2_scatter

    ll_total += -0.5 * np.sum((par_n - model_par_n)**2 / var_par_n + np.log(2 * np.pi * var_par_n))
    ll_total += -0.5 * np.sum(perp_n**2 / var_perp_n + np.log(2 * np.pi * var_perp_n))

    # Southern lobe
    dx_s = s_data[:, 1] - x_c
    dy_s = s_data[:, 2] - y_c
    ex_s = s_data[:, 3]
    ey_s = s_data[:, 4]

    th_s = np.radians(pa_s)
    sin_s, cos_s = np.sin(th_s), np.cos(th_s)
    par_s = dx_s * sin_s + dy_s * cos_s
    perp_s = dx_s * cos_s - dy_s * sin_s

    model_par_s = r_s_func(s_data[:, 0])
    var_par_s = (ex_s * sin_s)**2 + (ey_s * cos_s)**2
    var_perp_s = (ex_s * cos_s)**2 + (ey_s * sin_s)**2 + model_par_s**2 * tan2_scatter

    ll_total += -0.5 * np.sum((par_s - model_par_s)**2 / var_par_s + np.log(2 * np.pi * var_par_s))
    ll_total += -0.5 * np.sum(perp_s**2 / var_perp_s + np.log(2 * np.pi * var_perp_s))

    # Upper limits
    r_ul_n = r_n_func(ul_mjd)
    r_ul_s = np.abs(r_s_func(ul_mjd))
    excess_n = r_ul_n - ul_r
    excess_s = r_ul_s - ul_r
    if np.any(excess_n > 0):
        ll_total += -0.5 * np.sum((excess_n[excess_n > 0] / ul_sigma)**2)
    if np.any(excess_s > 0):
        ll_total += -0.5 * np.sum((excess_s[excess_s > 0] / ul_sigma)**2)

    return ll_total

# 3. BENCHMARK: MODEL 1 (BALLISTIC) AND MODEL 4 (PIECEWISE)
# Load existing Model 1 and Model 4 results from NPZ to ensure exact consistency
d1 = np.load("data/absolute_kinematics_plot_data_model1_with_limits.npz")
d4 = np.load("data/absolute_kinematics_plot_data_model4_with_limits.npz")

from src.plot_absolute_separation import unpack_params_abs, calc_trajectory
p1 = np.median(d1["flat_samples"], axis=0)
(t_ej_1, v1_n_1, v2_n_1, tb_n_1, v1_s_1, v2_s_1, tb_s_1, pa_n_1, pa_s_1, scat_1, xc_1, yc_1) = unpack_params_abs(p1, 1)

def traj_m1_n(t): return calc_trajectory(t, t_ej_1, v1_n_1, v2_n_1, tb_n_1)
def traj_m1_s(t): return calc_trajectory(t, t_ej_1, v1_s_1, v2_s_1, tb_s_1)
ll_m1 = log_likelihood_general(traj_m1_n, traj_m1_s, pa_n_1, pa_s_1, scat_1, xc_1, yc_1)
k_m1 = 8  # t_ej, v_n, v_s, pa_n, pa_s, theta_scat, xc, yc
bic_m1 = k_m1 * np.log(num_obs_total) - 2 * ll_m1

p4 = np.median(d4["flat_samples"], axis=0)
(t_ej_4, v1_n_4, v2_n_4, tb_n_4, v1_s_4, v2_s_4, tb_s_4, pa_n_4, pa_s_4, scat_4, xc_4, yc_4) = unpack_params_abs(p4, 4, anchor_date=float(d4["anchor_date"]))

def traj_m4_n(t): return calc_trajectory(t, t_ej_4, v1_n_4, v2_n_4, tb_n_4)
def traj_m4_s(t): return calc_trajectory(t, t_ej_4, v1_s_4, v2_s_4, tb_s_4)
ll_m4 = log_likelihood_general(traj_m4_n, traj_m4_s, pa_n_4, pa_s_4, scat_4, xc_4, yc_4)
k_m4 = 11  # v1_n, v2_n, tb_n, v1_s, v2_s, tb_s, pa_n, pa_s, theta_scat, xc, yc (t_ej anchored)
bic_m4 = k_m4 * np.log(num_obs_total) - 2 * ll_m4

print("\n--- BENCHMARK MODELS ---")
print(f"Model 1 (Ballistic): Log-Lik = {ll_m1:.2f}, k = {k_m1}, BIC = {bic_m1:.2f}")
print(f"Model 4 (Piecewise): Log-Lik = {ll_m4:.2f}, k = {k_m4}, BIC = {bic_m4:.2f}")

# 4. FIT SMOOTH CONTINUOUS DECELERATION MODEL
# r(t) = r_max * (1 - exp(-(t - t_ej) / tau))
# South has negative separation along the jet axis!
# Parameters:
# [rmax_n, tau_n, rmax_s, tau_s, pa_n, pa_s, theta_scatter, xc, yc]
# Anchor t_ej = 60645.47 (January 2025 flare)
t_ej_anchor = 60645.47

def smooth_traj(t, rmax, tau, t_ej):
    t = np.asarray(t)
    r = np.zeros_like(t, dtype=float)
    mask = t >= t_ej
    dt = t[mask] - t_ej
    r[mask] = rmax * (1.0 - np.exp(-dt / tau))
    return r

def obj_smooth_anchored(params):
    rmax_n, tau_n, rmax_s, tau_s, pa_n, pa_s, theta_scat, xc, yc = params
    # Priors / bounds penalty
    if rmax_n <= 0 or tau_n <= 5 or tau_n > 1000:
        return 1e9
    if rmax_s >= 0 or tau_s <= 5 or tau_s > 1000:  # rmax_s is negative
        return 1e9
    if theta_scat <= 0.5 or theta_scat > 30.0:
        return 1e9

    def r_n(t): return smooth_traj(t, rmax_n, tau_n, t_ej_anchor)
    def r_s(t): return smooth_traj(t, rmax_s, tau_s, t_ej_anchor)

    ll = log_likelihood_general(r_n, r_s, pa_n, pa_s, theta_scat, xc, yc)
    return -ll

# Multi-start optimization
best_res = None
best_nll = 1e9

for init_tau in [50.0, 100.0, 150.0, 250.0]:
    for init_rmax_n in [3.5, 4.5, 6.0]:
        for init_rmax_s in [-4.0, -5.0, -6.5]:
            p0 = [init_rmax_n, init_tau, init_rmax_s, init_tau, -41.0, -42.0, 5.5, -0.19, -0.30]
            res = minimize(obj_smooth_anchored, p0, method="Nelder-Mead", options={"maxiter": 4000})
            if res.fun < best_nll:
                best_nll = res.fun
                best_res = res

# Refine with Powell/BFGS
res_final = minimize(obj_smooth_anchored, best_res.x, method="Powell", options={"maxiter": 3000})
if res_final.fun < best_nll:
    best_res = res_final
    best_nll = res_final.fun

p_opt = best_res.x
rmax_n_opt, tau_n_opt, rmax_s_opt, tau_s_opt, pa_n_opt, pa_s_opt, scat_opt, xc_opt, yc_opt = p_opt
ll_smooth = -best_nll
k_smooth = 9  # rmax_n, tau_n, rmax_s, tau_s, pa_n, pa_s, theta_scat, xc, yc
bic_smooth = k_smooth * np.log(num_obs_total) - 2 * ll_smooth

v0_n = (rmax_n_opt / tau_n_opt) * 1000.0  # mas/day
v0_s = (abs(rmax_s_opt) / tau_s_opt) * 1000.0  # mas/day

print("\n--- SMOOTH DECELERATION TO STALL (Anchored Tej = 60645.5) ---")
print(f"Log-Likelihood: {ll_smooth:.2f}")
print(f"Parameters (k): {k_smooth}")
print(f"BIC:            {bic_smooth:.2f} (Delta BIC vs M1: {bic_smooth - bic_m1:+.2f}, vs M4: {bic_smooth - bic_m4:+.2f})")
print(f"North: r_max = {rmax_n_opt:.3f}\", tau = {tau_n_opt:.1f} d, initial v0 = {v0_n:.1f} mas/d")
print(f"South: r_max = {abs(rmax_s_opt):.3f}\", tau = {tau_s_opt:.1f} d, initial v0 = {v0_s:.1f} mas/d")
print(f"PA North: {pa_n_opt:.2f} deg, PA South: {pa_s_opt:.2f} deg, Scatter: {scat_opt:.2f} deg")

# 5. ALSO TEST HYBRID: NORTH PIECEWISE (or Ballistic), SOUTH SMOOTH STALL
# (Since the Southern lobe is the one exhibiting the pronounced plateau)
def obj_hybrid(params):
    v1_n, v2_n, tb_n, rmax_s, tau_s, pa_n, pa_s, theta_scat, xc, yc = params
    if v1_n <= 0 or v2_n <= 0 or tb_n <= t_ej_anchor:
        return 1e9
    if rmax_s >= 0 or tau_s <= 5 or tau_s > 1000:
        return 1e9
    if theta_scat <= 0.5 or theta_scat > 30.0:
        return 1e9

    def r_n(t): return calc_trajectory(t, t_ej_anchor, v1_n, v2_n, tb_n)
    def r_s(t): return smooth_traj(t, rmax_s, tau_s, t_ej_anchor)

    ll = log_likelihood_general(r_n, r_s, pa_n, pa_s, theta_scat, xc, yc)
    return -ll

res_hyb = minimize(obj_hybrid, [0.015, 0.007, 60775.0, -4.5, 120.0, -41.0, -42.0, 5.5, -0.19, -0.30],
                   method="Nelder-Mead", options={"maxiter": 4000})
ll_hyb = -res_hyb.fun
k_hyb = 10
bic_hyb = k_hyb * np.log(num_obs_total) - 2 * ll_hyb
print(f"\n--- HYBRID MODEL (North Piecewise, South Smooth Stall) ---")
print(f"Log-Likelihood: {ll_hyb:.2f}, k = {k_hyb}, BIC = {bic_hyb:.2f} (Delta BIC vs M1: {bic_hyb - bic_m1:+.2f})")

# 6. GENERATE VISUAL DIAGNOSTIC FIGURE
t_eval = np.linspace(60475, 61030, 600)
r_m1_n_eval = traj_m1_n(t_eval)
r_m1_s_eval = traj_m1_s(t_eval)

r_m4_n_eval = traj_m4_n(t_eval)
r_m4_s_eval = traj_m4_s(t_eval)

r_sm_n_eval = smooth_traj(t_eval, rmax_n_opt, tau_n_opt, t_ej_anchor)
r_sm_s_eval = smooth_traj(t_eval, rmax_s_opt, tau_s_opt, t_ej_anchor)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"hspace": 0.08})

# Northern
th_n = np.radians(pa_n_opt)
n_par = (n_data[:, 1] - xc_opt) * np.sin(th_n) + (n_data[:, 2] - yc_opt) * np.cos(th_n)
n_err = np.sqrt((n_data[:, 3] * np.sin(th_n))**2 + (n_data[:, 4] * np.cos(th_n))**2)

# Southern
th_s = np.radians(pa_s_opt)
s_par = (s_data[:, 1] - xc_opt) * np.sin(th_s) + (s_data[:, 2] - yc_opt) * np.cos(th_s)
s_err = np.sqrt((s_data[:, 3] * np.sin(th_s))**2 + (s_data[:, 4] * np.cos(th_s))**2)

# Plot Trajectories
ax1.axhline(0, color="gray", linestyle=":", lw=0.8)
ax1.plot(t_eval, r_m1_n_eval, color="gray", linestyle="--", lw=1.5, label=f"Model 1: Ballistic (BIC={bic_m1:.1f})")
ax1.plot(t_eval, r_m4_n_eval, color="orange", linestyle="-", lw=1.8, label=f"Model 4: Piecewise (BIC={bic_m4:.1f})")
ax1.plot(t_eval, r_sm_n_eval, color="purple", linestyle="-", lw=2.0, label=f"Smooth Stall (BIC={bic_smooth:.1f})")

ax1.plot(t_eval, r_m1_s_eval, color="gray", linestyle="--", lw=1.5)
ax1.plot(t_eval, r_m4_s_eval, color="orange", linestyle="-", lw=1.8)
ax1.plot(t_eval, r_sm_s_eval, color="purple", linestyle="-", lw=2.0)

ax1.errorbar(n_data[:, 0], n_par, yerr=n_err, fmt="^", color="dodgerblue", markeredgecolor="black", capsize=0, zorder=5, label=r"North Data ($\theta_\parallel^N$)")
ax1.errorbar(s_data[:, 0], s_par, yerr=s_err, fmt="v", color="crimson", markeredgecolor="black", capsize=0, zorder=5, label=r"South Data ($\theta_\parallel^S$)")

# Limits
ax1.errorbar(ul_mjd, ul_r, yerr=0.4, uplims=True, fmt="none", color="black", capsize=3, lw=1.2, zorder=6, label="Upper Limits")
ax1.errorbar(ul_mjd, -ul_r, yerr=0.4, lolims=True, fmt="none", color="black", capsize=3, lw=1.2, zorder=6)

ax1.set_ylabel(r"Separation ($^{\prime\prime}$)", fontsize=12)
ax1.legend(loc="upper left", fontsize=9, framealpha=0.9, ncol=2)
ax1.set_ylim(-5.5, 4.5)
ax1.grid(True, linestyle=":", alpha=0.5)
ax1.set_title("Comparison: Ballistic vs Piecewise Deceleration vs Smooth Stall Model", fontsize=12, fontweight="bold")

# Plot Instantaneous Velocities v(t)
v_m1_n = np.full_like(t_eval, v1_n_1 * 1000.0)
v_m1_s = np.full_like(t_eval, abs(v1_s_1) * 1000.0)

v_m4_n = np.where(t_eval < tb_n_4, v1_n_4, v2_n_4) * 1000.0
v_m4_s = np.where(t_eval < tb_s_4, abs(v1_s_4), abs(v2_s_4)) * 1000.0

# Smooth v(t) = v0 * exp(-dt / tau)
dt_eval = np.maximum(0.0, t_eval - t_ej_anchor)
v_sm_n = np.where(t_eval >= t_ej_anchor, (rmax_n_opt / tau_n_opt) * np.exp(-dt_eval / tau_n_opt) * 1000.0, 0.0)
v_sm_s = np.where(t_eval >= t_ej_anchor, (abs(rmax_s_opt) / tau_s_opt) * np.exp(-dt_eval / tau_s_opt) * 1000.0, 0.0)

ax2.plot(t_eval, v_m1_n, color="gray", linestyle="--", lw=1.2)
ax2.plot(t_eval, v_m4_n, color="orange", linestyle="-", lw=1.5, label="North Velocity")
ax2.plot(t_eval, v_sm_n, color="purple", linestyle="-", lw=1.8)

ax2.plot(t_eval, v_m1_s, color="gray", linestyle=":", lw=1.2)
ax2.plot(t_eval, v_m4_s, color="orange", linestyle="--", lw=1.5, label="South Velocity")
ax2.plot(t_eval, v_sm_s, color="purple", linestyle="--", lw=1.8)

ax2.set_xlabel("Time (MJD)", fontsize=12)
ax2.set_ylabel(r"Apparent Proper Motion $\mu$ (mas d$^{-1}$)", fontsize=11)
ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax2.set_ylim(-1, 28)
ax2.grid(True, linestyle=":", alpha=0.5)

out_fig = "figures/test_smooth_stall_comparison.png"
plt.savefig(out_fig, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved diagnostic figure to {out_fig}")
