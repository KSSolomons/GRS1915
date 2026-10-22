"""
Pipeline for Absolute Kinematics Modeling of GRS 1915+105.

This module models the jet kinematics based on the absolute coordinates (RA and Dec)
of the components relative to a fixed catalog position on the sky. Observational
upper limits can optionally be incorporated as a likelihood penalty to constrain
posteriors (controlled via --include-limits / --no-include-limits). It performs
MCMC sampling for parameter estimation and produces publication-ready astrometric
and flux evolution plots.
"""

import glob
import os
import re
import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")
import corner
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import argparse
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from epoch_limits import EJECTION_CAMPAIGN_START, blending_limits

UL_MJD, UL_R, _ = blending_limits()

# =========================================================================
# RUN CONFIGURATION (Edit N_STEPS directly here when running from script)
# =========================================================================
N_STEPS = 25000  # Number of MCMC steps (set e.g. 2000 for fast test runs)
# =========================================================================

parser = argparse.ArgumentParser(description="Pipeline for Absolute Kinematics Modeling with Limits")
parser.add_argument("--n-steps", type=int, default=N_STEPS, help="Number of MCMC steps")
parser.add_argument(
    "--anchor-date",
    type=float,
    default=60600.0,
    help="Anchored launch epoch (MJD) for Models 2 & 4 (default: 60600.0, Oct 2024 outburst onset)",
)
parser.add_argument(
    "--anchor-std",
    type=float,
    default=None,
    help="Gaussian prior standard deviation (days) for anchored launch epoch in Models 2 & 4. If None or <=0, anchor is hard-fixed to --anchor-date (default: None)",
)
parser.add_argument(
    "--model-id",
    type=int,
    default=5,
    choices=[1, 2, 3, 4, 5, 6],
    help="Model ID to adopt for MCMC sampling (1: Ballistic Free Tej, 2: Ballistic Anchored Tej, 3: Piecewise Decel Free Tej, 4: Piecewise Decel Anchored Tej, 5: Smooth Decel Free Tej, 6: Smooth Decel Anchored Tej; default: 4)",
)
parser.add_argument(
    "--plot-separation",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Automatically generate parallel/perpendicular separation plot (default: True, use --no-plot-separation to skip)",
)
parser.add_argument(
    "--show-separation-limits",
    "--separation-limits",
    dest="show_separation_limits",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Display upper limit arrows on the separation plot (default: matches --include-limits)",
)
parser.add_argument(
    "--hide-limits",
    "--hide-separation-limits",
    dest="show_separation_limits",
    action="store_false",
    help="Convenience alias: Hide upper limits arrows on the separation plot",
)
parser.add_argument(
    "--include-limits",
    dest="include_limits",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Include observational upper-limit penalty in the likelihood (default: True, use --no-include-limits to run without)",
)
parser.add_argument(
    "--ul-sigma",
    type=float,
    default=0.05,
    help="Gaussian likelihood penalty width (arcsec) for position upper limits (default: 0.05 arcsec)",
)
parser.add_argument(
    "--hdi-prob",
    type=float,
    default=0.90,
    help="Credible interval mass for Highest Density Interval (default: 0.90 for 90%% HDI)",
)
parser.add_argument(
    "--enforce-subluminal",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enforce physical subluminal intrinsic speed constraint beta <= max_beta (default: False)",
)
parser.add_argument(
    "--max-beta",
    type=float,
    default=1.0,
    help="Maximum physical intrinsic jet velocity beta = v/c (default: 1.0)",
)
parser.add_argument(
    "--sys-dist",
    type=float,
    default=9.4,
    help="System distance in kpc for intrinsic velocity calculation and physical prior (default: 9.4)",
)
parser.add_argument(
    "--use-cached",
    action="store_true",
    default=False,
    help="Load cached MCMC flat_samples from data/ if available to quickly re-generate kinematics & plots",
)

parser.add_argument(
    "--all-models",
    action="store_true",
    help="Run all models (1-4) with and without limits via run_and_compile_all_models.py",
)
parser.add_argument(
    "--compile-only",
    action="store_true",
    help="Compile tables and generate comparison figures from existing reports without re-running MCMC",
)
args = parser.parse_args()

if getattr(args, "all_models", False) or getattr(args, "compile_only", False):
    runner_script = os.path.join(os.path.dirname(__file__), "run_and_compile_all_models.py")
    cmd = [
        sys.executable,
        runner_script,
        "--n-steps", str(args.n_steps),
        "--anchor-date", str(args.anchor_date),
    ]
    if args.anchor_std is not None and args.anchor_std > 0:
        cmd.extend(["--anchor-std", str(args.anchor_std)])
    if getattr(args, "compile_only", False):
        cmd.append("--compile-only")
    print(f"--> Delegating to full suite runner: {' '.join(cmd)}\n", flush=True)
    res = subprocess.run(cmd)
    sys.exit(res.returncode)

if args.show_separation_limits is None:
    args.show_separation_limits = args.include_limits
_lim_suffix = "_with_limits" if args.include_limits else "_without_limits"




def compute_hdi(samples, credible_mass=None):
    if credible_mass is None:
        credible_mass = getattr(args, "hdi_prob", 0.90) if "args" in globals() else 0.90
    valid_samples = samples[np.isfinite(samples)]
    if len(valid_samples) == 0:
        return [np.nan, np.nan, np.nan]
    sorted_samples = np.sort(valid_samples)
    window_size = int(np.ceil(credible_mass * len(sorted_samples)))
    if window_size == 0 or window_size > len(sorted_samples):
        return [sorted_samples[0], np.median(valid_samples), sorted_samples[-1]]
    window_widths = (
        sorted_samples[window_size - 1 :] - sorted_samples[: -window_size + 1]
    )
    min_idx = np.argmin(window_widths)
    hdi_lower = sorted_samples[min_idx]
    hdi_upper = sorted_samples[min_idx + window_size - 1]
    return [hdi_lower, np.median(valid_samples), hdi_upper]


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# 1. LOAD AND PARSE DATA INTO ABSOLUTE OFFSETS FROM CATALOG
dir_path = "S-band-fits"
files = sorted(glob.glob(os.path.join(dir_path, "*2D_Fitting*.txt")))

act_ra_deg = (19 + 15 / 60.0 + 11.558 / 3600.0) * 15.0
act_dec_deg = 10 + 56 / 60.0 + 44.905 / 3600.0
cos_dec = np.cos(np.radians(act_dec_deg))

core_data_raw = []
n_data = []
s_data = []

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
                h, m_time, s, err_s = (
                    float(m_ra.group(1)),
                    float(m_ra.group(2)),
                    float(m_ra.group(3)),
                    float(m_ra.group(4)),
                )
                curr_comp["ra_deg"] = (h + m_time / 60.0 + s / 3600.0) * 15.0
                curr_comp["ra_err"] = err_s * 15.0 * cos_dec  # arcsec
        elif "Center Y" in line and curr_comp is not None:
            m_dec = re.search(r"=\s*([+-]?\d+):(\d+):(\d+\.\d+)\s*±\s*(\S+)", line)
            if m_dec:
                d, m_arc, s_arc, err_arcsec = (
                    float(m_dec.group(1)),
                    float(m_dec.group(2)),
                    float(m_dec.group(3)),
                    float(m_dec.group(4)),
                )
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
            if ts != 1764664923:  # Exclude Epoch 12 jet measurements
                n_data.append([mjd, dx, dy, ex, ey, flux, flux_err])
        else:
            if ts != 1764664923:  # Exclude Epoch 12 jet measurements
                s_data.append([mjd, dx, dy, ex, ey, flux, flux_err])

# Dynamically filter outliers based on Median Absolute Deviation (MAD) of spatial distances
core_data = []
omitted_core_data = []
if len(core_data_raw) > 0:
    temp_core = np.array(core_data_raw)
    med_dx = np.median(temp_core[:, 1])
    med_dy = np.median(temp_core[:, 2])
    
    distances = np.sqrt((temp_core[:, 1] - med_dx)**2 + (temp_core[:, 2] - med_dy)**2)
    mad = np.median(np.abs(distances - np.median(distances)))
    
    # Use a robust threshold of 5 MAD (with a 0.4 arcsec minimum floor to prevent over-filtering)
    threshold = max(5 * mad, 0.4)
    
    for i, c in enumerate(core_data_raw):
        if distances[i] > threshold:
            omitted_core_data.append(c)
        else:
            core_data.append(c)

core_data = np.array(core_data)
omitted_core_data = np.array(omitted_core_data) if len(omitted_core_data) > 0 else np.empty((0, 7))
n_data = np.array(n_data)
s_data = np.array(s_data)

# Keep only the 2025 campaign jets (older pre-2025 ejecta are a separate event)
n_data = n_data[np.argsort(n_data[:, 0])]
s_data = s_data[np.argsort(s_data[:, 0])]
n_data = n_data[n_data[:, 0] > EJECTION_CAMPAIGN_START]
s_data = s_data[s_data[:, 0] > EJECTION_CAMPAIGN_START]
num_obs_total = 2 * len(core_data) + 2 * len(n_data) + 2 * len(s_data)

print(
    f"Loaded {len(core_data)} core points, {len(n_data)} N points, {len(s_data)} S points"
)# 2. UNIVERSAL TRAJECTORY AND LIKELIHOOD FUNCTION
def calc_trajectory(t, t_ej, v1, v2, t_break):
    """
    Computes trajectory r(t) for piecewise linear motion.
    If v1 == v2 or t_break >= np.max(t), it reduces exactly to pure ballistic.
    t: array of MJD timestamps
    t_ej: launch epoch (MJD)
    v1: initial proper motion (arcsec/day)
    v2: post-break proper motion (arcsec/day)
    t_break: transition break epoch (MJD)
    """
    t = np.asarray(t)
    dt = t - t_ej
    r = np.zeros_like(dt, dtype=float)
    mask1 = (t >= t_ej) & (t <= t_break)
    r[mask1] = v1 * (t[mask1] - t_ej)
    mask2 = t > t_break
    r_break = v1 * max(0.0, t_break - t_ej)
    r[mask2] = r_break + v2 * (t[mask2] - t_break)
    return r


def calc_trajectory_smooth(t, t_ej, v0, tau):
    """
    Computes trajectory r(t) for smooth continuous deceleration to terminal stall radius:
    r(t) = (v0 * tau) * (1 - exp(-(t - t_ej) / tau)) for t >= t_ej, 0 otherwise.
    For southern approaching lobe, v0 is negative.
    """
    t = np.asarray(t)
    dt = t - t_ej
    r = np.zeros_like(dt, dtype=float)
    mask = t >= t_ej
    r[mask] = (v0 * tau) * (1.0 - np.exp(-(t[mask] - t_ej) / tau))
    return r


def calc_log_likelihood(
    t_ej,
    v1_n,
    v2_n,
    tb_n,
    v1_s,
    v2_s,
    tb_s,
    pa_n,
    pa_s,
    theta_scatter,
    x_c,
    y_c,
    include_limits=True,
    ul_sigma=None,
):
    tan2_scatter = np.tan(np.radians(theta_scatter)) ** 2

    ll_total = 0.0

    # --- CORE LIKELIHOOD (Standard Gaussian) ---
    dx_c = core_data[:, 1] - x_c
    dy_c = core_data[:, 2] - y_c
    ex_c = core_data[:, 3]
    ey_c = core_data[:, 4]

    ll_in_x = -0.5 * (dx_c**2 / ex_c**2 + np.log(2 * np.pi * ex_c**2))
    ll_in_y = -0.5 * (dy_c**2 / ey_c**2 + np.log(2 * np.pi * ey_c**2))
    ll_total += np.sum(ll_in_x + ll_in_y)

    # --- NORTHERN LOBE LIKELIHOOD ---
    dx_n = n_data[:, 1] - x_c
    dy_n = n_data[:, 2] - y_c
    ex_n = n_data[:, 3]
    ey_n = n_data[:, 4]

    theta_n = np.radians(pa_n)
    sin_n, cos_n = np.sin(theta_n), np.cos(theta_n)
    par_n = dx_n * sin_n + dy_n * cos_n
    perp_n = dx_n * cos_n - dy_n * sin_n

    model_par_n = calc_trajectory(n_data[:, 0], t_ej, v1_n, v2_n, tb_n)

    var_par_n = (ex_n * sin_n) ** 2 + (ey_n * cos_n) ** 2
    var_perp_n = (
        (ex_n * cos_n) ** 2 + (ey_n * sin_n) ** 2 + model_par_n**2 * tan2_scatter
    )

    ll_n = -0.5 * np.sum(
        (par_n - model_par_n) ** 2 / var_par_n + np.log(2 * np.pi * var_par_n)
    )
    ll_n += -0.5 * np.sum(perp_n**2 / var_perp_n + np.log(2 * np.pi * var_perp_n))
    ll_total += ll_n

    # --- SOUTHERN LOBE LIKELIHOOD ---
    dx_s = s_data[:, 1] - x_c
    dy_s = s_data[:, 2] - y_c
    ex_s = s_data[:, 3]
    ey_s = s_data[:, 4]

    theta_s = np.radians(pa_s)
    sin_s, cos_s = np.sin(theta_s), np.cos(theta_s)
    par_s = dx_s * sin_s + dy_s * cos_s
    perp_s = dx_s * cos_s - dy_s * sin_s

    model_par_s = calc_trajectory(s_data[:, 0], t_ej, v1_s, v2_s, tb_s)

    var_par_s = (ex_s * sin_s) ** 2 + (ey_s * cos_s) ** 2
    var_perp_s = (
        (ex_s * cos_s) ** 2 + (ey_s * sin_s) ** 2 + model_par_s**2 * tan2_scatter
    )

    ll_s = -0.5 * np.sum(
        (par_s - model_par_s) ** 2 / var_par_s + np.log(2 * np.pi * var_par_s)
    )
    ll_s += -0.5 * np.sum(perp_s**2 / var_perp_s + np.log(2 * np.pi * var_perp_s))
    ll_total += ll_s

    # --- UPPER LIMITS LIKELIHOOD PENALTY ---
    if include_limits:
        ul_mjd, ul_r = UL_MJD, UL_R
        model_r_n = calc_trajectory(ul_mjd, t_ej, v1_n, v2_n, tb_n)
        model_r_s = np.abs(calc_trajectory(ul_mjd, t_ej, v1_s, v2_s, tb_s))

        if ul_sigma is not None:
            ul_sigma_arr = np.atleast_1d(ul_sigma)
            if len(ul_sigma_arr) == 1:
                ul_sigma_arr = np.full_like(ul_r, ul_sigma_arr[0])
        else:
            ul_sigma_val = getattr(args, "ul_sigma", 0.05)
            ul_sigma_arr = np.full_like(ul_r, ul_sigma_val)
        excess_n = model_r_n - ul_r
        mask_n = excess_n > 0
        if np.any(mask_n):
            ll_total += -0.5 * np.sum((excess_n[mask_n] / ul_sigma_arr[mask_n]) ** 2)

        excess_s = model_r_s - ul_r
        mask_s = excess_s > 0
        if np.any(mask_s):
            ll_total += -0.5 * np.sum((excess_s[mask_s] / ul_sigma_arr[mask_s]) ** 2)

    return ll_total


def calc_log_likelihood_smooth(
    t_ej,
    v0_n,
    tau_n,
    v0_s,
    tau_s,
    pa_n,
    pa_s,
    theta_scatter,
    x_c,
    y_c,
    include_limits=True,
    ul_sigma=None,
):
    tan2_scatter = np.tan(np.radians(theta_scatter)) ** 2
    ll_total = 0.0

    # --- CORE LIKELIHOOD ---
    dx_c = core_data[:, 1] - x_c
    dy_c = core_data[:, 2] - y_c
    ex_c = core_data[:, 3]
    ey_c = core_data[:, 4]
    ll_total += np.sum(-0.5 * (dx_c**2 / ex_c**2 + np.log(2 * np.pi * ex_c**2)))
    ll_total += np.sum(-0.5 * (dy_c**2 / ey_c**2 + np.log(2 * np.pi * ey_c**2)))

    # --- NORTHERN LOBE LIKELIHOOD ---
    dx_n = n_data[:, 1] - x_c
    dy_n = n_data[:, 2] - y_c
    ex_n = n_data[:, 3]
    ey_n = n_data[:, 4]
    theta_n = np.radians(pa_n)
    sin_n, cos_n = np.sin(theta_n), np.cos(theta_n)
    par_n = dx_n * sin_n + dy_n * cos_n
    perp_n = dx_n * cos_n - dy_n * sin_n

    model_par_n = calc_trajectory_smooth(n_data[:, 0], t_ej, v0_n, tau_n)
    var_par_n = (ex_n * sin_n) ** 2 + (ey_n * cos_n) ** 2
    var_perp_n = (
        (ex_n * cos_n) ** 2 + (ey_n * sin_n) ** 2 + model_par_n**2 * tan2_scatter
    )
    ll_total += -0.5 * np.sum(
        (par_n - model_par_n) ** 2 / var_par_n + np.log(2 * np.pi * var_par_n)
    )
    ll_total += -0.5 * np.sum(perp_n**2 / var_perp_n + np.log(2 * np.pi * var_perp_n))

    # --- SOUTHERN LOBE LIKELIHOOD ---
    dx_s = s_data[:, 1] - x_c
    dy_s = s_data[:, 2] - y_c
    ex_s = s_data[:, 3]
    ey_s = s_data[:, 4]
    theta_s = np.radians(pa_s)
    sin_s, cos_s = np.sin(theta_s), np.cos(theta_s)
    par_s = dx_s * sin_s + dy_s * cos_s
    perp_s = dx_s * cos_s - dy_s * sin_s

    model_par_s = calc_trajectory_smooth(s_data[:, 0], t_ej, v0_s, tau_s)
    var_par_s = (ex_s * sin_s) ** 2 + (ey_s * cos_s) ** 2
    var_perp_s = (
        (ex_s * cos_s) ** 2 + (ey_s * sin_s) ** 2 + model_par_s**2 * tan2_scatter
    )
    ll_total += -0.5 * np.sum(
        (par_s - model_par_s) ** 2 / var_par_s + np.log(2 * np.pi * var_par_s)
    )
    ll_total += -0.5 * np.sum(perp_s**2 / var_perp_s + np.log(2 * np.pi * var_perp_s))

    # --- UPPER LIMITS LIKELIHOOD PENALTY ---
    if include_limits:
        ul_mjd, ul_r = UL_MJD, UL_R
        model_r_n = calc_trajectory_smooth(ul_mjd, t_ej, v0_n, tau_n)
        model_r_s = np.abs(calc_trajectory_smooth(ul_mjd, t_ej, v0_s, tau_s))

        if ul_sigma is not None:
            ul_sigma_arr = np.atleast_1d(ul_sigma)
            if len(ul_sigma_arr) == 1:
                ul_sigma_arr = np.full_like(ul_r, ul_sigma_arr[0])
        else:
            ul_sigma_val = getattr(args, "ul_sigma", 0.05)
            ul_sigma_arr = np.full_like(ul_r, ul_sigma_val)
        excess_n = model_r_n - ul_r
        mask_n = excess_n > 0
        if np.any(mask_n):
            ll_total += -0.5 * np.sum((excess_n[mask_n] / ul_sigma_arr[mask_n]) ** 2)

        excess_s = model_r_s - ul_r
        mask_s = excess_s > 0
        if np.any(mask_s):
            ll_total += -0.5 * np.sum((excess_s[mask_s] / ul_sigma_arr[mask_s]) ** 2)

    return ll_total


def log_prior_universal(
    t_ej,
    v1_n,
    v2_n,
    tb_n,
    v1_s,
    v2_s,
    tb_s,
    pa_n,
    pa_s,
    theta_scatter,
    x_c,
    y_c,
    is_anchored=False,
    is_piecewise=False,
    anchor_date=60600.0,
    anchor_std=None,
    enforce_subluminal=None,
    max_beta=None,
    sys_dist_kpc=None,
):
    if enforce_subluminal is None:
        enforce_subluminal = getattr(args, "enforce_subluminal", False) if "args" in globals() else False
    if max_beta is None:
        max_beta = getattr(args, "max_beta", 1.0) if "args" in globals() else 1.0
    if sys_dist_kpc is None:
        sys_dist_kpc = getattr(args, "sys_dist", 9.4) if "args" in globals() else 9.4

    effective_tej = anchor_date if (is_anchored and (anchor_std is None or anchor_std <= 0)) else t_ej
    lp = 0.0
    if not (is_anchored and (anchor_std is None or anchor_std <= 0)):
        if not (60000.0 < t_ej < 60800.0):
            return -np.inf
        if is_anchored and anchor_std is not None and anchor_std > 0:
            lp += -0.5 * ((t_ej - anchor_date) / anchor_std) ** 2

    if is_piecewise:
        if not (effective_tej < tb_n < 60850.0 and effective_tej < tb_s < 60850.0):
            return -np.inf
        if not (1e-4 < v1_n < 0.5 and 1e-4 < v2_n < 0.5):
            return -np.inf
        if not (-0.5 < v1_s < -1e-4 and -0.5 < v2_s < -1e-4):
            return -np.inf
        # Physical condition: Southern lobe is always approaching at launch (|v1_s| > v1_n)
        if not (np.abs(v1_s) > v1_n):
            return -np.inf
        # Physical condition: Deceleration upon encountering ISM
        if not (v2_n <= v1_n and np.abs(v2_s) <= np.abs(v1_s)):
            return -np.inf
    else:
        if not (1e-4 < v1_n < 0.5 and -0.5 < v1_s < -1e-4):
            return -np.inf
        # Physical condition: Southern lobe is always approaching (|v_s| > v_n)
        if not (np.abs(v1_s) > v1_n):
            return -np.inf

    if not (-180.0 < pa_n < 180.0 and -180.0 < pa_s < 180.0):
        return -np.inf
    if not (0.0 <= theta_scatter <= 45.0):
        return -np.inf
    if not (-2.0 < x_c < 2.0 and -2.0 < y_c < 2.0):
        return -np.inf

    # Relativistic subluminal constraint: true intrinsic launch velocity beta <= max_beta
    if enforce_subluminal:
        mu_app = np.abs(v1_s)
        mu_rec = v1_n
        if (mu_app + mu_rec) > 0:
            bct = (mu_app - mu_rec) / (mu_app + mu_rec)
            mu_app_rad_s = mu_app * (np.pi / (180.0 * 3600.0)) / 86400.0
            bst = mu_app_rad_s * (1.0 - bct) * (sys_dist_kpc * 3.085677581e19) / 2.99792458e8
            beta_sq = bct**2 + bst**2
            if beta_sq >= (max_beta ** 2):
                return -np.inf
    return lp


def log_prior_smooth(
    t_ej,
    v0_n,
    tau_n,
    v0_s,
    tau_s,
    pa_n,
    pa_s,
    theta_scatter,
    x_c,
    y_c,
    is_anchored=False,
    anchor_date=60600.0,
    anchor_std=None,
    enforce_subluminal=None,
    max_beta=None,
    sys_dist_kpc=None,
):
    if enforce_subluminal is None:
        enforce_subluminal = (
            getattr(args, "enforce_subluminal", False) if "args" in globals() else False
        )
    if max_beta is None:
        max_beta = getattr(args, "max_beta", 1.0) if "args" in globals() else 1.0
    if sys_dist_kpc is None:
        sys_dist_kpc = (
            getattr(args, "sys_dist", 9.4) if "args" in globals() else 9.4
        )

    effective_tej = (
        anchor_date if (is_anchored and (anchor_std is None or anchor_std <= 0)) else t_ej
    )
    lp = 0.0
    if not (is_anchored and (anchor_std is None or anchor_std <= 0)):
        if not (60000.0 < t_ej < 60800.0):
            return -np.inf
        if is_anchored and anchor_std is not None and anchor_std > 0:
            lp += -0.5 * ((t_ej - anchor_date) / anchor_std) ** 2

    if not (1e-4 < v0_n < 0.5 and 5.0 < tau_n < 5000.0):
        return -np.inf
    if not (-0.5 < v0_s < -1e-4 and 5.0 < tau_s < 5000.0):
        return -np.inf

    # Physical condition: Southern lobe is approaching at launch (|v0_s| > v0_n)
    if not (np.abs(v0_s) > v0_n):
        return -np.inf

    # Physical condition: Terminal stall radius doesn't shoot to unphysical extremes
    rmax_n = v0_n * tau_n
    rmax_s = np.abs(v0_s) * tau_s
    if not (0.05 < rmax_n < 20.0 and 0.05 < rmax_s < 20.0):
        return -np.inf

    if not (-180.0 < pa_n < 180.0 and -180.0 < pa_s < 180.0):
        return -np.inf
    if not (0.0 <= theta_scatter <= 45.0):
        return -np.inf
    if not (-2.0 < x_c < 2.0 and -2.0 < y_c < 2.0):
        return -np.inf

    if enforce_subluminal:
        mu_app = np.abs(v0_s)
        mu_rec = v0_n
        if (mu_app + mu_rec) > 0:
            bct = (mu_app - mu_rec) / (mu_app + mu_rec)
            mu_app_rad_s = mu_app * (np.pi / (180.0 * 3600.0)) / 86400.0
            bst = (
                mu_app_rad_s
                * (1.0 - bct)
                * (sys_dist_kpc * 3.085677581e19)
                / 2.99792458e8
            )
            beta_sq = bct**2 + bst**2
            if beta_sq >= (max_beta**2):
                return -np.inf

    return lp


# 3. QUANTITATIVE MODEL SELECTION (BIC & AIC COMPARISONS)
print("\n" + "=" * 70)
print(f"{'QUANTITATIVE MODEL SELECTION (BIC / AIC SUITE)':^70}")
print("=" * 70)

# Model 1: Ballistic, Free Tej (k=8)
# p = [t_ej, vn, vs, pa_n, pa_s, theta_scatter, x_c, y_c]
def nll_m1(p):
    lp = log_prior_universal(
        p[0], p[1], p[1], 1e6, p[2], p[2], 1e6, p[3], p[4], p[5], p[6], p[7],
        is_anchored=False, is_piecewise=False, anchor_date=args.anchor_date,
    )
    if not np.isfinite(lp):
        return 1e12
    return -(
        lp
        + calc_log_likelihood(
            p[0], p[1], p[1], 1e6, p[2], p[2], 1e6, p[3], p[4], p[5], p[6], p[7],
            include_limits=args.include_limits,
        )
    )

res_m1 = minimize(
    nll_m1,
    [60366.0, 0.0053, -0.0073, -42.8, -41.4, 4.4, -0.1914, -0.2984],
    method="Nelder-Mead",
    options={"maxiter": 10000},
)
ll_m1 = calc_log_likelihood(
    res_m1.x[0], res_m1.x[1], res_m1.x[1], 1e6, res_m1.x[2], res_m1.x[2], 1e6,
    res_m1.x[3], res_m1.x[4], res_m1.x[5], res_m1.x[6], res_m1.x[7],
    include_limits=args.include_limits,
)
k_m1 = 8
bic_m1 = k_m1 * np.log(num_obs_total) - 2 * ll_m1


# Model 2: Ballistic, Anchored Tej (k=7 or 8)
if args.anchor_std is not None and args.anchor_std > 0:
    # Model 2 with Gaussian anchor: p = [t_ej, vn, vs, pa_n, pa_s, theta_scatter, x_c, y_c]
    def nll_m2(p):
        lp = log_prior_universal(
            p[0], p[1], p[1], 1e6, p[2], p[2], 1e6, p[3], p[4], p[5], p[6], p[7],
            is_anchored=True, is_piecewise=False, anchor_date=args.anchor_date, anchor_std=args.anchor_std,
        )
        if not np.isfinite(lp):
            return 1e12
        return -(
            lp
            + calc_log_likelihood(
                p[0], p[1], p[1], 1e6, p[2], p[2], 1e6, p[3], p[4], p[5], p[6], p[7],
                include_limits=args.include_limits,
            )
        )

    res_m2 = minimize(
        nll_m2,
        [args.anchor_date, 0.0100, -0.0139, -41.7, -40.0, 4.4, -0.1914, -0.2984],
        method="Nelder-Mead",
        options={"maxiter": 10000},
    )
    ll_m2 = calc_log_likelihood(
        res_m2.x[0], res_m2.x[1], res_m2.x[1], 1e6, res_m2.x[2], res_m2.x[2], 1e6,
        res_m2.x[3], res_m2.x[4], res_m2.x[5], res_m2.x[6], res_m2.x[7],
        include_limits=args.include_limits,
    )
    k_m2 = 8
else:
    # Model 2 with hard-anchored Tej: p = [vn, vs, pa_n, pa_s, theta_scatter, x_c, y_c]
    def nll_m2(p):
        lp = log_prior_universal(
            args.anchor_date, p[0], p[0], 1e6, p[1], p[1], 1e6, p[2], p[3], p[4], p[5], p[6],
            is_anchored=True, is_piecewise=False, anchor_date=args.anchor_date,
        )
        if not np.isfinite(lp):
            return 1e12
        return -(
            lp
            + calc_log_likelihood(
                args.anchor_date, p[0], p[0], 1e6, p[1], p[1], 1e6, p[2], p[3], p[4], p[5], p[6],
                include_limits=args.include_limits,
            )
        )

    res_m2 = minimize(
        nll_m2,
        [0.0100, -0.0139, -41.7, -40.0, 4.4, -0.1914, -0.2984],
        method="Nelder-Mead",
        options={"maxiter": 10000},
    )
    ll_m2 = calc_log_likelihood(
        args.anchor_date, res_m2.x[0], res_m2.x[0], 1e6, res_m2.x[1], res_m2.x[1], 1e6,
        res_m2.x[2], res_m2.x[3], res_m2.x[4], res_m2.x[5], res_m2.x[6],
        include_limits=args.include_limits,
    )
    k_m2 = 7
bic_m2 = k_m2 * np.log(num_obs_total) - 2 * ll_m2


# Model 3: Piecewise Deceleration, Free Tej (k=12)
# p = [t_ej, v1n, v2n, tb_n, v1s, v2s, tb_s, pa_n, pa_s, theta_scatter, x_c, y_c]
def nll_m3(p):
    lp = log_prior_universal(
        p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11],
        is_anchored=False, is_piecewise=True, anchor_date=args.anchor_date,
    )
    if not np.isfinite(lp):
        return 1e12
    return -(
        lp
        + calc_log_likelihood(
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11],
            include_limits=args.include_limits,
        )
    )

init_m3 = [
    res_m1.x[0], 0.0121, 0.0072, 60752.0, -0.0238, -0.0054, 60719.0,
    res_m1.x[3], res_m1.x[4], res_m1.x[5], res_m1.x[6], res_m1.x[7],
]
res_m3 = minimize(
    nll_m3,
    init_m3,
    method="Nelder-Mead",
    options={"maxiter": 25000},
)
ll_m3 = calc_log_likelihood(*res_m3.x, include_limits=args.include_limits)
k_m3 = 12
bic_m3 = k_m3 * np.log(num_obs_total) - 2 * ll_m3


# Model 4: Piecewise Deceleration, Anchored Tej (k=11 or 12)
if args.anchor_std is not None and args.anchor_std > 0:
    # Model 4 with Gaussian anchor: p = [t_ej, v1n, v2n, tb_n, v1s, v2s, tb_s, pa_n, pa_s, theta_scatter, x_c, y_c]
    def nll_m4(p):
        lp = log_prior_universal(
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11],
            is_anchored=True, is_piecewise=True, anchor_date=args.anchor_date, anchor_std=args.anchor_std,
        )
        if not np.isfinite(lp):
            return 1e12
        return -(
            lp
            + calc_log_likelihood(
                p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11],
                include_limits=args.include_limits,
            )
        )

    init_m4 = [
        args.anchor_date, 0.0121, 0.0072, 60752.0, -0.0238, -0.0054, 60719.0,
        -42.5, -41.6, 4.4, -0.1914, -0.2984,
    ]
    res_m4 = minimize(
        nll_m4,
        init_m4,
        method="Nelder-Mead",
        options={"maxiter": 25000},
    )
    ll_m4 = calc_log_likelihood(
        *res_m4.x, include_limits=args.include_limits
    )
    k_m4 = 12
else:
    # Model 4 with hard-anchored Tej: p = [v1n, v2n, tb_n, v1s, v2s, tb_s, pa_n, pa_s, theta_scatter, x_c, y_c]
    def nll_m4(p):
        lp = log_prior_universal(
            args.anchor_date, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10],
            is_anchored=True, is_piecewise=True, anchor_date=args.anchor_date,
        )
        if not np.isfinite(lp):
            return 1e12
        return -(
            lp
            + calc_log_likelihood(
                args.anchor_date, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10],
                include_limits=args.include_limits,
            )
        )

    init_m4 = [
        0.0121, 0.0072, 60752.0, -0.0238, -0.0054, 60719.0,
        -42.5, -41.6, 4.4, -0.1914, -0.2984,
    ]
    res_m4 = minimize(
        nll_m4,
        init_m4,
        method="Nelder-Mead",
        options={"maxiter": 25000},
    )
    ll_m4 = calc_log_likelihood(
        args.anchor_date, *res_m4.x, include_limits=args.include_limits
    )
    k_m4 = 11
bic_m4 = k_m4 * np.log(num_obs_total) - 2 * ll_m4

# Model 5: Smooth Continuous Deceleration, Free Tej (k=10)
# p = [t_ej, rmax_n, tau_n, rmax_s, tau_s, pa_n, pa_s, theta_scatter, x_c, y_c]
def nll_m5(p):
    lp = log_prior_smooth(
        p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
        is_anchored=False, anchor_date=args.anchor_date,
    )
    if not np.isfinite(lp):
        return 1e12
    return -(
        lp
        + calc_log_likelihood_smooth(
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
            include_limits=args.include_limits,
        )
    )

best_res_m5 = None
best_nll_m5 = 1e12
trials_m5 = [
    (60660.0, res_m1.x[1], 400.0, res_m1.x[2], 300.0),
    (60600.0, res_m1.x[1], 500.0, res_m1.x[2], 400.0),
    (60550.0, res_m1.x[1], 800.0, res_m1.x[2], 600.0),
    (60500.0, 0.010, 600.0, -0.015, 450.0),
    (60600.0, 0.011, 350.0, -0.016, 250.0),
]
for init_t, v_n, tau_n, v_s, tau_s in trials_m5:
    init_m5 = [
        init_t, v_n, tau_n, v_s, tau_s,
        res_m1.x[3], res_m1.x[4], res_m1.x[5], res_m1.x[6], res_m1.x[7],
    ]
    if nll_m5(init_m5) >= 1e11:
        continue
    res_m5_trial = minimize(
        nll_m5,
        init_m5,
        method="Nelder-Mead",
        options={"maxiter": 20000},
    )
    if res_m5_trial.fun < best_nll_m5:
        best_nll_m5 = res_m5_trial.fun
        best_res_m5 = res_m5_trial

res_m5 = best_res_m5
ll_m5 = calc_log_likelihood_smooth(*res_m5.x, include_limits=args.include_limits)
k_m5 = 10
bic_m5 = k_m5 * np.log(num_obs_total) - 2 * ll_m5


# Model 6: Smooth Continuous Deceleration, Anchored Tej (k=9 or 10)
if args.anchor_std is not None and args.anchor_std > 0:
    # Model 6 with Gaussian anchor: p = [t_ej, v0_n, tau_n, v0_s, tau_s, pa_n, pa_s, theta_scatter, x_c, y_c]
    def nll_m6(p):
        lp = log_prior_smooth(
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
            is_anchored=True, anchor_date=args.anchor_date, anchor_std=args.anchor_std,
        )
        if not np.isfinite(lp):
            return 1e12
        return -(
            lp
            + calc_log_likelihood_smooth(
                p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
                include_limits=args.include_limits,
            )
        )

    best_res_m6 = None
    best_nll_m6 = 1e12
    trials_m6 = [
        (args.anchor_date, res_m1.x[1], 400.0, res_m1.x[2], 300.0),
        (args.anchor_date, res_m1.x[1], 800.0, res_m1.x[2], 600.0),
        (args.anchor_date, 0.010, 600.0, -0.015, 450.0),
    ]
    for init_t, v_n, tau_n, v_s, tau_s in trials_m6:
        init_m6 = [
            init_t, v_n, tau_n, v_s, tau_s,
            res_m1.x[3], res_m1.x[4], res_m1.x[5], res_m1.x[6], res_m1.x[7],
        ]
        if nll_m6(init_m6) >= 1e11:
            continue
        res_m6_trial = minimize(
            nll_m6,
            init_m6,
            method="Nelder-Mead",
            options={"maxiter": 25000},
        )
        if res_m6_trial.fun < best_nll_m6:
            best_nll_m6 = res_m6_trial.fun
            best_res_m6 = res_m6_trial
    res_m6 = best_res_m6
    ll_m6 = calc_log_likelihood_smooth(*res_m6.x, include_limits=args.include_limits)
    k_m6 = 10
else:
    # Model 6 with hard-anchored Tej: p = [v0_n, tau_n, v0_s, tau_s, pa_n, pa_s, theta_scatter, x_c, y_c]
    def nll_m6(p):
        lp = log_prior_smooth(
            args.anchor_date, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8],
            is_anchored=True, anchor_date=args.anchor_date,
        )
        if not np.isfinite(lp):
            return 1e12
        return -(
            lp
            + calc_log_likelihood_smooth(
                args.anchor_date, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8],
                include_limits=args.include_limits,
            )
        )

    best_res_m6 = None
    best_nll_m6 = 1e12
    trials_m6 = [
        (res_m1.x[1], 400.0, res_m1.x[2], 300.0),
        (res_m1.x[1], 800.0, res_m1.x[2], 600.0),
        (0.010, 600.0, -0.015, 450.0),
    ]
    for v_n, tau_n, v_s, tau_s in trials_m6:
        init_m6 = [
            v_n, tau_n, v_s, tau_s,
            res_m1.x[3], res_m1.x[4], res_m1.x[5], res_m1.x[6], res_m1.x[7],
        ]
        if nll_m6(init_m6) >= 1e11:
            continue
        res_m6_trial = minimize(
            nll_m6,
            init_m6,
            method="Nelder-Mead",
            options={"maxiter": 25000},
        )
        if res_m6_trial.fun < best_nll_m6:
            best_nll_m6 = res_m6_trial.fun
            best_res_m6 = res_m6_trial
    res_m6 = best_res_m6
    ll_m6 = calc_log_likelihood_smooth(
        args.anchor_date, *res_m6.x, include_limits=args.include_limits
    )
    k_m6 = 9
bic_m6 = k_m6 * np.log(num_obs_total) - 2 * ll_m6

_m2_anch_label = f"Tej={args.anchor_date:.0f}±{args.anchor_std:.0f}" if (args.anchor_std is not None and args.anchor_std > 0) else f"Tej={args.anchor_date:.0f}"
_m4_anch_label = f"Tej={args.anchor_date:.0f}±{args.anchor_std:.0f}" if (args.anchor_std is not None and args.anchor_std > 0) else f"Tej={args.anchor_date:.0f}"
_m6_anch_label = f"Tej={args.anchor_date:.0f}±{args.anchor_std:.0f}" if (args.anchor_std is not None and args.anchor_std > 0) else f"Tej={args.anchor_date:.0f}"

models = [
    {
        "id": 1,
        "name": "Ballistic (Free Tej)",
        "k": k_m1,
        "ll": ll_m1,
        "bic": bic_m1,
        "res": res_m1,
    },
    {
        "id": 2,
        "name": f"Ballistic (Anchored {_m2_anch_label})",
        "k": k_m2,
        "ll": ll_m2,
        "bic": bic_m2,
        "res": res_m2,
    },
    {
        "id": 3,
        "name": "Piecewise Deceleration (Free Tej)",
        "k": k_m3,
        "ll": ll_m3,
        "bic": bic_m3,
        "res": res_m3,
    },
    {
        "id": 4,
        "name": f"Piecewise Deceleration (Anchored {_m4_anch_label})",
        "k": k_m4,
        "ll": ll_m4,
        "bic": bic_m4,
        "res": res_m4,
    },
    {
        "id": 5,
        "name": "Smooth Continuous Deceleration (Free Tej)",
        "k": k_m5,
        "ll": ll_m5,
        "bic": bic_m5,
        "res": res_m5,
    },
    {
        "id": 6,
        "name": f"Smooth Continuous Deceleration (Anchored {_m6_anch_label})",
        "k": k_m6,
        "ll": ll_m6,
        "bic": bic_m6,
        "res": res_m6,
    },
]

if args.model_id is not None:
    best_model = next(m for m in models if m["id"] == args.model_id)
else:
    best_model = min(models, key=lambda m: m["bic"])

_model_suffix = f"_model{best_model['id']}"
_save_suffix = f"{_model_suffix}{_lim_suffix}"

min_bic = min(m["bic"] for m in models)
model_sel_lines = []
model_sel_lines.append("=" * 86)
model_sel_lines.append(f"{'QUANTITATIVE MODEL SELECTION (BIC / AIC SUITE)':^86}")
model_sel_lines.append("=" * 86)
model_sel_lines.append(
    f"{'Model Description':<54} {'k':<4} {'ln(L)':<10} {'BIC':<10} {'Delta BIC':<10}"
)
model_sel_lines.append("-" * 86)
for m in models:
    delta_bic = m["bic"] - min_bic
    model_sel_lines.append(
        f"{m['name']:<54} {m['k']:<4} {m['ll']:<10.2f} {m['bic']:<10.2f} {delta_bic:<10.2f}"
    )
model_sel_lines.append("-" * 86)
model_sel_lines.append(f"--> ADOPTED BEST MODEL: Model {best_model['id']} ({best_model['name']})")
model_sel_lines.append("=" * 86)
print("\n" + "\n".join(model_sel_lines) + "\n")


# 4. MCMC POSTERIOR SAMPLING FOR THE ADOPTED MODEL
if best_model["id"] == 1:
    ndim = 8
    param_labels = [
        r"$T_{\rm ej}$",
        r"$v_n$",
        r"$v_s$",
        r"${\rm PA}_n$",
        r"${\rm PA}_s$",
        r"$\theta_{\rm scatter}$",
        r"$x_c$",
        r"$y_c$",
    ]

    def log_prior(p):
        return log_prior_universal(
            p[0], p[1], p[1], 1e6, p[2], p[2], 1e6, p[3], p[4], p[5], p[6], p[7],
            is_anchored=False, is_piecewise=False, anchor_date=args.anchor_date,
        )

    def log_prob(p):
        lp = log_prior(p)
        if not np.isfinite(lp):
            return -np.inf
        return lp + calc_log_likelihood(
            p[0], p[1], p[1], 1e6, p[2], p[2], 1e6, p[3], p[4], p[5], p[6], p[7],
            include_limits=args.include_limits,
        )

elif best_model["id"] == 2:
    if args.anchor_std is not None and args.anchor_std > 0:
        ndim = 8
        param_labels = [
            r"$T_{\rm ej}$",
            r"$v_n$",
            r"$v_s$",
            r"${\rm PA}_n$",
            r"${\rm PA}_s$",
            r"$\theta_{\rm scatter}$",
            r"$x_c$",
            r"$y_c$",
        ]

        def log_prior(p):
            return log_prior_universal(
                p[0], p[1], p[1], 1e6, p[2], p[2], 1e6, p[3], p[4], p[5], p[6], p[7],
                is_anchored=True, is_piecewise=False, anchor_date=args.anchor_date, anchor_std=args.anchor_std,
            )

        def log_prob(p):
            lp = log_prior(p)
            if not np.isfinite(lp):
                return -np.inf
            return lp + calc_log_likelihood(
                p[0], p[1], p[1], 1e6, p[2], p[2], 1e6, p[3], p[4], p[5], p[6], p[7],
                include_limits=args.include_limits,
            )
    else:
        ndim = 7
        param_labels = [
            r"$v_n$",
            r"$v_s$",
            r"${\rm PA}_n$",
            r"${\rm PA}_s$",
            r"$\theta_{\rm scatter}$",
            r"$x_c$",
            r"$y_c$",
        ]

        def log_prior(p):
            return log_prior_universal(
                args.anchor_date, p[0], p[0], 1e6, p[1], p[1], 1e6, p[2], p[3], p[4], p[5], p[6],
                is_anchored=True, is_piecewise=False, anchor_date=args.anchor_date,
            )

        def log_prob(p):
            lp = log_prior(p)
            if not np.isfinite(lp):
                return -np.inf
            return lp + calc_log_likelihood(
                args.anchor_date, p[0], p[0], 1e6, p[1], p[1], 1e6, p[2], p[3], p[4], p[5], p[6],
                include_limits=args.include_limits,
            )

elif best_model["id"] == 3:
    ndim = 12
    param_labels = [
        r"$T_{\rm ej}$",
        r"$v_{1,n}$",
        r"$v_{2,n}$",
        r"$t_{b,n}$",
        r"$v_{1,s}$",
        r"$v_{2,s}$",
        r"$t_{b,s}$",
        r"${\rm PA}_n$",
        r"${\rm PA}_s$",
        r"$\theta_{\rm scatter}$",
        r"$x_c$",
        r"$y_c$",
    ]

    def log_prior(p):
        return log_prior_universal(
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11],
            is_anchored=False, is_piecewise=True, anchor_date=args.anchor_date,
        )

    def log_prob(p):
        lp = log_prior(p)
        if not np.isfinite(lp):
            return -np.inf
        return lp + calc_log_likelihood(
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11],
            include_limits=args.include_limits,
        )

elif best_model["id"] == 4:
    if args.anchor_std is not None and args.anchor_std > 0:
        ndim = 12
        param_labels = [
            r"$T_{\rm ej}$",
            r"$v_{1,n}$",
            r"$v_{2,n}$",
            r"$t_{b,n}$",
            r"$v_{1,s}$",
            r"$v_{2,s}$",
            r"$t_{b,s}$",
            r"${\rm PA}_n$",
            r"${\rm PA}_s$",
            r"$\theta_{\rm scatter}$",
            r"$x_c$",
            r"$y_c$",
        ]

        def log_prior(p):
            return log_prior_universal(
                p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11],
                is_anchored=True, is_piecewise=True, anchor_date=args.anchor_date, anchor_std=args.anchor_std,
            )

        def log_prob(p):
            lp = log_prior(p)
            if not np.isfinite(lp):
                return -np.inf
            return lp + calc_log_likelihood(
                p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11],
                include_limits=args.include_limits,
            )
    else:
        ndim = 11
        param_labels = [
            r"$v_{1,n}$",
            r"$v_{2,n}$",
            r"$t_{b,n}$",
            r"$v_{1,s}$",
            r"$v_{2,s}$",
            r"$t_{b,s}$",
            r"${\rm PA}_n$",
            r"${\rm PA}_s$",
            r"$\theta_{\rm scatter}$",
            r"$x_c$",
            r"$y_c$",
        ]

        def log_prior(p):
            return log_prior_universal(
                args.anchor_date, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10],
                is_anchored=True, is_piecewise=True, anchor_date=args.anchor_date,
            )

        def log_prob(p):
            lp = log_prior(p)
            if not np.isfinite(lp):
                return -np.inf
            return lp + calc_log_likelihood(
                args.anchor_date, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10],
                include_limits=args.include_limits,
            )

elif best_model["id"] == 5:
    ndim = 10
    param_labels = [
        r"$T_{\rm ej}$",
        r"$v_{0, n}$",
        r"$\tau_n$",
        r"$v_{0, s}$",
        r"$\tau_s$",
        r"${\rm PA}_n$",
        r"${\rm PA}_s$",
        r"$\theta_{\rm scatter}$",
        r"$x_c$",
        r"$y_c$",
    ]

    def log_prior(p):
        return log_prior_smooth(
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
            is_anchored=False, anchor_date=args.anchor_date,
        )

    def log_prob(p):
        lp = log_prior(p)
        if not np.isfinite(lp):
            return -np.inf
        return lp + calc_log_likelihood_smooth(
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
            include_limits=args.include_limits,
        )

elif best_model["id"] == 6:
    if args.anchor_std is not None and args.anchor_std > 0:
        ndim = 10
        param_labels = [
            r"$T_{\rm ej}$",
            r"$v_{0, n}$",
            r"$\tau_n$",
            r"$v_{0, s}$",
            r"$\tau_s$",
            r"${\rm PA}_n$",
            r"${\rm PA}_s$",
            r"$\theta_{\rm scatter}$",
            r"$x_c$",
            r"$y_c$",
        ]

        def log_prior(p):
            return log_prior_smooth(
                p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
                is_anchored=True, anchor_date=args.anchor_date, anchor_std=args.anchor_std,
            )

        def log_prob(p):
            lp = log_prior(p)
            if not np.isfinite(lp):
                return -np.inf
            return lp + calc_log_likelihood_smooth(
                p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
                include_limits=args.include_limits,
            )
    else:
        ndim = 9
        param_labels = [
            r"$v_{0, n}$",
            r"$\tau_n$",
            r"$v_{0, s}$",
            r"$\tau_s$",
            r"${\rm PA}_n$",
            r"${\rm PA}_s$",
            r"$\theta_{\rm scatter}$",
            r"$x_c$",
            r"$y_c$",
        ]

        def log_prior(p):
            return log_prior_smooth(
                args.anchor_date, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8],
                is_anchored=True, anchor_date=args.anchor_date,
            )

        def log_prob(p):
            lp = log_prior(p)
            if not np.isfinite(lp):
                return -np.inf
            return lp + calc_log_likelihood_smooth(
                args.anchor_date, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8],
                include_limits=args.include_limits,
            )


print(f"Sampling posteriors via MCMC for Model {best_model['id']}...")
opt_init = best_model["res"].x
nwalkers = 128
n_steps = args.n_steps
scales = np.abs(opt_init) * 0.05 + 1e-4
has_free_or_soft_tej = best_model["id"] in [1, 3, 5] or (
    best_model["id"] in [2, 4, 6] and args.anchor_std is not None and args.anchor_std > 0
)
if has_free_or_soft_tej:
    scales[0] = 2.0  # T_ej scale
if best_model["id"] == 3 or (best_model["id"] == 4 and args.anchor_std is not None and args.anchor_std > 0):
    scales[3] = 2.0  # tb_n scale
    scales[6] = 2.0  # tb_s scale
elif best_model["id"] == 4:
    scales[2] = 2.0  # tb_n scale
    scales[5] = 2.0  # tb_s scale
elif best_model["id"] in [5, 6]:
    if has_free_or_soft_tej:
        scales[1] = 0.0005  # v0_n scale
        scales[2] = 20.0    # tau_n scale
        scales[3] = 0.0005  # v0_s scale
        scales[4] = 20.0    # tau_s scale
    else:
        scales[0] = 0.0005  # v0_n scale
        scales[1] = 20.0    # tau_n scale
        scales[2] = 0.0005  # v0_s scale
        scales[3] = 20.0    # tau_s scale

pos_list = []
for _ in range(nwalkers):
    valid = False
    for _ in range(5000):
        trial = opt_init + scales * np.random.randn(ndim)
        if np.isfinite(log_prior(trial)):
            pos_list.append(trial)
            valid = True
            break
    if not valid:
        pos_list.append(opt_init)
pos = np.array(pos_list)

_cached_npz = f"data/absolute_kinematics_plot_data{_save_suffix}.npz"
if args.use_cached and os.path.exists(_cached_npz):
    print(f"Loading cached MCMC chains from {_cached_npz}...")
    _loaded = np.load(_cached_npz)
    flat_samples = _loaded["flat_samples"]
else:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(pos, n_steps, progress=False)
    burnin = int(n_steps * 0.2)
    thin = max(1, min(10, (n_steps - burnin) // 2))
    flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

p_res = [compute_hdi(flat_samples[:, i]) for i in range(ndim)]
p_res_1s = [compute_hdi(flat_samples[:, i], credible_mass=0.6827) for i in range(ndim)]


# 5. DERIVED KINEMATICS & OUTPUT
pa_n_idx = param_labels.index(r"${\rm PA}_n$")
pa_s_idx = param_labels.index(r"${\rm PA}_s$")
xc_idx, yc_idx = ndim - 2, ndim - 1

c_const = 2.99792458e8
sys_dist = getattr(args, "sys_dist", 9.4) if "args" in globals() else 9.4

from scipy.stats import truncnorm

def draw_physical_distance(mean_d, std_d, dmax_chain):
    # Vectorized truncated normal sampling conditioned on 0.5 kpc <= D <= D_max
    # Following Mirabel & Rodriguez (1994) and Bahramian et al. (2023).
    # If D_max comfortably exceeds the prior (e.g. Model 1 ballistic), use standard system distance.
    # If D_max is restrictive (e.g. Model 5 deceleration), adopt the Reid et al. (2014) trigonometric
    # parallax distance (8.6 +- 1.8 kpc) conditioned on D <= D_max to avoid artificial edge piling.
    if np.median(dmax_chain) >= 15.0:
        loc = mean_d
        scale = std_d
    else:
        loc = 8.6 if mean_d == 9.4 else mean_d
        scale = 1.8 if std_d == 1.0 else std_d
    a = (0.5 - loc) / scale
    b = (dmax_chain - loc) / scale
    b = np.maximum(a + 1e-4, b)
    d_samples = truncnorm.rvs(a, b, loc=loc, scale=scale)
    d_samples = np.minimum(d_samples, dmax_chain * 0.9999)
    return d_samples


if best_model["id"] == 1:
    t_ej_chain = flat_samples[:, 0]
    mu_rec_chain = flat_samples[:, 1]
    mu_app_chain = np.abs(flat_samples[:, 2])

    bct_chain = (mu_app_chain - mu_rec_chain) / (mu_app_chain + mu_rec_chain)
    mu_true_chain = (2 * mu_app_chain * mu_rec_chain) / (mu_app_chain + mu_rec_chain)
    mu_app_rad_s = mu_app_chain * (np.pi / (180.0 * 3600.0)) / 86400.0
    mu_rec_rad_s = mu_rec_chain * (np.pi / (180.0 * 3600.0)) / 86400.0
    D_max_kpc_chain = (c_const / np.sqrt(mu_app_rad_s * mu_rec_rad_s)) / 3.086e19

    D_kpc_chain = draw_physical_distance(sys_dist, 1.0, D_max_kpc_chain)
    D_m_chain = D_kpc_chain * 3.086e19

    # Apparent proper motions (beta_app = mu * D / c)
    beta_app_a_chain = mu_app_rad_s * D_m_chain / c_const
    beta_app_r_chain = mu_rec_rad_s * D_m_chain / c_const

    bst_chain = mu_app_rad_s * (1.0 - bct_chain) * D_m_chain / c_const
    beta_chain = np.sqrt(bct_chain**2 + bst_chain**2)
    valid_phys = beta_chain < 1.0
    gamma_chain = np.full_like(beta_chain, np.nan)
    gamma_chain[valid_phys] = 1.0 / np.sqrt(1.0 - beta_chain[valid_phys]**2)
    gamma_beta_chain = gamma_chain * beta_chain
    theta_deg_chain = np.degrees(np.arctan2(bst_chain, bct_chain))
    delta_app_chain = 1.0 / (gamma_chain * (1.0 - bct_chain))
    delta_rec_chain = 1.0 / (gamma_chain * (1.0 + bct_chain))
    delta_ratio_chain = (1.0 + bct_chain) / (1.0 - bct_chain)

elif best_model["id"] == 2:
    if args.anchor_std is not None and args.anchor_std > 0:
        t_ej_chain = flat_samples[:, 0]
        mu_rec_chain = flat_samples[:, 1]
        mu_app_chain = np.abs(flat_samples[:, 2])
    else:
        t_ej_chain = np.full(len(flat_samples), args.anchor_date)
        mu_rec_chain = flat_samples[:, 0]
        mu_app_chain = np.abs(flat_samples[:, 1])

    bct_chain = (mu_app_chain - mu_rec_chain) / (mu_app_chain + mu_rec_chain)
    mu_true_chain = (2 * mu_app_chain * mu_rec_chain) / (mu_app_chain + mu_rec_chain)
    mu_app_rad_s = mu_app_chain * (np.pi / (180.0 * 3600.0)) / 86400.0
    mu_rec_rad_s = mu_rec_chain * (np.pi / (180.0 * 3600.0)) / 86400.0
    D_max_kpc_chain = (c_const / np.sqrt(mu_app_rad_s * mu_rec_rad_s)) / 3.086e19
    D_kpc_chain = draw_physical_distance(sys_dist, 1.0, D_max_kpc_chain)
    D_m_chain = D_kpc_chain * 3.086e19

    # Apparent proper motions (beta_app = mu * D / c)
    beta_app_a_chain = mu_app_rad_s * D_m_chain / c_const
    beta_app_r_chain = mu_rec_rad_s * D_m_chain / c_const

    bst_chain = mu_app_rad_s * (1.0 - bct_chain) * D_m_chain / c_const
    beta_chain = np.sqrt(bct_chain**2 + bst_chain**2)
    valid_phys = beta_chain < 1.0
    gamma_chain = np.full_like(beta_chain, np.nan)
    gamma_chain[valid_phys] = 1.0 / np.sqrt(1.0 - beta_chain[valid_phys]**2)
    gamma_beta_chain = gamma_chain * beta_chain
    theta_deg_chain = np.degrees(np.arctan2(bst_chain, bct_chain))
    delta_app_chain = 1.0 / (gamma_chain * (1.0 - bct_chain))
    delta_rec_chain = 1.0 / (gamma_chain * (1.0 + bct_chain))
    delta_ratio_chain = (1.0 + bct_chain) / (1.0 - bct_chain)

elif best_model["id"] == 3:
    t_ej_chain = flat_samples[:, 0]
    mu_rec_1_chain = flat_samples[:, 1]
    mu_rec_chain = flat_samples[:, 2]
    t_break_n_chain = flat_samples[:, 3]
    mu_app_1_chain = np.abs(flat_samples[:, 4])
    mu_app_chain = np.abs(flat_samples[:, 5])
    t_break_s_chain = flat_samples[:, 6]

elif best_model["id"] == 4:
    if args.anchor_std is not None and args.anchor_std > 0:
        t_ej_chain = flat_samples[:, 0]
        mu_rec_1_chain = flat_samples[:, 1]
        mu_rec_chain = flat_samples[:, 2]
        t_break_n_chain = flat_samples[:, 3]
        mu_app_1_chain = np.abs(flat_samples[:, 4])
        mu_app_chain = np.abs(flat_samples[:, 5])
        t_break_s_chain = flat_samples[:, 6]
    else:
        t_ej_chain = np.full(len(flat_samples), args.anchor_date)
        mu_rec_1_chain = flat_samples[:, 0]
        mu_rec_chain = flat_samples[:, 1]
        t_break_n_chain = flat_samples[:, 2]
        mu_app_1_chain = np.abs(flat_samples[:, 3])
        mu_app_chain = np.abs(flat_samples[:, 4])
        t_break_s_chain = flat_samples[:, 5]

elif best_model["id"] in [5, 6]:
    if has_free_or_soft_tej:
        t_ej_chain = flat_samples[:, 0]
        v0_n_chain = flat_samples[:, 1]
        tau_n_chain = flat_samples[:, 2]
        v0_s_chain = flat_samples[:, 3]
        tau_s_chain = flat_samples[:, 4]
    else:
        t_ej_chain = np.full(len(flat_samples), args.anchor_date)
        v0_n_chain = flat_samples[:, 0]
        tau_n_chain = flat_samples[:, 1]
        v0_s_chain = flat_samples[:, 2]
        tau_s_chain = flat_samples[:, 3]

    mu_rec_chain = v0_n_chain
    mu_app_chain = np.abs(v0_s_chain)
    r_max_n_chain = v0_n_chain * tau_n_chain
    r_max_s_chain = np.abs(v0_s_chain) * tau_s_chain

    bct_chain = (mu_app_chain - mu_rec_chain) / (mu_app_chain + mu_rec_chain)
    mu_true_chain = (2 * mu_app_chain * mu_rec_chain) / (mu_app_chain + mu_rec_chain)
    mu_app_rad_s = mu_app_chain * (np.pi / (180.0 * 3600.0)) / 86400.0
    mu_rec_rad_s = mu_rec_chain * (np.pi / (180.0 * 3600.0)) / 86400.0
    D_max_kpc_chain = (c_const / np.sqrt(mu_app_rad_s * mu_rec_rad_s)) / 3.086e19
    D_kpc_chain = draw_physical_distance(sys_dist, 1.0, D_max_kpc_chain)
    D_m_chain = D_kpc_chain * 3.086e19

    beta_app_a_chain = mu_app_rad_s * D_m_chain / c_const
    beta_app_r_chain = mu_rec_rad_s * D_m_chain / c_const

    bst_chain = mu_app_rad_s * (1.0 - bct_chain) * D_m_chain / c_const
    beta_chain = np.sqrt(bct_chain**2 + bst_chain**2)
    valid_phys = beta_chain < 1.0
    gamma_chain = np.full_like(beta_chain, np.nan)
    gamma_chain[valid_phys] = 1.0 / np.sqrt(1.0 - beta_chain[valid_phys]**2)
    gamma_beta_chain = gamma_chain * beta_chain
    theta_deg_chain = np.degrees(np.arctan2(bst_chain, bct_chain))
    delta_app_chain = 1.0 / (gamma_chain * (1.0 - bct_chain))
    delta_rec_chain = 1.0 / (gamma_chain * (1.0 + bct_chain))
    delta_ratio_chain = (1.0 + bct_chain) / (1.0 - bct_chain)

    dt_final_n = np.maximum(0.0, 60897.0 - t_ej_chain)
    mu_final_r_chain = v0_n_chain * np.exp(-dt_final_n / tau_n_chain)
    mu_final_a_chain = np.abs(v0_s_chain) * np.exp(-dt_final_n / tau_s_chain)
    beta_app_final_r_chain = (mu_final_r_chain * (np.pi / (180.0 * 3600.0)) / 86400.0) * D_m_chain / c_const
    beta_app_final_a_chain = (mu_final_a_chain * (np.pi / (180.0 * 3600.0)) / 86400.0) * D_m_chain / c_const

if best_model["id"] in [3, 4]:
    # 1. Symmetric Launch Phase (calculates free system inclination theta)
    bct_chain = (mu_app_1_chain - mu_rec_1_chain) / (mu_app_1_chain + mu_rec_1_chain)
    mu_true_chain = (2 * mu_app_1_chain * mu_rec_1_chain) / (mu_app_1_chain + mu_rec_1_chain)
    mu_app_1_rad_s = mu_app_1_chain * (np.pi / (180.0 * 3600.0)) / 86400.0
    mu_rec_1_rad_s = mu_rec_1_chain * (np.pi / (180.0 * 3600.0)) / 86400.0
    D_max_kpc_chain = (c_const / np.sqrt(mu_app_1_rad_s * mu_rec_1_rad_s)) / 3.086e19
    D_kpc_chain = draw_physical_distance(sys_dist, 1.0, D_max_kpc_chain)
    D_m_chain = D_kpc_chain * 3.086e19

    # Apparent proper motions in Phase 1 (beta_app = mu * D / c)
    beta_app_1a_chain = mu_app_1_rad_s * D_m_chain / c_const
    beta_app_1r_chain = mu_rec_1_rad_s * D_m_chain / c_const

    bst_chain = mu_app_1_rad_s * (1.0 - bct_chain) * D_m_chain / c_const
    beta_1_chain = np.sqrt(bct_chain**2 + bst_chain**2)
    valid_phys_1 = beta_1_chain < 1.0
    gamma_1_chain = np.full_like(beta_1_chain, np.nan)
    gamma_1_chain[valid_phys_1] = 1.0 / np.sqrt(1.0 - beta_1_chain[valid_phys_1]**2)
    gamma_beta_1_chain = gamma_1_chain * beta_1_chain
    theta_deg_chain = np.degrees(np.arctan2(bst_chain, bct_chain))

    delta_app_1_chain = 1.0 / (gamma_1_chain * (1.0 - bct_chain))
    delta_rec_1_chain = 1.0 / (gamma_1_chain * (1.0 + bct_chain))
    delta_ratio_chain = (1.0 + bct_chain) / (1.0 - bct_chain)

    # For standardized reporting in Section 3
    beta_chain = beta_1_chain
    gamma_chain = gamma_1_chain
    gamma_beta_chain = gamma_beta_1_chain
    delta_app_chain = delta_app_1_chain
    delta_rec_chain = delta_rec_1_chain

    # 2. Post-Break Differential Deceleration Phase (given free system inclination theta)
    th_rad_chain = np.radians(theta_deg_chain)
    sin_th_chain = np.sin(th_rad_chain)
    cos_th_chain = np.cos(th_rad_chain)

    mu_app_2_rad_s = mu_app_chain * (np.pi / (180.0 * 3600.0)) / 86400.0
    mu_rec_2_rad_s = mu_rec_chain * (np.pi / (180.0 * 3600.0)) / 86400.0

    # Apparent proper motions in Phase 2 (beta_app = mu * D / c)
    beta_app_2a_chain = mu_app_2_rad_s * D_m_chain / c_const
    beta_app_2r_chain = mu_rec_2_rad_s * D_m_chain / c_const

    beta_app_a_chain = beta_app_2a_chain
    beta_app_r_chain = beta_app_2r_chain

    beta_2_app_chain = mu_app_2_rad_s / ((c_const * sin_th_chain / D_m_chain) + mu_app_2_rad_s * cos_th_chain)
    beta_2_rec_chain = mu_rec_2_rad_s / ((c_const * sin_th_chain / D_m_chain) - mu_rec_2_rad_s * cos_th_chain)

    valid_phys_2a = beta_2_app_chain < 1.0
    gamma_2_app_chain = np.full_like(beta_2_app_chain, np.nan)
    gamma_2_app_chain[valid_phys_2a] = 1.0 / np.sqrt(1.0 - beta_2_app_chain[valid_phys_2a]**2)

    valid_phys_2r = beta_2_rec_chain < 1.0
    gamma_2_rec_chain = np.full_like(beta_2_rec_chain, np.nan)
    gamma_2_rec_chain[valid_phys_2r] = 1.0 / np.sqrt(1.0 - beta_2_rec_chain[valid_phys_2r]**2)

    decel_ratio_app_chain = beta_2_app_chain / beta_1_chain
    decel_ratio_rec_chain = beta_2_rec_chain / beta_1_chain

    delta_app_2_chain = 1.0 / (gamma_2_app_chain * (1.0 - beta_2_app_chain * cos_th_chain))
    delta_rec_2_chain = 1.0 / (gamma_2_rec_chain * (1.0 + beta_2_rec_chain * cos_th_chain))

    r_brk_n_chain = mu_rec_1_chain * (t_break_n_chain - t_ej_chain)
    r_brk_s_chain = mu_app_1_chain * (t_break_s_chain - t_ej_chain)

p_Tej = compute_hdi(t_ej_chain)
p_Tej_1s = compute_hdi(t_ej_chain, credible_mass=0.6827)
p_v0_n = compute_hdi(mu_rec_chain)
p_v0_n_1s = compute_hdi(mu_rec_chain, credible_mass=0.6827)
p_v0_s = compute_hdi(mu_app_chain)
p_v0_s_1s = compute_hdi(mu_app_chain, credible_mass=0.6827)
p_bct = compute_hdi(bct_chain)
p_bct_1s = compute_hdi(bct_chain, credible_mass=0.6827)
p_beta = compute_hdi(beta_chain)
p_beta_1s = compute_hdi(beta_chain, credible_mass=0.6827)
p_gamma = compute_hdi(gamma_chain)
p_gamma_1s = compute_hdi(gamma_chain, credible_mass=0.6827)
p_gammabeta = compute_hdi(gamma_beta_chain)
p_gammabeta_1s = compute_hdi(gamma_beta_chain, credible_mass=0.6827)
p_theta = compute_hdi(theta_deg_chain)
p_theta_1s = compute_hdi(theta_deg_chain, credible_mass=0.6827)
p_delta_app = compute_hdi(delta_app_chain)
p_delta_app_1s = compute_hdi(delta_app_chain, credible_mass=0.6827)
p_delta_rec = compute_hdi(delta_rec_chain)
p_delta_rec_1s = compute_hdi(delta_rec_chain, credible_mass=0.6827)
p_mutrue = compute_hdi(mu_true_chain)
p_mutrue_1s = compute_hdi(mu_true_chain, credible_mass=0.6827)
p_Dmax = compute_hdi(D_max_kpc_chain)
p_Dmax_1s = compute_hdi(D_max_kpc_chain, credible_mass=0.6827)
p_D = compute_hdi(D_kpc_chain)
p_D_1s = compute_hdi(D_kpc_chain, credible_mass=0.6827)
p_dratio = compute_hdi(delta_ratio_chain)
p_dratio_1s = compute_hdi(delta_ratio_chain, credible_mass=0.6827)
p_pa_n = p_res[pa_n_idx]
p_pa_n_1s = p_res_1s[pa_n_idx]
p_pa_s = p_res[pa_s_idx]
p_pa_s_1s = p_res_1s[pa_s_idx]

p_beta_app_a = compute_hdi(beta_app_a_chain)
p_beta_app_a_1s = compute_hdi(beta_app_a_chain, credible_mass=0.6827)
p_beta_app_r = compute_hdi(beta_app_r_chain)
p_beta_app_r_1s = compute_hdi(beta_app_r_chain, credible_mass=0.6827)

avg_pa_chain = (flat_samples[:, pa_n_idx] + flat_samples[:, pa_s_idx]) / 2.0
p_avg_pa = compute_hdi(avg_pa_chain)
p_avg_pa_1s = compute_hdi(avg_pa_chain, credible_mass=0.6827)

if best_model["id"] in [3, 4]:
    p_tb_n = compute_hdi(t_break_n_chain)
    p_tb_n_1s = compute_hdi(t_break_n_chain, credible_mass=0.6827)
    p_tb_s = compute_hdi(t_break_s_chain)
    p_tb_s_1s = compute_hdi(t_break_s_chain, credible_mass=0.6827)
    p_v1_n = compute_hdi(mu_rec_1_chain)
    p_v1_n_1s = compute_hdi(mu_rec_1_chain, credible_mass=0.6827)
    p_v1_s = compute_hdi(mu_app_1_chain)
    p_v1_s_1s = compute_hdi(mu_app_1_chain, credible_mass=0.6827)
    p_r_brk_n = compute_hdi(r_brk_n_chain)
    p_r_brk_n_1s = compute_hdi(r_brk_n_chain, credible_mass=0.6827)
    p_r_brk_s = compute_hdi(r_brk_s_chain)
    p_r_brk_s_1s = compute_hdi(r_brk_s_chain, credible_mass=0.6827)
    p_beta_2_a = compute_hdi(beta_2_app_chain)
    p_beta_2_a_1s = compute_hdi(beta_2_app_chain, credible_mass=0.6827)
    p_gamma_2_a = compute_hdi(gamma_2_app_chain)
    p_gamma_2_a_1s = compute_hdi(gamma_2_app_chain, credible_mass=0.6827)
    p_ratio_a = compute_hdi(decel_ratio_app_chain)
    p_ratio_a_1s = compute_hdi(decel_ratio_app_chain, credible_mass=0.6827)
    p_beta_2_r = compute_hdi(beta_2_rec_chain)
    p_beta_2_r_1s = compute_hdi(beta_2_rec_chain, credible_mass=0.6827)
    p_gamma_2_r = compute_hdi(gamma_2_rec_chain)
    p_gamma_2_r_1s = compute_hdi(gamma_2_rec_chain, credible_mass=0.6827)
    p_ratio_r = compute_hdi(decel_ratio_rec_chain)
    p_ratio_r_1s = compute_hdi(decel_ratio_rec_chain, credible_mass=0.6827)
    p_beta_app_1a = compute_hdi(beta_app_1a_chain)
    p_beta_app_1a_1s = compute_hdi(beta_app_1a_chain, credible_mass=0.6827)
    p_beta_app_1r = compute_hdi(beta_app_1r_chain)
    p_beta_app_1r_1s = compute_hdi(beta_app_1r_chain, credible_mass=0.6827)
    p_beta_app_2a = compute_hdi(beta_app_2a_chain)
    p_beta_app_2a_1s = compute_hdi(beta_app_2a_chain, credible_mass=0.6827)
    p_beta_app_2r = compute_hdi(beta_app_2r_chain)
    p_beta_app_2r_1s = compute_hdi(beta_app_2r_chain, credible_mass=0.6827)

if best_model["id"] in [5, 6]:
    p_rmax_n = compute_hdi(r_max_n_chain)
    p_rmax_n_1s = compute_hdi(r_max_n_chain, credible_mass=0.6827)
    p_rmax_s = compute_hdi(r_max_s_chain)
    p_rmax_s_1s = compute_hdi(r_max_s_chain, credible_mass=0.6827)
    p_tau_n = compute_hdi(tau_n_chain)
    p_tau_n_1s = compute_hdi(tau_n_chain, credible_mass=0.6827)
    p_tau_s = compute_hdi(tau_s_chain)
    p_tau_s_1s = compute_hdi(tau_s_chain, credible_mass=0.6827)
    p_mu_final_r = compute_hdi(mu_final_r_chain)
    p_mu_final_r_1s = compute_hdi(mu_final_r_chain, credible_mass=0.6827)
    p_mu_final_a = compute_hdi(mu_final_a_chain)
    p_mu_final_a_1s = compute_hdi(mu_final_a_chain, credible_mass=0.6827)
    p_beta_final_r = compute_hdi(beta_app_final_r_chain)
    p_beta_final_r_1s = compute_hdi(beta_app_final_r_chain, credible_mass=0.6827)
    p_beta_final_a = compute_hdi(beta_app_final_a_chain)
    p_beta_final_a_1s = compute_hdi(beta_app_final_a_chain, credible_mass=0.6827)

report_lines = []
_report_tag = "WITH LIMITS" if args.include_limits else "WITHOUT LIMITS"
report_lines.append("=" * 70)
report_lines.append(f"{'GRS 1915+105 RELATIVISTIC KINEMATIC REPORT (' + _report_tag + ')':^70}")
report_lines.append("=" * 70)
report_lines.append(
    f"Statistically Preferred Model : Model {best_model['id']} ({best_model['name']})"
)
report_lines.append(f"Data Points Used (N_data)     : {num_obs_total}")
report_lines.append(f"Optimized BIC                 : {best_model['bic']:.2f}")
hdi_pct = int(round(getattr(args, "hdi_prob", 0.90) * 100))
report_lines.append(f"Credible Interval Bounds      : {hdi_pct}% HDI (primary) & 68.3% HDI (1-sigma equivalent)")
if getattr(args, "enforce_subluminal", False):
    report_lines.append(f"Physical Velocity Constraint  : Subluminal (beta <= {getattr(args, 'max_beta', 1.0):.2f} c at D = {getattr(args, 'sys_dist', 9.4):.1f} kpc)")
else:
    report_lines.append(f"Physical Velocity Constraint  : Unconstrained (D = {getattr(args, 'sys_dist', 9.4):.1f} +/- 1.0 kpc, no artificial capping)")
if args.include_limits:
    report_lines.append(f"Upper Limit Penalty Sigma     : {getattr(args, 'ul_sigma', 0.05):.3f} arcsec")

report_lines.append("\n[1] EJECTION PARAMETERS")
report_lines.append("-" * 70)
if best_model["id"] in [1, 3, 5]:
    report_lines.append(
        f"Extrapolated Launch Time (T_ej): MJD {p_Tej[1]:.2f} + {p_Tej[2]-p_Tej[1]:.2f} - {p_Tej[1]-p_Tej[0]:.2f} (1-sigma: + {p_Tej_1s[2]-p_Tej[1]:.2f} - {p_Tej[1]-p_Tej_1s[0]:.2f})"
    )
elif args.anchor_std is not None and args.anchor_std > 0:
    report_lines.append(
        f"Extrapolated Launch Time (T_ej): MJD {p_Tej[1]:.2f} + {p_Tej[2]-p_Tej[1]:.2f} - {p_Tej[1]-p_Tej[0]:.2f} (1-sigma: + {p_Tej_1s[2]-p_Tej[1]:.2f} - {p_Tej[1]-p_Tej_1s[0]:.2f}) (Anchored {args.anchor_date:.0f}±{args.anchor_std:.0f})"
    )
else:
    report_lines.append(
        f"Extrapolated Launch Time (T_ej): MJD {args.anchor_date:.2f} + 0.00 - 0.00 (Anchored)"
    )
report_lines.append(
    f"Core Offset RA (x_c)           : {p_res[xc_idx][1]:.4f} + {p_res[xc_idx][2]-p_res[xc_idx][1]:.4f} - {p_res[xc_idx][1]-p_res[xc_idx][0]:.4f} arcsec (1-sigma: + {p_res_1s[xc_idx][2]-p_res[xc_idx][1]:.4f} - {p_res[xc_idx][1]-p_res_1s[xc_idx][0]:.4f})"
)
report_lines.append(
    f"Core Offset Dec (y_c)          : {p_res[yc_idx][1]:.4f} + {p_res[yc_idx][2]-p_res[yc_idx][1]:.4f} - {p_res[yc_idx][1]-p_res[yc_idx][0]:.4f} arcsec (1-sigma: + {p_res_1s[yc_idx][2]-p_res[yc_idx][1]:.4f} - {p_res[yc_idx][1]-p_res_1s[yc_idx][0]:.4f})"
)

report_lines.append("\n[2] APPARENT PROPER MOTIONS & JET AXIS")
report_lines.append("-" * 70)
if best_model["id"] in [3, 4]:
    report_lines.append(
        f"Receding Break Time (t_break,r) : MJD {p_tb_n[1]:.2f} + {p_tb_n[2]-p_tb_n[1]:.2f} - {p_tb_n[1]-p_tb_n[0]:.2f} (1-sigma: + {p_tb_n_1s[2]-p_tb_n[1]:.2f} - {p_tb_n[1]-p_tb_n_1s[0]:.2f})"
    )
    report_lines.append(
        f"Receding Break Dist (r_b,r)     : {p_r_brk_n[1]:.3f} + {p_r_brk_n[2]-p_r_brk_n[1]:.3f} - {p_r_brk_n[1]-p_r_brk_n[0]:.3f} arcsec (1-sigma: + {p_r_brk_n_1s[2]-p_r_brk_n[1]:.3f} - {p_r_brk_n[1]-p_r_brk_n_1s[0]:.3f})"
    )
    report_lines.append(
        f"Approaching Break Time (t_break,a): MJD {p_tb_s[1]:.2f} + {p_tb_s[2]-p_tb_s[1]:.2f} - {p_tb_s[1]-p_tb_s[0]:.2f} (1-sigma: + {p_tb_s_1s[2]-p_tb_s[1]:.2f} - {p_tb_s[1]-p_tb_s_1s[0]:.2f})"
    )
    report_lines.append(
        f"Approaching Break Dist (r_b,a)  : {p_r_brk_s[1]:.3f} + {p_r_brk_s[2]-p_r_brk_s[1]:.3f} - {p_r_brk_s[1]-p_r_brk_s[0]:.3f} arcsec (1-sigma: + {p_r_brk_s_1s[2]-p_r_brk_s[1]:.3f} - {p_r_brk_s[1]-p_r_brk_s_1s[0]:.3f})"
    )
    report_lines.append(
        f"Initial Receding Speed (mu_1,r) : {p_v1_n[1]:.4e} + {p_v1_n[2]-p_v1_n[1]:.4e} - {p_v1_n[1]-p_v1_n[0]:.4e} arcsec/day (1-sigma: + {p_v1_n_1s[2]-p_v1_n[1]:.4e} - {p_v1_n[1]-p_v1_n_1s[0]:.4e})"
    )
    report_lines.append(
        f"Initial Receding Apparent Speed (beta_app,1r)   : {p_beta_app_1r[1]:>6.4f} + {p_beta_app_1r[2]-p_beta_app_1r[1]:.4f} - {p_beta_app_1r[1]-p_beta_app_1r[0]:.4f} c (1-sigma: + {p_beta_app_1r_1s[2]-p_beta_app_1r_1s[1]:.4f} - {p_beta_app_1r[1]-p_beta_app_1r_1s[0]:.4f})"
    )
    report_lines.append(
        f"Initial Approaching Speed (mu_1,a): {p_v1_s[1]:.4e} + {p_v1_s[2]-p_v1_s[1]:.4e} - {p_v1_s[1]-p_v1_s[0]:.4e} arcsec/day (1-sigma: + {p_v1_s_1s[2]-p_v1_s[1]:.4e} - {p_v1_s[1]-p_v1_s_1s[0]:.4e})"
    )
    report_lines.append(
        f"Initial Approaching Apparent Speed (beta_app,1a) : {p_beta_app_1a[1]:>6.4f} + {p_beta_app_1a[2]-p_beta_app_1a[1]:.4f} - {p_beta_app_1a[1]-p_beta_app_1a[0]:.4f} c (1-sigma: + {p_beta_app_1a_1s[2]-p_beta_app_1a_1s[1]:.4f} - {p_beta_app_1a[1]-p_beta_app_1a_1s[0]:.4f})"
    )
    report_lines.append(
        f"Post-Break Receding Speed (mu_2,r): {p_v0_n[1]:.4e} + {p_v0_n[2]-p_v0_n[1]:.4e} - {p_v0_n[1]-p_v0_n[0]:.4e} arcsec/day (1-sigma: + {p_v0_n_1s[2]-p_v0_n[1]:.4e} - {p_v0_n[1]-p_v0_n_1s[0]:.4e})"
    )
    report_lines.append(
        f"Post-Break Receding Apparent Speed (beta_app,2r) : {p_beta_app_2r[1]:>6.4f} + {p_beta_app_2r[2]-p_beta_app_2r[1]:.4f} - {p_beta_app_2r[1]-p_beta_app_2r[0]:.4f} c (1-sigma: + {p_beta_app_2r_1s[2]-p_beta_app_2r_1s[1]:.4f} - {p_beta_app_2r[1]-p_beta_app_2r_1s[0]:.4f})"
    )
    report_lines.append(
        f"Post-Break Approaching Speed (mu_2,a): {p_v0_s[1]:.4e} + {p_v0_s[2]-p_v0_s[1]:.4e} - {p_v0_s[1]-p_v0_s[0]:.4e} arcsec/day (1-sigma: + {p_v0_s_1s[2]-p_v0_s[1]:.4e} - {p_v0_s[1]-p_v0_s_1s[0]:.4e})"
    )
    report_lines.append(
        f"Post-Break Approaching Apparent Speed (beta_app,2a): {p_beta_app_2a[1]:>6.4f} + {p_beta_app_2a[2]-p_beta_app_2a[1]:.4f} - {p_beta_app_2a[1]-p_beta_app_2a[0]:.4f} c (1-sigma: + {p_beta_app_2a_1s[2]-p_beta_app_2a_1s[1]:.4f} - {p_beta_app_2a[1]-p_beta_app_2a_1s[0]:.4f})"
    )
elif best_model["id"] in [5, 6]:
    report_lines.append(
        f"Receding Stall Distance (r_max,r)   : {p_rmax_n[1]:.3f} + {p_rmax_n[2]-p_rmax_n[1]:.3f} - {p_rmax_n[1]-p_rmax_n[0]:.3f} arcsec (1-sigma: + {p_rmax_n_1s[2]-p_rmax_n[1]:.3f} - {p_rmax_n[1]-p_rmax_n_1s[0]:.3f})"
    )
    report_lines.append(
        f"Approaching Stall Distance (r_max,a): {p_rmax_s[1]:.3f} + {p_rmax_s[2]-p_rmax_s[1]:.3f} - {p_rmax_s[1]-p_rmax_s[0]:.3f} arcsec (1-sigma: + {p_rmax_s_1s[2]-p_rmax_s[1]:.3f} - {p_rmax_s[1]-p_rmax_s_1s[0]:.3f})"
    )
    report_lines.append(
        f"Receding Decel Timescale (tau_r)    : {p_tau_n[1]:.1f} + {p_tau_n[2]-p_tau_n[1]:.1f} - {p_tau_n[1]-p_tau_n[0]:.1f} days (1-sigma: + {p_tau_n_1s[2]-p_tau_n[1]:.1f} - {p_tau_n[1]-p_tau_n_1s[0]:.1f})"
    )
    report_lines.append(
        f"Approaching Decel Timescale (tau_a) : {p_tau_s[1]:.1f} + {p_tau_s[2]-p_tau_s[1]:.1f} - {p_tau_s[1]-p_tau_s[0]:.1f} days (1-sigma: + {p_tau_s_1s[2]-p_tau_s[1]:.1f} - {p_tau_s[1]-p_tau_s_1s[0]:.1f})"
    )
    report_lines.append(
        f"Initial Receding Launch Speed (mu_0,r) : {p_v0_n[1]:.4e} + {p_v0_n[2]-p_v0_n[1]:.4e} - {p_v0_n[1]-p_v0_n[0]:.4e} arcsec/day (1-sigma: + {p_v0_n_1s[2]-p_v0_n[1]:.4e} - {p_v0_n[1]-p_v0_n_1s[0]:.4e})"
    )
    report_lines.append(
        f"Initial Receding Apparent Speed (beta_app,0r): {p_beta_app_r[1]:>6.4f} + {p_beta_app_r[2]-p_beta_app_r[1]:.4f} - {p_beta_app_r[1]-p_beta_app_r[0]:.4f} c (1-sigma: + {p_beta_app_r_1s[2]-p_beta_app_r[1]:.4f} - {p_beta_app_r[1]-p_beta_app_r_1s[0]:.4f})"
    )
    report_lines.append(
        f"Initial Approaching Launch Speed (mu_0,a): {p_v0_s[1]:.4e} + {p_v0_s[2]-p_v0_s[1]:.4e} - {p_v0_s[1]-p_v0_s[0]:.4e} arcsec/day (1-sigma: + {p_v0_s_1s[2]-p_v0_s[1]:.4e} - {p_v0_s[1]-p_v0_s_1s[0]:.4e})"
    )
    report_lines.append(
        f"Initial Approaching Apparent Speed (beta_app,0a): {p_beta_app_a[1]:>6.4f} + {p_beta_app_a[2]-p_beta_app_a[1]:.4f} - {p_beta_app_a[1]-p_beta_app_a[0]:.4f} c (1-sigma: + {p_beta_app_a_1s[2]-p_beta_app_a[1]:.4f} - {p_beta_app_a[1]-p_beta_app_a_1s[0]:.4f})"
    )
    report_lines.append(
        f"Final Epoch Receding Speed (mu_end,r)  : {p_mu_final_r[1]:.4e} + {p_mu_final_r[2]-p_mu_final_r[1]:.4e} - {p_mu_final_r[1]-p_mu_final_r[0]:.4e} arcsec/day"
    )
    report_lines.append(
        f"Final Epoch Receding Apparent Speed    : {p_beta_final_r[1]:>6.4f} + {p_beta_final_r[2]-p_beta_final_r[1]:.4f} - {p_beta_final_r[1]-p_beta_final_r[0]:.4f} c"
    )
    report_lines.append(
        f"Final Epoch Approaching Speed (mu_end,a): {p_mu_final_a[1]:.4e} + {p_mu_final_a[2]-p_mu_final_a[1]:.4e} - {p_mu_final_a[1]-p_mu_final_a[0]:.4e} arcsec/day"
    )
    report_lines.append(
        f"Final Epoch Approaching Apparent Speed  : {p_beta_final_a[1]:>6.4f} + {p_beta_final_a[2]-p_beta_final_a[1]:.4f} - {p_beta_final_a[1]-p_beta_final_a[0]:.4f} c"
    )
# Provide standard keys for backward-compatibility with parsers
report_lines.append(
    f"Receding Lobe (North) Speed (mu_r): {p_v0_n[1]:.4e} + {p_v0_n[2]-p_v0_n[1]:.4e} - {p_v0_n[1]-p_v0_n[0]:.4e} arcsec/day (1-sigma: + {p_v0_n_1s[2]-p_v0_n[1]:.4e} - {p_v0_n[1]-p_v0_n_1s[0]:.4e})"
)
report_lines.append(
    f"Receding Apparent Speed (beta_app,r)            : {p_beta_app_r[1]:>6.4f} + {p_beta_app_r[2]-p_beta_app_r[1]:.4f} - {p_beta_app_r[1]-p_beta_app_r[0]:.4f} c (1-sigma: + {p_beta_app_r_1s[2]-p_beta_app_r_1s[1]:.4f} - {p_beta_app_r[1]-p_beta_app_r_1s[0]:.4f})"
)
report_lines.append(
    f"Approaching Lobe (South) Speed (mu_a): {p_v0_s[1]:.4e} + {p_v0_s[2]-p_v0_s[1]:.4e} - {p_v0_s[1]-p_v0_s[0]:.4e} arcsec/day (1-sigma: + {p_v0_s_1s[2]-p_v0_s[1]:.4e} - {p_v0_s[1]-p_v0_s_1s[0]:.4e})"
)
report_lines.append(
    f"Approaching Apparent Speed (beta_app,a)         : {p_beta_app_a[1]:>6.4f} + {p_beta_app_a[2]-p_beta_app_a[1]:.4f} - {p_beta_app_a[1]-p_beta_app_a[0]:.4f} c (1-sigma: + {p_beta_app_a_1s[2]-p_beta_app_a_1s[1]:.4f} - {p_beta_app_a[1]-p_beta_app_a_1s[0]:.4f})"
)
report_lines.append(
    f"Northern Position Angle (PA_n)    : {p_pa_n[1]:>8.2f} + {p_pa_n[2]-p_pa_n[1]:.2f} - {p_pa_n[1]-p_pa_n[0]:.2f} deg (1-sigma: + {p_pa_n_1s[2]-p_pa_n[1]:.2f} - {p_pa_n[1]-p_pa_n_1s[0]:.2f})"
)
report_lines.append(f"Standard Astronomical PA_n (EoN)  : {p_pa_n[1] + 180:>8.2f} deg")
report_lines.append(
    f"Southern Position Angle (PA_s)    : {p_pa_s[1]:>8.2f} + {p_pa_s[2]-p_pa_s[1]:.2f} - {p_pa_s[1]-p_pa_s[0]:.2f} deg (1-sigma: + {p_pa_s_1s[2]-p_pa_s[1]:.2f} - {p_pa_s[1]-p_pa_s_1s[0]:.2f})"
)
report_lines.append(f"Standard Astronomical PA_s (EoN)  : {p_pa_s[1] + 180:>8.2f} deg")
report_lines.append(
    f"Average Position Angle (PA_avg)   : {p_avg_pa[1]:>8.2f} + {p_avg_pa[2]-p_avg_pa[1]:.2f} - {p_avg_pa[1]-p_avg_pa[0]:.2f} deg (1-sigma: + {p_avg_pa_1s[2]-p_avg_pa[1]:.2f} - {p_avg_pa[1]-p_avg_pa_1s[0]:.2f})"
)
report_lines.append(
    f"Standard Astronomical PA_avg (EoN): {p_avg_pa[1] + 180:>8.2f} + {p_avg_pa[2]-p_avg_pa[1]:.2f} - {p_avg_pa[1]-p_avg_pa[0]:.2f} deg (1-sigma: + {p_avg_pa_1s[2]-p_avg_pa[1]:.2f} - {p_avg_pa[1]-p_avg_pa_1s[0]:.2f})"
)
p_theta_scatter = p_res[param_labels.index(r"$\theta_{\rm scatter}$")]
p_theta_scatter_1s = p_res_1s[param_labels.index(r"$\theta_{\rm scatter}$")]
report_lines.append(
    f"Scatter Angle (theta_scatter)     : {p_theta_scatter[1]:>8.2f} + {p_theta_scatter[2]-p_theta_scatter[1]:.2f} - {p_theta_scatter[1]-p_theta_scatter[0]:.2f} deg (1-sigma: + {p_theta_scatter_1s[2]-p_theta_scatter[1]:.2f} - {p_theta_scatter[1]-p_theta_scatter_1s[0]:.2f})"
)

report_lines.append("\n[3] INTRINSIC RELATIVISTIC KINEMATICS & JET GEOMETRY")
report_lines.append("-" * 70)
report_lines.append(
    f"Physical Distance (D <= D_max)    : {p_D[1]:>8.2f} + {p_D[2]-p_D[1]:.2f} - {p_D[1]-p_D[0]:.2f} kpc (prior: {sys_dist:.1f} +/- 1.0 kpc, 1-sigma: + {p_D_1s[2]-p_D[1]:.2f} - {p_D[1]-p_D_1s[0]:.2f})"
)
report_lines.append(
    f"Derived beta * cos(theta)         : {p_bct[1]:>8.4f} + {p_bct[2]-p_bct[1]:.4f} - {p_bct[1]-p_bct[0]:.4f} (1-sigma: + {p_bct_1s[2]-p_bct[1]:.4f} - {p_bct[1]-p_bct_1s[0]:.4f})"
)
report_lines.append(
    f"Intrinsic Jet Speed (beta = v/c)  : {p_beta[1]:>8.4f} + {p_beta[2]-p_beta[1]:.4f} - {p_beta[1]-p_beta[0]:.4f} c (1-sigma: + {p_beta_1s[2]-p_beta[1]:.4f} - {p_beta[1]-p_beta_1s[0]:.4f})"
)
report_lines.append(
    f"Lorentz Factor (Gamma)            : {p_gamma[1]:>8.4f} + {p_gamma[2]-p_gamma[1]:.4f} - {p_gamma[1]-p_gamma[0]:.4f} (1-sigma: + {p_gamma_1s[2]-p_gamma[1]:.4f} - {p_gamma[1]-p_gamma_1s[0]:.4f})"
)
report_lines.append(
    f"Proper 4-Velocity (Gamma * Beta)  : {p_gammabeta[1]:>8.4f} + {p_gammabeta[2]-p_gammabeta[1]:.4f} - {p_gammabeta[1]-p_gammabeta[0]:.4f} (1-sigma: + {p_gammabeta_1s[2]-p_gammabeta[1]:.4f} - {p_gammabeta[1]-p_gammabeta_1s[0]:.4f})"
)
report_lines.append(
    f"Jet Inclination Angle (theta)     : {p_theta[1]:>8.2f} + {p_theta[2]-p_theta[1]:.2f} - {p_theta[1]-p_theta[0]:.2f} deg (1-sigma: + {p_theta_1s[2]-p_theta[1]:.2f} - {p_theta[1]-p_theta_1s[0]:.2f})"
)
report_lines.append(
    f"Approaching Doppler Factor (delta_a): {p_delta_app[1]:>6.4f} + {p_delta_app[2]-p_delta_app[1]:.4f} - {p_delta_app[1]-p_delta_app[0]:.4f} (1-sigma: + {p_delta_app_1s[2]-p_delta_app_1s[1]:.4f} - {p_delta_app[1]-p_delta_app_1s[0]:.4f})"
)
report_lines.append(
    f"Receding Doppler Factor (delta_r) : {p_delta_rec[1]:>8.4f} + {p_delta_rec[2]-p_delta_rec[1]:.4f} - {p_delta_rec[1]-p_delta_rec[0]:.4f} (1-sigma: + {p_delta_rec_1s[2]-p_delta_rec[1]:.4f} - {p_delta_rec[1]-p_delta_rec_1s[0]:.4f})"
)
report_lines.append(
    f"Kinematic Doppler Factor Ratio    : {p_dratio[1]:>8.2f} + {p_dratio[2]-p_dratio[1]:.2f} - {p_dratio[1]-p_dratio[0]:.2f} (1-sigma: + {p_dratio_1s[2]-p_dratio[1]:.2f} - {p_dratio[1]-p_dratio_1s[0]:.2f})"
)
report_lines.append(
    f"True Intrinsic Angular Velocity   : {p_mutrue[1]:.4e} + {p_mutrue[2]-p_mutrue[1]:.4e} - {p_mutrue[1]-p_mutrue[0]:.4e} arcsec/day (1-sigma: + {p_mutrue_1s[2]-p_mutrue[1]:.4e} - {p_mutrue[1]-p_mutrue_1s[0]:.4e})"
)
report_lines.append(
    f"Maximum Physical Distance (D_max) : {p_Dmax[1]:>8.2f} + {p_Dmax[2]-p_Dmax[1]:.2f} - {p_Dmax[1]-p_Dmax[0]:.2f} kpc (1-sigma: + {p_Dmax_1s[2]-p_Dmax[1]:.2f} - {p_Dmax[1]-p_Dmax_1s[0]:.2f})"
)

if best_model["id"] in [3, 4]:
    report_lines.append("\n[4] POST-BREAK DIFFERENTIAL DECELERATION KINEMATICS")
    report_lines.append("-" * 70)
    report_lines.append(
        f"Approaching Post-Break Speed (beta_2,a): {p_beta_2_a[1]:>6.4f} + {p_beta_2_a[2]-p_beta_2_a[1]:.4f} - {p_beta_2_a[1]-p_beta_2_a[0]:.4f} c (1-sigma: + {p_beta_2_a_1s[2]-p_beta_2_a[1]:.4f} - {p_beta_2_a[1]-p_beta_2_a_1s[0]:.4f})"
    )
    report_lines.append(
        f"Approaching Post-Break Gamma (Gamma_2,a): {p_gamma_2_a[1]:>6.4f} + {p_gamma_2_a[2]-p_gamma_2_a[1]:.4f} - {p_gamma_2_a[1]-p_gamma_2_a[0]:.4f} (1-sigma: + {p_gamma_2_a_1s[2]-p_gamma_2_a[1]:.4f} - {p_gamma_2_a[1]-p_gamma_2_a_1s[0]:.4f})"
    )
    report_lines.append(
        f"Approaching Deceleration Ratio (b2/b1) : {p_ratio_a[1]:>6.2f} + {p_ratio_a[2]-p_ratio_a[1]:.2f} - {p_ratio_a[1]-p_ratio_a[0]:.2f} (1-sigma: + {p_ratio_a_1s[2]-p_ratio_a[1]:.2f} - {p_ratio_a[1]-p_ratio_a_1s[0]:.2f})"
    )
    report_lines.append(
        f"Receding Post-Break Speed (beta_2,r)   : {p_beta_2_r[1]:>6.4f} + {p_beta_2_r[2]-p_beta_2_r[1]:.4f} - {p_beta_2_r[1]-p_beta_2_r[0]:.4f} c (1-sigma: + {p_beta_2_r_1s[2]-p_beta_2_r[1]:.4f} - {p_beta_2_r[1]-p_beta_2_r_1s[0]:.4f})"
    )
    report_lines.append(
        f"Receding Post-Break Gamma (Gamma_2,r)  : {p_gamma_2_r[1]:>6.4f} + {p_gamma_2_r[2]-p_gamma_2_r[1]:.4f} - {p_gamma_2_r[1]-p_gamma_2_r[0]:.4f} (1-sigma: + {p_gamma_2_r_1s[2]-p_gamma_2_r[1]:.4f} - {p_gamma_2_r[1]-p_gamma_2_r_1s[0]:.4f})"
    )
    report_lines.append(
        f"Receding Deceleration Ratio (b2/b1)    : {p_ratio_r[1]:>6.2f} + {p_ratio_r[2]-p_ratio_r[1]:.2f} - {p_ratio_r[1]-p_ratio_r[0]:.2f} (1-sigma: + {p_ratio_r_1s[2]-p_ratio_r[1]:.2f} - {p_ratio_r[1]-p_ratio_r_1s[0]:.2f})"
    )
elif best_model["id"] in [5, 6]:
    report_lines.append("\n[4] TERMINAL STALL & EXPONENTIAL RELAXATION KINEMATICS")
    report_lines.append("-" * 70)
    report_lines.append(
        f"Receding Terminal Stall Radius (r_max,r)   : {p_rmax_n[1]:.4f} + {p_rmax_n[2]-p_rmax_n[1]:.4f} - {p_rmax_n[1]-p_rmax_n[0]:.4f} arcsec (1-sigma: + {p_rmax_n_1s[2]-p_rmax_n[1]:.4f} - {p_rmax_n[1]-p_rmax_n_1s[0]:.4f})"
    )
    report_lines.append(
        f"Approaching Terminal Stall Radius (r_max,a): {p_rmax_s[1]:.4f} + {p_rmax_s[2]-p_rmax_s[1]:.4f} - {p_rmax_s[1]-p_rmax_s[0]:.4f} arcsec (1-sigma: + {p_rmax_s_1s[2]-p_rmax_s[1]:.4f} - {p_rmax_s[1]-p_rmax_s_1s[0]:.4f})"
    )
    report_lines.append(
        f"Receding Relaxation Timescale (tau_r)      : {p_tau_n[1]:.1f} + {p_tau_n[2]-p_tau_n[1]:.1f} - {p_tau_n[1]-p_tau_n[0]:.1f} days (1-sigma: + {p_tau_n_1s[2]-p_tau_n[1]:.1f} - {p_tau_n[1]-p_tau_n_1s[0]:.1f})"
    )
    report_lines.append(
        f"Approaching Relaxation Timescale (tau_a)   : {p_tau_s[1]:.1f} + {p_tau_s[2]-p_tau_s[1]:.1f} - {p_tau_s[1]-p_tau_s[0]:.1f} days (1-sigma: + {p_tau_s_1s[2]-p_tau_s[1]:.1f} - {p_tau_s[1]-p_tau_s_1s[0]:.1f})"
    )
    report_lines.append(
        f"Final Epoch Receding Speed (mu_end,r)      : {p_mu_final_r[1]:.4e} + {p_mu_final_r[2]-p_mu_final_r[1]:.4e} - {p_mu_final_r[1]-p_mu_final_r[0]:.4e} arcsec/day"
    )
    report_lines.append(
        f"Final Epoch Receding Apparent Speed        : {p_beta_final_r[1]:>6.4f} + {p_beta_final_r[2]-p_beta_final_r[1]:.4f} - {p_beta_final_r[1]-p_beta_final_r[0]:.4f} c"
    )
    report_lines.append(
        f"Final Epoch Approaching Speed (mu_end,a)    : {p_mu_final_a[1]:.4e} + {p_mu_final_a[2]-p_mu_final_a[1]:.4e} - {p_mu_final_a[1]-p_mu_final_a[0]:.4e} arcsec/day"
    )
    report_lines.append(
        f"Final Epoch Approaching Apparent Speed      : {p_beta_final_a[1]:>6.4f} + {p_beta_final_a[2]-p_beta_final_a[1]:.4f} - {p_beta_final_a[1]-p_beta_final_a[0]:.4f} c"
    )

report_lines.append("=" * 70)

report_output_text = "\n".join(report_lines)
print("\n" + report_output_text + "\n")

out_report_file = f"data/kinematic_report{_save_suffix}.txt"
with open(out_report_file, "w") as f:
    f.write("\n".join(model_sel_lines) + "\n\n" + report_output_text + "\n")
with open(f"data/kinematic_report{_lim_suffix}.txt", "w") as f:
    f.write("\n".join(model_sel_lines) + "\n\n" + report_output_text + "\n")
print(f"Saved kinematic report to {out_report_file}\n")

print("\nGenerating prior-free empirical kinematics corner plot...")
_hdi_prob = getattr(args, "hdi_prob", 0.90) if "args" in globals() else 0.90
_q_tail = (1.0 - _hdi_prob) / 2.0
corner_quantiles = [_q_tail, 0.5, 1.0 - _q_tail]

mu_rec_mas = (mu_rec_1_chain if best_model["id"] in [3, 4] else mu_rec_chain) * 1000.0
mu_app_mas = (mu_app_1_chain if best_model["id"] in [3, 4] else mu_app_chain) * 1000.0

samples_empirical = np.vstack([
    t_ej_chain,
    mu_rec_mas,
    mu_app_mas,
    D_max_kpc_chain,
    bct_chain
]).T

is_decay = best_model["id"] in [3, 4, 5, 6]
kin_labels = [
    r"$T_{\mathrm{ej}}\ (\mathrm{MJD})$",
    r"$\mu_{0,r}\ (\mathrm{mas\,d^{-1}})$" if is_decay else r"$\mu_{r}\ (\mathrm{mas\,d^{-1}})$",
    r"$\mu_{0,a}\ (\mathrm{mas\,d^{-1}})$" if is_decay else r"$\mu_{a}\ (\mathrm{mas\,d^{-1}})$",
    r"$D_{\max}\ (\mathrm{kpc})$",
    r"$\beta\cos\theta$"
]
kin_titles = [
    r"$T_{\mathrm{ej}}$",
    r"$\mu_{0,r}$" if is_decay else r"$\mu_{r}$",
    r"$\mu_{0,a}$" if is_decay else r"$\mu_{a}$",
    r"$D_{\max}$",
    r"$\beta\cos\theta$"
]

fig_corner_kin = corner.corner(
    samples_empirical,
    labels=kin_labels,
    titles=kin_titles,
    quantiles=corner_quantiles,
    show_titles=True,
    title_kwargs={"fontsize": 11},
    label_kwargs={"fontsize": 11},
    title_fmt=".2f",
)
fig_corner_kin.savefig(
    f"figures/absolute_corner_plot_kinematics{_save_suffix}.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig_corner_kin)
print(
    f"Saved prior-free kinematic corner plot to figures/absolute_corner_plot_kinematics{_save_suffix}.png"
)

# Generate plots
fig_corner = corner.corner(
    flat_samples,
    labels=param_labels,
    quantiles=corner_quantiles,
    show_titles=True,
    title_kwargs={"fontsize": 10},
    title_fmt=".3f",
)
fig_corner.savefig(f"figures/absolute_corner_plot{_save_suffix}.png", dpi=300)
plt.close(fig_corner)

fig_core, ax = plt.subplots(figsize=(8, 8))
ax.errorbar(
    core_data[:, 1],
    core_data[:, 2],
    xerr=core_data[:, 3],
    yerr=core_data[:, 4],
    fmt="o",
    alpha=0.5,
    label="Filtered Core Data",
)
ax.scatter(
    p_res[xc_idx][1],
    p_res[yc_idx][1],
    color="red",
    marker="*",
    s=300,
    label="Global Fitted Core Position",
    zorder=5,
)
ax.set_xlabel(r"$\Delta \alpha \cos \delta$ (arcsec)", fontsize=14)
ax.set_ylabel(r"$\Delta \delta$ (arcsec)", fontsize=14)
ax.axhline(0, color="grey", ls="--")
ax.axvline(0, color="grey", ls="--")
if len(omitted_core_data) > 0:
    ax.scatter(
        omitted_core_data[:, 1],
        omitted_core_data[:, 2],
        marker="x",
        color="red",
        s=100,
        label="Omitted Core Data",
        zorder=4,
    )
ax.legend()
plt.savefig("figures/core_mixture_fit.png", dpi=300)
plt.close(fig_core)

# -----------------------------------------------------------------------------
# 6. RA/DEC KINEMATICS PLOT (MERGED)
# -----------------------------------------------------------------------------
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator

print("\nGenerating RA/Dec Kinematics Plot...")

pa_n_deg = p_res[pa_n_idx][1]
pa_s_deg = p_res[pa_s_idx][1]
theta_scatter_deg = p_res[param_labels.index(r"$\theta_{\rm scatter}$")][1]

weighted_mean_pa_deg = (pa_n_deg + pa_s_deg) / 2.0
weighted_std_deg = theta_scatter_deg

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
gs = gridspec.GridSpec(
    2, 2, width_ratios=[1, 0.06], height_ratios=[1, 1], wspace=0.08, hspace=0.15
)

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
ax.plot(x_line, y_line, color="black", linestyle="--", lw=1.8, zorder=1.5)

pa_hi = avg_pa_rad + std_pa_rad
pa_lo = avg_pa_rad - std_pa_rad

ax.fill(
    [0, L * np.sin(pa_lo), L * np.sin(pa_hi)],
    [0, L * np.cos(pa_lo), L * np.cos(pa_hi)],
    color="gray",
    alpha=0.22,
    zorder=1,
    edgecolor="none",
)
ax.fill(
    [0, -L * np.sin(pa_lo), -L * np.sin(pa_hi)],
    [0, -L * np.cos(pa_lo), -L * np.cos(pa_hi)],
    color="gray",
    alpha=0.22,
    zorder=1,
    edgecolor="none",
)

ax.scatter(
    0, 0, marker="*", s=180, color="gold", edgecolors="black", linewidth=1.2, zorder=4
)

ax.errorbar(
    n_da,
    n_dd,
    xerr=n_da_err,
    yerr=n_dd_err,
    fmt="none",
    ecolor="gray",
    capsize=0,
    zorder=2,
    alpha=0.6,
    elinewidth=1.2,
)
ax.errorbar(
    s_da,
    s_dd,
    xerr=s_da_err,
    yerr=s_dd_err,
    fmt="none",
    ecolor="gray",
    capsize=0,
    zorder=2,
    alpha=0.6,
    elinewidth=1.2,
)


scatter_north = ax.scatter(
    n_da,
    n_dd,
    c=n_mjd - 60000,
    cmap=cmap_north,
    norm=norm,
    marker="^",
    edgecolors="black",
    s=60,
    zorder=3,
    label=r"$\Delta\theta^N$ Data",
)
scatter_south = ax.scatter(
    s_da,
    s_dd,
    c=s_mjd - 60000,
    cmap=cmap_south,
    norm=norm,
    marker="v",
    edgecolors="black",
    s=60,
    zorder=3,
    label=r"$\Delta\theta^S$ Data",
)


ax.set_xlim(-5.0, 5.0)
ax.set_ylim(-5.0, 5.0)
ax.set_xlabel(r"$\Delta\alpha \cos\delta$ ($^{\prime\prime}$)", fontsize=22)
ax.set_ylabel(r"$\Delta\delta$ ($^{\prime\prime}$)", fontsize=22)

ax.xaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(2))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
ax.minorticks_on()
ax.tick_params(
    axis="both",
    which="major",
    direction="in",
    top=True,
    right=True,
    labelsize=16,
    length=7,
    width=1.4,
)
ax.tick_params(
    axis="both",
    which="minor",
    direction="in",
    top=True,
    right=True,
    length=3.5,
    width=1.0,
)
ax.invert_xaxis()
ax.grid(True, linestyle=":", alpha=0.5)

cb_north = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap_north), cax=cax_north)
cb_north.set_label("MJD - 60000 (NW)", fontsize=14)
cb_north.ax.tick_params(labelsize=12)

cb_south = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap_south), cax=cax_south)
cb_south.set_label("MJD - 60000 (SE)", fontsize=14)
cb_south.ax.tick_params(labelsize=12)
cb_south.ax.invert_yaxis()

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    plt.tight_layout()
plt.savefig(f"figures/ra_dec_plot{_save_suffix}.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Successfully generated and saved merged figures/ra_dec_plot{_save_suffix}.png")

# -----------------------------------------------------------------------------
# 6.5. SAVE DATA FOR STANDALONE SEPARATION PLOT
# -----------------------------------------------------------------------------
print("\nSaving data for standalone parallel/perpendicular separation plotting...")
os.makedirs("data", exist_ok=True)
np.savez(
    f"data/absolute_kinematics_plot_data{_save_suffix}.npz",
    flat_samples=flat_samples,
    best_model_id=best_model["id"],
    p_Tej=p_Tej,
    best_model_res_x=best_model["res"].x,
    anchor_date=args.anchor_date,
    n_mjd=n_mjd,
    n_da=n_da,
    n_dd=n_dd,
    n_da_err=n_da_err,
    n_dd_err=n_dd_err,
    s_mjd=s_mjd,
    s_da=s_da,
    s_dd=s_dd,
    s_da_err=s_da_err,
    s_dd_err=s_dd_err,
    c_mjd=c_mjd,
)
np.savez(
    f"data/absolute_kinematics_plot_data{_lim_suffix}.npz",
    flat_samples=flat_samples,
    best_model_id=best_model["id"],
    p_Tej=p_Tej,
    best_model_res_x=best_model["res"].x,
    anchor_date=args.anchor_date,
    n_mjd=n_mjd,
    n_da=n_da,
    n_dd=n_dd,
    n_da_err=n_da_err,
    n_dd_err=n_dd_err,
    s_mjd=s_mjd,
    s_da=s_da,
    s_dd=s_dd,
    s_da_err=s_da_err,
    s_dd_err=s_dd_err,
    c_mjd=c_mjd,
)
print(f"Saved plot data to data/absolute_kinematics_plot_data{_save_suffix}.npz")

if args.plot_separation:
    print("\nGenerating Parallel/Perpendicular Separation Plot...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plot_absolute_separation import plot_absolute_separation

    if args.show_separation_limits == args.include_limits:
        out_fig = f"figures/absolute_parallel_perp_fit_plot{_save_suffix}.png"
    elif args.show_separation_limits:
        out_fig = f"figures/absolute_parallel_perp_fit_plot{_save_suffix}_with_arrows.png"
    else:
        out_fig = f"figures/absolute_parallel_perp_fit_plot{_save_suffix}_no_arrows.png"

    plot_absolute_separation(
        with_limits=args.include_limits,
        show_limits=args.show_separation_limits,
        out_fig=out_fig,
        flat_samples=flat_samples,
        best_model_id=best_model["id"],
        p_Tej=p_Tej,
        best_model_res_x=best_model["res"].x,
        anchor_date=args.anchor_date,
        n_mjd=n_mjd,
        n_da=n_da,
        n_dd=n_dd,
        n_da_err=n_da_err,
        n_dd_err=n_dd_err,
        s_mjd=s_mjd,
        s_da=s_da,
        s_dd=s_dd,
        s_da_err=s_da_err,
        s_dd_err=s_dd_err,
        c_mjd=c_mjd,
    )

# -----------------------------------------------------------------------------
# 7. FLUX EVOLUTION AND EXTRAPOLATION
# -----------------------------------------------------------------------------
print("\nGenerating Flux Evolution Plot...")


def exp_model(t, a, b):
    return a * np.exp(b * t)


def pl_model(t, a, b):
    # offset by 10 days to avoid singularity near t=0 if t is relative to outburst
    return a * (t + 10) ** b


import datetime

from scipy.optimize import curve_fit

fig_flux, ax_flux = plt.subplots(figsize=(10, 6))

target_dates = [
    "2025-01-11 06:03:36",
    "2025-02-02 06:13:52",
    "2025-03-02 02:44:41",
    "2025-03-24 01:35:33",
]
target_ts = [
    datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S").timestamp() for d in target_dates
]
t0_mjd = target_ts[0] / 86400.0 + 40587.0
target_mjd = np.array(target_ts) / 86400.0 + 40587.0

colors = {"Core": "black", "NW Lobe": "blue", "SE Lobe": "red"}
markers = {"Core": "o", "NW Lobe": "s", "SE Lobe": "^"}

for label, dataset in [("Core", core_data), ("NW Lobe", n_data), ("SE Lobe", s_data)]:
    if len(dataset) < 3:
        continue

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
        if label == "Core":
            # 4th order polynomial fit in log10 space
            poly_coeffs = np.polyfit(t_days, np.log10(f_data), 4)
            t_fit = np.linspace(min(t_data) - 5, max(t_data) + 5, 200)
            f_fit = 10 ** np.polyval(poly_coeffs, t_fit - t0_mjd)
        else:
            popt, _ = curve_fit(
                exp_model,
                t_days,
                f_data,
                sigma=err_data,
                absolute_sigma=True,
                p0=[f_data[0], -0.01],
            )
            f_extrap = exp_model(target_days, *popt)
            t_fit = np.linspace(min(target_mjd) - 5, max(t_data) + 5, 200)
            f_fit = exp_model(t_fit - t0_mjd, *popt)

        ax_flux.errorbar(
            t_data - 60000,
            f_data,
            yerr=err_data,
            fmt=markers[label],
            color=colors[label],
            label=f"{label} Data",
            capsize=3,
        )
        ax_flux.plot(
            t_fit - 60000, f_fit, color=colors[label], linestyle="--", alpha=0.7
        )
        if label != "Core":
            ax_flux.scatter(
                target_mjd - 60000,
                f_extrap,
                facecolors="none",
                edgecolors=colors[label],
                marker=markers[label],
                s=100,
                linewidth=2,
                label=f"{label} Extrapolated",
            )
    except Exception as e:
        print(f"Failed to fit {label}: {e}")

ax_flux.set_yscale("log")
ax_flux.set_xlabel("Time (MJD - 60000)", fontsize=14)
ax_flux.set_ylabel("Flux (Jy)", fontsize=14)
ax_flux.set_title("Flux Evolution and Extrapolation", fontsize=16)
ax_flux.grid(True, which="both", linestyle="--", alpha=0.5)

if len(omitted_core_data) > 0:
    ax_flux.scatter(
        omitted_core_data[:, 0] - 60000,
        omitted_core_data[:, 5],
        marker="x",
        color="red",
        s=100,
        label="Omitted Core Data",
        zorder=5,
    )

ax_flux.legend(loc="best")
plt.tight_layout()
plt.savefig("figures/flux_evolution_with_extrap.png", dpi=300)
plt.close(fig_flux)
print("Successfully generated and saved figures/flux_evolution_with_extrap.png")

# Auto-sync plain-text copies to scripts_txt/
try:
    from sync_txt_scripts import sync_all_scripts
    sync_all_scripts(verbose=False)
except Exception:
    pass

