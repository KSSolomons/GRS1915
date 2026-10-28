"""
Physical (light-travel-time) kinematics modelling of the GRS 1915+105 2025 ejecta.

Unlike pipeline_absolute_kinematics_with_limits.py, which fits apparent proper
motions for each lobe independently, this module parameterises the ejection in
intrinsic quantities at a fixed source distance D:

    launch epoch T_ej, launch speed beta_0, inclination theta (shared by both sides),
    and an intrinsic deceleration timescale tau (per side, or shared).

Each ejecton travels a radial distance R(t') in coordinate time t' since launch,

    beta(t') = beta_0 exp(-t'/tau),      R(t') = beta_0 c tau (1 - exp(-t'/tau)),

and is observed at t_obs = T_ej + t' -/+ R(t') cos(theta) / c (approaching / receding),
with projected separation R(t') sin(theta) / D. The light-travel-time mapping is
integrated exactly (numerically inverted), so an intrinsically symmetric
deceleration produces different apparent timescales on the two sides without any
extra parameters.

The launch speed is fitted as lg_gb0 = log10(Gamma_0 beta_0) with a uniform prior on
LOG_GB0_RANGE. This keeps beta_0 < 1 by construction and turns the approach to the
speed of light into a long tail in lg_gb0 instead of a pile-up at a hard wall in
beta_0; a fit that reaches the upper end of LOG_GB0_RANGE means beta_0 is unconstrained
from above (flagged in the report).

Models (k at fixed D):
    ballistic   : tau -> infinity                                      (k = 8)
    decel_sym   : one intrinsic tau for both sides                     (k = 9)
    decel_asym  : independent tau_a, tau_r (different environments)  (k = 10)
    coast_decel : constant speed for t_coast, then decel_sym           (k = 10, only with --coast)

The core likelihood, perpendicular scatter term and one-sided beam upper-limit
penalty follow the absolute-kinematics pipeline so the BIC values are comparable.
The blending limits come from epoch_limits.py.

With --profile, the baseline configuration is refitted at fixed beta_0 values
(--profile-beta0) to show how much the data constrain the launch speed.
"""

import argparse
import glob
import multiprocessing as mp
import os
import re
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import lambertw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from epoch_limits import EJECTION_CAMPAIGN_START, blending_limits
from frame_offsets import apply_frame_offset, load_frame_offsets

parser = argparse.ArgumentParser(description="Physical light-travel-time kinematics fits")
parser.add_argument(
    "--distances",
    type=float,
    nargs="+",
    default=[8.4, 9.4, 10.4],
    help="Fixed source distances (kpc) to fit at (default: 8.4 9.4 10.4)",
)
parser.add_argument(
    "--ul-scales",
    type=float,
    nargs="+",
    default=[0.5, 1.0, 1.5],
    help="Multiples of the projected beam FWHM used as the blending upper limit (default: 0.5 1.0 1.5)",
)
parser.add_argument(
    "--ul-sigmas",
    type=float,
    nargs="+",
    default=[0.05, 0.25],
    help="Upper-limit penalty widths (arcsec) to test (default: 0.05 0.25)",
)
parser.add_argument(
    "--no-limits-run",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Also fit every model without the upper-limit penalty (default: True)",
)
parser.add_argument(
    "--epoch12-north",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Fit the Epoch 12 N lobe (the Epoch 12 S feature is never fitted; default: True, --no-epoch12-north "
    "reproduces the fits before 2026-10-28)",
)
parser.add_argument(
    "--ul-mjd",
    type=float,
    nargs="+",
    default=None,
    help="Custom upper-limit epochs (MJD). With --ul-radius and --ul-width this replaces the beam-FWHM grid",
)
parser.add_argument("--ul-radius", type=float, nargs="+", default=None, help="Custom limit radius per epoch (arcsec), e.g. the peel radius")
parser.add_argument("--ul-width", type=float, nargs="+", default=None, help="Custom penalty width per epoch (arcsec), e.g. sigma_pos")
parser.add_argument(
    "--report",
    default=None,
    help="Report path (default: data/physical_kinematics_report.txt, or ..._custom_limits.txt with --ul-mjd). "
    "The maximum-likelihood fits are saved beside it with .npz in place of .txt",
)
parser.add_argument("--n-starts", type=int, default=24, help="Random Nelder-Mead restarts per fit")
parser.add_argument(
    "--jobs",
    type=int,
    default=os.cpu_count(),
    help="Parallel worker processes for the grid and the profile (default: all cores; 1 = serial)",
)
parser.add_argument("--coast", action="store_true", help="Also fit the coast-then-decelerate model (k = 10)")
parser.add_argument(
    "--profile",
    action="store_true",
    help="Profile likelihood in beta_0 at the baseline configuration (D=9.4 kpc, 1.0 x FWHM, 0.05 arcsec)",
)
parser.add_argument(
    "--profile-beta0",
    type=float,
    nargs="+",
    default=[0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999],
    help="Fixed beta_0 values for --profile (default: 0.3 0.5 0.7 0.9 0.95 0.99 0.999)",
)
parser.add_argument(
    "--profile-no-limits",
    action="store_true",
    help="Profile and MCMC the no-limits fit at --profile-distance instead of the limited baseline (needs --no-limits-run)",
)
parser.add_argument("--profile-distance", type=float, default=9.4, help="Distance (kpc) of the profile and MCMC configuration (default: 9.4)")
parser.add_argument(
    "--beta0-range",
    type=float,
    nargs=2,
    default=None,
    metavar=("LO", "HI"),
    help="Restrict the prior on beta_0 to (LO, HI), applied to every fit and to --mcmc (default: 0.03 to 0.99995)",
)
parser.add_argument(
    "--mcmc",
    action="store_true",
    help="Sample posteriors with emcee for the baseline configuration (D=9.4 kpc, 1.0 x FWHM, 0.05 arcsec)",
)
parser.add_argument("--mcmc-models", nargs="+", default=None, help="Models to sample with --mcmc (default: all)")
parser.add_argument("--samples-suffix", default="", help="Appended to the MCMC samples file names, e.g. _b0.99")
parser.add_argument("--n-steps", type=int, default=10000, help="MCMC steps per walker")
parser.add_argument("--n-walkers", type=int, default=128, help="MCMC walkers")
parser.add_argument("--seed", type=int, default=1915, help="Random seed")
args = parser.parse_args()

rng = np.random.default_rng(args.seed)

C_LD_PER_DAY = 1.0  # work in light-days, so c = 1
AU_PER_LD = 173.144633  # astronomical units per light-day
EPOCH12_TS = 1764664923  # Epoch 12: N lobe detected (fitted unless --no-epoch12-north), S feature not a detection

LOG_GB0_RANGE = (-1.5, 2.0)  # prior on log10(Gamma_0 beta_0): beta_0 from 0.03 to 0.99995
TAU_RANGE = (5.0, 5000.0)  # prior on the deceleration timescales (days)
T_COAST_RANGE = (0.0, 1000.0)  # prior on the coasting time (days)
BASELINE_D, BASELINE_UL_SCALE, BASELINE_UL_SIGMA = args.profile_distance, 1.0, 0.05

# Blending upper limits of the blocks flagged "in_fit" in epoch_limits.py (projected beam FWHM along the jet axis)
UL_MJD, UL_FWHM, _ = blending_limits()

# Custom per-epoch limits (e.g. peel radius and sigma_pos = theta_beam / (2 SNR) from the peeled-image rms)
CUSTOM_LIMITS = args.ul_mjd is not None
if CUSTOM_LIMITS:
    if not (args.ul_radius and args.ul_width and len(args.ul_mjd) == len(args.ul_radius) == len(args.ul_width)):
        parser.error("--ul-mjd, --ul-radius and --ul-width must be given together with one value per epoch")
    UL_MJD = np.array(args.ul_mjd)
    UL_FWHM = np.array(args.ul_radius)  # used directly as the limit radius (ul_scale = 1)


# 1. LOAD DATA (same parsing and frame correction as the absolute-kinematics pipeline)
dir_path = "S-band-fits"
files = sorted(glob.glob(os.path.join(dir_path, "*2D_Fitting*.txt")))

act_ra_deg = (19 + 15 / 60.0 + 11.558 / 3600.0) * 15.0
act_dec_deg = 10 + 56 / 60.0 + 44.905 / 3600.0
cos_dec = np.cos(np.radians(act_dec_deg))

FRAME_OFFSETS = load_frame_offsets()
core_data_raw = []
n_data = []
s_data = []
no_offset = set()

for f in files:
    m = re.search(r"img_(\d+)", os.path.basename(f))
    if not m:
        continue
    ts = int(m.group(1))
    mjd = ts / 86400.0 + 40587.0

    comps = []
    curr = None
    with open(f, "r") as fp:
        for line in fp:
            if line.startswith("Component #"):
                curr = {}
                comps.append(curr)
            elif "Center X" in line and curr is not None:
                m_ra = re.search(r"=\s*(\d+):(\d+):(\d+\.\d+)\s*±\s*(\S+)", line)
                if m_ra:
                    h, mi, s, e = (float(g) for g in m_ra.groups())
                    curr["ra_deg"] = (h + mi / 60.0 + s / 3600.0) * 15.0
                    curr["ra_err"] = e * 15.0 * cos_dec
            elif "Center Y" in line and curr is not None:
                m_dec = re.search(r"=\s*([+-]?\d+):(\d+):(\d+\.\d+)\s*±\s*(\S+)", line)
                if m_dec:
                    d, mi, s, e = m_dec.groups()
                    sign = -1 if "-" in d else 1
                    curr["dec_deg"] = sign * (abs(float(d)) + float(mi) / 60.0 + float(s) / 3600.0)
                    curr["dec_err"] = float(e)

    for c in comps:
        if "ra_deg" not in c or "dec_deg" not in c:
            continue
        dx = (c["ra_deg"] - act_ra_deg) * 3600.0 * cos_dec
        dy = (c["dec_deg"] - act_dec_deg) * 3600.0
        # Tie the epoch to the common frame; epochs without a measured offset are left out
        corrected = apply_frame_offset(FRAME_OFFSETS, ts, dx, dy, c["ra_err"], c["dec_err"])
        if corrected is None:
            no_offset.add(round(mjd, 2))
            continue
        row = [mjd, *corrected]
        if 288.7979 < c["ra_deg"] < 288.7982:
            core_data_raw.append(row)
        elif mjd > EJECTION_CAMPAIGN_START and (ts != EPOCH12_TS or (dy > 0 and args.epoch12_north)):
            (n_data if dy > 0 else s_data).append(row)

if no_offset:
    print(f"No frame offset for MJD {sorted(no_offset)}; those epochs are left out")
core_data = np.array(sorted(core_data_raw))
n_data = np.array(sorted(n_data))
s_data = np.array(sorted(s_data))
num_obs_total = 2 * (len(core_data) + len(n_data) + len(s_data))

print(f"Loaded {len(core_data)} core, {len(n_data)} N (receding), {len(s_data)} S (approaching) points; N_data = {num_obs_total}")


# 2. PHYSICAL TRAJECTORY WITH EXACT LIGHT-TRAVEL-TIME MAPPING
def radial_distance(tp, beta0, tau, t_coast=0.0):
    """Intrinsic distance travelled (light-days) after coordinate time tp (days)."""
    if not np.isfinite(tau):
        return beta0 * tp
    t_dec = np.clip(tp - t_coast, 0.0, None)
    return beta0 * np.minimum(tp, t_coast) + beta0 * tau * (-np.expm1(-t_dec / tau))


def speed(tp, beta0, tau, t_coast=0.0):
    """Intrinsic speed beta at coordinate time tp (days)."""
    if not np.isfinite(tau):
        return beta0 * np.ones_like(np.asarray(tp, dtype=float))
    return beta0 * np.exp(-np.clip(tp - t_coast, 0.0, None) / tau)


def intrinsic_time(dt_obs, beta0, theta_deg, tau, side, t_coast=0.0):
    """
    Coordinate time since launch (days) at observer times T_ej + dt_obs (dt_obs > 0).
    side = +1 receding, -1 approaching.

    Exact inversion of t_obs - T_ej = t' + side * R(t') cos(theta) / c. During the coast
    (t' <= t_coast) this is linear. After it, with s = t' - t_coast and
    A = side * beta0 * tau * cos(theta) / c, the remaining interval obeys
        dt = s + A (1 - exp(-s / tau)),
    whose solution is s = dt - A + tau W0[(A / tau) exp(-(dt - A) / tau)] with W0 the
    principal Lambert W branch (|A| / tau = beta0 cos(theta) < 1 keeps the argument above -1/e).
    """
    dt_obs = np.asarray(dt_obs, dtype=float)
    cth = np.cos(np.radians(theta_deg))
    doppler = 1.0 + side * beta0 * cth
    if not np.isfinite(tau):
        return dt_obs / doppler
    dt_coast = t_coast * doppler  # observer time at the end of the coast
    dt_dec = np.clip(dt_obs - dt_coast, 0.0, None)
    A = side * beta0 * tau * cth / C_LD_PER_DAY
    s = dt_dec - A + tau * np.real(lambertw((A / tau) * np.exp(-(dt_dec - A) / tau)))
    return np.where(dt_obs <= dt_coast, dt_obs / doppler, t_coast + s)


def apparent_separation(t_obs, t_ej, beta0, theta_deg, tau, dist_kpc, side, t_coast=0.0):
    """
    Projected separation (arcsec) at observer times t_obs for one ejecton.
    side = +1 receding, -1 approaching.
    """
    t_obs = np.atleast_1d(np.asarray(t_obs, dtype=float))
    dt_obs = t_obs - t_ej
    out = np.zeros_like(dt_obs)
    pos = dt_obs > 0
    if not np.any(pos):
        return out
    tp = intrinsic_time(dt_obs[pos], beta0, theta_deg, tau, side, t_coast)
    out[pos] = radial_distance(tp, beta0, tau, t_coast) * np.sin(np.radians(theta_deg)) * AU_PER_LD / (dist_kpc * 1000.0)
    return out


# 3. MODELS, PRIORS AND LIKELIHOOD
MODELS = {
    "ballistic": ["t_ej", "lg_gb0", "theta", "pa_n", "pa_s", "theta_scatter", "x_c", "y_c"],
    "decel_sym": ["t_ej", "lg_gb0", "theta", "tau", "pa_n", "pa_s", "theta_scatter", "x_c", "y_c"],
    "decel_asym": ["t_ej", "lg_gb0", "theta", "tau_a", "tau_r", "pa_n", "pa_s", "theta_scatter", "x_c", "y_c"],
}
if args.coast:
    MODELS["coast_decel"] = ["t_ej", "lg_gb0", "theta", "tau", "t_coast", "pa_n", "pa_s", "theta_scatter", "x_c", "y_c"]


def gb_to_beta(gb):
    return gb / np.sqrt(1.0 + gb**2)


def beta_to_lg_gb(beta):
    return np.log10(beta / np.sqrt(1.0 - beta**2))


if args.beta0_range is not None:
    LOG_GB0_RANGE = tuple(float(beta_to_lg_gb(b)) for b in args.beta0_range)


def unpack(model, p):
    d = dict(zip(MODELS[model], p))
    d["beta0"] = gb_to_beta(10 ** d["lg_gb0"])
    d.setdefault("t_coast", 0.0)
    if model == "ballistic":
        d["tau_a"] = d["tau_r"] = np.inf
    elif model in ("decel_sym", "coast_decel"):
        d["tau_a"] = d["tau_r"] = d["tau"]
    return d


def log_prior(model, d):
    if not (60000.0 < d["t_ej"] < 60800.0):
        return -np.inf
    if not (LOG_GB0_RANGE[0] < d["lg_gb0"] < LOG_GB0_RANGE[1]):
        return -np.inf
    if not (0.0 < d["theta"] < 90.0):  # theta < 90 deg makes the SE (southern) ejecton the approaching one
        return -np.inf
    for key in ("tau_a", "tau_r"):
        if np.isfinite(d[key]) and not (TAU_RANGE[0] < d[key] < TAU_RANGE[1]):
            return -np.inf
    if not (T_COAST_RANGE[0] <= d["t_coast"] < T_COAST_RANGE[1]):
        return -np.inf
    if not (-180.0 < d["pa_n"] < 180.0 and -180.0 < d["pa_s"] < 180.0):
        return -np.inf
    if not (0.0 <= d["theta_scatter"] <= 45.0):
        return -np.inf
    if not (-2.0 < d["x_c"] < 2.0 and -2.0 < d["y_c"] < 2.0):
        return -np.inf
    return 0.0


def lobe_log_likelihood(data, model_par, pa, tan2_scatter, x_c, y_c):
    dx = data[:, 1] - x_c
    dy = data[:, 2] - y_c
    ex, ey = data[:, 3], data[:, 4]
    sin_p, cos_p = np.sin(np.radians(pa)), np.cos(np.radians(pa))
    par = dx * sin_p + dy * cos_p
    perp = dx * cos_p - dy * sin_p
    var_par = (ex * sin_p) ** 2 + (ey * cos_p) ** 2
    var_perp = (ex * cos_p) ** 2 + (ey * sin_p) ** 2 + model_par**2 * tan2_scatter
    ll = -0.5 * np.sum((par - model_par) ** 2 / var_par + np.log(2 * np.pi * var_par))
    ll += -0.5 * np.sum(perp**2 / var_perp + np.log(2 * np.pi * var_perp))
    return ll


def log_likelihood(d, dist_kpc, ul_scale=1.0, ul_sigma=0.05, include_limits=True):
    tan2_scatter = np.tan(np.radians(d["theta_scatter"])) ** 2
    x_c, y_c = d["x_c"], d["y_c"]

    ex_c, ey_c = core_data[:, 3], core_data[:, 4]
    ll = np.sum(-0.5 * ((core_data[:, 1] - x_c) ** 2 / ex_c**2 + np.log(2 * np.pi * ex_c**2)))
    ll += np.sum(-0.5 * ((core_data[:, 2] - y_c) ** 2 / ey_c**2 + np.log(2 * np.pi * ey_c**2)))

    # Receding NW lobe moves along +parallel, approaching SE lobe along -parallel (PA convention of the main pipeline)
    kin = (d["t_ej"], d["beta0"], d["theta"])
    model_n = apparent_separation(n_data[:, 0], *kin, d["tau_r"], dist_kpc, +1, d["t_coast"])
    model_s = -apparent_separation(s_data[:, 0], *kin, d["tau_a"], dist_kpc, -1, d["t_coast"])
    ll += lobe_log_likelihood(n_data, model_n, d["pa_n"], tan2_scatter, x_c, y_c)
    ll += lobe_log_likelihood(s_data, model_s, d["pa_s"], tan2_scatter, x_c, y_c)

    if include_limits:
        ul_r = ul_scale * UL_FWHM
        for tau, side in ((d["tau_r"], +1), (d["tau_a"], -1)):
            excess = apparent_separation(UL_MJD, *kin, tau, dist_kpc, side, d["t_coast"]) - ul_r
            ll += -0.5 * np.sum((np.clip(excess, 0.0, None) / ul_sigma) ** 2)
    return ll


def log_prob(p, model, dist_kpc, ul_scale, ul_sigma, include_limits):
    d = unpack(model, p)
    lp = log_prior(model, d)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(d, dist_kpc, ul_scale, ul_sigma, include_limits)
    return lp + ll if np.isfinite(ll) else -np.inf


# 4. MAXIMUM-LIKELIHOOD FITS WITH RESTARTS
# Random starts in lg_gb0 from (-0.5, 0.7), or from the whole prior when --beta0-range lies outside that window
LG_START = (max(-0.5, LOG_GB0_RANGE[0]), min(0.7, LOG_GB0_RANGE[1]))
if LG_START[0] >= LG_START[1]:
    LG_START = LOG_GB0_RANGE
def random_start(model):
    p = {
        "t_ej": rng.uniform(60450.0, 60720.0),
        "lg_gb0": rng.uniform(*LG_START),
        "theta": rng.uniform(50.0, 85.0),
        "tau": 10 ** rng.uniform(1.7, 3.3),
        "tau_a": 10 ** rng.uniform(1.7, 3.3),
        "tau_r": 10 ** rng.uniform(1.7, 3.3),
        "t_coast": rng.uniform(0.0, 200.0),
        "pa_n": rng.normal(-42.0, 2.0),
        "pa_s": rng.normal(-41.0, 2.0),
        "theta_scatter": rng.uniform(2.0, 8.0),
        "x_c": -0.191,
        "y_c": -0.298,
    }
    return np.array([p[name] for name in MODELS[model]])


def fit_ml(model, dist_kpc, ul_scale, ul_sigma, include_limits, seeds=(), fixed=None, n_starts=None):
    """
    Maximum-likelihood fit with Nelder-Mead restarts from the seeds plus n_starts random
    points (default --n-starts). fixed = {name: value} holds parameters at given values
    (used for the profile likelihood); k counts only free ones.
    """
    n_starts = args.n_starts if n_starts is None else n_starts
    names = MODELS[model]
    fixed = fixed or {}
    free = [i for i, n in enumerate(names) if n not in fixed]

    def full(q):
        p = np.array([fixed.get(n, 0.0) for n in names], dtype=float)
        p[free] = q
        return p

    def nll(q):
        val = log_prob(full(q), model, dist_kpc, ul_scale, ul_sigma, include_limits)
        return -val if np.isfinite(val) else 1e12

    starts = [np.asarray(s)[free] for s in seeds] + [random_start(model)[free] for _ in range(n_starts)]
    best = None
    for q0 in starts:
        if nll(q0) >= 1e11:
            continue
        res = minimize(nll, q0, method="Nelder-Mead", options={"maxiter": 20000, "maxfev": 20000, "xatol": 1e-6, "fatol": 1e-6})
        res = minimize(nll, res.x, method="Nelder-Mead", options={"maxiter": 20000, "maxfev": 20000, "xatol": 1e-7, "fatol": 1e-7})
        if best is None or res.fun < best.fun:
            best = res
    if best is None:
        return None
    p = full(best.x)
    d = unpack(model, p)
    ll = log_likelihood(d, dist_kpc, ul_scale, ul_sigma, include_limits)
    k = len(free)
    return {"model": model, "p": p, "d": d, "ll": ll, "k": k, "bic": k * np.log(num_obs_total) - 2 * ll}


def derived(d, dist_kpc):
    """Physical summary quantities for one parameter set."""
    beta0, theta = d["beta0"], d["theta"]
    out = {
        "gamma_beta0": 10 ** d["lg_gb0"],
        "beta0_cos": beta0 * np.cos(np.radians(theta)),
    }
    t_last = max(n_data[:, 0].max(), s_data[:, 0].max())
    for tag, tau, side in (("a", d["tau_a"], -1), ("r", d["tau_r"], +1)):
        tp = intrinsic_time(t_last - d["t_ej"], beta0, theta, tau, side, d["t_coast"])
        beta_end = float(speed(tp, beta0, tau, d["t_coast"]))
        out[f"beta_end_{tag}"] = beta_end
        out[f"gamma_beta_end_{tag}"] = beta_end / np.sqrt(1.0 - beta_end**2)
        # Asymptotic projected separation as t' -> infinity (not the separation at the last epoch)
        out[f"rmax_{tag}"] = (
            np.inf if not np.isfinite(tau)
            else beta0 * (d["t_coast"] + tau) * np.sin(np.radians(theta)) * AU_PER_LD / (dist_kpc * 1000.0)
        )
    return out


def boundary_flags(d):
    """Parameters within a small margin of a prior edge, so the fit is set by the prior there."""
    flags = []
    if d["lg_gb0"] > LOG_GB0_RANGE[1] - 0.05:
        flags.append("b0->1")
    if d["lg_gb0"] < LOG_GB0_RANGE[0] + 0.05:
        flags.append("b0->0")
    if d["theta"] > 89.0:
        flags.append("th->90")
    if d["theta"] < 1.0:
        flags.append("th->0")
    for tag in ("a", "r"):
        tau = d[f"tau_{tag}"]
        if np.isfinite(tau) and tau > 0.98 * TAU_RANGE[1]:
            flags.append(f"tau_{tag}->max")
        if np.isfinite(tau) and tau < 1.1 * TAU_RANGE[0]:
            flags.append(f"tau_{tag}->min")
    if "t_coast" in d and d["t_coast"] > 0.98 * T_COAST_RANGE[1]:
        flags.append("tc->max")
    return ",".join(dict.fromkeys(flags)) or "-"


def ul_penalty(d, dist_kpc, ul_scale, ul_sigma):
    return log_likelihood(d, dist_kpc, ul_scale, ul_sigma, True) - log_likelihood(d, dist_kpc, include_limits=False)


# 5. RUN THE GRID
if CUSTOM_LIMITS:
    configs = [(D, 1.0, np.array(args.ul_width), True) for D in args.distances]
else:
    configs = [(D, s, sig, True) for D in args.distances for s in args.ul_scales for sig in args.ul_sigmas]
if args.no_limits_run:
    configs += [(D, np.nan, np.nan, False) for D in args.distances]

def limit_labels(ul_scale, ul_sigma, use_ul):
    """Short (limit, width) labels for the progress line and report."""
    if not use_ul:
        return "none", "--"
    if CUSTOM_LIMITS:
        return "custom", "/".join(f"{s:.2f}" for s in np.atleast_1d(ul_sigma))
    return f"{ul_scale:.1f}xFWHM", f"{ul_sigma:.2f}"


def is_baseline(r):
    if args.profile_no_limits:
        return not r["limits"] and r["D"] == BASELINE_D
    if not (r["limits"] and r["D"] == BASELINE_D):
        return False
    return CUSTOM_LIMITS or (r["ul_scale"] == BASELINE_UL_SCALE and r["ul_sigma"] == BASELINE_UL_SIGMA)


def run_parallel(func, tasks):
    """Map func over tasks with --jobs worker processes (fork, so workers see the loaded data)."""
    if args.jobs <= 1 or len(tasks) <= 1:
        return [func(t) for t in tasks]
    with mp.get_context("fork").Pool(min(args.jobs, len(tasks))) as pool:
        return pool.map(func, tasks)


N_RESEED = 3  # pass 2 restarts from this many of the best other-configuration optima


def fit_task(task):
    """Fit one model at one grid configuration from the seeds plus n_starts random starts."""
    idx, model, (D, ul_scale, ul_sigma, use_ul), seeds, n_starts = task
    global rng
    rng = np.random.default_rng([args.seed, idx])  # independent random starts per task
    if seeds:
        # Keep only the seeds that are already best at this configuration (one likelihood call each)
        score = [log_prob(s, model, D, ul_scale, ul_sigma, use_ul) for s in seeds]
        seeds = [seeds[i] for i in np.argsort(score)[::-1][:N_RESEED] if np.isfinite(score[i])]
    return fit_ml(model, D, ul_scale, ul_sigma, use_ul, seeds=seeds, n_starts=n_starts)


# One task per (configuration, model), so the work spreads evenly over the workers.
# Pass 1: random restarts. Pass 2: restart from the best pass-1 optima of the other
# configurations, so a configuration stuck in a local optimum is repaired by a neighbour's solution.
pairs = [(c, m) for c in configs for m in MODELS]
first = run_parallel(fit_task, [(i, m, c, [], args.n_starts) for i, (c, m) in enumerate(pairs)])
bank = {m: [f["p"] for (c, mm), f in zip(pairs, first) if mm == m] for m in MODELS}
second = run_parallel(fit_task, [(len(pairs) + i, m, c, bank[m], 0) for i, (c, m) in enumerate(pairs)])
best_fit = {(i // len(MODELS), m): max((f1, f2), key=lambda f: f["ll"])
            for i, ((c, m), f1, f2) in enumerate(zip(pairs, first, second))}

results = []
for i, (D, ul_scale, ul_sigma, use_ul) in enumerate(configs):
    row = {"D": D, "ul_scale": ul_scale, "ul_sigma": ul_sigma, "limits": use_ul, "fits": {}}
    for model in MODELS:
        fit = best_fit[(i, model)]
        fit["derived"] = derived(fit["d"], D)
        fit["ul_penalty"] = ul_penalty(fit["d"], D, ul_scale, ul_sigma) if use_ul else 0.0
        row["fits"][model] = fit
    results.append(row)
    best_bic = min(f["bic"] for f in row["fits"].values())
    ul_txt, sig_txt = limit_labels(ul_scale, ul_sigma, use_ul)
    tag = f"D={D:4.1f} kpc  " + (f"UL={ul_txt} sig={sig_txt}\"" if use_ul else "no upper limits      ")
    print(
        tag + "  " + "  ".join(
            f"{m}: lnL={f['ll']:8.2f} dBIC={f['bic'] - best_bic:5.2f}" for m, f in row["fits"].items()
        ),
        flush=True,
    )


# 6. REPORT
lines = []
lines.append("=" * 130)
lines.append("PHYSICAL LIGHT-TRAVEL-TIME KINEMATICS: MODEL COMPARISON GRID".center(130))
lines.append("=" * 130)
lines.append(f"N_data = {num_obs_total} ({len(core_data)} core, {len(n_data)} N, {len(s_data)} S epochs; Epoch 12 S excluded{'' if args.epoch12_north else ', Epoch 12 N excluded'})")
lines.append("BIC = k ln N - 2 ln L at the maximum likelihood; dBIC relative to the best model in each row.")
lines.append("UL penalty = contribution of the beam upper-limit term to ln L at the best fit (0 = limits satisfied).")
lines.append(
    "Limits: " + ", ".join(f"MJD {m:.2f}: r < {r:.3f}\"" for m, r in zip(UL_MJD, UL_FWHM))
    + ("" if CUSTOM_LIMITS else " (x the UL scale; from epoch_limits.py)")
)
if CUSTOM_LIMITS:
    lines.append("Custom limit widths: " + ", ".join(f"{w:.3f}\"" for w in args.ul_width))
lines.append(f"Priors: log10(Gb0) in {LOG_GB0_RANGE}, tau in {TAU_RANGE} d, theta in (0, 90) deg, T_ej in (60000, 60800).")
lines.append("Columns: Gb0 = Gamma_0 beta_0 at launch; Gb_end_a/r = Gamma beta of the approaching/receding ejecton at the")
lines.append("  last tracked epoch; rinf_a/r = asymptotic projected separation (arcsec) as t' -> infinity, NOT the separation")
lines.append("  at the last epoch; edge = parameters at a prior edge (b0->1: Gb0 at the top of its range, so beta_0 is only a")
lines.append("  lower limit; th->90: no light-travel asymmetry; tau_x->max: deceleration not constrained).")
lines.append("")
header = (
    f"{'D':>5} {'UL':>9} {'sigUL':>6} | {'model':<11} {'k':>2} {'lnL':>8} {'dBIC':>6} {'ULpen':>6} | {'T_ej':>8} {'beta0':>7} {'theta':>6} "
    f"{'tau_a':>7} {'tau_r':>7} {'Gb0':>7} {'Gb_end_a':>8} {'Gb_end_r':>8} {'rinf_a':>6} {'rinf_r':>6} | edge"
)
lines.append(header)
lines.append("-" * len(header))
for row in results:
    best_bic = min(f["bic"] for f in row["fits"].values())
    ul_txt, sig_txt = limit_labels(row["ul_scale"], row["ul_sigma"], row["limits"])
    for model, f in row["fits"].items():
        d, dv = f["d"], f["derived"]
        fmt_tau = lambda t: "inf" if not np.isfinite(t) else f"{t:7.0f}"
        fmt_r = lambda r: "--" if not np.isfinite(r) else f"{r:6.2f}"
        lines.append(
            f"{row['D']:5.1f} {ul_txt:>9} {sig_txt:>6} | {model:<11} {f['k']:2d} {f['ll']:8.2f} {f['bic'] - best_bic:6.2f} {f['ul_penalty']:6.2f} | "
            f"{d['t_ej']:8.1f} {d['beta0']:7.4f} {d['theta']:6.1f} {fmt_tau(d['tau_a']):>7} {fmt_tau(d['tau_r']):>7} "
            f"{dv['gamma_beta0']:7.2f} {dv['gamma_beta_end_a']:8.2f} {dv['gamma_beta_end_r']:8.2f} {fmt_r(dv['rmax_a']):>6} {fmt_r(dv['rmax_r']):>6} | "
            f"{boundary_flags(d)}"
        )
        if model == "coast_decel":
            lines[-1] += f"  (t_coast = {d['t_coast']:.0f} d)"
    lines.append("")

os.makedirs("data", exist_ok=True)
report_file = args.report or ("data/physical_kinematics_report_custom_limits.txt" if CUSTOM_LIMITS else "data/physical_kinematics_report.txt")
os.makedirs(os.path.dirname(report_file) or ".", exist_ok=True)
with open(report_file, "w") as fp:
    fp.write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"Saved report to {report_file}")

# Every maximum-likelihood fit, keyed c<config>_<model>_<field>; config arrays give the grid
fits_file = os.path.splitext(report_file)[0] + ".npz"
store = {
    "models": np.array(list(MODELS)),
    "D": np.array([r["D"] for r in results]),
    "ul_scale": np.array([r["ul_scale"] for r in results]),
    "ul_sigma": np.array([np.mean(r["ul_sigma"]) for r in results]),
    "limits": np.array([r["limits"] for r in results]),
    "ul_mjd": np.array(UL_MJD),
    "ul_radius": np.array(UL_FWHM),
    "n_data": num_obs_total,
}
for i, row in enumerate(results):
    for model, f in row["fits"].items():
        store[f"c{i}_{model}_names"] = np.array(MODELS[model])
        store[f"c{i}_{model}_p"] = f["p"]
        store[f"c{i}_{model}_lnL"] = f["ll"]
        store[f"c{i}_{model}_bic"] = f["bic"]
        store[f"c{i}_{model}_k"] = f["k"]
np.savez(fits_file, **store)
print(f"Saved fits to {fits_file}")


# 7. OPTIONAL PROFILE LIKELIHOOD IN beta_0 AT THE BASELINE CONFIGURATION
if args.profile:
    base = next(r for r in results if is_baseline(r))
    base_args = (BASELINE_D, base["ul_scale"], base["ul_sigma"], base["limits"])
    prof = []
    prof.append("\nPROFILE LIKELIHOOD IN beta_0 (D = {} kpc, UL = {}, sigma_UL = {}\")".format(
        BASELINE_D, *limit_labels(base["ul_scale"], base["ul_sigma"], base["limits"])))
    prof.append("Each row refits all other parameters with beta_0 held fixed. dlnL is relative to the free fit; values of")
    prof.append("beta_0 with 2 dlnL > -3.84 lie inside the 95% profile interval (one parameter), so a flat profile up to")
    prof.append("beta_0 -> 1 means only a lower limit, and a profile that stays within that bound everywhere means no constraint.")
    prof.append(f"{'model':<11} {'beta0':>7} {'Gb0':>7} {'lnL':>9} {'2dlnL':>7} {'T_ej':>8} {'theta':>6} {'tau_a':>7} {'tau_r':>7} {'Gb_end_a':>8} {'Gb_end_r':>8} | edge")
    def profile_fit(task):
        idx, model, lg, seed = task
        global rng
        rng = np.random.default_rng([args.seed, 10000 + idx])
        return fit_ml(model, *base_args, seeds=[seed], fixed={"lg_gb0": lg})

    tasks = []
    for model, f_free in base["fits"].items():
        for b in sorted(args.profile_beta0):
            lg = float(beta_to_lg_gb(b))
            if not (LOG_GB0_RANGE[0] < lg < LOG_GB0_RANGE[1]):
                continue
            seed = np.array(f_free["p"], dtype=float)
            seed[MODELS[model].index("lg_gb0")] = lg
            tasks.append((len(tasks), model, lg, seed))
    profile_fits = run_parallel(profile_fit, tasks)

    for model, f_free in base["fits"].items():
        rows = [(1, fp_) for t, fp_ in zip(tasks, profile_fits) if t[1] == model and fp_ is not None]
        for b, fp_ in rows + [(None, f_free)]:
            d, dv = fp_["d"], derived(fp_["d"], BASELINE_D)
            fmt_tau = lambda t: "inf" if not np.isfinite(t) else f"{t:7.0f}"
            label = f"{d['beta0']:7.4f}" if b is not None else "   free"
            prof.append(
                f"{model:<11} {label} {dv['gamma_beta0']:7.2f} {fp_['ll']:9.2f} {2 * (fp_['ll'] - f_free['ll']):7.2f} "
                f"{d['t_ej']:8.1f} {d['theta']:6.1f} {fmt_tau(d['tau_a']):>7} {fmt_tau(d['tau_r']):>7} "
                f"{dv['gamma_beta_end_a']:8.2f} {dv['gamma_beta_end_r']:8.2f} | {boundary_flags(d)}"
            )
        prof.append("")
        print("\n".join(prof[-len(rows) - 2:]), flush=True)
    with open(report_file, "a") as fp:
        fp.write("\n".join(prof) + "\n")


# 8. OPTIONAL MCMC FOR THE BASELINE CONFIGURATION
if args.mcmc:
    import emcee

    base = next(r for r in results if is_baseline(r))
    base_args = (BASELINE_D, base["ul_scale"], base["ul_sigma"], base["limits"])
    base_txt = "UL={}, sigma_UL={}\"".format(*limit_labels(base["ul_scale"], base["ul_sigma"], base["limits"]))
    hdi_lines = []
    for model, f in base["fits"].items():
        if args.mcmc_models and model not in args.mcmc_models:
            continue
        ndim = len(MODELS[model])
        scale = np.array([{"t_ej": 5.0, "lg_gb0": 0.01, "theta": 1.0}.get(n, 0.01 * max(abs(v), 1.0)) for n, v in zip(MODELS[model], f["p"])])
        p0 = []
        while len(p0) < args.n_walkers:
            trial = f["p"] + scale * rng.normal(size=ndim) * 0.1
            if np.isfinite(log_prob(trial, model, *base_args)):
                p0.append(trial)
        sampler = emcee.EnsembleSampler(args.n_walkers, ndim, log_prob, args=(model, *base_args))
        sampler.run_mcmc(np.array(p0), args.n_steps, progress=False)
        burnin = int(0.2 * args.n_steps)
        chain = sampler.get_chain(discard=burnin, thin=10, flat=True)
        try:
            tau_ac = sampler.get_autocorr_time(discard=burnin, quiet=True)
        except Exception:
            tau_ac = np.full(ndim, np.nan)

        # Derived quantities on a subsample of the chain (the trajectory inversion is the slow part)
        sub = chain[rng.choice(len(chain), size=min(5000, len(chain)), replace=False)]
        dvs = [derived(unpack(model, p), BASELINE_D) for p in sub]
        derived_names = ["beta0", "gamma_beta_end_a", "gamma_beta_end_r"]
        derived_samples = np.array(
            [[unpack(model, p)["beta0"], dv["gamma_beta_end_a"], dv["gamma_beta_end_r"]] for p, dv in zip(sub, dvs)]
        )
        np.savez(
            f"data/physical_kinematics_samples_{model}{'_nolimits' if args.profile_no_limits else ''}{args.samples_suffix}.npz",
            samples=chain, names=MODELS[model], autocorr=tau_ac,
            derived_samples=derived_samples, derived_names=derived_names,
        )
        hdi_lines.append(f"\n{model} (D={BASELINE_D} kpc, {base_txt}): median [16th, 84th]; max autocorr = {np.nanmax(tau_ac):.0f} steps")
        for i, name in enumerate(MODELS[model]):
            q16, q50, q84 = np.percentile(chain[:, i], [16, 50, 84])
            hdi_lines.append(f"  {name:<16} {q50:10.4f}  [{q16:10.4f}, {q84:10.4f}]")
        for i, name in enumerate(derived_names):
            q5, q16, q50, q84 = np.percentile(derived_samples[:, i], [5, 16, 50, 84])
            hdi_lines.append(f"  {name:<16} {q50:10.4f}  [{q16:10.4f}, {q84:10.4f}]  (5th percentile {q5:.4f})")
        print("\n".join(hdi_lines[-(ndim + len(derived_names) + 1):]), flush=True)
    with open(report_file, "a") as fp:
        fp.write("\nMCMC POSTERIORS\n" + "\n".join(hdi_lines) + "\n")
