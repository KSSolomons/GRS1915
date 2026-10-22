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
extra parameters. beta_0 < 1 is enforced at the adopted distance.

Models (k at fixed D):
    ballistic   : tau -> infinity                                      (k = 8)
    decel_sym   : one intrinsic tau for both sides                     (k = 9)
    decel_asym  : independent tau_a, tau_r (different environments)  (k = 10)

The core likelihood, perpendicular scatter term and one-sided beam upper-limit
penalty follow the absolute-kinematics pipeline so the BIC values are comparable.
"""

import argparse
import glob
import os
import re

import numpy as np
from scipy.optimize import minimize

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
    "--ul-mjd",
    type=float,
    nargs="+",
    default=None,
    help="Custom upper-limit epochs (MJD). With --ul-radius and --ul-width this replaces the beam-FWHM grid",
)
parser.add_argument("--ul-radius", type=float, nargs="+", default=None, help="Custom limit radius per epoch (arcsec), e.g. the peel radius")
parser.add_argument("--ul-width", type=float, nargs="+", default=None, help="Custom penalty width per epoch (arcsec), e.g. sigma_pos")
parser.add_argument("--n-starts", type=int, default=24, help="Random Nelder-Mead restarts per fit")
parser.add_argument(
    "--mcmc",
    action="store_true",
    help="Sample posteriors with emcee for the baseline configuration (D=9.4 kpc, 1.0 x FWHM, 0.05 arcsec)",
)
parser.add_argument("--n-steps", type=int, default=10000, help="MCMC steps per walker")
parser.add_argument("--n-walkers", type=int, default=128, help="MCMC walkers")
parser.add_argument("--seed", type=int, default=1915, help="Random seed")
args = parser.parse_args()

rng = np.random.default_rng(args.seed)

C_LD_PER_DAY = 1.0  # work in light-days, so c = 1
AU_PER_LD = 173.144633  # astronomical units per light-day
EJECTION_CAMPAIGN_START = 60600.0  # jets are only taken from the 2025 campaign
EPOCH12_TS = 1764664923  # Epoch 12 jet features are unreliable (likely artefacts)

# Blending upper limits from calc_upper_limits.py (projected beam FWHM along the jet axis)
UL_MJD = np.array([60736.11478, 60758.0668])
UL_FWHM = np.array([2.423, 2.460])

# Custom per-epoch limits (e.g. peel radius and sigma_pos = theta_beam / (2 SNR) from the peeled-image rms)
CUSTOM_LIMITS = args.ul_mjd is not None
if CUSTOM_LIMITS:
    if not (args.ul_radius and args.ul_width and len(args.ul_mjd) == len(args.ul_radius) == len(args.ul_width)):
        parser.error("--ul-mjd, --ul-radius and --ul-width must be given together with one value per epoch")
    UL_MJD = np.array(args.ul_mjd)
    UL_FWHM = np.array(args.ul_radius)  # used directly as the limit radius (ul_scale = 1)


# 1. LOAD DATA (same parsing and core MAD filter as the absolute-kinematics pipeline)
dir_path = "S-band-fits"
files = sorted(glob.glob(os.path.join(dir_path, "*2D_Fitting*.txt")))

act_ra_deg = (19 + 15 / 60.0 + 11.558 / 3600.0) * 15.0
act_dec_deg = 10 + 56 / 60.0 + 44.905 / 3600.0
cos_dec = np.cos(np.radians(act_dec_deg))

core_data_raw = []
n_data = []
s_data = []

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
        row = [mjd, dx, dy, c["ra_err"], c["dec_err"]]
        if 288.7979 < c["ra_deg"] < 288.7982:
            core_data_raw.append(row)
        elif mjd > EJECTION_CAMPAIGN_START and ts != EPOCH12_TS:
            (n_data if dy > 0 else s_data).append(row)

temp_core = np.array(core_data_raw)
dist = np.hypot(temp_core[:, 1] - np.median(temp_core[:, 1]), temp_core[:, 2] - np.median(temp_core[:, 2]))
mad = np.median(np.abs(dist - np.median(dist)))
core_data = temp_core[dist <= max(5 * mad, 0.4)]
n_data = np.array(sorted(n_data))
s_data = np.array(sorted(s_data))
num_obs_total = 2 * (len(core_data) + len(n_data) + len(s_data))

print(f"Loaded {len(core_data)} core, {len(n_data)} N (receding), {len(s_data)} S (approaching) points; N_data = {num_obs_total}")


# 2. PHYSICAL TRAJECTORY WITH EXACT LIGHT-TRAVEL-TIME MAPPING
def radial_distance(tp, beta0, tau):
    """Intrinsic distance travelled (light-days) after coordinate time tp (days)."""
    if not np.isfinite(tau):
        return beta0 * tp
    return beta0 * tau * (-np.expm1(-tp / tau))


def apparent_separation(t_obs, t_ej, beta0, theta_deg, tau, dist_kpc, side):
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
    cth, sth = np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))
    if not np.isfinite(tau):
        tp = dt_obs[pos] / (1.0 + side * beta0 * cth)
    else:
        # t_obs - T_ej = t' + side * R(t') cos(theta) is monotonic in t' because beta cos(theta) < 1
        tp_max = dt_obs[pos].max() / (1.0 - beta0 * cth) + 1.0
        tp_grid = np.linspace(0.0, tp_max, 4000)
        tobs_grid = tp_grid + side * radial_distance(tp_grid, beta0, tau) * cth / C_LD_PER_DAY
        tp = np.interp(dt_obs[pos], tobs_grid, tp_grid)
    out[pos] = radial_distance(tp, beta0, tau) * sth * AU_PER_LD / (dist_kpc * 1000.0)
    return out


def intrinsic_time(dt_obs, beta0, theta_deg, tau, side):
    """Coordinate time since launch (days) at observer time T_ej + dt_obs."""
    cth = np.cos(np.radians(theta_deg))
    if not np.isfinite(tau):
        return dt_obs / (1.0 + side * beta0 * cth)
    tp_grid = np.linspace(0.0, dt_obs / (1.0 - beta0 * cth) + 1.0, 4000)
    return np.interp(dt_obs, tp_grid + side * radial_distance(tp_grid, beta0, tau) * cth, tp_grid)


# 3. MODELS, PRIORS AND LIKELIHOOD
MODELS = {
    "ballistic": ["t_ej", "beta0", "theta", "pa_n", "pa_s", "theta_scatter", "x_c", "y_c"],
    "decel_sym": ["t_ej", "beta0", "theta", "tau", "pa_n", "pa_s", "theta_scatter", "x_c", "y_c"],
    "decel_asym": ["t_ej", "beta0", "theta", "tau_a", "tau_r", "pa_n", "pa_s", "theta_scatter", "x_c", "y_c"],
}


def unpack(model, p):
    d = dict(zip(MODELS[model], p))
    if model == "ballistic":
        d["tau_a"] = d["tau_r"] = np.inf
    elif model == "decel_sym":
        d["tau_a"] = d["tau_r"] = d["tau"]
    return d


def log_prior(model, d):
    if not (60000.0 < d["t_ej"] < 60800.0):
        return -np.inf
    if not (0.0 < d["beta0"] < 1.0):
        return -np.inf
    if not (0.0 < d["theta"] < 90.0):  # theta < 90 deg makes the SE (southern) ejecton the approaching one
        return -np.inf
    for key in ("tau_a", "tau_r"):
        if np.isfinite(d[key]) and not (5.0 < d[key] < 5000.0):
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
    model_n = apparent_separation(n_data[:, 0], *kin, d["tau_r"], dist_kpc, side=+1)
    model_s = -apparent_separation(s_data[:, 0], *kin, d["tau_a"], dist_kpc, side=-1)
    ll += lobe_log_likelihood(n_data, model_n, d["pa_n"], tan2_scatter, x_c, y_c)
    ll += lobe_log_likelihood(s_data, model_s, d["pa_s"], tan2_scatter, x_c, y_c)

    if include_limits:
        ul_r = ul_scale * UL_FWHM
        for tau, side in ((d["tau_r"], +1), (d["tau_a"], -1)):
            excess = apparent_separation(UL_MJD, *kin, tau, dist_kpc, side) - ul_r
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
def random_start(model):
    p = {
        "t_ej": rng.uniform(60450.0, 60720.0),
        "beta0": rng.uniform(0.3, 0.98),
        "theta": rng.uniform(50.0, 85.0),
        "tau": 10 ** rng.uniform(1.7, 3.3),
        "tau_a": 10 ** rng.uniform(1.7, 3.3),
        "tau_r": 10 ** rng.uniform(1.7, 3.3),
        "pa_n": rng.normal(-42.0, 2.0),
        "pa_s": rng.normal(-41.0, 2.0),
        "theta_scatter": rng.uniform(2.0, 8.0),
        "x_c": -0.191,
        "y_c": -0.298,
    }
    return np.array([p[name] for name in MODELS[model]])


def fit_ml(model, dist_kpc, ul_scale, ul_sigma, include_limits, seeds=()):
    def nll(p):
        val = log_prob(p, model, dist_kpc, ul_scale, ul_sigma, include_limits)
        return -val if np.isfinite(val) else 1e12

    starts = [np.asarray(s) for s in seeds] + [random_start(model) for _ in range(args.n_starts)]
    best = None
    for p0 in starts:
        if nll(p0) >= 1e11:
            continue
        res = minimize(nll, p0, method="Nelder-Mead", options={"maxiter": 20000, "maxfev": 20000, "xatol": 1e-6, "fatol": 1e-6})
        res = minimize(nll, res.x, method="Nelder-Mead", options={"maxiter": 20000, "maxfev": 20000, "xatol": 1e-7, "fatol": 1e-7})
        if best is None or res.fun < best.fun:
            best = res
    d = unpack(model, best.x)
    ll = log_likelihood(d, dist_kpc, ul_scale, ul_sigma, include_limits)
    k = len(MODELS[model])
    return {"model": model, "p": best.x, "d": d, "ll": ll, "k": k, "bic": k * np.log(num_obs_total) - 2 * ll}


def derived(d, dist_kpc):
    """Physical summary quantities for one parameter set."""
    beta0, theta = d["beta0"], d["theta"]
    out = {
        "gamma_beta0": beta0 / np.sqrt(1.0 - beta0**2),
        "beta0_cos": beta0 * np.cos(np.radians(theta)),
    }
    t_last = max(n_data[:, 0].max(), s_data[:, 0].max())
    for tag, tau, side in (("a", d["tau_a"], -1), ("r", d["tau_r"], +1)):
        tp = intrinsic_time(t_last - d["t_ej"], beta0, theta, tau, side)
        beta_end = beta0 if not np.isfinite(tau) else beta0 * np.exp(-tp / tau)
        out[f"beta_end_{tag}"] = beta_end
        out[f"gamma_beta_end_{tag}"] = beta_end / np.sqrt(1.0 - beta_end**2)
        out[f"rmax_{tag}"] = (
            np.inf if not np.isfinite(tau)
            else beta0 * tau * np.sin(np.radians(theta)) * AU_PER_LD / (dist_kpc * 1000.0)
        )
    return out


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


results = []
seed_bank = {m: [] for m in MODELS}
for D, ul_scale, ul_sigma, use_ul in configs:
    row = {"D": D, "ul_scale": ul_scale, "ul_sigma": ul_sigma, "limits": use_ul, "fits": {}}
    for model in MODELS:
        fit = fit_ml(model, D, ul_scale, ul_sigma, use_ul, seeds=seed_bank[model][-3:])
        seed_bank[model].append(fit["p"])
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
lines.append("=" * 118)
lines.append("PHYSICAL LIGHT-TRAVEL-TIME KINEMATICS: MODEL COMPARISON GRID".center(118))
lines.append("=" * 118)
lines.append(f"N_data = {num_obs_total} ({len(core_data)} core, {len(n_data)} N, {len(s_data)} S epochs; Epoch 12 jets excluded)")
lines.append("BIC = k ln N - 2 ln L at the maximum likelihood; dBIC relative to the best model in each row.")
lines.append("UL penalty = contribution of the beam upper-limit term to ln L at the best fit (0 = limits satisfied).")
if CUSTOM_LIMITS:
    lines.append(
        "Custom limits: " + ", ".join(f"MJD {m:.2f}: r < {r:.2f}\" (width {w:.3f}\")" for m, r, w in zip(UL_MJD, UL_FWHM, args.ul_width))
    )
lines.append("")
header = f"{'D':>5} {'UL':>9} {'sigUL':>6} | {'model':<10} {'k':>2} {'lnL':>8} {'dBIC':>6} {'ULpen':>6} | {'T_ej':>8} {'beta0':>6} {'theta':>6} {'tau_a':>7} {'tau_r':>7} {'Gb0':>6} {'Gb_end_a':>8} {'Gb_end_r':>8} {'rmax_a':>6} {'rmax_r':>6}"
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
            f"{row['D']:5.1f} {ul_txt:>9} {sig_txt:>6} | {model:<10} {f['k']:2d} {f['ll']:8.2f} {f['bic'] - best_bic:6.2f} {f['ul_penalty']:6.2f} | "
            f"{d['t_ej']:8.1f} {d['beta0']:6.3f} {d['theta']:6.1f} {fmt_tau(d['tau_a']):>7} {fmt_tau(d['tau_r']):>7} "
            f"{dv['gamma_beta0']:6.2f} {dv['gamma_beta_end_a']:8.2f} {dv['gamma_beta_end_r']:8.2f} {fmt_r(dv['rmax_a']):>6} {fmt_r(dv['rmax_r']):>6}"
        )
    lines.append("")

os.makedirs("data", exist_ok=True)
report_file = "data/physical_kinematics_report_custom_limits.txt" if CUSTOM_LIMITS else "data/physical_kinematics_report.txt"
with open(report_file, "w") as fp:
    fp.write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"Saved report to {report_file}")


# 7. OPTIONAL MCMC FOR THE BASELINE CONFIGURATION
if args.mcmc:
    import emcee

    def is_baseline(r):
        if not (r["limits"] and r["D"] == 9.4):
            return False
        return CUSTOM_LIMITS or (r["ul_scale"] == 1.0 and r["ul_sigma"] == 0.05)

    base = next(r for r in results if is_baseline(r))
    base_args = (9.4, base["ul_scale"], base["ul_sigma"], True)
    base_txt = "UL={}, sigma_UL={}\"".format(*limit_labels(base["ul_scale"], base["ul_sigma"], True))
    hdi_lines = []
    for model, f in base["fits"].items():
        ndim = len(MODELS[model])
        scale = np.array([{"t_ej": 5.0, "beta0": 0.01, "theta": 1.0}.get(n, 0.01 * max(abs(v), 1.0)) for n, v in zip(MODELS[model], f["p"])])
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
        np.savez(f"data/physical_kinematics_samples_{model}.npz", samples=chain, names=MODELS[model], autocorr=tau_ac)
        hdi_lines.append(f"\n{model} (D=9.4 kpc, {base_txt}): median [16th, 84th]; max autocorr = {np.nanmax(tau_ac):.0f} steps")
        for i, name in enumerate(MODELS[model]):
            q16, q50, q84 = np.percentile(chain[:, i], [16, 50, 84])
            hdi_lines.append(f"  {name:<14} {q50:10.4f}  [{q16:10.4f}, {q84:10.4f}]")
        print(hdi_lines[-ndim - 1 :] and "\n".join(hdi_lines[-ndim - 1 :]), flush=True)
    with open(report_file, "a") as fp:
        fp.write("\nMCMC POSTERIORS\n" + "\n".join(hdi_lines) + "\n")
