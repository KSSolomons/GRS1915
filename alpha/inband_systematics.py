"""
Channel-to-channel systematics and spectral curvature of the in-band spectra.

Run locally (ordinary Python, no CASA) on the outputs of get_alpha.py:

    python alpha/inband_systematics.py --channels inband_channel_flux.txt \\
        --reference inband_reference_flux.txt

1. Reference sources (inband_reference_flux.txt). The channel images are not
   primary-beam corrected, so each flux is divided by the Gaussian beam
   attenuation A(nu) = A0 ** ((nu / nu_A0) ** 2), where A0 is the attenuation
   measured in the MFS image (--pb-atten at --pb-freq); for a Gaussian beam with
   FWHM proportional to 1/nu, ln A scales as nu^2 at a fixed position. A power
   law is then fitted per epoch, and from the residuals:
     - sigma_sys: the fractional per-channel error that, added in quadrature,
       brings the pooled reduced chi^2 of the reference fits to one (or a 95%
       upper limit when the scatter is already consistent with the noise);
     - the common pattern: the weighted mean residual per channel over all
       epochs, i.e. a channel-dependent calibration error shared by the epochs.
2. Core (inband_channel_flux.txt). Per epoch, optionally after dividing out the
   common pattern (--pattern), and with sigma_sys added in quadrature, the
   channel fluxes are fitted with a power law and with a log-parabola
       S = S0 (nu / nu0) ** (alpha + beta ln(nu / nu0)),
   in linear flux space like get_alpha.py. beta measures curvature; dBIC < 0
   favours the curved spectrum.
3. Lobes (inband_alpha_lobes.txt). Per epoch and lobe, a power law with
   sigma_sys added, and the weighted mean over --lobe-epochs (by default Epochs
   6-8: in Epoch 5 the N lobe sits on a core wing brighter than itself, and from
   Epoch 9 the lobe spectra are too faint to constrain alpha).
4. A residual figure (reference pattern, and the core about its power law), a
   text report, and the final table (--final) of core and lobe alpha for the paper.

Channels in --drop-channels (default 0: 2.68 GHz reads ~3% low in every source
and epoch) are left out everywhere, as is any channel whose error exceeds
--max-err-ratio times the median of its epoch (a nearly fully flagged channel).
"""

import argparse
import os
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq, curve_fit
from scipy.stats import chi2 as chi2_dist

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 10

HERE = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description="In-band channel systematics and curvature test")
parser.add_argument("--channels", default=os.path.join(HERE, "inband_channel_flux.txt"), help="core channel fluxes from get_alpha.py")
parser.add_argument("--reference", default=os.path.join(HERE, "inband_reference_flux.txt"), help="reference-source channel fluxes from get_alpha.py")
parser.add_argument("--pb-atten", type=float, default=0.893, help="primary-beam attenuation of the reference in the MFS image (pre-peel / pbcor = 5.78 / 6.47 mJy)")
parser.add_argument("--pb-freq", type=float, default=3.0624, help="frequency (GHz) at which --pb-atten applies (MFS reference frequency)")
parser.add_argument("--sys", type=float, default=None, help="fractional per-channel systematic to use instead of the measured one")
parser.add_argument("--pattern", action=argparse.BooleanOptionalAction, default=False, help="divide the core channel fluxes by the common reference pattern (default: off)")
parser.add_argument("--drop-channels", default="0", help="comma-separated channel indices left out of every fit (default: 0; '' keeps all)")
parser.add_argument("--max-err-ratio", type=float, default=5.0, help="leave out channels whose error exceeds this multiple of the epoch median (default: 5)")
parser.add_argument("--alpha", default=os.path.join(HERE, "inband_alpha.txt"), help="epoch table from get_alpha.py (for the MJDs)")
parser.add_argument("--lobes", default=os.path.join(HERE, "inband_alpha_lobes.txt"), help="lobe spectra from get_alpha.py")
parser.add_argument("--lobe-epochs", default="1749586570,1751137270,1753549996", help="epochs entering the mean lobe alpha (default: Epochs 6-8)")
parser.add_argument("--final", default=os.path.join(HERE, "inband_alpha_final.txt"), help="final core and lobe table")
parser.add_argument("--plot-epochs", type=int, default=5, help="number of earliest epochs shown in the core residual panel (default: 5)")
parser.add_argument("--report", default=os.path.join(HERE, "inband_systematics_report.txt"))
parser.add_argument("--figure", default=os.path.join(HERE, "..", "figures", "spectral", "inband_residuals.png"))
args = parser.parse_args()


def read_table(path, ncols):
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts or parts[0].startswith("#") or len(parts) < ncols:
                continue
            rows.append(parts[:ncols])
    return rows


DROP = {int(c) for c in args.drop_channels.split(",") if c.strip()}


def usable(rows):
    """Rows (chan, nu, S, err, ...) without DROP channels and error outliers."""
    rows = [r for r in sorted(rows) if r[0] not in DROP]
    if rows and args.max_err_ratio:
        med = np.median([r[3] for r in rows])
        rows = [r for r in rows if r[3] <= args.max_err_ratio * med]
    return rows


def power_law(x, s0, alpha):
    return s0 * np.exp(alpha * x)


def log_parabola(x, s0, alpha, beta):
    return s0 * np.exp(alpha * x + beta * x**2)


def fit(model, x, s, err, p0):
    """Weighted fit in linear flux space. Returns (params, errors, chi2)."""
    p, cov = curve_fit(model, x, s, p0=p0, sigma=err, absolute_sigma=True, maxfev=20000)
    chi2 = float(np.sum(((s - model(x, *p)) / err) ** 2))
    return p, np.sqrt(np.diag(cov)), chi2


# 1. REFERENCE SOURCES
ref = defaultdict(list)  # (epoch, label) -> [(chan, nu, S, err)]
for epoch, label, chan, nu, s, err, _rms, _sep in read_table(args.reference, 8):
    ref[(epoch, label)].append((int(chan), float(nu), float(s) * 1e-6, float(err) * 1e-6))

ref_resid = []  # (epoch, label, chan, nu, r, sigma_r)
ref_lines = []
for (epoch, label), rows in sorted(ref.items()):
    chan, nu, s, err = (np.array(c) for c in zip(*usable(rows)))
    atten = args.pb_atten ** ((nu / args.pb_freq) ** 2)
    s, err = s / atten, err / atten
    nu0 = np.sqrt(nu.min() * nu.max())
    x = np.log(nu / nu0)
    p, perr, chi2 = fit(power_law, x, s, err, [np.mean(s), -0.7])
    model = power_law(x, *p)
    for c, n, r, sr in zip(chan, nu, s / model - 1.0, err / model):
        ref_resid.append((epoch, label, c, n, r, sr))
    ref_lines.append(f"{epoch:<16} {label:<5} N={len(s)}  S0={p[0] * 1e3:7.3f} mJy  alpha={p[1]:+.3f} +/- {perr[1]:.3f}  chi2/dof={chi2 / (len(s) - 2):5.2f}")

r = np.array([row[4] for row in ref_resid])
sr = np.array([row[5] for row in ref_resid])
dof_ref = len(r) - 2 * len(ref)
chi2_ref = float(np.sum((r / sr) ** 2))

# Common pattern: weighted mean residual per channel index over epochs
by_chan = defaultdict(list)
for _e, _l, c, n, ri, si in ref_resid:
    by_chan[c].append((n, ri, si))
pattern = {}
for c, vals in sorted(by_chan.items()):
    n, ri, si = (np.array(v) for v in zip(*vals))
    w = 1.0 / si**2
    pattern[c] = (float(np.mean(n)), float(np.sum(w * ri) / np.sum(w)), float(1.0 / np.sqrt(np.sum(w))), len(ri))
chi2_pattern = sum((m / e) ** 2 for _n, m, e, _k in pattern.values())

# Residual scatter after removing the pattern, and the systematic needed for chi2/dof = 1
r_clean = np.array([ri - pattern[c][1] for _e, _l, c, _n, ri, _s in ref_resid])
dof_clean = dof_ref - len(pattern)


def sys_for(res, sig, dof):
    """Fractional systematic giving chi2 = dof, or (0, 95% upper limit) if already below."""
    f = lambda s: float(np.sum(res**2 / (sig**2 + s**2))) - dof
    if f(0.0) <= 0.0:
        target = chi2_dist.ppf(0.05, dof)
        g = lambda s: float(np.sum(res**2 / (sig**2 + s**2))) - target
        upper = brentq(g, 0.0, 1.0) if g(0.0) > 0 else 0.0
        return 0.0, upper
    return brentq(f, 0.0, 1.0), None


sys_raw, sys_raw_ul = sys_for(r, sr, dof_ref)
sys_clean, sys_clean_ul = sys_for(r_clean, sr, dof_clean)
if args.sys is not None:
    sys_used = args.sys
elif args.pattern:
    sys_used = sys_clean if sys_clean > 0 else (sys_clean_ul or 0.0)
else:
    sys_used = sys_raw if sys_raw > 0 else (sys_raw_ul or 0.0)


# 2. CORE SPECTRA
core = defaultdict(list)  # epoch -> [(chan, nu, S, err)]
for epoch, chan, nu, _bmaj, _rms, _zero, s, err in read_table(args.channels, 8):
    core[epoch].append((int(chan), float(nu), float(s) * 1e-6, float(err) * 1e-6))

core_lines = []
core_resid = {}
mjd = {row[0]: float(row[1]) for row in read_table(args.alpha, 2)}
core_fits = {}
for epoch, rows in sorted(core.items()):
    chan, nu, s, err = (np.array(c) for c in zip(*usable(rows)))
    if args.pattern:
        corr = np.array([1.0 + pattern[c][1] if c in pattern else 1.0 for c in chan])
        corr_err = np.array([pattern[c][2] if c in pattern else 0.0 for c in chan])
        s = s / corr
        err = np.hypot(err, s * corr_err)
    err_tot = np.hypot(err, sys_used * np.abs(s))
    nu0 = np.sqrt(nu.min() * nu.max())
    x = np.log(nu / nu0)
    try:
        p1, e1, c1 = fit(power_law, x, s, err_tot, [np.mean(s), -0.7])
        p2, e2, c2 = fit(log_parabola, x, s, err_tot, [p1[0], p1[1], 0.0])
    except RuntimeError:
        core_lines.append(f"{epoch:<16} fit failed")
        continue
    n = len(s)
    dbic = (c2 + 3 * np.log(n)) - (c1 + 2 * np.log(n))
    core_fits[epoch] = (n, nu0, p1[0], e1[0], p1[1], e1[1] * max(1.0, np.sqrt(c1 / (n - 2))), c1 / (n - 2), p2[2], e2[2], dbic)
    core_resid[epoch] = (nu, s / power_law(x, *p1) - 1.0, err_tot / power_law(x, *p1))
    core_lines.append(
        f"{epoch:<16} N={n}  S0={p1[0] * 1e3:9.3f} mJy  PL: alpha={p1[1]:+.3f} +/- {e1[1]:.3f} chi2/dof={c1 / (n - 2):7.2f}  |  "
        f"curved: alpha={p2[1]:+.3f} +/- {e2[1]:.3f} beta={p2[2]:+.2f} +/- {e2[2]:.2f} ({abs(p2[2]) / e2[2]:4.1f} sigma) "
        f"chi2/dof={c2 / (n - 3):7.2f}  dBIC={dbic:+6.2f}"
    )


# 3. LOBE SPECTRA (channel section of inband_alpha_lobes.txt: epoch, lobe, chan, nu, S, err)
lobe = defaultdict(list)
with open(args.lobes) as fh:
    in_channels = False
    for line in fh:
        if line.startswith("# Channel fluxes"):
            in_channels = True
            continue
        parts = line.split()
        if in_channels and len(parts) == 6 and not parts[0].startswith("#"):
            lobe[(parts[0], parts[1])].append((int(parts[2]), float(parts[3]), float(parts[4]) * 1e-6, float(parts[5]) * 1e-6))

lobe_fits = {}
for (epoch, label), rows in sorted(lobe.items()):
    chan, nu, s, err = (np.array(c) for c in zip(*usable(rows)))
    err_tot = np.hypot(err, sys_used * np.abs(s))
    nu0 = np.sqrt(nu.min() * nu.max())
    x = np.log(nu / nu0)
    try:
        p, e, c = fit(power_law, x, s, err_tot, [np.mean(s), -0.7])
    except RuntimeError:
        continue
    n = len(s)
    lobe_fits[(epoch, label)] = (n, nu0, p[0], e[0], p[1], e[1] * max(1.0, np.sqrt(c / (n - 2))), c / (n - 2))


def weighted_mean(vals):
    a, e = np.array(vals).T
    w = 1.0 / e**2
    m = float(np.sum(w * a) / np.sum(w))
    chi2 = float(np.sum(w * (a - m) ** 2))
    return m, float(1.0 / np.sqrt(np.sum(w))), chi2, len(a) - 1


mean_epochs = [e.strip() for e in args.lobe_epochs.split(",") if e.strip()]
lobe_means = {}
for label in ("N", "S", "N+S"):
    vals = [(v[4], v[5]) for (ep, lab), v in lobe_fits.items()
            if ep[:10] in mean_epochs and (label == "N+S" or lab == label)]
    if len(vals) > 1:
        lobe_means[label] = weighted_mean(vals)

lobe_lines = [
    f"{ep:<16} {lab}  N={v[0]}  S0={v[2] * 1e3:7.3f} mJy  alpha={v[4]:+.3f} +/- {v[5]:.3f}  chi2/dof={v[6]:5.2f}"
    + ("  *" if ep[:10] in mean_epochs else "")
    for (ep, lab), v in sorted(lobe_fits.items())
]
lobe_lines += [f"Mean over * epochs, lobe {lab}: alpha = {m:+.3f} +/- {e:.3f} (chi2 = {c:.2f} for {d} dof)"
               for lab, (m, e, c, d) in lobe_means.items()]


# 4. REPORT
def fmt_sys(val, ul):
    return f"{val * 100:.2f}%" if val > 0 else f"0 (95% upper limit {ul * 100:.2f}%)"


lines = [
    "IN-BAND CHANNEL SYSTEMATICS AND CURVATURE",
    f"Channels left out everywhere: {sorted(DROP) or 'none'}; also any channel with error > {args.max_err_ratio}x its epoch median.",
    f"Reference fluxes divided by A(nu) = {args.pb_atten} ** ((nu / {args.pb_freq} GHz) ** 2) (Gaussian primary beam).",
    "",
    "Reference sources, power law per epoch:",
    *ref_lines,
    "",
    f"Pooled reference chi2/dof = {chi2_ref:.1f} / {dof_ref} = {chi2_ref / dof_ref:.2f}; per-channel systematic for chi2/dof = 1: {fmt_sys(sys_raw, sys_raw_ul)}",
    f"Common channel pattern (weighted mean residual over epochs), chi2 against zero = {chi2_pattern:.1f} for {len(pattern)} channels:",
    *[f"  chan {c:2d}  nu = {n:.3f} GHz  {m * 100:+.2f}% +/- {e * 100:.2f}%  ({k} epochs)" for c, (n, m, e, k) in sorted(pattern.items())],
    f"After removing the pattern: systematic for chi2/dof = 1: {fmt_sys(sys_clean, sys_clean_ul)}",
    "",
    f"Core fits: pattern {'divided out' if args.pattern else 'not applied'}; per-channel systematic used = {sys_used * 100:.2f}%"
    + (" (set by --sys)" if args.sys is not None else ""),
    "dBIC = BIC(curved) - BIC(power law); negative favours curvature, < -6 strongly.",
    *core_lines,
    "",
    f"Lobe fits: per-channel systematic {sys_used * 100:.2f}%; alpha errors rescaled by sqrt(chi2/dof) where that exceeds one.",
    *lobe_lines,
]
with open(args.report, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"Saved report to {args.report}")

with open(args.final, "w") as fh:
    fh.write("# Final in-band spectral indices, S(nu) = S0 (nu/nu0)^alpha, written by inband_systematics.py\n")
    fh.write(f"# Channels {sorted(DROP)} left out, and any channel with error > {args.max_err_ratio}x its epoch median; "
             f"per-channel systematic {sys_used * 100:.2f}% (from the reference source) added in quadrature;\n")
    fh.write("# Err_a rescaled by sqrt(chi2/dof) where that exceeds one. beta, dBIC: log-parabola curvature test (dBIC < 0 favours curvature).\n")
    fh.write("# CORE\n# {:<14} {:>10} {:>3} {:>7} {:>10} {:>8} {:>7} {:>6} {:>8} {:>6} {:>7} {:>7}\n".format(
        "Epoch", "MJD", "N", "nu0", "S0(mJy)", "S0err", "alpha", "Err_a", "chi2/dof", "beta", "Err_b", "dBIC"))
    for ep, v in sorted(core_fits.items(), key=lambda kv: mjd.get(kv[0], np.inf)):
        n, nu0, s0, s0e, a, ae, chi, b, be, db = v
        fh.write(f"{ep:<16} {mjd.get(ep, np.nan):>10.4f} {n:>3d} {nu0:>7.4f} {s0 * 1e3:>10.4f} {s0e * 1e3:>8.4f} {a:>7.3f} {ae:>6.3f} {chi:>8.2f} {b:>6.2f} {be:>7.2f} {db:>7.2f}\n")
    fh.write("# LOBES (* = in the mean)\n# {:<14} {:>4} {:>10} {:>3} {:>7} {:>10} {:>8} {:>7} {:>6} {:>8}\n".format(
        "Epoch", "Lobe", "MJD", "N", "nu0", "S0(mJy)", "S0err", "alpha", "Err_a", "chi2/dof"))
    for (ep, lab), v in sorted(lobe_fits.items(), key=lambda kv: (mjd.get(kv[0][0], np.inf), kv[0][1])):
        n, nu0, s0, s0e, a, ae, chi = v
        fh.write(f"{ep:<16} {lab:>4} {mjd.get(ep, np.nan):>10.4f} {n:>3d} {nu0:>7.4f} {s0 * 1e3:>10.4f} {s0e * 1e3:>8.4f} {a:>7.3f} {ae:>6.3f} {chi:>8.2f}"
                 + ("  *" if ep[:10] in mean_epochs else "") + "\n")
    for lab, (m, e, c, d) in lobe_means.items():
        fh.write(f"# Mean lobe alpha ({lab}, * epochs): {m:+.3f} +/- {e:.3f}  (chi2 = {c:.2f} for {d} dof)\n")
print(f"Saved final table to {args.final}")


# 4. FIGURE: reference pattern (top) and core residuals about the power law (bottom)
# Categorical slots 1-5 of the reference palette in fixed order, plus a marker per epoch
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.8, 6.4), sharex=True, dpi=200)
for ax in (ax1, ax2):
    ax.axhline(0.0, color=MUTED, lw=1.0, zorder=1)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.tick_params(colors=INK)
    for spine in ax.spines.values():
        spine.set_color(MUTED)

for (epoch, label) in sorted(ref):
    pts = [(n, ri) for e, l, _c, n, ri, _s in ref_resid if e == epoch and l == label]
    n, ri = zip(*pts)
    ax1.plot(n, np.array(ri) * 100, color="#b9b8b0", lw=0.8, zorder=2)
pn = np.array([v[0] for v in pattern.values()])
pm = np.array([v[1] for v in pattern.values()]) * 100
pe = np.array([v[2] for v in pattern.values()]) * 100
ax1.errorbar(pn, pm, yerr=pe, fmt="o-", color=INK, ms=6, lw=2, capsize=3, zorder=3, label="mean over epochs")
ax1.plot([], [], color="#b9b8b0", lw=0.8, label="individual epochs")
ax1.set_ylabel("Reference residual (%)", color=INK)
ax1.set_title(f"Reference source: per-channel systematic {fmt_sys(sys_raw, sys_raw_ul)}", fontsize=10, color=INK, loc="left")
ax1.legend(frameon=False, fontsize=8, loc="best")

for i, epoch in enumerate(sorted(core_resid)[: args.plot_epochs]):
    nu, ri, si = core_resid[epoch]
    ax2.errorbar(nu, ri * 100, yerr=si * 100, fmt=MARKERS[i % 5] + "-", color=COLORS[i % 5], ms=6, lw=2, capsize=2,
                 markeredgecolor="white", markeredgewidth=1.0, label=epoch, zorder=3 + i)
ax2.set_xlabel("Frequency (GHz)", color=INK)
ax2.set_ylabel("Core residual about power law (%)", color=INK)
ax2.set_title(f"GRS 1915+105 core, earliest {args.plot_epochs} epochs (systematic {sys_used * 100:.2f}% included)", fontsize=10, color=INK, loc="left")
ax2.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Epoch", title_fontsize=8)
fig.tight_layout()
os.makedirs(os.path.dirname(os.path.abspath(args.figure)), exist_ok=True)
fig.savefig(args.figure, facecolor="white")
print(f"Saved figure to {args.figure}")
