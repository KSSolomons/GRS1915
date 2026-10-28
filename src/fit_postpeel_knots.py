"""
Fit beam-shaped components (fixed BMAJ, BMIN, BPA) to the residuals around the peeled core of a post-peel image,
to test whether the structure left near the core is jet knots or calibration ripple.

Models (all within --fit-radius of the given core position):
  core   : one component at the core (signed amplitude <= 5 mJy, position free within 1")
  jet    : core + two positive components (<= 10 mJy) at r >= 1.5", within +/-40 deg of PA --jet-pa and +180
  control: the same along the perpendicular axis
  anti   : the same as jet, but with the two components forced negative
           (a ripple has negative lobes as strong as the positive ones; real knots do not)

chi2 is computed per pixel with the 2-8" MAD noise of the image and divided by the beam area in pixels,
so dchi2 is roughly per independent beam.

  python src/fit_postpeel_knots.py FITS/peel/ep3/<...>postpeel-MFS-image.pbcor.fits 19:15:11.5449 +10:56:44.612 \
      --out figures/peel/epoch3_postpeel_knot_fit.png [--expect E_SE N_SE E_NW N_NW]
"""

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("ra")
parser.add_argument("dec")
parser.add_argument("--out", required=True)
parser.add_argument("--jet-pa", type=float, default=138.0)
parser.add_argument("--fit-radius", type=float, default=8.0)
parser.add_argument("--expect", type=float, nargs=4, default=None, metavar=("E_SE", "N_SE", "E_NW", "N_NW"),
                    help="Expected knot offsets (arcsec) to mark on the maps")
parser.add_argument("--knot-flux", type=float, nargs=2, default=None, metavar=("SE", "NW"), help="Expected knot fluxes (mJy), for the summary")
parser.add_argument("--psf", default=None, help="Post-peel MFS PSF image: also fit the residual as a * PSF (shift free), with and without dPSF/dE, dPSF/dN")
parser.add_argument("--ripple-at", type=float, nargs="+", default=None, metavar=("R", "PA"),
                    help="With --psf: pairs of separation (arcsec) and PA (deg) at which to report the residual left by each model")
args = parser.parse_args()

plt.rcParams["font.family"] = "serif"
h = fits.open(args.image, memmap=True)[0]
hd = h.header
w = WCS(hd).celestial
pix = abs(hd["CDELT2"]) * 3600
core_c = SkyCoord(args.ra, args.dec, unit=("hourangle", "deg"))
x0, y0 = (float(v) for v in w.world_to_pixel_values(core_c.ra.deg, core_c.dec.deg))
xi, yi = int(round(x0)), int(round(y0))
n = 30
img = np.array(h.data[0, 0][yi - n : yi + n + 1, xi - n : xi + n + 1]) * 1e3
yy, xx = np.mgrid[-n : n + 1, -n : n + 1]
E = -(xx - (x0 - xi)) * pix  # offsets from the given core position (east positive)
N = (yy - (y0 - yi)) * pix
r = np.hypot(E, N)
bmaj, bmin, bpa = hd["BMAJ"] * 3600, hd["BMIN"] * 3600, np.radians(hd["BPA"])
f2s = 1 / (2 * np.sqrt(2 * np.log(2)))
beam_px = 1.1331 * bmaj * bmin / pix**2
mad = lambda v: 1.4826 * np.median(np.abs(v - np.median(v)))
noise = mad(img[(r > 2) & (r < 8)])
bkg = mad(img[(r > 12) & (r < 15)])
sel = r < args.fit_radius


def beam(e0, n0):
    de, dn = E - e0, N - n0
    u = de * np.sin(bpa) + dn * np.cos(bpa)
    v = de * np.cos(bpa) - dn * np.sin(bpa)
    return np.exp(-0.5 * ((u / (bmaj * f2s)) ** 2 + (v / (bmin * f2s)) ** 2))


def knot_en(rr, pa_deg):
    return rr * np.sin(np.radians(pa_deg)), rr * np.cos(np.radians(pa_deg))


def model(p):
    """p = [core amp, core E, core N] + [amp, r, PA] per extra component."""
    m = p[0] * beam(p[1], p[2])
    for k in range(1, len(p) // 3):
        m = m + p[3 * k] * beam(*knot_en(p[3 * k + 1], p[3 * k + 2]))
    return m


def fit(p0, lo, hi):
    f = least_squares(lambda p: (model(p) - img)[sel] / noise, p0, bounds=(lo, hi))
    cov = np.linalg.pinv(f.jac.T @ f.jac)
    return f.x, np.sqrt(np.diag(cov)), (f.fun**2).sum() / beam_px


core_lo, core_hi = [-5, -1.0, -1.0], [5, 1.0, 1.0]
pc, ec, chi_core = fit([np.clip(img[n, n], -4.9, 4.9), 0, 0], core_lo, core_hi)


def two_comp(pa0, sign, radii=(1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0), dpa=40.0):
    """Best of a grid of starting radii for two components within +/-dpa of pa0 and pa0 + 180, r >= 1.5"."""
    best = None
    amp_lo, amp_hi = (0, 10) if sign > 0 else (-10, 0)
    for r1 in radii:
        for r2 in radii:
            p0 = list(pc) + [sign * 0.5, r1, pa0, sign * 0.5, r2, pa0 + 180]
            lo = core_lo + [amp_lo, 1.5, pa0 - dpa, amp_lo, 1.5, pa0 + 180 - dpa]
            hi = core_hi + [amp_hi, args.fit_radius, pa0 + dpa, amp_hi, args.fit_radius, pa0 + 180 + dpa]
            res = fit(p0, lo, hi)
            if best is None or res[2] < best[2]:
                best = res
    return best


fits_ = {"core": (pc, ec, chi_core), "jet": two_comp(args.jet_pa, +1), "control": two_comp(args.jet_pa + 90, +1),
         "anti": two_comp(args.jet_pa, -1)}

print(f"{args.image.split('/')[-1]}")
print(f"beam {bmaj:.2f}\" x {bmin:.2f}\" PA {hd['BPA']:.1f}; beam area {beam_px:.1f} px; "
      f"noise 2-8\" {noise * 1e3:.0f} uJy/b (12-15\": {bkg * 1e3:.0f}); fit radius {args.fit_radius:.0f}\"")
print("chi2 is per independent beam (pixel chi2 / beam area); dchi2 relative to the core-only model")
for name, (p, e, chi) in fits_.items():
    print(f"\n[{name}] chi2 {chi:.1f}  dchi2 {chi_core - chi:+.1f}")
    for k in range(len(p) // 3):
        a, de, dn = p[3 * k : 3 * k + 3]
        if k > 0:
            de, dn = knot_en(de, dn)
        lab = "core" if k == 0 else f"comp {k}"
        print(f"  {lab:>7}: {a:+6.2f} +/- {e[3 * k]:.2f} mJy/b at E{de:+.2f} N{dn:+.2f} "
              f"(sep {np.hypot(de, dn):.2f}\", PA {np.degrees(np.arctan2(de, dn)) % 360:5.1f}, S/N {abs(a) / noise:.1f})")
if args.knot_flux:
    print(f"\nexpected knots: SE {args.knot_flux[0]:.2f} mJy, NW {args.knot_flux[1]:.2f} mJy")



# PSF test: is the structure left near the core a ripple (the response to a small error in the peel model of the core)?
if args.psf:
    from scipy.interpolate import RectBivariateSpline

    ph = fits.open(args.psf, memmap=True)[0]
    pd = ph.data[0, 0]
    py, px = np.unravel_index(np.nanargmax(pd), pd.shape)
    m = int(round((args.fit_radius + 2.5) / pix)) + 2  # cut-out half-width covering the fit region plus a 1.5" shift
    cut_ = np.array(pd[py - m : py + m + 1, px - m : px + m + 1], dtype=float)
    spl = RectBivariateSpline(np.arange(-m, m + 1), np.arange(-m, m + 1), cut_, kx=3, ky=3)
    print(f"\nPSF {args.psf.split('/')[-1]}: peak {pd[py, px]:.3f} at pixel ({px}, {py}); cut-out +/-{m} px")

    def psf_at(e0, n0, deriv=None):
        """PSF centred at offset (e0, n0) from the core (arcsec), on the image grid; deriv = 'E' or 'N' gives d/dE or d/dN (per arcsec)."""
        xp, yp = -(E - e0) / pix, (N - n0) / pix
        if deriv == "E":
            return -spl.ev(yp, xp, dx=1) / pix
        if deriv == "N":
            return spl.ev(yp, xp, dy=1) / pix
        return spl.ev(yp, xp)

    def fit_psf(with_deriv):
        def res(p):
            mod = p[0] * psf_at(p[1], p[2])
            if with_deriv:
                mod = mod + p[3] * psf_at(p[1], p[2], "E") + p[4] * psf_at(p[1], p[2], "N")
            return (mod - img)[sel] / noise

        p0 = [pc[0], 0.0, 0.0] + ([0.0, 0.0] if with_deriv else [])
        lo = [-5, -1, -1] + ([-5, -5] if with_deriv else [])
        hi = [5, 1, 1] + ([5, 5] if with_deriv else [])
        f = least_squares(res, p0, bounds=(lo, hi))
        mod = f.x[0] * psf_at(f.x[1], f.x[2])
        if with_deriv:
            mod = mod + f.x[3] * psf_at(f.x[1], f.x[2], "E") + f.x[4] * psf_at(f.x[1], f.x[2], "N")
        return f.x, (f.fun**2).sum() / beam_px, mod

    psf_fits = {"PSF": fit_psf(False), "PSF + dPSF/dE, dPSF/dN": fit_psf(True)}
    print(f"chi2 per beam within {args.fit_radius:.0f}\": core-only (beam-shaped) {chi_core:.1f}")
    for name, (p, chi, _) in psf_fits.items():
        extra = f", dipole ({p[3]:+.2f}, {p[4]:+.2f}) mJy/b per arcsec" if len(p) > 3 else ""
        print(f"  {name:<24} {chi:7.1f}  (amp {p[0]:+.2f} mJy/b, shift E{p[1]:+.2f} N{p[2]:+.2f}\"{extra})")
    noise_ring = lambda a: mad(a[(r > 2) & (r < 8)])
    resids = {"core-only": img - model(pc), "PSF": img - psf_fits["PSF"][2], "PSF + deriv": img - psf_fits["PSF + dPSF/dE, dPSF/dN"][2]}
    print(f"rms-equivalent noise (MAD) in the 2-8\" ring: " + ", ".join(f"{k} {noise_ring(v) * 1e3:.0f}" for k, v in resids.items()) + f" uJy/b (12-15\": {bkg * 1e3:.0f})")
    if args.ripple_at:
        print("residual (mJy/b), peak within one beam-FWHM radius of each position:")
        pos_list = list(zip(args.ripple_at[0::2], args.ripple_at[1::2]))
        for rr_, pa_ in pos_list:
            e_, n_ = knot_en(rr_, pa_)
            near = np.hypot(E - e_, N - n_) < 0.5 * np.sqrt(bmaj * bmin)
            print(f"  at r={rr_:.2f}\" PA {pa_:.0f}: " + "  ".join(
                f"{k}: {v[near][np.argmax(np.abs(v[near]))]:+.2f}" for k, v in resids.items()))

# extrema along the jet axis, to show whether the structure alternates in sign
s = np.linspace(-args.fit_radius, args.fit_radius, 4 * int(args.fit_radius / pix) + 1)


def cut(a, pa0):
    col = (x0 - xi) + n - s * np.sin(np.radians(pa0)) / pix
    row = (y0 - yi) + n + s * np.cos(np.radians(pa0)) / pix
    return map_coordinates(a, [row, col], order=1)


cj = cut(img, args.jet_pa)
ext_i = [i for i in range(1, len(s) - 1) if (cj[i] - cj[i - 1]) * (cj[i + 1] - cj[i]) < 0 and abs(cj[i]) > 2 * bkg]
print("\nextrema along the jet axis (+ = SE):", "  ".join(f"{s[i]:+.1f}\":{cj[i]:+.2f}" for i in ext_i))

fig, ax = plt.subplots(1, 4, figsize=(21, 5.2))
ext = [E[0, 0] + pix / 2, E[0, -1] - pix / 2, N[0, 0] - pix / 2, N[-1, 0] + pix / 2]
vm = max(np.abs(img[r < 3]).max(), 0.3)
pj = fits_["jet"][0]
for a, (data, t) in zip(ax, [(img, "post-peel image"), (img - model(pc), "minus core-only fit"), (img - model(pj), "minus core + 2 jet comps")]):
    im = a.imshow(data, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vm, vmax=vm)
    for pa0 in (args.jet_pa, args.jet_pa + 180):
        a.plot([0, 8 * np.sin(np.radians(pa0))], [0, 8 * np.cos(np.radians(pa0))], "k--", lw=0.6)
    a.add_patch(plt.Circle((0, 0), 3, fill=False, color="0.4", lw=0.8))
    for k in (1, 2):
        a.plot(*knot_en(pj[3 * k + 1], pj[3 * k + 2]), "x", color="k", ms=9, mew=1.8)
    if args.expect:
        a.plot(args.expect[0::2], args.expect[1::2], "+", color="lime", ms=14, mew=2)
    a.set_xlim(args.fit_radius, -args.fit_radius)
    a.set_ylim(-args.fit_radius, args.fit_radius)
    a.set_title(t, fontsize=10)
    a.set_xlabel("East offset from core (arcsec)")
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.03, label="mJy/beam")
ax[0].set_ylabel("North offset (arcsec)")
a = ax[3]
a.plot(s, cj, "k", label="jet axis (PA %.0f)" % args.jet_pa)
a.plot(s, cut(img, args.jet_pa + 90), color="orange", lw=1, label="control axis (PA %.0f)" % (args.jet_pa + 90))
a.plot(s, cut(model(pj), args.jet_pa), "b--", lw=1, label="core + 2 jet comps")
if args.knot_flux:
    a.axhline(args.knot_flux[0], color="r", ls=":", lw=0.8, label="expected knot flux")
a.axhline(0, color="0.6", lw=0.5)
a.set_xlabel("distance along the axis (arcsec; + = SE / PA %.0f)" % (args.jet_pa + 90))
a.set_ylabel("mJy/beam")
a.legend(fontsize=8)
fig.tight_layout()
fig.savefig(args.out, dpi=120)
print("saved", args.out)
