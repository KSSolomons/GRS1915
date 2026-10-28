"""
Check for jet knots next to the core in a pbcor image by subtracting a beam-shaped core fit.

Reports positive peaks of the core-subtracted image within --r-max, the same scan along a control axis
(perpendicular to the jet), an optional free fit of core + two knots started from given offsets, and
makes a four-panel figure (image, core-subtracted, after core + knots, cut along the jet axis).

  python src/check_knots.py FITS/pcalmask/img_<id>_...pbcor.fits --out figures/peel/epoch3_knot_check.png \
      [--knots E_SE N_SE E_NW N_NW --knot-flux SE_MJY NW_MJY]
"""

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates, maximum_filter
from scipy.optimize import least_squares

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("--out", required=True)
parser.add_argument("--jet-pa", type=float, default=138.0)
parser.add_argument("--r-max", type=float, default=7.0)
parser.add_argument("--fit-radius", type=float, default=9.0, help="Only pixels within this radius enter the fits")
parser.add_argument("--knots", type=float, nargs=4, default=None, metavar=("E_SE", "N_SE", "E_NW", "N_NW"),
                    help="Starting offsets (arcsec, east/north of the core) for a free core + 2 knot fit")
parser.add_argument("--knot-flux", type=float, nargs=2, default=(3.0, 3.0), metavar=("SE", "NW"), help="Starting knot fluxes (mJy)")
parser.add_argument("--mark", type=float, nargs="*", default=[], help="Extra E N pairs to mark on the maps (e.g. model predictions)")
args = parser.parse_args()

plt.rcParams["font.family"] = "serif"
CAT = SkyCoord("19:15:11.558", "10:56:44.905", unit=("hourangle", "deg"))
h = fits.open(args.image, memmap=True)[0]
hd = h.header
w = WCS(hd).celestial
pix = abs(hd["CDELT2"]) * 3600
x0, y0 = (float(v) for v in w.world_to_pixel_values(CAT.ra.deg, CAT.dec.deg))
n = 40
xi, yi = int(round(x0)), int(round(y0))
big = np.array(h.data[0, 0][yi - 20 : yi + 21, xi - 20 : xi + 21]) * 1e3
jy, jx = np.unravel_index(np.argmax(big), big.shape)  # peak pixel in a +/-10" box of the catalogue position
xi, yi = xi - 20 + jx, yi - 20 + jy
img = np.array(h.data[0, 0][yi - n : yi + n + 1, xi - n : xi + n + 1]) * 1e3
yy, xx = np.mgrid[-n : n + 1, -n : n + 1]
bmaj, bmin, bpa = hd["BMAJ"] * 3600, hd["BMIN"] * 3600, np.radians(hd["BPA"])
f2s = 1 / (2 * np.sqrt(2 * np.log(2)))


def beam(e0, n0, E, N):
    de, dn = E - e0, N - n0
    u = de * np.sin(bpa) + dn * np.cos(bpa)
    v = de * np.cos(bpa) - dn * np.sin(bpa)
    return np.exp(-0.5 * ((u / (bmaj * f2s)) ** 2 + (v / (bmin * f2s)) ** 2))


# core fit with a free centre
E0, N0 = -xx * pix, yy * pix
box = np.hypot(E0, N0) < 4
c1 = least_squares(lambda p: (p[0] * beam(p[1], p[2], E0, N0) - img)[box], [img.max(), 0, 0]).x
ce, cn = c1[1], c1[2]
E, N = E0 - ce, N0 - cn  # offsets from the fitted core
r = np.hypot(E, N)
pa = np.degrees(np.arctan2(E, N)) % 360
core = c1[0] * beam(0, 0, E, N)
res = img - core
mad = lambda v: 1.4826 * np.median(np.abs(v - np.median(v)))
bkg = mad(img[(r > 15) & (r < 20)])
near = mad(res[(r > 2) & (r < 8)])  # noise of the core-subtracted image close to the core
core_sky = w.pixel_to_world_values(xi + ce / (-pix), yi + cn / pix)
print(f"core {c1[0]:.1f} mJy/b; beam {bmaj:.2f}x{bmin:.2f} PA {hd['BPA']:.1f}; local rms 15-20\" {bkg:.3f} mJy; "
      f"core-only residual rms within {args.fit_radius:.0f}\": {np.sqrt((res[r < args.fit_radius] ** 2).mean()):.3f}")
print("core RA/Dec:", SkyCoord(float(core_sky[0]) % 360, float(core_sky[1]), unit="deg").to_string("hmsdms", sep=":", precision=3))

# peaks of the core-subtracted image, jet axis versus control axis
def sector_peak(pa0, rmin=1.0):
    dpa = np.abs((pa - pa0 + 180) % 360 - 180)
    s = (dpa <= 20) & (r > rmin) & (r < args.r_max)
    j = np.argmax(np.where(s, res, -np.inf))
    j, i = np.unravel_index(j, res.shape)
    return res[j, i], r[j, i], E[j, i], N[j, i]

print(f"noise of the core-subtracted image at 2-8\" (MAD): {near:.3f} mJy, i.e. {near / bkg:.1f} x the 15-20\" rms")
print("sector peaks of the core-subtracted image (+/-20 deg, 1\" < r < %.0f\"):" % args.r_max)
for lab, p0 in (("SE (PA %.0f)" % args.jet_pa, args.jet_pa), ("NW (PA %.0f)" % (args.jet_pa + 180), args.jet_pa + 180),
                ("control PA %.0f" % (args.jet_pa + 90), args.jet_pa + 90), ("control PA %.0f" % (args.jet_pa + 270), args.jet_pa + 270)):
    v, rr, e, nn = sector_peak(p0 % 360)
    print(f"  {lab:>16}: {v:6.2f} mJy/b ({v / bkg:5.1f} x local rms) at r = {rr:.2f}\" E{e:+.2f} N{nn:+.2f}")
pk = (res == maximum_filter(res, size=5)) & (res > 3 * bkg) & (r < args.r_max)
print("all positive peaks > 3 x local rms within %.0f\":" % args.r_max)
for j, i in zip(*np.nonzero(pk)):
    print(f"   {res[j, i]:.2f} mJy/b at E{E[j, i]:+.1f} N{N[j, i]:+.1f}  (r {r[j, i]:.1f}\", PA {pa[j, i]:.0f}, {res[j, i] / bkg:.1f} x rms)")

# optional free fit of core + two knots
model_img = None
if args.knots:
    sel = r < args.fit_radius
    p0 = [c1[0], ce, cn, args.knot_flux[0], args.knots[0] + ce, args.knots[1] + cn, args.knot_flux[1], args.knots[2] + ce, args.knots[3] + cn]
    mdl = lambda p: sum(p[3 * k] * beam(p[3 * k + 1], p[3 * k + 2], E0, N0) for k in range(3))
    lo = [0, ce - 3, cn - 3, 0, -9, -9, 0, -9, -9]  # fluxes >= 0, positions within the cutout
    hi = [1e4, ce + 3, cn + 3, 50, 9, 9, 50, 9, 9]
    fit = least_squares(lambda p: (mdl(p) - img)[sel], p0, bounds=(lo, hi))
    p = fit.x
    jac = fit.jac
    cov = np.linalg.pinv(jac.T @ jac) * bkg**2
    err = np.sqrt(np.diag(cov))
    model_img = mdl(p)
    print("free core + 2 knots fit (beam-shaped, errors from local rms; no systematic from the core model):")
    for k, lab in enumerate(("core", "SE knot", "NW knot")):
        e, nn = p[3 * k + 1] - ce, p[3 * k + 2] - cn
        print(f"  {lab:>8}: {p[3 * k]:6.2f} +/- {err[3 * k]:.2f} mJy at E{e:+.2f} N{nn:+.2f} (+/-{err[3 * k + 1]:.2f},{err[3 * k + 2]:.2f}) sep {np.hypot(e, nn):.2f}\" PA {np.degrees(np.arctan2(e, nn)) % 360:.0f}")
    print(f"  rms within {args.fit_radius:.0f}\": {np.sqrt(((img - model_img)[sel] ** 2).mean()):.3f} mJy (core only {np.sqrt((res[sel] ** 2).mean()):.3f})")

fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
ext = [E[0, 0] + pix / 2, E[0, -1] - pix / 2, N[0, 0] - pix / 2, N[-1, 0] + pix / 2]
panels = [(img, "pbcor image (mJy/beam, sqrt)", None), (res, "after subtracting a core-only fit", 3 * max(bkg * 3, 1.0))]
if model_img is not None:
    panels.append((img - model_img, "after subtracting core + 2 knots", 3 * max(bkg * 3, 1.0)))
for ax, (data, t, vm) in zip(axes, panels):
    if vm is None:
        im = ax.imshow(data, origin="lower", extent=ext, cmap="inferno", norm=matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=data.max()))
    else:
        im = ax.imshow(data, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vm, vmax=vm)
    ax.plot(0, 0, "x", color="k" if vm else "w", ms=10, mew=2)
    for pa0 in (args.jet_pa, args.jet_pa + 180):
        ax.plot([0, 6 * np.sin(np.radians(pa0))], [0, 6 * np.cos(np.radians(pa0))], "k--" if vm else "w--", lw=0.7)
    for e, nn in zip(args.mark[0::2], args.mark[1::2]):
        ax.plot(e, nn, "+", color="lime", ms=14, mew=2)
    ax.set_xlim(8, -8)
    ax.set_ylim(-8, 8)
    ax.set_title(t, fontsize=10)
    ax.set_xlabel("East offset from fitted core (arcsec)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
axes[0].set_ylabel("North offset (arcsec)")
ax = axes[3]
s = np.linspace(-7, 7, 281)


def prof(a, pa0):
    col = -(s * np.sin(np.radians(pa0)) / pix) + n + (ce / -pix)
    row = (s * np.cos(np.radians(pa0)) / pix) + n + cn / pix
    return map_coordinates(a, [row, col], order=1)


ax.plot(s, prof(img, args.jet_pa), "k", label="image")
ax.plot(s, prof(core, args.jet_pa), "0.5", ls="--", label="core-only fit")
ax.plot(s, prof(res, args.jet_pa) * 5, "b", lw=1, label="core-subtracted x5 (jet axis)")
ax.plot(s, prof(res, args.jet_pa + 90) * 5, color="orange", lw=1, label="core-subtracted x5 (control axis)")
ax.axhline(5 * bkg, color="r", ls=":", lw=0.8, label="5 x local rms (x5 scale)")
ax.set_xlabel("distance along the axis (arcsec; + = SE for the jet axis)")
ax.set_ylabel("mJy/beam")
ax.legend(fontsize=8)
ax.set_title("cuts through the core", fontsize=10)
fig.tight_layout()
fig.savefig(args.out, dpi=130)
print("saved", args.out)
