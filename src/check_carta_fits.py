"""
Can the CARTA centre fits (S-band-fits/*2D_Fitting*.txt) be reproduced outside CARTA?

For every epoch with a fit file, fits the MFS image (FITS/pcalmask/) with the same model: one 2D Gaussian per CARTA
component, shape frozen to the image's restoring beam (major, minor, PA from the header), centre and amplitude free,
background fixed at the CARTA value, pixels inside a ~10" region around the core. Two starts are used:
  carta  the CARTA centres and amplitudes (does the minimum stay put?);
  blind  the core at the image peak and the lobes at +/-3" along the jet axis (is it the same minimum?).
Positions are compared to CARTA in arcsec and in units of the CARTA 1 sigma; amplitudes in per cent.

    python src/check_carta_fits.py [--radius 10] [--free-background]

Writes data/carta_fit_check.txt.
"""

import argparse
import glob
import itertools
import os
import re

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from scipy.optimize import least_squares

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
JET_PA = np.radians(137.6)  # East of North, SE side (src/epoch_limits.py JET_PA_DEG)

p = argparse.ArgumentParser()
p.add_argument("--radius", type=float, default=10.0, help="Fit region radius about the CARTA core (arcsec)")
p.add_argument("--free-background", action="store_true", help="Fit the background instead of fixing it at the CARTA value")
args = p.parse_args()


def num(text):
    return float(text.split("(")[0].split("±")[0].strip())


def parse_carta(path):
    """Components (sky, sigma_x_s, sigma_y_as, amp, beam) and background of a CARTA 2D fit file."""
    comps, bg, cur = [], 0.0, None
    for line in open(path):
        line = line.strip()
        if line.startswith("Component #"):
            cur = {}
            comps.append(cur)
        elif cur is not None and line.startswith("Center X") and ":" in line:
            val = line.split("=")[1]
            cur["ra"] = val.split("±")[0].strip().split("(")[0].strip()
            cur["sx"] = num(val.split("±")[1]) if "±" in val else np.nan  # seconds of time
        elif cur is not None and line.startswith("Center Y") and ":" in line:
            val = line.split("=")[1]
            cur["dec"] = val.split("±")[0].strip()
            cur["sy"] = num(val.split("±")[1]) if "±" in val else np.nan  # arcsec
        elif cur is not None and line.startswith("Amplitude"):
            cur["amp"] = num(line.split("=")[1])
            cur["samp"] = num(line.split("±")[1]) if "±" in line else np.nan
        elif line.startswith("Background"):
            bg = num(line.split("=")[1])
    for c in comps:
        c["sky"] = SkyCoord(c["ra"].replace(":", "h", 1).replace(":", "m", 1) + "s",
                            c["dec"].replace(":", "d", 1).replace(":", "m", 1) + "s", frame="fk5")
    return comps, bg


def model(par, nbeam, u_east, v_north, bmaj, bmin, pa, bg):
    """Sum of beam-shaped Gaussians. par = (x0, y0, amp) per component, offsets in arcsec (east, north)."""
    out = np.full_like(u_east, bg, dtype=float)
    s1, s2 = bmaj * FWHM_TO_SIGMA, bmin * FWHM_TO_SIGMA
    for k in range(nbeam):
        x0, y0, a = par[3 * k:3 * k + 3]
        dx, dy = u_east - x0, v_north - y0
        u = dx * np.sin(pa) + dy * np.cos(pa)
        v = -dx * np.cos(pa) + dy * np.sin(pa)
        out += a * np.exp(-0.5 * ((u / s1) ** 2 + (v / s2) ** 2))
    return out


rows = []
files = sorted(glob.glob(os.path.join(ROOT, "S-band-fits", "*2D_Fitting*.txt")))
report = []
for f in files:
    ts = re.match(r"img_(\d{10})", os.path.basename(f)).group(1)
    imgs = glob.glob(os.path.join(ROOT, "FITS", "pcalmask", f"img_{ts}*pcalmask-MFS-image.pbcor.fits"))
    if not imgs:
        report.append(f"{ts}: no image in FITS/pcalmask, skipped")
        continue
    comps, bg_carta = parse_carta(f)
    hdu = fits.open(imgs[0])[0]
    data = hdu.data
    while data.ndim > 2:
        data = data[0]
    wcs = WCS(hdu.header).celestial
    pix = abs(hdu.header["CDELT1"]) * 3600
    bmaj, bmin = hdu.header["BMAJ"] * 3600, hdu.header["BMIN"] * 3600
    pa = np.radians(hdu.header["BPA"])
    # The core is the brightest component; the region is centred on it
    comps = sorted(comps, key=lambda c: -c["amp"])  # core first, then the lobes by amplitude
    core = comps[0]
    cut = Cutout2D(data, core["sky"], 2 * args.radius / pix + 1, wcs=wcs)
    yy, xx = np.mgrid[0:cut.data.shape[0], 0:cut.data.shape[1]]
    sky = cut.wcs.pixel_to_world(xx, yy)
    east = (sky.ra.deg - core["sky"].ra.deg) * np.cos(np.radians(core["sky"].dec.deg)) * 3600
    north = (sky.dec.deg - core["sky"].dec.deg) * 3600
    inside = np.hypot(east, north) <= args.radius
    z = cut.data[inside]
    ue, vn = east[inside], north[inside]
    ok = np.isfinite(z)
    z, ue, vn = z[ok], ue[ok], vn[ok]
    bg0 = bg_carta

    def offsets(c):
        return ((c["sky"].ra.deg - core["sky"].ra.deg) * np.cos(np.radians(core["sky"].dec.deg)) * 3600,
                (c["sky"].dec.deg - core["sky"].dec.deg) * 3600)

    n = len(comps)
    starts = {}
    p0 = []
    for c in comps:
        x, y = offsets(c)
        p0 += [x, y, c["amp"]]
    starts["carta"] = p0
    # blind start: the core at the image peak, lobes at +/-3" on the jet axis
    peak = np.argmax(z)
    p0 = [ue[peak], vn[peak], z[peak]]
    for k in range(1, n):
        sgn = 1 if k == 1 else -1
        p0 += [sgn * 3 * np.sin(JET_PA), sgn * 3 * np.cos(JET_PA), 0.001]
    starts["blind"] = p0

    def resid(par):
        bg = par[-1] if args.free_background else bg0
        return model(par[:3 * n], n, ue, vn, bmaj, bmin, pa, bg) - z

    for label, p0 in starts.items():
        x0 = np.array(p0 + ([bg0] if args.free_background else []), float)
        lo = np.array(sum([[-args.radius, -args.radius, -0.1] for _ in range(n)], []) + ([-0.01] if args.free_background else []))
        hi = np.array(sum([[args.radius, args.radius, 2.0] for _ in range(n)], []) + ([0.01] if args.free_background else []))
        x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
        sol = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac")
        dof = max(len(z) - len(x0), 1)
        s2 = 2 * sol.cost / dof
        cov = np.linalg.pinv(sol.jac.T @ sol.jac) * s2
        err = np.sqrt(np.diag(cov))
        # Match each refit component to the CARTA component it landed on (lobes can come back swapped: the two
        # lobes are interchangeable in the model, and the blind start does not know which is brighter)
        fit = [sol.x[3 * k:3 * k + 3] for k in range(n)]
        best = min(itertools.permutations(range(1, n)),
                   key=lambda pm: sum(np.hypot(fit[j][0] - offsets(comps[k])[0], fit[j][1] - offsets(comps[k])[1])
                                      for k, j in zip(range(1, n), pm)))
        match = [0] + list(best)
        for k, c in enumerate(comps):
            x, y, a = fit[match[k]]
            cx, cy = offsets(c)
            sx_as = c["sx"] * 15 * np.cos(np.radians(core["sky"].dec.deg))  # seconds of time -> arcsec on the sky
            d = np.hypot(x - cx, y - cy)
            sig = np.hypot(c["sx"] * 15 * np.cos(np.radians(c["sky"].dec.deg)), c["sy"])
            rows.append(dict(ts=ts, start=label, comp="core" if k == 0 else f"lobe{k}", d=d, sig=sig,
                             dx=x - cx, dy=y - cy, a_carta=c["amp"] * 1e3, a=a * 1e3,
                             ea=err[3 * match[k] + 2] * 1e3, ex=err[3 * match[k]], ey=err[3 * match[k] + 1]))
    report.append(f"{ts}: {n} component(s), beam {bmaj:.3f} x {bmin:.3f} arcsec, PA {np.degrees(pa):.2f} deg, "
                  f"{len(z)} pixels within {args.radius:g} arcsec, background {'free' if args.free_background else f'fixed {bg0 * 1e3:.4f} mJy/beam'}")

lines = ["# CARTA 2D fits vs a beam-frozen refit (src/check_carta_fits.py)",
         f"# region radius {args.radius:g} arcsec about the CARTA core; background "
         f"{'free' if args.free_background else 'fixed at the CARTA value'}",
         "# d = offset of the refit centre from the CARTA centre (arcsec); sig = CARTA 1 sigma on the centre (arcsec);",
         "# dA = refit minus CARTA amplitude (mJy/beam); eA = refit 1 sigma on the amplitude.", ""] + report + [""]
lines.append(f"{'epoch':11s} {'start':6s} {'comp':6s} {'d (as)':>8s} {'d/sig':>7s} {'dx':>8s} {'dy':>8s} {'A_carta':>9s} {'A_fit':>9s} {'dA':>8s} {'eA':>7s}")
for r in rows:
    lines.append(f"{r['ts']:11s} {r['start']:6s} {r['comp']:6s} {r['d']:8.3f} {r['d'] / r['sig']:7.2f} {r['dx']:8.3f} {r['dy']:8.3f} "
                 f"{r['a_carta']:9.3f} {r['a']:9.3f} {r['a'] - r['a_carta']:8.3f} {r['ea']:7.3f}")
text = "\n".join(lines)
print(text)
out = os.path.join(ROOT, "data", "carta_fit_check.txt")
open(out, "w").write(text + "\n")
print(f"\nSaved {out}")
