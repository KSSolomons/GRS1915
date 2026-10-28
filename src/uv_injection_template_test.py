"""
Epoch 4 uv injection (src/inject_knot_uv.py): how much of the injected knot survives the peel, and how strongly the
original post-peel image disfavours a real knot of S_exp, using the peeled knot itself as the template.

The surviving knot is not beam-shaped (its inner part is absorbed into the dE solution and over-subtracted, the rest is
shifted outwards), so a "surviving fraction" depends on the estimator. This script prints several, and then avoids the
question with a matched filter: template T = injected post-peel image - original post-peel image (the response of the
peel to a knot of S_inj), amplitude a = sum(T * img) / sum(T^2) over a mask, so a = 1 for an S_inj knot on a blank
background. The noise is the scatter of a over rotated copies of the original image (template >= 40 deg from the jet
axis), and the significance of an S_inj knot is (1 + mean_noise - a_obs) / sigma_noise. Neighbouring rotations are
correlated (0.94 at 10 deg, zero only at 45 deg; about 4 independent samples), so sigma_noise, and hence the
significance, is uncertain by ~40 per cent.

    python src/uv_injection_template_test.py  ->  data/peel_regions/uv_injection_ep4_template.txt
"""

import glob

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

CORE = "19:15:11.5085 +10:56:43.507"  # data/peel_regions/epoch4_1742780174_peel_3as.reg
R_INJ, PA_INJ, S_INJ = 2.8, 137.6, 2.83
JET_PA = 137.6
PIX = 0.5
OUT = "data/peel_regions/uv_injection_ep4_template.txt"
core = SkyCoord(CORE, unit=(u.hourangle, u.deg))


def load(pattern):
    h = fits.open(glob.glob(pattern)[0], memmap=True)[0]
    x0, y0 = (float(v) for v in WCS(h.header).celestial.world_to_pixel_values(core.ra.deg, core.dec.deg))
    return np.asarray(h.data[0, 0]), x0, y0


O, xo, yo = load("FITS/peel/ep4/*postpeel-MFS-image.pbcor.fits")
I, xi, yi = load("FITS/peel/ep4_inj/*postpeel-MFS-image.fits")
P, _, _ = load("FITS/peel/ep4/*postpeel-MFS-psf.fits")
py, px = np.unravel_index(np.argmax(P), P.shape)

g = np.arange(-8, 8.01, 0.25)
E, N = np.meshgrid(g, g)


def samp(img, xc, yc, e, n, rot=0.0):
    """img (x1e3 unless PSF) at offsets (e, n) arcsec from (xc, yc), after rotating the sky by rot deg (PA sense)."""
    a = np.radians(rot)
    er, nr = e * np.cos(a) + n * np.sin(a), -e * np.sin(a) + n * np.cos(a)
    return map_coordinates(img, [yc + nr / PIX, xc - er / PIX], order=1)


o = samp(O, xo, yo, E, N) * 1e3
T = samp(I, xi, yi, E, N) * 1e3 - o
ek, nk = R_INJ * np.sin(np.radians(PA_INJ)), R_INJ * np.cos(np.radians(PA_INJ))
psf_k = samp(P, px, py, E - ek, N - nk)
r = np.hypot(E, N)
pa = np.degrees(np.arctan2(E, N)) % 360
dpa = lambda a, b: np.abs((a - b + 180) % 360 - 180)
lines = ["Epoch 4 uv injection: surviving fraction and matched-filter test (src/uv_injection_template_test.py)", ""]

# 1. Surviving fraction by estimator
along = lambda img, s: float(map_coordinates(img, [[(s * np.cos(np.radians(PA_INJ)) + 8) / 0.25],
                                                    [(s * np.sin(np.radians(PA_INJ)) + 8) / 0.25]], order=1)[0])
lines.append("Cut along the jet axis (mJy/beam): expected if the whole knot survived (S_inj x PSF) vs measured difference")
for s in (1.0, 1.35, 2.0, 2.5, 2.8, 3.1, 3.45, 4.0, 5.0):
    lines.append(f"  {s:4.2f}\"  expected {along(S_INJ * psf_k, s):+.2f}  measured {along(T, s):+.2f}")
sel = r < 5
f_k = least_squares(lambda q: (q[0] * S_INJ * psf_k - T)[sel], [0.5])
peak = np.unravel_index(np.argmax(np.where(sel, T, -np.inf)), T.shape)
lines += ["", "Surviving fraction by estimator:",
          f"  pixel value at the injected position      {along(T, R_INJ) / S_INJ:.2f}",
          f"  peak of the difference ({r[peak]:.2f}\", PA {pa[peak]:.0f})      {T[peak] / S_INJ:.2f}",
          f"  PSF-shaped knot fitted within 5\"          {f_k.x[0]:.2f}",
          "  beam-shaped fit with free core term       0.51  (src/measure_uv_injection.py; the core term absorbs the trough)"]

# 2. Matched filter with the peeled knot as template
rots = [x for x in np.arange(0, 360, 10) if min(dpa(PA_INJ + x, JET_PA), dpa(PA_INJ + x, JET_PA + 180)) >= 40]
masks = {"within 1.5\" of (3.1\", PA 138)": np.hypot(E - 3.1 * np.sin(np.radians(138)), N - 3.1 * np.cos(np.radians(138))) < 1.5,
         "1.5-5\" from the core, within 40 deg of the SE axis": (r > 1.5) & (r < 5) & (dpa(pa, JET_PA) < 40)}
lines += ["", f"Matched filter (template = peeled knot of {S_INJ} mJy; noise from {len(rots)} rotations >= 40 deg from the jet axis)"]
for label, m in masks.items():
    t = T[m]
    amp = lambda img: (t * img[m]).sum() / (t * t).sum()
    a_obs = amp(o)
    noise = np.array([amp(samp(O, xo, yo, E, N, rot=x) * 1e3) for x in rots])
    sd, mu = noise.std(ddof=1), noise.mean()
    sig = (1 + mu - a_obs) / sd
    lim5 = (a_obs - mu + 5 * sd) * S_INJ
    lines.append(f"  mask {label}:")
    lines.append(f"    a_obs = {a_obs:.3f} (as an S_inj-scaled knot: {a_obs * S_INJ:.2f} mJy); noise sigma {sd:.3f}, mean {mu:+.3f}")
    lines.append(f"    knot of {S_INJ} mJy disfavoured at {sig:.1f} sigma; for 2.97 mJy (SE fade) {(2.97 / S_INJ + mu - a_obs) / sd:.1f} sigma;"
                 f" 5 sigma upper limit {lim5:.2f} mJy")

open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"Saved {OUT}")
