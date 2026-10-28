"""
Lobe detection test for Epochs 5-12, and checks that the Epoch 12 N lobe is not a sidelobe or artefact of the core.

1. Detection. A beam-shaped component held at the separation the linear model predicts for each lobe (fitted
   together with the core and the other lobe, within 12" of the core). Background level and noise sigma_bg from
   beam-shaped fits at 400 random positions 10-30" from the core and more than 20 deg from the jet axis (outside the
   core and the jets): level = their mean, sigma_bg = their standard deviation. A lobe is detected if
   (amp - level)/sigma_bg >= 3. The predictions use the linear-model fit without Epoch 12 (52 coordinates), so the
   Epoch 12 test does not use the Epoch 12 positions.
   For Epoch 12 the same significance is also given with the level and noise from (a) a ring at the lobe's own
   radius (24 PAs > 30 deg from the axis; picks up the core's wings in the elongated Epoch 12 beam), (b) 30-90"
   (robust scatter; one field source lies in it) and (c) the pixel rms at 15-40", with and without its median.
2. Field-source test. Beam-shaped residual at the Epoch 12 lobe offsets around compact field sources S1-S3 of
   data/frame_offsets.txt (S4 and S5 lie 17-20' out and have a bright neighbour / high noise), as a fraction of
   their peak: what a compact source leaves at these offsets in the same image (PSF sidelobes and calibration
   artefacts), to compare with the lobe/core ratio.
3. Mirror test. Amplitude (with a local constant) at each Epoch 12 feature and at the point reflected through the
   core; a symmetric artefact (like the post-peel ripple of Epochs 3-4) would be as bright at both.

The Epoch 12 image was CLEANed to 1 uJy inside the mask (WSClean HISTORY), and the lobe positions lie inside it
(masks/epoch1/0_S_GRS1915+105.mask0.fits, the same mask for every epoch), so the core leaves no dirty-beam
sidelobes. The Epoch 12 PSF image is no longer on ilifu.

  python src/epoch12_checks.py [--out data/epoch12_checks.txt]
"""

import argparse
import glob
import re

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import least_squares

p = argparse.ArgumentParser()
p.add_argument("--out", default="data/epoch12_checks.txt")
p.add_argument("--seed", type=int, default=3)
args = p.parse_args()
rng = np.random.default_rng(args.seed)

TEJ, MUA, MUR, PAS, PAN = 60370.94, 7.2306e-3, 5.4870e-3, 137.59, 318.24  # linear model, 52-coordinate fit
EPOCHS = {5: "1745709012", 6: "1749586570", 7: "1751137270", 8: "1753549996", 9: "1756479610",
          10: "1759581193", 11: "1761391938", 12: "1764664923"}
FIELD = {"S1": "19h14m44.71s +10d57m56.26s", "S2": "19h15m57.12s +11d05m18.16s", "S3": "19h15m40.47s +11d09m17.92s"}
EP12_FREE = {"NW": (4.00, 335.0), "SE": (4.97, 162.5)}  # free-position fit of the Epoch 12 features (r ", PA deg)
F2S = 1 / (2 * np.sqrt(2 * np.log(2)))
mad = lambda v: 1.4826 * np.median(np.abs(v - np.median(v)))
dpa = lambda a, b: np.abs((a - b + 180) % 360 - 180)
en = lambda r, pa: (r * np.sin(np.radians(pa)), r * np.cos(np.radians(pa)))


class Image:
    def __init__(self, ts, centre, n=200):
        h = fits.open(glob.glob(f"FITS/pcalmask/img_{ts}*.fits")[0], memmap=True)[0]
        hd = h.header
        w = WCS(hd).celestial
        pix = abs(hd["CDELT2"]) * 3600
        self.bmaj, self.bmin, self.bpa = hd["BMAJ"] * 3600, hd["BMIN"] * 3600, np.radians(hd["BPA"])
        x0, y0 = (float(v) for v in w.world_to_pixel_values(centre.ra.deg, centre.dec.deg))
        xi, yi = int(round(x0)), int(round(y0))
        self.img = np.array(h.data[0, 0][yi - n : yi + n + 1, xi - n : xi + n + 1]) * 1e6  # uJy/beam
        yy, xx = np.mgrid[-n : n + 1, -n : n + 1]
        self.E = -(xx - (x0 - xi)) * pix
        self.N = (yy - (y0 - yi)) * pix
        self.r = np.hypot(self.E, self.N)

    def beam(self, e0, n0, sel):
        de, dn = self.E[sel] - e0, self.N[sel] - n0
        u = de * np.sin(self.bpa) + dn * np.cos(self.bpa)
        v = de * np.cos(self.bpa) - dn * np.sin(self.bpa)
        return np.exp(-0.5 * ((u / (self.bmaj * F2S)) ** 2 + (v / (self.bmin * F2S)) ** 2))

    def lsq(self, positions, sel, const=False, img=None):
        d = (self.img if img is None else img)[sel]
        cols = [self.beam(*q, sel) for q in positions] + ([np.ones(sel.sum())] if const else [])
        return np.linalg.lstsq(np.vstack(cols).T, d, rcond=None)[0]

    def single(self, e0, n0):
        """Amplitude of one beam-shaped component (no core), from the pixels within 10" of it."""
        m = (np.abs(self.E - e0) < 10) & (np.abs(self.N - n0) < 10)
        b = self.beam(e0, n0, m)
        return (b @ self.img[m]) / (b @ b)

    def annulus(self, r1, r2, k=400, avoid=20.0):
        out = []
        while len(out) < k:
            rr, pa = np.sqrt(rng.uniform(r1**2, r2**2)), rng.uniform(0, 360)
            if min(dpa(pa, PAS), dpa(pa, PAN)) >= avoid:
                out.append(self.single(*en(rr, pa)))
        return np.array(out)


def core_position(ts):
    t = open(glob.glob(f"S-band-fits/img_{ts}*")[0]).read()
    x, y = re.findall(r"Center X\s*=\s*(\S+).*?Center Y\s*=\s*(\S+)", t, re.S)[0]
    return SkyCoord(x, y, unit=("hourangle", "deg"))


lines = ["Lobe detection test and Epoch 12 checks (src/epoch12_checks.py); amplitudes in uJy/beam.", "",
         "1. Fixed-position amplitude at the linear-model separation; level and sigma_bg from beam fits at 400 random",
         "   positions 10-30\" from the core, > 20 deg from the jet axis. Detected if (amp - level)/sigma_bg >= 3.",
         f"{'epoch':>5} {'lobe':>4} {'r_pred':>6} {'amp':>7} {'level':>6} {'sig_bg':>6} {'signif':>6}"]
ep12 = None
for ep, ts in EPOCHS.items():
    mjd = int(ts) / 86400.0 + 40587.0
    im = Image(ts, core_position(ts))
    near = im.r < 12
    dt = mjd - TEJ
    pred = {"SE": (MUA * dt, PAS), "NW": (MUR * dt, PAN)}
    a = im.lsq([(0, 0), en(*pred["SE"]), en(*pred["NW"])], near)
    bg = im.annulus(10, 30)
    for j, lobe in ((1, "SE"), (2, "NW")):
        sig = (a[j] - bg.mean()) / bg.std(ddof=1)
        lines.append(f"{ep:5d} {lobe:>4} {pred[lobe][0]:6.2f} {a[j]:7.1f} {bg.mean():+6.1f} {bg.std(ddof=1):6.1f} "
                     f"{sig:6.1f}{'' if sig >= 3 else '  not detected'}")
    if ep == 12:
        ep12 = (im, pred, a, near)

# Epoch 12 with other background estimates
im, pred, a, near = ep12
lines += ["", "Epoch 12 significance with other level/noise estimates:"]
far = im.annulus(30, 90)
ring_pix = (im.r > 15) & (im.r < 40)
pix_rms, pix_med = mad(im.img[ring_pix]), np.median(im.img[ring_pix])
for j, lobe in ((1, "SE"), (2, "NW")):
    r_, _ = pred[lobe]
    ring = np.array([im.lsq([(0, 0), en(*pred["SE"]), en(*pred["NW"]), en(r_, s)], near)[-1]
                     for s in np.arange(36) * 10.0 if min(dpa(s, PAS), dpa(s, PAN)) > 30])
    lines.append(f"  {lobe}: ring at {r_:.2f}\" {(a[j] - ring.mean()) / ring.std(ddof=1):.1f}; 30-90\" (robust) "
                 f"{(a[j] - np.median(far)) / mad(far):.1f}; pixel rms 15-40\" minus median "
                 f"{(a[j] - pix_med) / pix_rms:.1f}, without {a[j] / pix_rms:.1f}")
lines.append(f"  (30-90\": median {np.median(far):+.1f}, robust sigma {mad(far):.1f}; pixels 15-40\": median {pix_med:+.1f}, "
             f"rms {pix_rms:.1f})")

# 3. mirror test on the free-fit features
core = im.lsq([(0, 0)], near)[0]
lines += ["", "3. Mirror test (Epoch 12 free-fit features; amplitude with a local constant):"]
for lobe, (r_, pa) in EP12_FREE.items():
    m1 = im.lsq([(0, 0), en(r_, pa)], near, const=True)[1]
    m2 = im.lsq([(0, 0), en(r_, pa + 180)], near, const=True)[1]
    ring = np.array([im.lsq([(0, 0), en(r_, s)], near, const=True)[1] for s in np.arange(0, 360, 10.0)])
    lines.append(f"  {lobe} at {r_:.2f}\", PA {pa:.1f} ({dpa(pa, PAS if lobe == 'SE' else PAN):.0f} deg off axis): {m1:.1f}; "
                 f"mirrored (PA {(pa + 180) % 360:.1f}) {m2:.1f}; ring median {np.median(ring):.1f}")

# 2. field sources
ts12 = EPOCHS[12]
lines += ["", f"2. Field sources, Epoch 12: residual at the lobe offsets as a fraction of the source peak "
              f"(lobe/core: NW {a[2] / core:.2f}, SE {a[1] / core:.2f})"]
for name, s in FIELD.items():
    fim = Image(ts12, SkyCoord(s), n=50)
    sel = fim.r < 3
    f = least_squares(lambda q: q[0] * fim.beam(q[1], q[2], sel) + q[3] - fim.img[sel],
                      [fim.img[50, 50], 0, 0, 0]).x
    A, e0, n0 = f[:3]
    res = fim.img - A * fim.beam(e0, n0, np.ones_like(fim.r, dtype=bool)).reshape(fim.img.shape)
    close = np.hypot(fim.E - e0, fim.N - n0) < 12
    out = []
    for lobe, (r_, pa) in EP12_FREE.items():
        de, dn = en(r_, pa)
        out.append(f"{lobe} {fim.lsq([(e0 + de, n0 + dn)], close, const=True, img=res)[0] / A:+.3f}")
    lines.append(f"  {name} (peak {A / 1e3:.2f} mJy/beam): " + ", ".join(out))

open(args.out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
