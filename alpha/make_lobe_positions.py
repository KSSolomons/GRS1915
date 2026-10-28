"""
Write the per-epoch component positions for get_alpha.py.

Run from the repository root with ordinary Python (no CASA needed):

    python alpha/make_lobe_positions.py

For every 2025 epoch it lists, labelled by component:

  C     the core, from S-band-fits/*2D_Fitting*.txt (the fitted component
        nearest the catalogue position);
  N, S  each non-core component of the same fits, north (receding) or south
        (approaching) of the catalogue position, for the epochs with resolved
        jets (Epoch 12 excluded, as in the kinematics pipelines);
  REF1  the reference source, from a Gaussian fit to the MFS image in FITS/pcalmask/.

get_alpha.py holds every component at these positions in the channel images.
The positions must come from each epoch's own image: the images are offset from
the catalogue frame by 0.3-0.9 arcsec, and Epoch 4 by 1.6 arcsec, and a
beam-shaped Gaussian held at the wrong position loses flux, more so at high
frequency, which biases the flux low and alpha steep (20 per cent and roughly
-0.4 for the core in Epoch 4). Copy alpha/lobe_positions.txt next to
get_alpha.py on the machine with the channel images.
"""

import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from epoch_limits import EJECTION_CAMPAIGN_START

EPOCH12_TS = 1764664923  # Epoch 12 jet features are unreliable (likely artefacts)
CORE_RA_DEG = (288.7979, 288.7982)  # same core selection as the kinematics pipelines
CAT_DEC_DEG = 10 + 56 / 60.0 + 44.905 / 3600.0
CAT_RA_DEG = (19 + 15 / 60.0 + 11.558 / 3600.0) * 15.0
CORE_MAX_SEP_ARCSEC = 3.0
REFERENCE_SOURCES = [("REF1", "19h14m44.703s", "+10d57m56.151s")]  # as in get_alpha.py
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lobe_positions.txt")


def sep_arcsec(ra_deg, dec_deg, ra0_deg, dec0_deg):
    dra = (ra_deg - ra0_deg) * np.cos(np.radians(dec0_deg))
    return 3600.0 * np.hypot(dra, dec_deg - dec0_deg)


def hms(ra_deg):
    h = ra_deg / 15.0
    return "{}h{:02d}m{:.4f}s".format(int(h), int(h * 60) % 60, (h * 3600) % 60)


def dms(dec_deg):
    sign = "-" if dec_deg < 0 else "+"
    d = abs(dec_deg)
    return "{}{}d{:02d}m{:.4f}s".format(sign, int(d), int(d * 60) % 60, (d * 3600) % 60)


def reference_positions(ts):
    """Fit each reference source in the epoch's MFS image (FITS/pcalmask/); [(label, ra, dec, snr)]."""
    files = glob.glob(os.path.join("FITS", "pcalmask", "img_{}*pcalmask-MFS-image.pbcor.fits".format(ts)))
    if not files:
        return []
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.wcs import WCS
    from scipy.optimize import curve_fit

    def g2(xy, amp, x0, y0, sx, sy, th, z):
        x, y = xy
        a = np.cos(th) ** 2 / (2 * sx**2) + np.sin(th) ** 2 / (2 * sy**2)
        b = np.sin(2 * th) * (1 / (4 * sy**2) - 1 / (4 * sx**2))
        c = np.sin(th) ** 2 / (2 * sx**2) + np.cos(th) ** 2 / (2 * sy**2)
        return (amp * np.exp(-(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2)) + z).ravel()

    header = fits.getheader(files[0])
    wcs = WCS(header).celestial
    data = fits.getdata(files[0], memmap=True)
    data = data.reshape(data.shape[-2:])
    out = []
    for label, ra, dec in REFERENCE_SOURCES:
        px, py = (int(round(float(v))) for v in wcs.world_to_pixel(SkyCoord(ra, dec)))
        cut = np.nan_to_num(np.array(data[py - 12:py + 13, px - 12:px + 13]))
        yy, xx = np.mgrid[-12:13, -12:13]
        j = np.unravel_index(np.argmax(cut), cut.shape)
        try:
            p, cov = curve_fit(g2, (xx, yy), cut.ravel(), p0=[cut[j], xx[j], yy[j], 3, 3, 0, 0], maxfev=20000)
        except RuntimeError:
            print("  {}: {} fit failed, not listed".format(ts, label))
            continue
        pos = wcs.pixel_to_world(px + p[1], py + p[2])
        out.append((label, pos.ra.deg, pos.dec.deg, p[0] / np.sqrt(cov[0, 0])))
    return out


rows = []
for f in sorted(glob.glob(os.path.join("S-band-fits", "*2D_Fitting*.txt"))):
    m = re.search(r"img_(\d+)", os.path.basename(f))
    if not m:
        continue
    ts = int(m.group(1))
    if ts / 86400.0 + 40587.0 < EJECTION_CAMPAIGN_START:
        continue
    comps, curr = [], None
    with open(f) as fp:
        for line in fp:
            if line.startswith("Component #"):
                curr = {}
                comps.append(curr)
            elif curr is None:
                continue
            elif "Center X" in line:
                h, mi, s = re.search(r"=\s*(\d+):(\d+):(\d+\.\d+)", line).groups()
                curr["ra"] = "{}h{}m{:.4f}s".format(h, mi, float(s))
                curr["ra_deg"] = (float(h) + float(mi) / 60.0 + float(s) / 3600.0) * 15.0
            elif "Center Y" in line:
                d, mi, s = re.search(r"=\s*([+-]?\d+):(\d+):(\d+\.\d+)", line).groups()
                sign = "-" if d.startswith("-") else "+"
                curr["dec"] = "{}{}d{}m{:.4f}s".format(sign, abs(int(d)), mi, float(s))
                curr["dec_deg"] = (-1 if sign == "-" else 1) * (abs(float(d)) + float(mi) / 60.0 + float(s) / 3600.0)
            elif "Amplitude" in line:
                a, e = re.search(r"=\s*([\d.eE+-]+)\s*±\s*([\d.eE+-]+)", line).groups()
                curr["snr"] = float(a) / float(e)
    comps = [c for c in comps if "ra_deg" in c and "dec_deg" in c]
    seps = [sep_arcsec(c["ra_deg"], c["dec_deg"], CAT_RA_DEG, CAT_DEC_DEG) for c in comps]
    if seps and min(seps) < CORE_MAX_SEP_ARCSEC:
        core = comps[int(np.argmin(seps))]
        rows.append((str(ts), "C", core["ra"], core["dec"], core.get("snr", float("nan"))))
    else:
        print("  {}: no core component within {}\" of the catalogue position".format(ts, CORE_MAX_SEP_ARCSEC))
    if ts != EPOCH12_TS:
        for c in comps:
            if CORE_RA_DEG[0] < c["ra_deg"] < CORE_RA_DEG[1]:
                continue
            rows.append((str(ts), "N" if c["dec_deg"] > CAT_DEC_DEG else "S", c["ra"], c["dec"], c.get("snr", float("nan"))))
    for label, ra_deg, dec_deg, snr in reference_positions(ts):
        rows.append((str(ts), label, hms(ra_deg), dms(dec_deg), snr))

with open(OUTPUT, "w") as fp:
    fp.write("# Per-epoch component positions for get_alpha.py, written by make_lobe_positions.py\n")
    fp.write("# C = core and N = north (receding), S = south (approaching) lobes, from the S-band MFS fits (S-band-fits/);\n")
    fp.write("# REF1 = reference source, from a Gaussian fit to the MFS image (FITS/pcalmask/). SNR is amplitude / error.\n")
    fp.write("# {:<12} {:<5} {:<20} {:<20} {:>6}\n".format("Block", "Label", "RA (J2000)", "Dec (J2000)", "SNR"))
    for r in rows:
        fp.write("{:<14} {:<5} {:<20} {:<20} {:>6.1f}\n".format(*r))
print("Wrote {} positions for {} epochs to {}".format(len(rows), len(set(r[0] for r in rows)), OUTPUT))
