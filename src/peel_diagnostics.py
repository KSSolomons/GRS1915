"""
Peel-region and upper-limit diagnostics for the unresolved Epochs 1-4 of GRS 1915+105.

Given an S-band MFS image (ideally the post-peel image), this script:
  1. Locates the core with a fixed-beam Gaussian fit around the catalogue position.
  2. Measures the background rms (robust, 40-120 arcsec annulus, as in alpha/get_alpha.py).
  3. Measures the local rms and the peak value in sectors along the jet axis
     (PA_jet and PA_jet + 180 deg) in 0.5 arcsec radial steps outside the peel circle.
     Peaks above n_sigma x local rms are listed as candidate knots.
  4. If expected knot fluxes are given, finds the limit radius r_lim beyond which a knot of
     that flux would be detected at n_sigma, and its positional accuracy
     sigma_pos = theta_beam / (2 SNR).
  5. With --model (the WSClean *-MFS-model.fits), suggests the smallest peel circle that holds
     a given fraction of the core's clean-component flux, and lists model components that sit
     off-centre along the jet axis (possible knots that a larger circle would peel away).
  6. With --subtract-core, subtracts the fitted beam-shaped core in the image plane first
     (a rough look before a real peel).

Outputs (prefix from --out-prefix): a text report, a DS9 circle region for the peel (with --model
or --peel-radius), a diagnostic PNG, and optionally a 5-sigma FITS mask (--write-mask).
"""

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser(description="Peel-region and upper-limit diagnostics around the GRS 1915+105 core")
parser.add_argument("image", help="MFS image (post-peel preferred)")
parser.add_argument("--model", default=None, help="WSClean MFS model image used to size the peel circle")
parser.add_argument("--peel-radius", type=float, default=None, help="Radius (arcsec) of the peel circle already used; sectors start outside it")
parser.add_argument("--jet-pa", type=float, default=138.0, help="Jet position angle of the approaching (SE) side, deg E of N")
parser.add_argument("--sector-half-width", type=float, default=20.0, help="Half opening angle of the jet-axis sectors (deg)")
parser.add_argument("--r-max", type=float, default=6.0, help="Outer radius of the sector scan (arcsec)")
parser.add_argument("--n-sigma", type=float, default=5.0, help="Detection threshold in units of the local rms")
parser.add_argument("--knot-flux", type=float, nargs=2, default=None, metavar=("NW_MJY", "SE_MJY"), help="Expected knot flux densities (mJy)")
parser.add_argument("--model-fraction", type=float, default=0.99, help="Fraction of core model flux the suggested peel circle must hold")
parser.add_argument("--subtract-core", action="store_true", help="Subtract the fitted beam-shaped core in the image plane before the scan")
parser.add_argument(
    "--centre",
    nargs=2,
    default=None,
    metavar=("RA", "DEC"),
    help="Fixed scan centre (sexagesimal or deg) instead of fitting the core; needed for post-peel images where the core is gone",
)
parser.add_argument("--write-mask", action="store_true", help="Write a full-size FITS mask of pixels above n_sigma x background rms")
parser.add_argument(
    "--write-peel-mask",
    action="store_true",
    help="Write a full-size FITS peel mask (1 inside the peel circle around the fitted core, 0 elsewhere) on the image grid",
)
parser.add_argument("--out-prefix", default=None, help="Output prefix (default: data/peel_diag_<image id>)")
args = parser.parse_args()

CAT_RA = (19 + 15 / 60.0 + 11.558 / 3600.0) * 15.0
CAT_DEC = 10 + 56 / 60.0 + 44.905 / 3600.0
CUTOUT_HALF = 150.0  # arcsec; covers the 40-120 arcsec background annulus


def load_cutout(path):
    """Return (2D cutout, celestial WCS of the cutout, header, (y0, x0) cutout origin in the full image)."""
    with fits.open(path, memmap=True) as hdul:
        header = hdul[0].header
        data = hdul[0].data
        while data.ndim > 2:
            data = data[0]
        wcs = WCS(header).celestial
        xc, yc = wcs.world_to_pixel_values(CAT_RA, CAT_DEC)
        pix = abs(header["CDELT2"]) * 3600.0
        half = int(np.ceil(CUTOUT_HALF / pix))
        x0, y0 = int(round(float(xc))) - half, int(round(float(yc))) - half
        cut = np.array(data[y0 : y0 + 2 * half + 1, x0 : x0 + 2 * half + 1], dtype=float)
    return cut, wcs[y0 : y0 + 2 * half + 1, x0 : x0 + 2 * half + 1], header, (y0, x0)


def robust_rms(values):
    values = values[np.isfinite(values)]
    if values.size < 5:
        return np.nan
    return 1.4826 * np.median(np.abs(values - np.median(values)))


def beam_gaussian(coords, amp, x0, y0, sx, sy, theta):
    x, y = coords
    a = np.cos(theta) ** 2 / (2 * sx**2) + np.sin(theta) ** 2 / (2 * sy**2)
    b = np.sin(2 * theta) / (4 * sy**2) - np.sin(2 * theta) / (4 * sx**2)
    c = np.sin(theta) ** 2 / (2 * sx**2) + np.cos(theta) ** 2 / (2 * sy**2)
    return amp * np.exp(-(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2))


image, wcs, header, origin = load_cutout(args.image)
pix = abs(header["CDELT2"]) * 3600.0  # arcsec per pixel
bmaj, bmin, bpa = header["BMAJ"] * 3600.0, header["BMIN"] * 3600.0, header["BPA"]
theta_beam = np.sqrt(bmaj * bmin)
fwhm2sig = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

# Beam orientation in pixel space: PA is E of N and RA increases to -x
sx, sy = bmin * fwhm2sig / pix, bmaj * fwhm2sig / pix
theta_pix = -np.radians(bpa)

ny, nx = image.shape
yy, xx = np.mgrid[0:ny, 0:nx]
xcat, ycat = (float(v) for v in wcs.world_to_pixel_values(CAT_RA, CAT_DEC))

# 1. Fixed-beam core fit in a 4-beam box
box = (np.abs(xx - xcat) < 2 * bmaj / pix) & (np.abs(yy - ycat) < 2 * bmaj / pix)
p0 = [np.nanmax(image[box]), xcat, ycat]
fit_fn = lambda c, amp, x0, y0: beam_gaussian(c, amp, x0, y0, sx, sy, theta_pix)
if args.centre:
    from astropy.coordinates import SkyCoord

    unit = ("hourangle", "deg") if ":" in args.centre[0] else ("deg", "deg")
    c = SkyCoord(args.centre[0], args.centre[1], unit=unit)
    core_x, core_y = (float(v) for v in wcs.world_to_pixel_values(c.ra.deg, c.dec.deg))
    core_amp = image[int(round(core_y)), int(round(core_x))]
    popt = [core_amp, core_x, core_y]
else:
    popt, _ = curve_fit(fit_fn, (xx[box], yy[box]), image[box], p0=p0)
    core_amp, core_x, core_y = popt
core_ra, core_dec = (float(v) for v in wcs.pixel_to_world_values(core_x, core_y))
core_ra %= 360.0  # header CRVAL1 is stored as a negative angle
if args.subtract_core:
    image = image - fit_fn((xx, yy), *popt)

# Sky offsets (arcsec, east and north) and PA of every pixel relative to the fitted core
east = -(xx - core_x) * pix
north = (yy - core_y) * pix
radius = np.hypot(east, north)
pa = np.degrees(np.arctan2(east, north)) % 360.0

# 2. Background rms
bkg_rms = robust_rms(image[(radius > 40.0) & (radius < 120.0)])

# 3-4. Sector scan along the jet axis
r_start = max(args.peel_radius or 0.0, 0.5)
edges = np.arange(r_start, args.r_max + 0.5, 0.5)
sides = {"SE": args.jet_pa % 360.0, "NW": (args.jet_pa + 180.0) % 360.0}
expected = {"NW": args.knot_flux[0] / 1e3, "SE": args.knot_flux[1] / 1e3} if args.knot_flux else {}

scan = {s: [] for s in sides}
for lo, hi in zip(edges[:-1], edges[1:]):
    ring = (radius >= lo) & (radius < hi)
    ring_rms = robust_rms(image[ring])
    for side, pa_side in sides.items():
        dpa = np.abs((pa - pa_side + 180.0) % 360.0 - 180.0)
        sel = ring & (dpa <= args.sector_half_width)
        vals = image[sel]
        if vals.size == 0:  # sector too thin to hold a pixel at this radius (0.5" pixels)
            continue
        imax = np.nanargmax(vals)
        scan[side].append(
            {
                "r": 0.5 * (lo + hi),
                "ring_rms": ring_rms,
                "sector_rms": robust_rms(vals),
                "max": vals[imax],
                "min": np.nanmin(vals),
                "max_east": east[sel][imax],
                "max_north": north[sel][imax],
            }
        )

lines = [f"Image: {os.path.basename(args.image)}" + ("  [core subtracted in image plane]" if args.subtract_core else "")]
lines.append(f"Beam: {bmaj:.2f}\" x {bmin:.2f}\" at {bpa:.1f} deg; theta_beam = {theta_beam:.2f}\"; pixel = {pix:.2f}\"")
lines.append(f"Fitted core: RA {core_ra:.6f} Dec {core_dec:.6f} deg, peak {core_amp * 1e3:.2f} mJy/beam")
lines.append(f"Background rms (40-120\" annulus, MAD): {bkg_rms * 1e6:.1f} uJy/beam")
lines.append(f"Jet sectors: PA {sides['SE']:.0f} (SE) and {sides['NW']:.0f} (NW), +/-{args.sector_half_width:.0f} deg; threshold {args.n_sigma:.0f} x local rms")
lines.append("")

limits = {}
for side in sides:
    lines.append(f"--- {side} side ---")
    lines.append(f"{'r(as)':>6} {'ring rms':>9} {'sect rms':>9} {'max':>9} {'min':>9}  (uJy/beam)  flag")
    r_lim = None
    for row in scan[side]:
        local = row["ring_rms"]
        flag = ""
        if row["max"] > args.n_sigma * local:
            flag = f"CANDIDATE ({row['max'] / local:.1f} sigma at E{row['max_east']:+.2f}\" N{row['max_north']:+.2f}\")"
        lines.append(
            f"{row['r']:6.2f} {local * 1e6:9.1f} {row['sector_rms'] * 1e6:9.1f} {row['max'] * 1e6:9.1f} {row['min'] * 1e6:9.1f}  {flag}"
        )
    if side in expected:
        detectable = [expected[side] > args.n_sigma * row["ring_rms"] for row in scan[side]]
        # r_lim: inner edge of the outermost run of radii at which the expected knot is detectable
        for i in range(len(detectable)):
            if all(detectable[i:]):
                r_lim = scan[side][i]["r"] - 0.25
                snr = expected[side] / scan[side][i]["ring_rms"]
                limits[side] = (r_lim, snr, theta_beam / (2.0 * snr))
                break
        if side in limits:
            r_lim, snr, spos = limits[side]
            lines.append(
                f"Expected knot {expected[side] * 1e3:.2f} mJy: detectable at >= {args.n_sigma:.0f} sigma beyond r_lim = {r_lim:.2f}\" "
                f"(SNR {snr:.1f} there, sigma_pos = {spos:.3f}\")"
            )
        else:
            lines.append(f"Expected knot {expected[side] * 1e3:.2f} mJy is not detectable at {args.n_sigma:.0f} sigma out to {args.r_max:.1f}\"")
    lines.append("")

# 5. Peel circle from the clean-component model
peel_r = args.peel_radius
if args.model:
    model, _, _, _ = load_cutout(args.model)
    near = (radius < 5.0) & (model != 0)
    order = np.argsort(radius[near])
    cum = np.cumsum(np.abs(model[near][order]))
    peel_r = radius[near][order][np.searchsorted(cum, args.model_fraction * cum[-1])] + 0.5 * pix
    lines.append(f"Model: {near.sum()} clean-component pixels within 5\"; total {model[near].sum() * 1e3:.2f} mJy")
    lines.append(f"Suggested peel radius (holds {100 * args.model_fraction:.0f}% of the model flux within 5\"): {peel_r:.2f}\"")
    for side, pa_side in sides.items():
        dpa = np.abs((pa - pa_side + 180.0) % 360.0 - 180.0)
        off = near & (radius > 0.75) & (dpa <= args.sector_half_width + 5.0)
        if off.any():
            lines.append(f"  {side}: {off.sum()} model pixels off-centre along the jet axis, {model[off].sum() * 1e3:.2f} mJy, "
                         f"radii {radius[off].min():.2f}-{radius[off].max():.2f}\" -> possible knots; keep them outside the peel circle")
    lines.append("")

prefix = args.out_prefix or os.path.join("data", "peel_diag_" + os.path.basename(args.image).split("_")[1].split("-")[0])
os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
if peel_r is not None:
    from astropy.coordinates import SkyCoord

    centre = SkyCoord(core_ra, core_dec, unit="deg")
    ra_hms = centre.ra.to_string(unit="hourangle", sep=":", precision=4, pad=True)
    dec_dms = centre.dec.to_string(sep=":", precision=3, alwayssign=True, pad=True)
    with open(prefix + "_peel.reg", "w") as fp:
        fp.write("# Region file format: DS9 version 4.1\nglobal color=green\nfk5\n")
        fp.write(f"circle({ra_hms},{dec_dms},{peel_r:.3f}\")\n")
    lines.append(f"Wrote DS9 peel circle (r = {peel_r:.2f}\") to {prefix}_peel.reg")

if args.write_peel_mask:
    if peel_r is None:
        parser.error("--write-peel-mask needs --peel-radius or --model")
    with fits.open(args.image, memmap=True) as hdul:
        shape = hdul[0].data.shape
        peel_mask = np.zeros(shape, dtype=np.float32)
        ny_full, nx_full = shape[-2:]
        y0, x0 = origin
        cx, cy = core_x + x0, core_y + y0  # core pixel position in the full image
        half = int(np.ceil(peel_r / pix)) + 2
        ys, xs = np.mgrid[int(cy) - half : int(cy) + half + 1, int(cx) - half : int(cx) + half + 1]
        inside = np.hypot((xs - cx) * pix, (ys - cy) * pix) <= peel_r
        peel_mask[..., ys[inside], xs[inside]] = 1.0
        mask_header = hdul[0].header.copy()
        mask_header["BUNIT"] = ""
        mask_header["HISTORY"] = f"Peel mask: circle r={peel_r:.3f} arcsec at RA {core_ra:.6f} Dec {core_dec:.6f} (peel_diagnostics.py)"
        fits.writeto(prefix + "_peel_mask.fits", peel_mask, mask_header, overwrite=True)
    lines.append(f"Wrote peel mask ({int(inside.sum())} pixels, r = {peel_r:.2f}\") to {prefix}_peel_mask.fits")

# 6. Optional 5-sigma mask (full image size, for WSClean -fits-mask)
if args.write_mask:
    with fits.open(args.image, memmap=True) as hdul:
        full = hdul[0].data
        mask = (full > args.n_sigma * bkg_rms).astype(np.float32)
        fits.writeto(prefix + "_mask.fits", mask, hdul[0].header, overwrite=True)
    lines.append(f"Wrote {args.n_sigma:.0f}-sigma mask to {prefix}_mask.fits")

# Diagnostic figure: residual cutout with sectors, and local threshold versus radius
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
zoom = radius < args.r_max + 2.0
ext = (args.r_max + 2.0) / pix
sub = image[int(core_y - ext) : int(core_y + ext), int(core_x - ext) : int(core_x + ext)]
vlim = 10 * bkg_rms
ax1.imshow(sub, origin="lower", cmap="inferno", vmin=-vlim, vmax=vlim,
           extent=[(args.r_max + 2.0), -(args.r_max + 2.0), -(args.r_max + 2.0), (args.r_max + 2.0)])
for side, pa_side in sides.items():
    for d in (-args.sector_half_width, args.sector_half_width):
        ang = np.radians(pa_side + d)
        ax1.plot([0, args.r_max * np.sin(ang)], [0, args.r_max * np.cos(ang)], color="cyan", lw=0.8)
if peel_r:
    ax1.add_patch(plt.Circle((0, 0), peel_r, fill=False, color="lime"))
ax1.set_xlabel("East offset (arcsec)")
ax1.set_ylabel("North offset (arcsec)")
ax1.set_title("Image around core (+/-10 x background rms)")
for side, colour in (("SE", "tab:blue"), ("NW", "tab:red")):
    r = [row["r"] for row in scan[side]]
    ax2.plot(r, [args.n_sigma * row["ring_rms"] * 1e3 for row in scan[side]], color=colour, label=f"{side}: {args.n_sigma:.0f} x local rms")
    ax2.plot(r, [row["max"] * 1e3 for row in scan[side]], color=colour, ls=":", label=f"{side}: sector max")
    if side in expected:
        ax2.axhline(expected[side] * 1e3, color=colour, ls="--", lw=0.8, label=f"{side}: expected knot")
ax2.set_yscale("log")
ax2.set_xlabel("Distance from core (arcsec)")
ax2.set_ylabel("mJy/beam")
ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig(prefix + "_diag.png", dpi=150)
lines.append(f"Saved figure to {prefix}_diag.png")

report = "\n".join(lines)
print(report)
with open(prefix + "_report.txt", "w") as fp:
    fp.write(report + "\n")
