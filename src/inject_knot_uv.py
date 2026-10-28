"""
uv-plane injection test of the Epoch 3-4 peel: does the peel (CubiCal dE on the 3" region, core-only mask) remove the flux
of a knot that sits inside the peel region?

A point source of the flux the exponential fade predicts (S_exp) is added to the visibilities of a COPY of the measurement
set, at the knot position the kinematic fits predict (SE side, data/peel_regions/limits_posterior_check.txt). The copy is
then peeled and imaged exactly like the original, and src/measure_uv_injection.py compares the two post-peel images.

Runs on ilifu inside CASA 6 (modular casatools/casatasks):

    casa --nogui -c /path/to/inject_knot_uv.py --epoch 4 --vis-in 1742780174_sdp_l0_1024ch_GRS1915+105.ms \
        --vis-out 1742780174_sdp_l0_1024ch_GRS1915+105_inj.ms

The script path must exist from the working directory: if it does not, "casa -c" runs the argument as a Python
statement and fails with NameError. Plain python also works where casatools and casatasks are installed. Steps:

  1. This script copies --vis-in to --vis-out (never touches the original), predicts the point source into MODEL_DATA of
     the copy with ft, adds MODEL_DATA to --column (default CORRECTED_DATA), and restores the copy's original MODEL_DATA.
  2. Rerun the whole oxkat 3GC peel job on the copy in a separate folder, with the same core-only mask1, the same 3"
     peel region and the same parset (time-chunk, dE time-int) as the original Epoch 3/4 peel.
  3. Image it with the same WSClean settings, so the output is <ms>_postpeel-MFS-image.pbcor.fits.
  4. Copy that image to FITS/peel/ep<N>_inj/ and run src/measure_uv_injection.py --epoch N.

--vis-in must be the measurement set as it was BEFORE the original peel. The oxkat 3GC peel job images CORRECTED_DATA,
then copies CORRECTED_DATA to DATA, and only then runs CubiCal on DATA (the parset's "[data] column"). A knot added to
DATA is therefore overwritten before the peel, and a post-peel MS has the core already removed from CORRECTED_DATA.
"""

import argparse
import os
import shutil
import sys

import numpy as np
from casatasks import ft
from casatools import componentlist, table

# Per epoch: core (peel-region centre, data/peel_regions/epoch*_peel_3as.reg), predicted SE knot separation (median of
# the four models in data/peel_regions/limits_posterior_check.txt, rounded) and S_exp (src/epoch_limits.py, s_jet_jy)
EPOCHS = {
    3: dict(core=("19:15:11.5449", "+10:56:44.611"), r=2.6, flux_mjy=4.13),
    4: dict(core=("19:15:11.5085", "+10:56:43.507"), r=2.8, flux_mjy=2.83),
}
JET_PA_SE = 137.6  # deg East of North (src/epoch_limits.py JET_PA_DEG)
LOBE_ALPHA = -1.06  # in-band lobe spectral index, Epochs 6-8 (alpha/inband_alpha_final.txt)
REF_FREQ = "3.06GHz"  # centre of the 2.68-3.44 GHz band, where the MFS fluxes of the fade fit apply
CHUNK = 200000  # rows per read/write

p = argparse.ArgumentParser()
p.add_argument("--epoch", type=int, choices=sorted(EPOCHS), required=True)
p.add_argument("--vis-in", required=True, help="Measurement set to copy (left untouched)")
p.add_argument("--vis-out", required=True, help="Copy that receives the knot; must not exist yet (see --reuse-copy)")
p.add_argument("--reuse-copy", action="store_true",
               help="--vis-out is already an untouched copy of --vis-in (e.g. after a failed run): skip the copy")
p.add_argument("--column", default="CORRECTED_DATA",
               help="Column the 3GC peel job starts from (CORRECTED_DATA; it is copied to DATA before CubiCal)")
p.add_argument("--r", type=float, default=None, help="Knot separation from the core (arcsec; default per epoch)")
p.add_argument("--pa", type=float, default=JET_PA_SE, help="Knot position angle (deg East of North)")
p.add_argument("--flux-mjy", type=float, default=None, help="Knot flux density at REF_FREQ (mJy; default S_exp)")
p.add_argument("--alpha", type=float, default=LOBE_ALPHA)
# Under "casa -c" sys.argv can also hold CASA's own options, so parse only what follows this script's name
argv = sys.argv[1:]
mine = [i for i, a in enumerate(sys.argv) if os.path.basename(a) == "inject_knot_uv.py"]
if mine:
    argv = sys.argv[mine[0] + 1 :]
args = p.parse_args(argv)

cfg = EPOCHS[args.epoch]
r = cfg["r"] if args.r is None else args.r
flux = (cfg["flux_mjy"] if args.flux_mjy is None else args.flux_mjy) * 1e-3

column = args.column
print(f"Injecting into column {column}")


def sexa_to_deg(ra, dec):
    h, m, s = (float(v) for v in ra.split(":"))
    sign = -1.0 if dec.strip().startswith("-") else 1.0
    d, dm, ds = (abs(float(v)) for v in dec.split(":"))
    return 15.0 * (h + m / 60 + s / 3600), sign * (d + dm / 60 + ds / 3600)


def deg_to_sexa(ra, dec):
    h = ra / 15.0
    hh, mm = int(h), int((h - int(h)) * 60)
    ss = (h - hh - mm / 60) * 3600
    sign = "-" if dec < 0 else "+"
    a = abs(dec)
    dd, dm = int(a), int((a - int(a)) * 60)
    ds = (a - dd - dm / 60) * 3600
    return f"{hh:02d}h{mm:02d}m{ss:07.4f}s", f"{sign}{dd:02d}d{dm:02d}m{ds:06.3f}s"


ra0, dec0 = sexa_to_deg(*cfg["core"])
de, dn = r * np.sin(np.radians(args.pa)), r * np.cos(np.radians(args.pa))
ra_k = ra0 + de / 3600.0 / np.cos(np.radians(dec0))
dec_k = dec0 + dn / 3600.0
ra_s, dec_s = deg_to_sexa(ra_k, dec_k)
print(f"Epoch {args.epoch}: knot at r = {r:.2f}\", PA = {args.pa:.1f} deg -> J2000 {ra_s} {dec_s}, "
      f"{flux * 1e3:.2f} mJy at {REF_FREQ}, alpha = {args.alpha}")

if args.reuse_copy:
    if not os.path.exists(args.vis_out):
        raise SystemExit(f"--reuse-copy given but {args.vis_out} does not exist")
    print(f"Reusing the existing copy {args.vis_out}")
else:
    if os.path.exists(args.vis_out):
        raise SystemExit(f"{args.vis_out} exists; remove it, choose another name, or use --reuse-copy if it is untouched")
    print(f"Copying {args.vis_in} -> {args.vis_out}")
    shutil.copytree(args.vis_in, args.vis_out)

cl_path = args.vis_out.rstrip("/") + ".knot.cl"
if os.path.exists(cl_path):
    shutil.rmtree(cl_path)  # left over from an earlier run
cl = componentlist()
cl.addcomponent(dir=f"J2000 {ra_s} {dec_s}", flux=flux, fluxunit="Jy", freq=REF_FREQ, shape="point",
                spectrumtype="spectral index", index=args.alpha)
cl.rename(cl_path)
cl.close()

# Not called "tb", which would replace the CASA shell's own table tool when run with exec()
mst = table()
mst.open(args.vis_out, nomodify=False)
nrow = mst.nrows()
cols = mst.colnames()
if column not in cols:
    mst.close()
    raise SystemExit(f"{column} not in {args.vis_out} (columns: {cols})")
if "MODEL_BACKUP" in cols:
    mst.close()
    raise SystemExit(f"{args.vis_out} already has MODEL_BACKUP: an earlier injection stopped part-way; start from a fresh copy")
had_model = "MODEL_DATA" in cols
if had_model:
    # Keep the copy's MODEL_DATA so the peel sees the same model column as the original run
    desc = mst.getcoldesc("MODEL_DATA")
    desc["comment"] = "backup before knot injection"
    # Let CASA give the backup its own default storage manager instead of MODEL_DATA's data-manager group
    desc.pop("dataManagerType", None)
    desc.pop("dataManagerGroup", None)
    mst.addcols({"MODEL_BACKUP": desc})
    for i in range(0, nrow, CHUNK):
        n = min(CHUNK, nrow - i)
        mst.putcol("MODEL_BACKUP", mst.getcol("MODEL_DATA", i, n), i, n)
mst.close()

ft(vis=args.vis_out, complist=cl_path, usescratch=True, incremental=False)

mst.open(args.vis_out, nomodify=False)
peak = 0.0
for i in range(0, nrow, CHUNK):
    n = min(CHUNK, nrow - i)
    knot = mst.getcol("MODEL_DATA", i, n)
    peak = max(peak, np.abs(knot).max())
    mst.putcol(column, mst.getcol(column, i, n) + knot, i, n)
    if had_model:
        mst.putcol("MODEL_DATA", mst.getcol("MODEL_BACKUP", i, n), i, n)
mst.removecols("MODEL_BACKUP" if had_model else "MODEL_DATA")  # the knot must not stay in a model column
mst.close()

print(f"Added the knot to {column} of {args.vis_out} ({nrow} rows; max |V_knot| = {peak * 1e3:.3f} mJy, "
      f"~{flux * 1e3:.2f} mJy at {REF_FREQ}, more at lower frequencies). "
      f"MODEL_DATA {'restored' if had_model else 'removed (it did not exist before)'}.")
print("Next: peel and image this MS exactly like the original, then run src/measure_uv_injection.py.")
