"""
Is the knot added by src/inject_knot_uv.py actually in the injected measurement set?

Compares every data column of the injected copy with the original, row by row, on a sample of rows. A column that
carries the knot differs by |V_knot| ~ S_inj (point source: the same amplitude on every baseline, ~2.8 mJy at
3.06 GHz for Epoch 4, more at lower frequencies); a column without it differs by ~0 (or by the noise of a rerun
calibration if the column was rewritten). Also lists the columns of both sets, so a column the peel recreated shows up.

Runs on ilifu inside CASA 6, or plain python with casatools:

    casa --nogui -c /path/to/check_uv_injection.py --orig 1742780174_sdp_l0_1024ch_GRS1915+105.ms \
        --inj 1742780174_sdp_l0_1024ch_GRS1915+105_inj.ms
"""

import argparse
import os
import sys

import numpy as np
from casatools import table

p = argparse.ArgumentParser()
p.add_argument("--orig", required=True, help="Original measurement set")
p.add_argument("--inj", required=True, help="Injected copy")
p.add_argument("--nsample", type=int, default=20, help="Number of row blocks to sample across the set")
p.add_argument("--block", type=int, default=500, help="Rows per sampled block (1024 ch x 4 pol each; keep small)")
p.add_argument("--epoch", type=int, choices=(3, 4), default=4, help="Knot position and flux as in inject_knot_uv.py")
p.add_argument("--r", type=float, default=None, help="Knot separation (arcsec; default per epoch)")
p.add_argument("--pa", type=float, default=137.6, help="Knot position angle (deg East of North)")
p.add_argument("--alpha", type=float, default=-1.06)
# Under "casa -c" sys.argv can also hold CASA's own options, so parse only what follows this script's name
argv = sys.argv[1:]
mine = [i for i, a in enumerate(sys.argv) if os.path.basename(a) == "check_uv_injection.py"]
if mine:
    argv = sys.argv[mine[0] + 1 :]
args = p.parse_args(argv)

to = table()
ti = table()
to.open(args.orig)
ti.open(args.inj)
cols_o, cols_i = to.colnames(), ti.colnames()
print(f"original columns: {cols_o}")
print(f"injected columns: {cols_i}")
if to.nrows() != ti.nrows():
    print(f"Row counts differ ({to.nrows()} vs {ti.nrows()}): the copy was changed (split/averaged?); stopping")
    sys.exit(1)

nrow = to.nrows()
starts = np.linspace(0, max(nrow - args.block, 0), args.nsample).astype(int)
# Rows must be the same visibilities in both sets, or every difference below is meaningless
mismatch, duvw = 0, 0.0
for s in starts:
    n = min(args.block, nrow - s)
    for c in ("TIME", "ANTENNA1", "ANTENNA2"):
        mismatch += int(np.sum(ti.getcol(c, s, n) != to.getcol(c, s, n)))
    duvw = max(duvw, np.abs(ti.getcol("UVW", s, n) - to.getcol("UVW", s, n)).max())
print(f"row alignment: {mismatch} TIME/ANTENNA mismatches, max |dUVW| = {duvw:.3g} m")
data_cols = [c for c in ("DATA", "CORRECTED_DATA", "MODEL_DATA") if c in cols_o and c in cols_i]
for c in data_cols:
    diffs, amps = [], []
    for s in starts:
        n = min(args.block, nrow - s)
        fo = to.getcol("FLAG", s, n)
        vo = to.getcol(c, s, n)
        d = ti.getcol(c, s, n) - vo
        diffs.append(np.abs(d[~fo]))
        amps.append(np.abs(vo[~fo]))
    d = np.concatenate(diffs)
    print(f"{c:15s} |V_inj - V_orig| (mJy): median {np.median(d) * 1e3:8.4f}  "
          f"p95 {np.percentile(d, 95) * 1e3:8.4f}  max {d.max() * 1e3:8.4f}  "
          f"(median |V_orig| {np.median(np.concatenate(amps)) * 1e3:8.4f})", flush=True)
for c in sorted(set(cols_i) - set(cols_o)):
    print(f"{c}: only in the injected copy")

# Fit the knot's own visibility pattern to V_inj - V_orig, jointly with a point source at the core (the peel changes
# the core most, and its fringes are not orthogonal to the knot's on the sampled baselines): complex least squares
# d = a_k m_k + a_c m_c, with m the visibilities of 1 Jy (at 3.06 GHz) point sources. Changes elsewhere have other
# phase patterns and average out, so Re a_k ~ S_inj if the knot is in the column and ~0 if not. Both phase-sign
# conventions are tried; the right one gives ~S_inj. XX and YY only (Stokes I point source: XX = YY = I).
EPOCHS = {  # as in inject_knot_uv.py
    3: dict(core=("19:15:11.5449", "+10:56:44.611"), r=2.6, flux_mjy=4.13),
    4: dict(core=("19:15:11.5085", "+10:56:43.507"), r=2.8, flux_mjy=2.83),
}
cfg = EPOCHS[args.epoch]
r = cfg["r"] if args.r is None else args.r
h, m_, s_ = (float(v) for v in cfg["core"][0].split(":"))
d_, dm, ds = (float(v) for v in cfg["core"][1].split(":"))
ra0, dec0 = np.radians(15.0 * (h + m_ / 60 + s_ / 3600)), np.radians(d_ + dm / 60 + ds / 3600)
de, dn = r * np.sin(np.radians(args.pa)), r * np.cos(np.radians(args.pa))
ra_k = ra0 + np.radians(de / 3600.0) / np.cos(dec0)
dec_k = dec0 + np.radians(dn / 3600.0)

tf = table()
tf.open(os.path.join(args.inj, "FIELD"))
phase_dirs = tf.getcol("PHASE_DIR")  # (2, 1, nfield)
tf.close()
tf.open(os.path.join(args.inj, "SPECTRAL_WINDOW"))
chan_freq = tf.getcol("CHAN_FREQ")  # (nchan, nspw)
tf.close()
tf.open(os.path.join(args.inj, "DATA_DESCRIPTION"))
dd_spw = tf.getcol("SPECTRAL_WINDOW_ID")
tf.close()

C = 299792458.0


def lm(ra, dec, pra, pdec):
    l = np.cos(dec) * np.sin(ra - pra)
    m = np.sin(dec) * np.cos(pdec) - np.cos(dec) * np.sin(pdec) * np.cos(ra - pra)
    return l, m


def fit_knot(c, diff):
    """Joint knot + core fit to V_inj - V_orig (diff) or to V_inj alone (model-only columns, e.g. DIR1_DATA)."""
    A = {sg: np.zeros((2, 2), complex) for sg in (+1, -1)}
    b = {sg: np.zeros(2, complex) for sg in (+1, -1)}
    for s in starts:
        n = min(args.block, nrow - s)
        ok = ~ti.getcol("FLAG", s, n)[[0, -1]]  # (2, nchan, nrow)
        d = ti.getcol(c, s, n)[[0, -1]]
        if diff:
            d = d - to.getcol(c, s, n)[[0, -1]]
        uvw = ti.getcol("UVW", s, n)  # (3, nrow), metres
        fld = ti.getcol("FIELD_ID", s, n)
        ddid = ti.getcol("DATA_DESC_ID", s, n)
        if len(set(fld)) > 1 or len(set(ddid)) > 1:
            raise SystemExit("Sampled block spans several fields or spectral windows; this check assumes one")
        pra, pdec = phase_dirs[0, 0, fld[0]], phase_dirs[1, 0, fld[0]]
        nu = chan_freq[:, dd_spw[ddid[0]]]  # (nchan,)
        spec = (nu / 3.06e9)[:, None]
        for sg in (+1, -1):
            mods = []
            for (ra, dec, alpha) in ((ra_k, dec_k, args.alpha), (ra0, dec0, 0.0)):
                l, m = lm(ra, dec, pra, pdec)
                geom = (uvw[0] * l + uvw[1] * m)[None, :] * nu[:, None] / C  # (nchan, nrow), turns
                mods.append(np.where(ok, (spec ** alpha * np.exp(sg * 2j * np.pi * geom))[None], 0))
            for i in range(2):
                b[sg][i] += np.sum(np.conj(mods[i]) * np.where(ok, d, 0))
                for j in range(2):
                    A[sg][i, j] += np.sum(np.conj(mods[i]) * mods[j])
    out = []
    for sg in (+1, -1):
        ak, ac = np.linalg.solve(A[sg], b[sg])
        out.append(f"sign {sg:+d}: knot {ak.real * 1e3:7.3f} (imag {ak.imag * 1e3:6.3f}), core |a| {abs(ac) * 1e3:7.2f}")
    label = f"{c} (inj - orig)" if diff else f"{c} (inj alone)"
    print(f"{label:28s} fitted (mJy at 3.06 GHz; injected knot {cfg['flux_mjy']:.2f}): {'; '.join(out)}", flush=True)


for c in data_cols:
    fit_knot(c, diff=True)
# Model-only columns need no reference: a knot here means the peel subtracted it as part of the core direction
for c in ("MODEL_DATA", "DIR1_DATA"):
    if c in cols_i:
        fit_knot(c, diff=False)
to.close()
ti.close()
