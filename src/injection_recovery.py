"""
Injection-recovery in the post-peel Epoch 3 and 4 images: the radius r_lim(S) beyond which a knot of flux S would have been detected.

A beam-shaped knot of flux S (x S_expected = 1, 0.5, 0.25) is added to the image at radius r = 1.0-4.0" (0.25" steps) on each side of
the core (PA 138 and 318, jittered by +/-10 deg, random sub-pixel position), and recovered with the fit of fit_postpeel_knots.py
(core residual plus one free component started at the injected position, position within +/-1"). A detection needs
amplitude / sigma_emp >= 5 (sigma_emp = scatter of fixed-position amplitudes at that radius in the uninjected image) and a recovered
position within half the projected beam FWHM of the injected one. r_lim = smallest r with completeness >= 90% at that r and every larger r;
sigma_pos = b / (2 SNR), b = sqrt(BMAJ x BMIN), SNR = S / sigma_emp at r_lim.

This is an image-plane test: it does not show whether the peel itself (dE solve, CLEAN) absorbs knot flux.

  python src/injection_recovery.py [--trials 20] [--threshold 5]
"""

import argparse
import os
import sys
from multiprocessing import Pool

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from postpeel_tools import EPOCHS, JET_PA, PostPeelImage, en

p = argparse.ArgumentParser()
p.add_argument("--trials", type=int, default=20)
p.add_argument("--threshold", type=float, default=5.0)
p.add_argument("--completeness", type=float, default=0.9)
p.add_argument("--s-factors", type=float, nargs="+", default=[1.0, 0.5, 0.25])
p.add_argument("--seed", type=int, default=1915)
p.add_argument("--out", default="data/peel_regions/limits_ep34.txt")
p.add_argument("--fig", default="figures/peel/limit_completeness_ep34.png")
args = p.parse_args()

RADII = np.arange(1.0, 4.001, 0.25)
SIDES = {"SE": JET_PA, "NW": JET_PA + 180.0}
IMGS = {ep: PostPeelImage(ep) for ep in EPOCHS}
SIGMA = {ep: {round(r, 3): im.sigma_emp(r)[0] for r in RADII} for ep, im in IMGS.items()}


def trial(task):
    ep, side, sf, r, k = task
    im = IMGS[ep]
    rng = np.random.default_rng([args.seed, ep, ord(side[0]), int(sf * 100), int(r * 100), k])
    pa = SIDES[side] + rng.uniform(-10, 10)
    pos = en(r + rng.uniform(-0.5, 0.5) * im.pix, pa)  # sub-pixel jitter
    s = sf * im.s_expected
    inj = im.img.copy()
    e, n = im.E - pos[0], im.N - pos[1]
    u = e * np.sin(im.bpa) + n * np.cos(im.bpa)
    v = e * np.cos(im.bpa) - n * np.sin(im.bpa)
    inj += s * np.exp(-0.5 * ((u / (im.bmaj * im.f2s)) ** 2 + (v / (im.bmin * im.f2s)) ** 2))
    amp, rec = im.fit_free(pos, inj)
    ok = amp / SIGMA[ep][round(r, 3)] >= args.threshold and np.hypot(*(rec - pos)) <= 0.5 * im.fwhm_jet
    return ep, side, sf, r, ok


if __name__ == "__main__":
    tasks = [(ep, side, sf, r, k) for ep in EPOCHS for side in SIDES for sf in args.s_factors for r in RADII for k in range(args.trials)]
    with Pool() as pool:
        out = pool.map(trial, tasks, chunksize=20)
    comp = {}
    for ep, side, sf, r, ok in out:
        comp.setdefault((ep, side, sf), {}).setdefault(round(r, 3), []).append(ok)
    comp = {k: np.array([np.mean(v[round(r, 3)]) for r in RADII]) for k, v in comp.items()}

    lines = [
        "Injection-recovery limits in the post-peel images (image plane; the peel itself is not tested).",
        f"Detection: amp/sigma_emp >= {args.threshold:g} and recovered position within half the projected beam FWHM; {args.trials} trials per (S, r, side).",
        f"r_lim = smallest r with completeness >= {args.completeness:.0%} at that r and every larger r in {RADII[0]:.2f}-{RADII[-1]:.2f}\" ('>4.00' = none).",
        "sigma_pos = b / (2 SNR), b = sqrt(BMAJ BMIN), SNR = S / sigma_emp at r_lim.",
        f"{'epoch':>5} {'side':>4} {'S/S_exp':>7} {'S_mJy':>6} {'r_lim':>6} {'sigma_pos':>9} {'sigma_emp':>9}",
    ]
    for (ep, side, sf), c in sorted(comp.items()):
        im = IMGS[ep]
        good = c >= args.completeness
        idx = next((i for i in range(len(RADII)) if good[i:].all()), None)
        s = sf * im.s_expected
        if idx is None:
            lines.append(f"{ep:5d} {side:>4} {sf:7.2f} {s:6.2f} {'>4.00':>6} {'--':>9} {'--':>9}")
        else:
            sg = SIGMA[ep][round(RADII[idx], 3)]
            lines.append(f"{ep:5d} {side:>4} {sf:7.2f} {s:6.2f} {RADII[idx]:6.2f} {im.b / (2 * s / sg):9.3f} {sg:9.3f}")
    lines += ["", "sigma_emp (mJy/beam) by radius:", f"{'r':>5} " + " ".join(f"ep{ep}" .rjust(8) for ep in EPOCHS)]
    lines += [f"{r:5.2f} " + " ".join(f"{SIGMA[ep][round(r, 3)]:8.3f}" for ep in EPOCHS) for r in RADII]
    lines += ["", "Completeness by radius:"]
    for (ep, side, sf), c in sorted(comp.items()):
        lines.append(f"ep{ep} {side} S x{sf:.2f}: " + " ".join(f"{x:.2f}" for x in c))
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    plt.rcParams["font.family"] = "serif"
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, ep in zip(axs, EPOCHS):
        for j, sf in enumerate(args.s_factors):
            for side, ls in (("SE", "-"), ("NW", "--")):
                ax.plot(RADII, comp[(ep, side, sf)], ls, color=f"C{j}", marker="o", ms=3,
                        label=f"{side}, S = {sf * IMGS[ep].s_expected:.2f} mJy ({sf:g} S_exp)")
        ax.axhline(args.completeness, color="0.5", lw=0.6, ls=":")
        ax.set_title(f"Epoch {ep}")
        ax.set_xlabel("radius from the core (arcsec)")
    axs[0].set_ylabel("completeness")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.fig), exist_ok=True)
    fig.savefig(args.fig, dpi=150)
