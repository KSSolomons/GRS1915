"""
Summary of the physical kinematics grids by limit type (data/kinematics_limit_summary.txt).

One row per (limit type, distance): no limits; the beam-FWHM limits (x0.5, 1.0, 1.5, sigma_UL = 0.05"); and the measured limits at
S = 1.0, 0.5, 0.25 x S_exp. Read from the reports written by pipeline_physical_kinematics.py:
    data/physical_kinematics_report.txt                      FWHM grid and no-limits fits
    data/physical_kinematics_report_measured_S<factor>.txt   measured limits (--ul-mjd/--ul-radius/--ul-width)
dBIC = BIC(model) - BIC(ballistic): negative means the model is preferred. ULpen = upper-limit term in ln L at the best fit
(0 = the limit is satisfied without changing the fit, i.e. it does not bind).

  python src/summarise_limits.py [--out data/kinematics_limit_summary.txt] [--sigma-ul 0.05]
"""

import argparse
import os
import re

p = argparse.ArgumentParser()
p.add_argument("--out", default="data/kinematics_limit_summary.txt")
p.add_argument("--sigma-ul", default="0.05", help="FWHM-grid penalty width to list")
p.add_argument("--s-factors", nargs="+", default=["1.0", "0.5", "0.25"])
args = p.parse_args()


def parse(path):
    """{(label, width, D): {model: dict}} from a report."""
    out = {}
    for line in open(path):
        if line.count("|") != 3 or line.lstrip().startswith("D "):
            continue
        left, mid, right, edge = (x.strip() for x in line.split("|"))
        D, label, width = left.split()
        model, k, lnl, dbic, ulpen = mid.split()
        v = right.split()
        out.setdefault((label, width, float(D)), {})[model] = dict(
            bic=float(dbic), ulpen=float(ulpen), t_ej=float(v[0]), beta0=float(v[1]), theta=float(v[2]), tau=v[3],
            gb0=float(v[5]), edge=re.sub(r"\s*\(t_coast.*", "", edge))
    return out


rows = []  # (limit label, D, fits)
main = parse("data/physical_kinematics_report.txt")
for (label, width, D), fits in sorted(main.items(), key=lambda kv: (kv[0][2], kv[0][0])):
    if label == "none":
        rows.append(("none", D, fits))
    elif width == args.sigma_ul:
        rows.append((f"FWHM x{label[:3]}", D, fits))
for sf in args.s_factors:
    path = f"data/physical_kinematics_report_measured_S{sf}.txt"
    if not os.path.exists(path):
        continue
    for (label, width, D), fits in sorted(parse(path).items(), key=lambda kv: kv[0][2]):
        if label == "custom":
            rows.append((f"measured S x{sf}", D, fits))

order = {"none": 0, "FWHM x0.5": 1, "FWHM x1.0": 2, "FWHM x1.5": 3, "measured S x1.0": 4, "measured S x0.5": 5, "measured S x0.25": 6}
rows.sort(key=lambda r: (r[1], order[r[0]]))
hdr = (f"{'limit':<17}{'D':>5} | {'dBIC dsym':>9} {'dBIC coast':>10} {'ULpen bal':>9} {'ULpen dsym':>10} | "
       f"{'bal T_ej':>9} {'beta0':>6} {'theta':>6} | {'dsym T_ej':>9} {'theta':>6} {'tau':>5} {'Gb0':>7} | edge flags (bal; dsym)")
lines = [
    "Kinematic fits by limit type. dBIC = BIC(model) - BIC(ballistic) (negative = model preferred); dsym = decel_sym, coast = coast_decel.",
    f"ULpen = upper-limit term in ln L at the best fit (0 = the limit does not bind). FWHM rows use sigma_UL = {args.sigma_ul}\".",
    "", hdr, "-" * len(hdr),
]
for label, D, f in rows:
    b, d, c = f["ballistic"], f["decel_sym"], f.get("coast_decel")
    cb = f"{c['bic'] - b['bic']:10.2f}" if c else f"{'--':>10}"
    lines.append(
        f"{label:<17}{D:5.1f} | {d['bic'] - b['bic']:9.2f} {cb} {b['ulpen']:9.2f} {d['ulpen']:10.2f} | "
        f"{b['t_ej']:9.1f} {b['beta0']:6.3f} {b['theta']:6.1f} | {d['t_ej']:9.1f} {d['theta']:6.1f} {d['tau']:>5} {d['gb0']:7.2f} | {b['edge']}; {d['edge']}"
    )
open(args.out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
