"""
Write the lobe positions for get_alpha.py from the S-band MFS component fits.

Run from the repository root with ordinary Python (no CASA needed):

    python alpha/make_lobe_positions.py

For every 2025 epoch with resolved jets (Epoch 12 excluded, as in the kinematics
pipelines) it lists each non-core component of S-band-fits/*2D_Fitting*.txt with
its position, labelled N (north of the catalogue position, receding) or S
(approaching), and its MFS signal-to-noise. Copy alpha/lobe_positions.txt next
to get_alpha.py on the machine with the channel images; get_alpha.py then fits
those positions as extra fixed components in every channel image.
"""

import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from epoch_limits import EJECTION_CAMPAIGN_START

EPOCH12_TS = 1764664923  # Epoch 12 jet features are unreliable (likely artefacts)
CORE_RA_DEG = (288.7979, 288.7982)  # same core selection as the kinematics pipelines
CAT_DEC_DEG = 10 + 56 / 60.0 + 44.905 / 3600.0
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lobe_positions.txt")

rows = []
for f in sorted(glob.glob(os.path.join("S-band-fits", "*2D_Fitting*.txt"))):
    m = re.search(r"img_(\d+)", os.path.basename(f))
    if not m:
        continue
    ts = int(m.group(1))
    if ts / 86400.0 + 40587.0 < EJECTION_CAMPAIGN_START or ts == EPOCH12_TS:
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
    for c in comps:
        if "ra_deg" not in c or "dec_deg" not in c or CORE_RA_DEG[0] < c["ra_deg"] < CORE_RA_DEG[1]:
            continue
        rows.append((str(ts), "N" if c["dec_deg"] > CAT_DEC_DEG else "S", c["ra"], c["dec"], c.get("snr", float("nan"))))

with open(OUTPUT, "w") as fp:
    fp.write("# Lobe positions for get_alpha.py, from the S-band MFS fits (S-band-fits/), written by make_lobe_positions.py\n")
    fp.write("# N = north (receding), S = south (approaching); SNR is amplitude / error in the MFS image.\n")
    fp.write("# {:<12} {:<5} {:<20} {:<20} {:>6}\n".format("Block", "Label", "RA (J2000)", "Dec (J2000)", "SNR"))
    for r in rows:
        fp.write("{:<14} {:<5} {:<20} {:<20} {:>6.1f}\n".format(*r))
print("Wrote {} lobe positions for {} epochs to {}".format(len(rows), len(set(r[0] for r in rows)), OUTPUT))
