"""
Per-epoch astrometric frame offsets of the S-band images.

data/frame_offsets.txt (written by measure_frame_offsets.py) gives, for every campaign
image, the shift of the whole image measured from compact field sources. The kinematics
pipelines subtract it from every position measured in that image (core and lobes) and add
its uncertainty in quadrature, which ties all epochs to one frame.
"""

import os

import numpy as np

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "frame_offsets.txt")


def load_frame_offsets(path=PATH):
    """{scheduling-block timestamp: (dx, dy, err_x, err_y)} in arcsec."""
    offsets = {}
    in_frame = False
    with open(path) as fh:
        for line in fh:
            if line.startswith("# FRAME"):
                in_frame = True
                continue
            if line.startswith("# PER SOURCE"):
                break
            parts = line.split()
            if in_frame and len(parts) >= 9 and not parts[0].startswith("#"):
                offsets[int(parts[0])] = tuple(float(v) for v in parts[2:6])
    return offsets


def apply_frame_offset(offsets, ts, dx, dy, ex, ey):
    """
    Position (dx, dy) with errors (ex, ey) corrected for the frame offset of epoch ts,
    or None if that epoch has no measured offset.
    """
    if ts not in offsets:
        return None
    fx, fy, efx, efy = offsets[ts]
    return dx - fx, dy - fy, float(np.hypot(ex, efx)), float(np.hypot(ey, efy))
