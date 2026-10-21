from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Dynamic output directory relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Jet geometry & real data (EoN)
true_pa = 137.95  # degrees (from Model 4 / preferred model fits, global average)
rad_pa = np.deg2rad(true_pa + 180)  # +180 to point to the North-West lobe in Panel 1

# Real distances and PAs (Northern Lobe)
distances = np.array([2.226, 2.537, 2.684, 2.880, 3.079, 3.472, 3.598])
measured_pas = np.array([-48.1, -43.9, -49.3, -45.5, -35.4, -40.9, -30.9]) + 180
measured_pa_rad = np.deg2rad(measured_pas + 180)  # +180 to point North-West in Panel 1

point_colors = [
    "#cc33ff",
    "#ff33ff",
    "#ff3399",
    "#ff6666",
    "#ffcc00",
    "#99ff33",
    "#00ffcc",
]

# Vectorized position calculations
x_centers = distances * np.sin(rad_pa)
y_centers = distances * np.cos(rad_pa)
x_peaks = distances * np.sin(measured_pa_rad)
y_peaks = distances * np.cos(measured_pa_rad)

# Set up figure
plt.style.use("dark_background")
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.3]}
)

# --- Panel 1: Schematic of the jet ---
# ax1.set_title("Physical Picture:\nDiscrete Ejecta Skewed by Interaction", fontsize=16, pad=15, weight='bold')

# Core
ax1.plot(0, 0, marker="*", color="white", markersize=25, label="Core")

# True jet axis (expected linear progression)
ax1.plot(
    [0, 4.0 * np.sin(rad_pa)],
    [0, 4.0 * np.cos(rad_pa)],
    color="white",
    linestyle="-",
    alpha=0.6,
    lw=2,
    label="Expected Linear Path",
)

# Blobs, centers, and measured peaks
for i in range(len(distances)):
    # Expanding blob ellipse (scaled to intermediate size)
    blob_w = 0.8 + i * 0.2
    blob_h = 0.4 + i * 0.15
    blob = Ellipse(
        (x_centers[i], y_centers[i]),
        width=blob_w,
        height=blob_h,
        angle=-(true_pa + 180),
        facecolor="#1f77b4",
        alpha=0.22,
        edgecolor="#66b3ff",
        lw=1.5,
    )
    ax1.add_patch(blob)

    # Bulk center marker
    ax1.plot(
        x_centers[i],
        y_centers[i],
        "+",
        color="white",
        markersize=10,
        alpha=0.7,
        label="Bulk Center" if i == 0 else None,
    )

    # Extended line from core through measured peak
    ext_x = 4.2 * np.sin(measured_pa_rad[i])
    ext_y = 4.2 * np.cos(measured_pa_rad[i])
    ax1.plot(
        [0, ext_x], [0, ext_y], color=point_colors[i], linestyle="--", alpha=0.3, lw=1.5
    )

    # Measured peak marker
    ax1.plot(
        x_peaks[i],
        y_peaks[i],
        "o",
        color=point_colors[i],
        markersize=9,
        label="Measured Peak" if i == 0 else None,
    )

ax1.set_xlim(-3.5, 0.5)
ax1.set_ylim(-0.5, 3.5)
ax1.set_aspect("equal")
ax1.axis("off")

# Legend
ax1.legend(loc="lower left", frameon=False, fontsize=12)

# --- Panel 2: Measured PA vs Distance ---
# ax2.set_title("Model Comparison:\nLinear Drift vs. Apparent Wobble", fontsize=16, pad=15, weight='bold')

distances_smooth = np.linspace(2.0, 3.8, 100)

# 1. True PA line
ax2.axhline(
    true_pa, color="white", linestyle="--", alpha=0.7, lw=2, label="Constant-PA Model"
)

# 2. Fit a linear model to represent the "Linear PA Drift Model"
poly = np.polyfit(distances, measured_pas, 1)
p1 = np.poly1d(poly)
ax2.plot(
    distances_smooth,
    p1(distances_smooth),
    color="#ff3333",
    lw=3,
    label="Linear PA Drift Model",
    path_effects=[pe.Stroke(linewidth=5, foreground="black"), pe.Normal()],
)

# 3. Measured data points
for i in range(len(distances)):
    lbl = "Measured Position (Northern Lobe)" if i == 0 else None
    ax2.plot(
        distances[i],
        measured_pas[i],
        "o",
        color=point_colors[i],
        markersize=12,
        label=lbl,
    )

ax2.set_xlabel("Separation from Core (arcsec)", fontsize=14)
ax2.set_ylabel("Measured Position Angle (°)", fontsize=14)
ax2.tick_params(axis="both", which="major", labelsize=12)
ax2.grid(True, alpha=0.15, ls="--")

margin = 5
ax2.set_ylim(min(np.min(measured_pas), true_pa) - margin, np.max(measured_pas) + margin)
ax2.legend(loc="lower right", fontsize=12, framealpha=0.6, edgecolor="#444444")

plt.tight_layout()

# Save figures and close
png_path = FIG_DIR / "pa_wobble_explanation.png"
pdf_path = FIG_DIR / "pa_wobble_explanation.pdf"
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

print(f"Plots saved successfully to {FIG_DIR}.")
