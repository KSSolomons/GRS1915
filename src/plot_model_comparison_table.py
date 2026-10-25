import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Set style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.titlesize": 16,
    "mathtext.fontset": "dejavuserif",
    "font.family": "serif",
})

# Define the 12 configurations (6 models x 2 limits settings)
runs = [
    {"id": "m1_lim", "model": 1, "name": "M1 (Free)\n+Limits", "limits": True, "label": "Model 1 (With Limits)", "file": "data/absolute_kinematics/kinematic_report_model1_with_limits.txt", "color": "#1f77b4"},
    {"id": "m1_nolim", "model": 1, "name": "M1 (Free)\nNo Limits", "limits": False, "label": "Model 1 (No Limits)", "file": "data/absolute_kinematics/kinematic_report_model1_without_limits.txt", "color": "#6baed6"},
    {"id": "m2_lim", "model": 2, "name": "M2 (Anch)\n+Limits", "limits": True, "label": "Model 2 (With Limits)", "file": "data/absolute_kinematics/kinematic_report_model2_with_limits.txt", "color": "#d62728"},
    {"id": "m2_nolim", "model": 2, "name": "M2 (Anch)\nNo Limits", "limits": False, "label": "Model 2 (No Limits)", "file": "data/absolute_kinematics/kinematic_report_model2_without_limits.txt", "color": "#fb6a4a"},
    {"id": "m3_lim", "model": 3, "name": "M3 (Free)\n+Limits", "limits": True, "label": "Model 3 (With Limits)", "file": "data/absolute_kinematics/kinematic_report_model3_with_limits.txt", "color": "#9467bd"},
    {"id": "m3_nolim", "model": 3, "name": "M3 (Free)\nNo Limits", "limits": False, "label": "Model 3 (No Limits)", "file": "data/absolute_kinematics/kinematic_report_model3_without_limits.txt", "color": "#c5b0d5"},
    {"id": "m4_lim", "model": 4, "name": "M4 (Anch)\n+Limits", "limits": True, "label": "Model 4 (With Limits)", "file": "data/absolute_kinematics/kinematic_report_model4_with_limits.txt", "color": "#2ca02c"},
    {"id": "m4_nolim", "model": 4, "name": "M4 (Anch)\nNo Limits", "limits": False, "label": "Model 4 (No Limits)", "file": "data/absolute_kinematics/kinematic_report_model4_without_limits.txt", "color": "#74c476"},
    {"id": "m5_lim", "model": 5, "name": "M5 (Free)\n+Limits", "limits": True, "label": "Model 5 (With Limits)", "file": "data/absolute_kinematics/kinematic_report_model5_with_limits.txt", "color": "#e377c2"},
    {"id": "m5_nolim", "model": 5, "name": "M5 (Free)\nNo Limits", "limits": False, "label": "Model 5 (No Limits)", "file": "data/absolute_kinematics/kinematic_report_model5_without_limits.txt", "color": "#f7b6d2"},
    {"id": "m6_lim", "model": 6, "name": "M6 (Anch)\n+Limits", "limits": True, "label": "Model 6 (With Limits)", "file": "data/absolute_kinematics/kinematic_report_model6_with_limits.txt", "color": "#ff7f0e"},
    {"id": "m6_nolim", "model": 6, "name": "M6 (Anch)\nNo Limits", "limits": False, "label": "Model 6 (No Limits)", "file": "data/absolute_kinematics/kinematic_report_model6_without_limits.txt", "color": "#ffbb78"},
]

def parse_report(filepath, model_id):
    data = {}
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found. Using placeholder values.")
        return {
            "Tej": (60600.0, 0.0, 0.0),
            "mu_r": (0.007, 0.001, 0.001),
            "mu_a": (0.010, 0.001, 0.001),
            "beta": (0.8, 0.1, 0.1),
            "gamma": (1.5, 0.2, 0.2),
            "theta": (70.0, 2.0, 2.0),
            "dBIC": 0.0,
        }

    with open(filepath, "r") as f:
        text = f.read()

    def get_val_err(pattern):
        match = re.search(pattern, text)
        if match:
            val = float(match.group(1))
            err_plus = float(match.group(2))
            err_minus = float(match.group(3))
            return val, err_plus, err_minus
        return None, None, None

    data["Tej"] = get_val_err(r"Extrapolated Launch Time \(T_ej\):\s*MJD\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")
    if data["Tej"][0] is None:
        m_anch = re.search(r"Anchored Launch Time \(T_ej\)\s*:\s*MJD\s*([\d\.]+)", text)
        if m_anch:
            data["Tej"] = (float(m_anch.group(1)), 0.0, 0.0)

    data["mu_r"] = get_val_err(r"Receding Lobe \(North\) Speed \(mu_r\):\s*([\d\.eE\+-]+)\s*\+\s*([\d\.eE\+-]+)\s*-\s*([\d\.eE\+-]+)")
    data["mu_a"] = get_val_err(r"Approaching Lobe \(South\) Speed \(mu_a\):\s*([\d\.eE\+-]+)\s*\+\s*([\d\.eE\+-]+)\s*-\s*([\d\.eE\+-]+)")
    data["beta"] = get_val_err(r"Intrinsic Jet Speed \(beta = v/c\)\s*:\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")
    data["gamma"] = get_val_err(r"Lorentz Factor \(Gamma\)\s*:\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")
    data["theta"] = get_val_err(r"Jet Inclination Angle \(theta\)\s*:\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")
    data["bct"] = get_val_err(r"Derived beta \* cos\(theta\)\s*:\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")
    data["dratio"] = get_val_err(r"Kinematic Doppler Factor Ratio\s*:\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")
    data["Dmax"] = get_val_err(r"Maximum Physical Distance \(D_max\)\s*:\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")

    return data

def get_unified_dbic(model_id, limits):
    ref_file = "data/absolute_kinematics/kinematic_report_model5_with_limits.txt" if limits else "data/absolute_kinematics/kinematic_report_model5_without_limits.txt"
    if not os.path.exists(ref_file):
        return 0.0
    with open(ref_file, "r") as f:
        ref_text = f.read()
    for line in ref_text.splitlines():
        if line.startswith("Ballistic (Free Tej)") and model_id == 1:
            return float(line.split()[-1])
        elif line.startswith("Ballistic (Anchored") and model_id == 2:
            return float(line.split()[-1])
        elif line.startswith("Piecewise Deceleration (Free") and model_id == 3:
            return float(line.split()[-1])
        elif line.startswith("Piecewise Deceleration (Anchored") and model_id == 4:
            return float(line.split()[-1])
        elif line.startswith("Smooth Continuous Deceleration (Free") and model_id == 5:
            return float(line.split()[-1])
        elif line.startswith("Smooth Continuous Deceleration (Anchored") and model_id == 6:
            return float(line.split()[-1])
    return 0.0

parsed = []
for r in runs:
    p = parse_report(r["file"], r["model"])
    p["dBIC"] = get_unified_dbic(r["model"], r["limits"])
    parsed.append(p)

# Create Figure
fig, axes = plt.subplots(2, 3, figsize=(19, 11))
fig.suptitle("GRS 1915+105 Kinematic Model Suite: Comprehensive Comparison (Models 1–6)", fontsize=16, weight="bold", y=0.98)

x_pos = np.arange(len(runs))
width = 0.65

# -------------------------------------------------------------
# Panel 1: Delta BIC (Model Selection)
# -------------------------------------------------------------
ax1 = axes[0, 0]
dbic_vals = [p["dBIC"] for p in parsed]
bars = ax1.bar(x_pos, dbic_vals, width=width, color=[r["color"] for r in runs], edgecolor="black", linewidth=1.2)
for i, r in enumerate(runs):
    if not r["limits"]:
        bars[i].set_hatch("//")
        bars[i].set_alpha(0.85)

ax1.axhline(0, color="gray", linestyle="-", linewidth=1)
ax1.axhline(2, color="orange", linestyle="--", linewidth=1.2, label=r"$\Delta{\rm BIC} = 2$ (Positive)")
ax1.axhline(6, color="red", linestyle=":", linewidth=1.2, label=r"$\Delta{\rm BIC} = 6$ (Strong)")
ax1.axhline(10, color="darkred", linestyle="-.", linewidth=1.2, label=r"$\Delta{\rm BIC} = 10$ (Decisive)")
ax1.set_ylabel(r"$\Delta{\rm BIC}$ (relative to best model)")
ax1.set_title(r"(a) Model Selection Penalty ($\Delta{\rm BIC}$)", weight="bold")
ax1.set_xticks(x_pos)
ax1.set_xticklabels([r["name"] for r in runs], fontsize=7.5)
ax1.set_ylim(-0.5, 32.0)
ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
ax1.grid(axis="y", linestyle=":", alpha=0.6)

for i, v in enumerate(dbic_vals):
    ax1.text(i, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, weight="bold")


# -------------------------------------------------------------
# Panel 2: Launch Time T_ej (MJD)
# -------------------------------------------------------------
ax2 = axes[0, 1]
ax2.axhspan(60550, 60650, color="gold", alpha=0.25, label="Radio/X-ray Flare Window\n(Aug-Nov 2024, ~60600)", zorder=0)

for i, (p, r) in enumerate(zip(parsed, runs)):
    val, e_plus, e_minus = p["Tej"]
    marker = "o" if r["limits"] else "s"
    mfc = r["color"] if r["limits"] else "white"
    ax2.errorbar(i, val, yerr=[[e_minus], [e_plus]], fmt=marker, color=r["color"], mfc=mfc,
                 mec=r["color"], mew=2, markersize=8, capsize=4, capthick=1.5, elinewidth=2, zorder=5)

ax2.set_ylabel(r"Launch Time $T_{\rm ej}$ (MJD)")
ax2.set_title(r"(b) Ejection Epoch ($T_{\rm ej}$)", weight="bold")
ax2.set_xticks(x_pos)
ax2.set_xticklabels([r["name"] for r in runs], fontsize=7.5)
ax2.set_ylim(60250, 60750)
ax2.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
ax2.grid(axis="y", linestyle=":", alpha=0.6)

ax2_twin = ax2.twinx()
ax2_twin.set_ylim(ax2.get_ylim())
mjd_ticks = [60300, 60400, 60500, 60600, 60700]
dates_labels = ["Jan 2024", "Apr 2024", "Jul 2024", "Oct 2024", "Jan 2025"]
ax2_twin.set_yticks(mjd_ticks)
ax2_twin.set_yticklabels(dates_labels, fontsize=8.5, color="dimgray")


# -------------------------------------------------------------
# Panel 3: Apparent Speeds (mu_a and mu_r)
# -------------------------------------------------------------
ax3 = axes[0, 2]
for i, (p, r) in enumerate(zip(parsed, runs)):
    val_a, ea_plus, ea_minus = p["mu_a"]
    val_r, er_plus, er_minus = p["mu_r"]
    
    offset = 0.15
    ax3.errorbar(i - offset, val_a * 1e3, yerr=[[ea_minus * 1e3], [ea_plus * 1e3]],
                 fmt="v", color="#d62728", mfc="#d62728" if r["limits"] else "white",
                 mec="#d62728", mew=1.8, markersize=7, capsize=3.5, elinewidth=1.8)
    ax3.errorbar(i + offset, val_r * 1e3, yerr=[[er_minus * 1e3], [er_plus * 1e3]],
                 fmt="^", color="#1f77b4", mfc="#1f77b4" if r["limits"] else "white",
                 mec="#1f77b4", mew=1.8, markersize=7, capsize=3.5, elinewidth=1.8)

ax3.plot([], [], marker="v", color="#d62728", linestyle="none", markersize=7, label=r"Approaching $\mu_a$ (South)")
ax3.plot([], [], marker="^", color="#1f77b4", linestyle="none", markersize=7, label=r"Receding $\mu_r$ (North)")
ax3.set_ylabel(r"Proper Motion (mas day$^{-1}$)")
ax3.set_title(r"(c) Late Proper Motions ($\mu_a, \mu_r$)", weight="bold")
ax3.set_xticks(x_pos)
ax3.set_xticklabels([r["name"] for r in runs], fontsize=7.5)
ax3.set_ylim(0, 22)
ax3.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
ax3.grid(axis="y", linestyle=":", alpha=0.6)


# -------------------------------------------------------------
# Panel 4: Intrinsic Jet Speed beta (v/c)
# -------------------------------------------------------------
ax4 = axes[1, 0]
ax4.axhline(1.0, color="black", linestyle="-", linewidth=1.2, alpha=0.7, label=r"Light Speed $c$")
ax4.axhline(0.92, color="purple", linestyle="--", linewidth=1.2, alpha=0.8, label="Historical GRS 1915 (Mirabel 1994)")

for i, (p, r) in enumerate(zip(parsed, runs)):
    val, e_plus, e_minus = p["beta"]
    err_high = min(e_plus, 0.999 - val) if val < 1.0 else e_plus
    marker = "o" if r["limits"] else "s"
    mfc = r["color"] if r["limits"] else "white"
    ax4.errorbar(i, val, yerr=[[e_minus], [err_high]], fmt=marker, color=r["color"], mfc=mfc,
                 mec=r["color"], mew=2, markersize=8, capsize=4, capthick=1.5, elinewidth=2, zorder=5)

ax4.set_ylabel(r"Intrinsic Speed $\beta = v/c$")
ax4.set_title(r"(d) Late Intrinsic Jet Speed ($\beta$)", weight="bold")
ax4.set_xticks(x_pos)
ax4.set_xticklabels([r["name"] for r in runs], fontsize=7.5)
ax4.set_ylim(0.2, 1.05)
ax4.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
ax4.grid(axis="y", linestyle=":", alpha=0.6)


# -------------------------------------------------------------
# Panel 5: Lorentz Factor Gamma
# -------------------------------------------------------------
ax5 = axes[1, 1]
ax5.axhline(1.0, color="gray", linestyle="-", linewidth=1)
ax5.axhline(2.5, color="purple", linestyle="--", linewidth=1.2, alpha=0.8, label=r"Historical GRS 1915 ($\Gamma \approx 2.5$)")

for i, (p, r) in enumerate(zip(parsed, runs)):
    val, e_plus, e_minus = p["gamma"]
    err_high = min(e_plus, 5.0)
    marker = "o" if r["limits"] else "s"
    mfc = r["color"] if r["limits"] else "white"
    ax5.errorbar(i, val, yerr=[[e_minus], [err_high]], fmt=marker, color=r["color"], mfc=mfc,
                 mec=r["color"], mew=2, markersize=8, capsize=4, capthick=1.5, elinewidth=2, zorder=5)

ax5.set_ylabel(r"Lorentz Factor $\Gamma$")
ax5.set_title(r"(e) Bulk Lorentz Factor ($\Gamma$)", weight="bold")
ax5.set_xticks(x_pos)
ax5.set_xticklabels([r["name"] for r in runs], fontsize=7.5)
ax5.set_ylim(0.8, 4.5)
ax5.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
ax5.grid(axis="y", linestyle=":", alpha=0.6)


# -------------------------------------------------------------
# Panel 6: Jet Inclination Angle theta (deg)
# -------------------------------------------------------------
ax6 = axes[1, 2]
ax6.axhspan(66, 71, color="purple", alpha=0.15, label="Historical Angle (66°–71°)", zorder=0)

for i, (p, r) in enumerate(zip(parsed, runs)):
    val, e_plus, e_minus = p["theta"]
    marker = "o" if r["limits"] else "s"
    mfc = r["color"] if r["limits"] else "white"
    ax6.errorbar(i, val, yerr=[[e_minus], [e_plus]], fmt=marker, color=r["color"], mfc=mfc,
                 mec=r["color"], mew=2, markersize=8, capsize=4, capthick=1.5, elinewidth=2, zorder=5)

ax6.set_ylabel(r"Inclination Angle $\theta$ (deg)")
ax6.set_title(r"(f) Jet Inclination ($\theta$)", weight="bold")
ax6.set_xticks(x_pos)
ax6.set_xticklabels([r["name"] for r in runs], fontsize=7.5)
ax6.set_ylim(55, 88)
ax6.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
ax6.grid(axis="y", linestyle=":", alpha=0.6)

# Marker Legend
fig.text(0.5, 0.015, "Marker Legend:  ● Filled Circle = With Upper Limits  |  □ Open Square / Hatch = Without Upper Limits  (Error bars denote 90% HDI)", 
         ha="center", fontsize=10.5, weight="bold", bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="gray"))

plt.tight_layout(rect=[0, 0.035, 1, 0.96])
out_fig = "figures/kinematics/model_comparison_parameter_summary.png"
os.makedirs("figures/kinematics", exist_ok=True)
plt.savefig(out_fig, dpi=300)
plt.close(fig)
print(f"Saved visual comparison figure to {out_fig}")

