import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Arc
from astropy.time import Time
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from epoch_limits import blending_limits

from src.plot_absolute_separation import (
    unpack_params_abs,
    calc_trajectory,
    unpack_params_smooth,
    calc_trajectory_smooth,
)

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

def make_combined_figure(out_fig="figures/combined_kinematics_fit_plot.png"):
    np.random.seed(42)

    # 1. Load data
    f_m1 = "data/absolute_kinematics_plot_data_model1_with_limits.npz"
    f_m5 = "data/absolute_kinematics_plot_data_model5_with_limits.npz"
    
    if not os.path.exists(f_m1) or not os.path.exists(f_m5):
        raise FileNotFoundError("Model data files not found in data/")
        
    d1 = np.load(f_m1)
    d5 = np.load(f_m5)

    # Astrometric data
    n_mjd = d1["n_mjd"]
    n_da = d1["n_da"]
    n_dd = d1["n_dd"]
    n_da_err = d1["n_da_err"]
    n_dd_err = d1["n_dd_err"]

    s_mjd = d1["s_mjd"]
    s_da = d1["s_da"]
    s_dd = d1["s_dd"]
    s_da_err = d1["s_da_err"]
    s_dd_err = d1["s_dd_err"]

    c_mjd = d1["c_mjd"]

    # Best-fit parameters for RA/Dec plot
    best_fit_file = "data/best_fit_params.txt"
    if os.path.exists(best_fit_file):
        params = {}
        with open(best_fit_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    params[parts[0]] = float(parts[1])
        weighted_mean_pa_deg = (params["pa_n_deg"] + params["pa_s_deg"]) / 2.0
        weighted_std_deg = params["theta_cone_deg"]
    else:
        weighted_mean_pa_deg = -41.5
        weighted_std_deg = 5.6

    avg_pa_rad = np.radians(weighted_mean_pa_deg)
    std_pa_rad = np.radians(weighted_std_deg)

    # Figure setup: slender left column with ample x-padding and snug right column
    fig = plt.figure(figsize=(12.0, 7.2))
    
    gs = gridspec.GridSpec(2, 2, width_ratios=[0.26, 0.74], height_ratios=[1.0, 1.0],
                           left=0.07, right=0.98, top=0.91, bottom=0.155,
                           wspace=0.25, hspace=0.0)

    ax_radec = fig.add_subplot(gs[:, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_bot = fig.add_subplot(gs[1, 1], sharex=ax_top)

    # -------------------------------------------------------------------------
    # PANEL A: RA / DEC SKY PLOT (FAITHFUL BAHRAMIAN ET AL. STYLE)
    # -------------------------------------------------------------------------
    mjd_cb_min = 60800.0
    mjd_cb_max = 60960.0
    norm = Normalize(vmin=mjd_cb_min, vmax=mjd_cb_max)

    # Custom sequential colormaps matching Bahramian et al. (light at launch, dark at late epochs)
    cmap_north = LinearSegmentedColormap.from_list(
        "north_cb", ["#a6d2f5", "#5c9ee5", "#2563eb", "#1e40af", "#0f172a"]
    )
    cmap_south = LinearSegmentedColormap.from_list(
        "south_cb", ["#fde047", "#f97316", "#dc2626", "#991b1b", "#450a0a"]
    )

    L = 6.0
    x_line = np.array([L * np.sin(avg_pa_rad), -L * np.sin(avg_pa_rad)])
    y_line = np.array([L * np.cos(avg_pa_rad), -L * np.cos(avg_pa_rad)])
    ax_radec.plot(x_line, y_line, color="black", linestyle="--", lw=1.2, zorder=1.5)

    pa_hi = avg_pa_rad + std_pa_rad
    pa_lo = avg_pa_rad - std_pa_rad

    # Conical scatter envelopes
    ax_radec.fill(
        [0, L * np.sin(pa_lo), L * np.sin(pa_hi)],
        [0, L * np.cos(pa_lo), L * np.cos(pa_hi)],
        color="gray", alpha=0.25, zorder=1, edgecolor="none",
    )
    ax_radec.fill(
        [0, -L * np.sin(pa_lo), -L * np.sin(pa_hi)],
        [0, -L * np.cos(pa_lo), -L * np.cos(pa_hi)],
        color="gray", alpha=0.25, zorder=1, edgecolor="none",
    )

    # Annotate \theta_{scatter} with an arc across the cone at R=2.2 arcsec
    thetas_arc = np.linspace(pa_lo, pa_hi, 40)
    R_arc = 2.2
    arc_x = -R_arc * np.sin(thetas_arc)
    arc_y = -R_arc * np.cos(thetas_arc)
    ax_radec.plot(arc_x, arc_y, color="dimgray", lw=0.9, zorder=2.5)
    ax_radec.text(np.mean(arc_x) - 0.55, np.mean(arc_y), r"$\theta_{\rm scatter}$",
                  fontsize=10.5, color="black", alpha=0.8, fontstyle="italic", zorder=3, va="center", ha="right")

    # Core at (0,0)
    ax_radec.scatter(0, 0, marker="*", s=210, color="gold", edgecolors="black", linewidth=1.1, zorder=4)

    # Data errorbars
    ax_radec.errorbar(n_da, n_dd, xerr=n_da_err, yerr=n_dd_err, fmt="none", ecolor="gray", capsize=0, zorder=2, alpha=0.6, elinewidth=1.1)
    ax_radec.errorbar(s_da, s_dd, xerr=s_da_err, yerr=s_dd_err, fmt="none", ecolor="gray", capsize=0, zorder=2, alpha=0.6, elinewidth=1.1)

    # Scatter points colored by time (matching right panel)
    ax_radec.scatter(n_da, n_dd, c=n_mjd, cmap=cmap_north, norm=norm, marker="^", edgecolors="black", s=60, zorder=3)
    ax_radec.scatter(s_da, s_dd, c=s_mjd, cmap=cmap_south, norm=norm, marker="v", edgecolors="black", s=60, zorder=3)

    # Comfortable x-axis padding around ejecta points
    ax_radec.set_xlim(3.4, -3.0)
    ax_radec.set_ylim(-4.4, 3.6)
    
    # Clean labels without cos delta
    ax_radec.set_xlabel(r"$\Delta\alpha$ ($^{\prime\prime}$)", fontsize=11.5, labelpad=5)
    ax_radec.set_ylabel(r"$\Delta\delta$ ($^{\prime\prime}$)", fontsize=11.5, labelpad=5)

    ax_radec.xaxis.set_major_locator(MultipleLocator(1))
    ax_radec.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax_radec.yaxis.set_major_locator(MultipleLocator(1))
    ax_radec.yaxis.set_minor_locator(MultipleLocator(0.2))
    ax_radec.minorticks_on()
    ax_radec.tick_params(axis="both", which="major", direction="in", top=True, right=True, labelsize=10, length=5, width=1.1)
    ax_radec.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=2.5, width=0.8)
    ax_radec.grid(True, linestyle=":", alpha=0.4)

    # CONTINUOUS FLUSH DUAL COLORBARS (extended to almost meet at 50% midpoint)
    cax_north = inset_axes(ax_radec, width="6.5%", height="100%", loc="lower left",
                           bbox_to_anchor=(1.012, 0.510, 1.0, 0.490), bbox_transform=ax_radec.transAxes, borderpad=0)
    cax_south = inset_axes(ax_radec, width="6.5%", height="100%", loc="lower left",
                           bbox_to_anchor=(1.012, 0.0, 1.0, 0.490), bbox_transform=ax_radec.transAxes, borderpad=0)

    cb_north = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap_north), cax=cax_north)
    cb_north.set_ticks([60800, 60850, 60900, 60950])
    cb_north.set_label("Time (Northern lobe)", fontsize=9.5)
    cb_north.ax.tick_params(labelsize=8.5, direction="out", pad=2, length=3.5, width=0.9)

    cb_south = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap_south), cax=cax_south)
    cb_south.set_ticks([60800, 60850, 60900, 60950])
    cb_south.set_label("Time (Southern lobe)", fontsize=9.5)
    cb_south.ax.tick_params(labelsize=8.5, direction="out", pad=2, length=3.5, width=0.9)
    cb_south.ax.invert_yaxis()

    # -------------------------------------------------------------------------
    # RIGHT PANELS: SEPARATION TRAJECTORIES (MODEL 1 TOP, MODEL 4 BOTTOM)
    # -------------------------------------------------------------------------
    t_start = 60475.0
    t_end = max(np.max(n_mjd), np.max(s_mjd), np.max(c_mjd)) + 20.0
    t_smooth = np.linspace(t_start, t_end, 500)

    def plot_model_separation(ax, data_npz, model_id, is_top=False):
        flat_samples = data_npz["flat_samples"]
        anchor_date = float(data_npz["anchor_date"]) if "anchor_date" in data_npz else 60600.0

        ax.axhline(0, color="gray", linestyle=":", linewidth=0.9)

        # Trajectories
        models_n = []
        models_s = []
        inds = np.random.randint(len(flat_samples), size=4000)
        if model_id in [5, 6]:
            for i in inds:
                (t_ej_s, v0_n_s, tau_n_s, v0_s_s, tau_s_s,
                 pa_n_s, pa_s_s, theta_scatter_s, x_c_s, y_c_s) = unpack_params_smooth(flat_samples[i], model_id, anchor_date=anchor_date)
                models_n.append(calc_trajectory_smooth(t_smooth, t_ej_s, v0_n_s, tau_n_s))
                models_s.append(calc_trajectory_smooth(t_smooth, t_ej_s, v0_s_s, tau_s_s))

            mcmc_med_p = np.median(flat_samples, axis=0)
            (med_t_ej, med_v0_n, med_tau_n, med_v0_s, med_tau_s,
             med_pa_n, med_pa_s, med_scat, med_xc, med_yc) = unpack_params_smooth(mcmc_med_p, model_id, anchor_date=anchor_date)
        else:
            for i in inds:
                (t_ej_s, v1_n_s, v2_n_s, tb_n_s, v1_s_s, v2_s_s, tb_s_s,
                 pa_n_s, pa_s_s, theta_scatter_s, x_c_s, y_c_s) = unpack_params_abs(flat_samples[i], model_id, anchor_date=anchor_date)
                models_n.append(calc_trajectory(t_smooth, t_ej_s, v1_n_s, v2_n_s, tb_n_s))
                models_s.append(calc_trajectory(t_smooth, t_ej_s, v1_s_s, v2_s_s, tb_s_s))

            mcmc_med_p = np.median(flat_samples, axis=0)
            (med_t_ej, med_v1_n, med_v2_n, med_tb_n, med_v1_s, med_v2_s, med_tb_s,
             med_pa_n, med_pa_s, med_scat, med_xc, med_yc) = unpack_params_abs(mcmc_med_p, model_id, anchor_date=anchor_date)

        q_n = np.percentile(np.array(models_n), [15.865, 50, 84.135], axis=0)
        q_s = np.percentile(np.array(models_s), [15.865, 50, 84.135], axis=0)

        t_plot = t_smooth[t_smooth >= med_t_ej]
        mask_plot = t_smooth >= med_t_ej
        t_plot_line = np.concatenate(([med_t_ej], t_plot))
        med_n_line = np.concatenate(([0.0], q_n[1][mask_plot]))
        med_s_line = np.concatenate(([0.0], q_s[1][mask_plot]))

        # Median & CI (Bahramian orange/yellow style)
        ax.plot(t_plot_line, med_n_line, color="#e66101", linestyle="-", linewidth=1.7, zorder=2.5, label="Model Posterior Median")
        ax.fill_between(t_smooth, q_n[0], q_n[2], color="#fdb863", alpha=0.55, zorder=1.4, label=r"Model Posterior $1\sigma$ CI")

        ax.plot(t_plot_line, med_s_line, color="#e66101", linestyle="-", linewidth=1.7, zorder=2.5)
        ax.fill_between(t_smooth, q_s[0], q_s[2], color="#fdb863", alpha=0.55, zorder=1.4)

        # Break lines for piecewise models
        if model_id in [3, 4]:
            ax.axvline(med_tb_n, color="#1e40af", linestyle="--", linewidth=1.2, zorder=1.8, label=f"North Break ({med_tb_n:.0f})")
            ax.axvline(med_tb_s, color="#b91c1c", linestyle="--", linewidth=1.2, zorder=1.8, label=f"South Break ({med_tb_s:.0f})")

        # Project data points
        th_n_opt = np.radians(med_pa_n)
        sin_n_opt, cos_n_opt = np.sin(th_n_opt), np.cos(th_n_opt)
        n_par_opt = n_da * sin_n_opt + n_dd * cos_n_opt
        n_perp_opt = n_da * cos_n_opt - n_dd * sin_n_opt
        n_par_err_opt = np.sqrt((n_da_err * sin_n_opt)**2 + (n_dd_err * cos_n_opt)**2)
        n_perp_err_opt = np.sqrt((n_da_err * cos_n_opt)**2 + (n_dd_err * sin_n_opt)**2)

        th_s_opt = np.radians(med_pa_s)
        sin_s_opt, cos_s_opt = np.sin(th_s_opt), np.cos(th_s_opt)
        s_par_opt = s_da * sin_s_opt + s_dd * cos_s_opt
        s_perp_opt = s_da * cos_s_opt - s_dd * sin_s_opt
        s_par_err_opt = np.sqrt((s_da_err * sin_s_opt)**2 + (s_dd_err * cos_s_opt)**2)
        s_perp_err_opt = np.sqrt((s_da_err * cos_s_opt)**2 + (s_dd_err * sin_s_opt)**2)

        # Errorbars for data points
        ax.errorbar(n_mjd, n_par_opt, yerr=n_par_err_opt, fmt="none", ecolor="black", capsize=0, elinewidth=1.0, zorder=3)
        ax.errorbar(s_mjd, s_par_opt, yerr=s_par_err_opt, fmt="none", ecolor="black", capsize=0, elinewidth=1.0, zorder=3)

        # Colored markers matching left plot time colormaps (Bahramian et al. style)
        ax.scatter(n_mjd, n_par_opt, c=n_mjd, cmap=cmap_north, norm=norm, marker="^",
                   edgecolors="black", s=55, zorder=4, label=r"$\theta_\parallel^N$")
        ax.scatter(s_mjd, s_par_opt, c=s_mjd, cmap=cmap_south, norm=norm, marker="v",
                   edgecolors="black", s=55, zorder=4, label=r"$\theta_\parallel^S$")

        # Perpendicular motion (gray triangles along zero line)
        ax.errorbar(n_mjd, n_perp_opt, yerr=n_perp_err_opt, fmt="none", ecolor="silver", capsize=0, elinewidth=0.8, zorder=2)
        ax.errorbar(s_mjd, s_perp_opt, yerr=s_perp_err_opt, fmt="none", ecolor="silver", capsize=0, elinewidth=0.8, zorder=2)
        ax.scatter(n_mjd, n_perp_opt, color="lightgray", edgecolors="gray", marker="^", s=45, zorder=2.2, label=r"$\theta_\perp^N$")
        ax.scatter(s_mjd, s_perp_opt, color="lightgray", edgecolors="gray", marker="v", s=45, zorder=2.2, label=r"$\theta_\perp^S$")

        # Non-detection Upper limits with positional accuracy sigma_pos = theta_beam / (2 * SNR)
        ul_mjd, ul_r, ul_sigma_pos = blending_limits()
        ax.errorbar(ul_mjd, ul_r, yerr=ul_sigma_pos,
                    uplims=True, fmt="none", color="black", ecolor="black", capsize=3, elinewidth=1.2, zorder=5, label="Upper Limits")
        ax.errorbar(ul_mjd, -ul_r, yerr=ul_sigma_pos,
                    lolims=True, fmt="none", color="black", ecolor="black", capsize=3, elinewidth=1.2, zorder=5)

        ax.set_xlim(t_start, t_end)
        ax.set_ylim(-5.2, 4.4)

        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(20))
        ax.minorticks_on()
        ax.tick_params(axis="both", which="major", direction="in", right=True, top=True, labelsize=10, length=5, width=1.1)
        ax.tick_params(axis="both", which="minor", direction="in", right=True, top=True, length=2.5, width=0.8)

        if is_top:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.tick_params(axis="x", which="both", direction="in", bottom=True)

    # Plot Model 1 and Model 5
    plot_model_separation(ax_top, d1, model_id=1, is_top=True)
    plot_model_separation(ax_bot, d5, model_id=5, is_top=False)
    ax_bot.set_xlabel("Time (MJD)", fontsize=11.5)

    # Common shared y-axis label for right panels
    ax_big = fig.add_subplot(gs[:, 1], frameon=False)
    ax_big.tick_params(labelcolor="none", which="both", top=False, bottom=False, left=False, right=False)
    ax_big.grid(False)
    ax_big.set_ylabel(r"Separation ($^{\prime\prime}$)", fontsize=12, labelpad=2)

    # Secondary calendar date axis on top (Bahramian et al. style)
    ax_date = ax_top.twiny()
    ax_date.set_xlim(ax_top.get_xlim())
    date_mjds = np.array([60500, 60600, 60700, 60800, 60900, 61000])
    date_labels = [Time(m, format="mjd").to_datetime().strftime("%Y-%m-%d") for m in date_mjds]
    ax_date.set_xticks(date_mjds)
    ax_date.set_xticklabels(date_labels, rotation=40, ha="left", fontsize=8.5)
    ax_date.tick_params(axis="x", direction="in", length=4, width=1.0)

    # Subplot titles
    ax_top.text(0.02, 0.90, "Ballistic Model (Constant Velocity)", transform=ax_top.transAxes,
                fontsize=10.5, fontweight="bold", verticalalignment="top",
                bbox=dict(boxstyle="square,pad=0.25", facecolor="white", alpha=0.9, edgecolor="none"))

    ax_bot.text(0.02, 0.90, "Continuous Deceleration Model", transform=ax_bot.transAxes,
                fontsize=10.5, fontweight="bold", verticalalignment="top",
                bbox=dict(boxstyle="square,pad=0.25", facecolor="white", alpha=0.9, edgecolor="none"))

    # Unified Bahramian et al. style legend below bottom panel
    handles, labels = ax_bot.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    leg_order = [
        r"$\theta_\parallel^N$", r"$\theta_\parallel^S$",
        r"$\theta_\perp^N$", r"$\theta_\perp^S$",
        "Model Posterior Median", r"Model Posterior $1\sigma$ CI",
        "Upper Limits"
    ]
    final_h = [by_label[l] for l in leg_order if l in by_label]
    final_l = [l for l in leg_order if l in by_label]

    ax_bot.legend(final_h, final_l, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  ncol=4, fontsize=8.8, frameon=True, edgecolor="#cccccc",
                  facecolor="white", framealpha=0.95, columnspacing=1.1, handletextpad=0.35)

    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Successfully generated {out_fig}")

if __name__ == "__main__":
    make_combined_figure()
