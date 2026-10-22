import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from epoch_limits import blending_limits

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


def compute_hdi(samples, credible_mass=0.90):
    valid_samples = samples[np.isfinite(samples)]
    if len(valid_samples) == 0:
        return [np.nan, np.nan, np.nan]
    sorted_samples = np.sort(valid_samples)
    window_size = int(np.ceil(credible_mass * len(sorted_samples)))
    if window_size == 0 or window_size > len(sorted_samples):
        return [sorted_samples[0], np.median(valid_samples), sorted_samples[-1]]
    window_widths = (
        sorted_samples[window_size - 1 :] - sorted_samples[: -window_size + 1]
    )
    min_idx = np.argmin(window_widths)
    hdi_lower = sorted_samples[min_idx]
    hdi_upper = sorted_samples[min_idx + window_size - 1]
    return [hdi_lower, np.median(valid_samples), hdi_upper]


def calc_trajectory(t, t_ej, v1, v2, t_break):
    """
    Computes trajectory r(t) for piecewise linear motion.
    If v1 == v2 or t_break >= np.max(t), it reduces exactly to pure ballistic.
    """
    t = np.asarray(t)
    dt = t - t_ej
    r = np.zeros_like(dt, dtype=float)
    mask1 = (t >= t_ej) & (t <= t_break)
    r[mask1] = v1 * (t[mask1] - t_ej)
    mask2 = t > t_break
    r_break = v1 * max(0.0, t_break - t_ej)
    r[mask2] = r_break + v2 * (t[mask2] - t_break)
    return r


def calc_trajectory_smooth(t, t_ej, v0, tau):
    """
    Computes trajectory r(t) for smooth continuous deceleration to terminal stall radius:
    r(t) = (v0 * tau) * (1 - exp(-(t - t_ej) / tau)) for t >= t_ej, 0 otherwise.
    For southern approaching lobe, v0 is negative.
    """
    t = np.asarray(t)
    dt = t - t_ej
    r = np.zeros_like(dt, dtype=float)
    mask = t >= t_ej
    r[mask] = (v0 * tau) * (1.0 - np.exp(-(t[mask] - t_ej) / tau))
    return r


def unpack_params_smooth(sample, model_id=5, anchor_date=60600.0):
    """
    Unpacks parameter vector for Models 5-6 into:
    (t_ej, v0_n, tau_n, v0_s, tau_s, pa_n, pa_s, theta_scatter, x_c, y_c)
    """
    n = len(sample)
    if n == 10:
        return (
            sample[0],
            sample[1],
            sample[2],
            sample[3],
            sample[4],
            sample[5],
            sample[6],
            sample[7],
            sample[8],
            sample[9],
        )
    elif n == 9:
        return (
            anchor_date,
            sample[0],
            sample[1],
            sample[2],
            sample[3],
            sample[4],
            sample[5],
            sample[6],
            sample[7],
            sample[8],
        )
    else:
        raise ValueError(f"Unexpected parameter count {n} for smooth deceleration model.")


def unpack_params_abs(sample, model_id, anchor_date=60600.0):
    """
    Unpacks parameter vector for Models 1-4 into:
    (t_ej, v1_n, v2_n, tb_n, v1_s, v2_s, tb_s, pa_n, pa_s, theta_scatter, x_c, y_c)
    """
    if model_id in [5, 6]:
        return unpack_params_smooth(sample, model_id, anchor_date=anchor_date)
    n = len(sample)
    if n == 12:
        # Piecewise with sampled Tej (Model 3 or Model 4 with Gaussian anchor):
        # [t_ej, v1n, v2n, tb_n, v1s, v2s, tb_s, pa_n, pa_s, theta_scatter, x_c, y_c]
        return (
            sample[0],
            sample[1],
            sample[2],
            sample[3],
            sample[4],
            sample[5],
            sample[6],
            sample[7],
            sample[8],
            sample[9],
            sample[10],
            sample[11],
        )
    elif n == 11:
        # Piecewise with hard-anchored Tej (Model 4):
        # [v1n, v2n, tb_n, v1s, v2s, tb_s, pa_n, pa_s, theta_scatter, x_c, y_c]
        return (
            anchor_date,
            sample[0],
            sample[1],
            sample[2],
            sample[3],
            sample[4],
            sample[5],
            sample[6],
            sample[7],
            sample[8],
            sample[9],
            sample[10],
        )
    elif n == 8:
        # Ballistic with sampled Tej (Model 1 or Model 2 with Gaussian anchor):
        # [t_ej, vn, vs, pa_n, pa_s, theta_scatter, x_c, y_c]
        return (
            sample[0],
            sample[1],
            sample[1],
            1e6,
            sample[2],
            sample[2],
            1e6,
            sample[3],
            sample[4],
            sample[5],
            sample[6],
            sample[7],
        )
    elif n == 7:
        # Ballistic with hard-anchored Tej (Model 2):
        # [vn, vs, pa_n, pa_s, theta_scatter, x_c, y_c]
        return (
            anchor_date,
            sample[0],
            sample[0],
            1e6,
            sample[1],
            sample[1],
            1e6,
            sample[2],
            sample[3],
            sample[4],
            sample[5],
            sample[6],
        )
    elif model_id == 3:
        return (
            sample[0],
            sample[1],
            sample[2],
            sample[3],
            sample[4],
            sample[5],
            sample[6],
            sample[7],
            sample[8],
            sample[9],
            sample[10],
            sample[11],
        )
    elif model_id == 4:
        return (
            anchor_date,
            sample[0],
            sample[1],
            sample[2],
            sample[3],
            sample[4],
            sample[5],
            sample[6],
            sample[7],
            sample[8],
            sample[9],
            sample[10],
        )
    else:
        return (
            sample[0],
            sample[1],
            sample[1],
            1e6,
            sample[2],
            sample[2],
            1e6,
            sample[3],
            sample[4],
            sample[5],
            sample[6],
            sample[7],
        )

def plot_absolute_separation(
    with_limits=False,
    show_limits=None,
    model_id=None,
    data_file=None,
    out_fig=None,
    flat_samples=None,
    best_model_id=None,
    p_Tej=None,
    best_model_res_x=None,
    anchor_date=60600.0,
    n_mjd=None,
    n_da=None,
    n_dd=None,
    n_da_err=None,
    n_dd_err=None,
    s_mjd=None,
    s_da=None,
    s_dd=None,
    s_da_err=None,
    s_dd_err=None,
    c_mjd=None,
):
    if show_limits is None:
        show_limits = with_limits

    if flat_samples is None:
        if data_file is None:
            lim_str = "_with_limits" if with_limits else "_without_limits"
            if model_id is not None:
                candidate = f"data/absolute_kinematics_plot_data_model{model_id}{lim_str}.npz"
                if os.path.exists(candidate):
                    data_file = candidate
                else:
                    data_file = f"data/absolute_kinematics_plot_data{lim_str}.npz"
            else:
                found = None
                for m in [4, 3, 2, 1]:
                    c = f"data/absolute_kinematics_plot_data_model{m}{lim_str}.npz"
                    if os.path.exists(c):
                        found = c
                        break
                if found:
                    data_file = found
                else:
                    data_file = (
                        f"data/absolute_kinematics_plot_data{lim_str}.npz"
                        if os.path.exists(f"data/absolute_kinematics_plot_data{lim_str}.npz")
                        else "data/absolute_kinematics_plot_data.npz"
                    )
        if not os.path.exists(data_file):
            print(
                f"Error: {data_file} not found. Please run the absolute kinematics pipeline first to generate the data."
            )
            return None

        data = np.load(data_file)
        flat_samples = data["flat_samples"]
        best_model_id = int(data["best_model_id"])
        p_Tej = data["p_Tej"]
        best_model_res_x = data["best_model_res_x"]
        anchor_date = float(data["anchor_date"]) if "anchor_date" in data else 60600.0
        n_mjd = data["n_mjd"]
        n_da = data["n_da"]
        n_dd = data["n_dd"]
        n_da_err = data["n_da_err"]
        n_dd_err = data["n_dd_err"]
        s_mjd = data["s_mjd"]
        s_da = data["s_da"]
        s_dd = data["s_dd"]
        s_da_err = data["s_da_err"]
        s_dd_err = data["s_dd_err"]
        c_mjd = data["c_mjd"]

    print("Generating Parallel and Perpendicular Separation Plot...")

    fig_sep, ax_sep = plt.subplots(figsize=(10, 6))
    ax_sep.axhline(0, color="darkgrey", linestyle=":", linewidth=1.0)

    # t_start = (
    #     min(60400 60550.0)
    #     if best_model_id in [2, 4]
    #     else min(p_Tej[1] - 50.0, 60300.0)
    # )
    t_start = 60400
    t_end = max(np.max(n_mjd), np.max(s_mjd), np.max(c_mjd)) + 30
    t_smooth = np.linspace(t_start, t_end, 500)

    models_n = []
    models_s = []

    inds = np.random.randint(len(flat_samples), size=5000)
    for i in inds:
        if best_model_id in [5, 6]:
            (
                t_ej_s,
                v0_n_s,
                tau_n_s,
                v0_s_s,
                tau_s_s,
                pa_n_s,
                pa_s_s,
                theta_scatter_s,
                x_c_s,
                y_c_s,
            ) = unpack_params_smooth(flat_samples[i], best_model_id, anchor_date=anchor_date)
            m_n = calc_trajectory_smooth(t_smooth, t_ej_s, v0_n_s, tau_n_s)
            m_s = calc_trajectory_smooth(t_smooth, t_ej_s, v0_s_s, tau_s_s)
        else:
            (
                t_ej_s,
                v1_n_s,
                v2_n_s,
                tb_n_s,
                v1_s_s,
                v2_s_s,
                tb_s_s,
                pa_n_s,
                pa_s_s,
                theta_scatter_s,
                x_c_s,
                y_c_s,
            ) = unpack_params_abs(flat_samples[i], best_model_id, anchor_date=anchor_date)
            m_n = calc_trajectory(t_smooth, t_ej_s, v1_n_s, v2_n_s, tb_n_s)
            m_s = calc_trajectory(t_smooth, t_ej_s, v1_s_s, v2_s_s, tb_s_s)
        models_n.append(m_n)
        models_s.append(m_s)

    # Equal-tailed 1-sigma equivalent credible band (15.87th-84.13th percentile, ~68.3% CI).
    # A column-wise HDI is NOT used here: with a soft/free T_ej, samples not yet launched
    # pile up at exactly r=0, and the HDI discontinuously jumps once that point-mass
    # fraction drops below (1 - credible_mass). Percentile bands evolve smoothly in time.
    q_n = np.percentile(np.array(models_n), [15.865, 50, 84.135], axis=0)
    q_s = np.percentile(np.array(models_s), [15.865, 50, 84.135], axis=0)

    mcmc_med_p = np.median(flat_samples, axis=0)
    if best_model_id in [5, 6]:
        (
            med_t_ej,
            med_v0_n,
            med_tau_n,
            med_v0_s,
            med_tau_s,
            med_pa_n,
            med_pa_s,
            med_scat,
            med_xc,
            med_yc,
        ) = unpack_params_smooth(mcmc_med_p, best_model_id, anchor_date=anchor_date)
        med_rmax_n = med_v0_n * med_tau_n
        med_rmax_s = med_v0_s * med_tau_s
    else:
        (
            med_t_ej,
            med_v1_n,
            med_v2_n,
            med_tb_n,
            med_v1_s,
            med_v2_s,
            med_tb_s,
            med_pa_n,
            med_pa_s,
            med_scat,
            med_xc,
            med_yc,
        ) = unpack_params_abs(mcmc_med_p, best_model_id, anchor_date=anchor_date)

    t_plot = t_smooth[t_smooth >= med_t_ej]
    mask_plot = t_smooth >= med_t_ej

    # Use pointwise posterior median from MCMC ensemble trajectories, anchored at ejection time
    t_plot_line = np.concatenate(([med_t_ej], t_plot))
    med_n_line = np.concatenate(([0.0], q_n[1][mask_plot]))
    med_s_line = np.concatenate(([0.0], q_s[1][mask_plot]))

    ax_sep.plot(
        t_plot_line - 60000,
        med_n_line,
        color="orange",
        linestyle="-",
        linewidth=1.5,
        zorder=1.5,
        label="Model Posterior Median",
    )
    ax_sep.fill_between(
        t_smooth - 60000,
        q_n[0],
        q_n[2],
        color="navajowhite",
        alpha=0.5,
        zorder=1.4,
        label=r"Model Posterior $1\sigma$ CI",
    )

    ax_sep.plot(
        t_plot_line - 60000, med_s_line, color="orange", linestyle="-", linewidth=1.5, zorder=1.5
    )
    ax_sep.fill_between(
        t_smooth - 60000, q_s[0], q_s[2], color="navajowhite", alpha=0.5, zorder=1.4
    )

    if best_model_id in [3, 4]:
        ax_sep.axvline(
            med_tb_n - 60000,
            color="royalblue",
            linestyle="--",
            linewidth=1.2,
            zorder=1.3,
            label=f"North Break ($t={med_tb_n:.0f}$)",
        )
        ax_sep.axvline(
            med_tb_s - 60000,
            color="crimson",
            linestyle="--",
            linewidth=1.2,
            zorder=1.3,
            label=f"South Break ($t={med_tb_s:.0f}$)",
        )
    elif best_model_id in [5, 6]:
        ax_sep.axhline(
            med_rmax_n,
            color="royalblue",
            linestyle=":",
            linewidth=1.2,
            zorder=1.3,
            label=f"North Stall ($r_\\infty={med_rmax_n:.2f}''$)",
        )
        ax_sep.axhline(
            med_rmax_s,
            color="crimson",
            linestyle=":",
            linewidth=1.2,
            zorder=1.3,
            label=f"South Stall ($r_\\infty={med_rmax_s:.2f}''$)",
        )

    th_n_opt = np.radians(med_pa_n)
    sin_n_opt, cos_n_opt = np.sin(th_n_opt), np.cos(th_n_opt)
    n_par_opt = n_da * sin_n_opt + n_dd * cos_n_opt
    n_perp_opt = n_da * cos_n_opt - n_dd * sin_n_opt
    n_par_err_opt = np.sqrt((n_da_err * sin_n_opt) ** 2 + (n_dd_err * cos_n_opt) ** 2)
    n_perp_err_opt = np.sqrt((n_da_err * cos_n_opt) ** 2 + (n_dd_err * sin_n_opt) ** 2)

    th_s_opt = np.radians(med_pa_s)
    sin_s_opt, cos_s_opt = np.sin(th_s_opt), np.cos(th_s_opt)
    s_par_opt = s_da * sin_s_opt + s_dd * cos_s_opt
    s_perp_opt = s_da * cos_s_opt - s_dd * sin_s_opt
    s_par_err_opt = np.sqrt((s_da_err * sin_s_opt) ** 2 + (s_dd_err * cos_s_opt) ** 2)
    s_perp_err_opt = np.sqrt((s_da_err * cos_s_opt) ** 2 + (s_dd_err * sin_s_opt) ** 2)

    ax_sep.errorbar(
        n_mjd - 60000,
        n_par_opt,
        yerr=n_par_err_opt,
        fmt="^",
        color="dodgerblue",
        markeredgecolor="black",
        markersize=9,
        zorder=2,
        ecolor="black",
        capsize=0,
        elinewidth=1.0,
        label=r"$\Delta\theta_\parallel^N$",
    )
    ax_sep.errorbar(
        s_mjd - 60000,
        s_par_opt,
        yerr=s_par_err_opt,
        fmt="v",
        color="crimson",
        markeredgecolor="black",
        markersize=9,
        zorder=2,
        ecolor="black",
        capsize=0,
        elinewidth=1.0,
        label=r"$\Delta\theta_\parallel^S$",
    )

    ax_sep.errorbar(
        n_mjd - 60000,
        n_perp_opt,
        yerr=n_perp_err_opt,
        fmt="^",
        color="lightgray",
        markeredgecolor="lightgray",
        markersize=9,
        zorder=1,
        ecolor="lightgray",
        capsize=0,
        elinewidth=1.0,
        label=r"$\Delta\theta_\perp^N$",
    )
    ax_sep.errorbar(
        s_mjd - 60000,
        s_perp_opt,
        yerr=s_perp_err_opt,
        fmt="v",
        color="lightgray",
        markeredgecolor="lightgray",
        markersize=9,
        zorder=1,
        ecolor="lightgray",
        capsize=0,
        elinewidth=1.0,
        label=r"$\Delta\theta_\perp^S$",
    )

    # Upper Limits (only drawn if show_limits is True)
    if show_limits:
        # Blending limits of the blocks flagged "in_fit" in epoch_limits.py (projected beam FWHM
        # along the jet axis), with error bars sigma_pos = theta_beam / 6 (positional accuracy at 3 sigma)
        ul_mjd, ul_r, ul_sigma_pos = blending_limits()

        ax_sep.errorbar(
            ul_mjd - 60000,
            ul_r,
            yerr=ul_sigma_pos,
            uplims=True,
            fmt="none",
            color="black",
            ecolor="black",
            capsize=3,
            elinewidth=1.5,
            zorder=5,
            label="Upper Limits",
        )
        ax_sep.errorbar(
            ul_mjd - 60000,
            -ul_r,
            yerr=ul_sigma_pos,
            lolims=True,
            fmt="none",
            color="black",
            ecolor="black",
            capsize=3,
            elinewidth=1.5,
            zorder=5,
        )

    ax_sep.set_ylabel(r"Separation ($^{\prime\prime}$)", fontsize=18)
    ax_sep.set_xlabel("Time (MJD - 60000)", fontsize=18)
    ax_sep.set_xlim(t_start - 60000, t_end - 60000)
    ax_sep.set_ylim(-6.2, 5.0)

    handles, labels = ax_sep.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    preferred_order = [
        r"$\Delta\theta_\parallel^N$",
        r"$\Delta\theta_\parallel^S$",
        r"$\Delta\theta_\perp^N$",
        r"$\Delta\theta_\perp^S$",
        "Model Posterior Median",
        r"Model Posterior $1\sigma$ CI",
        "Model Posterior 90% CI",
        "Upper Limits",
    ]
    for lbl in list(by_label.keys()):
        if "Decel Break" in lbl:
            preferred_order.append(lbl)

    final_handles = [by_label[lbl] for lbl in preferred_order if lbl in by_label]
    final_labels = [lbl for lbl in preferred_order if lbl in by_label]

    ax_sep.legend(
        final_handles,
        final_labels,
        loc="lower left",
        ncol=4 if len(final_handles) >= 7 else 3,
        fontsize=11,
    )

    ax_sep.minorticks_on()
    ax_sep.tick_params(
        axis="both",
        which="major",
        direction="in",
        right=True,
        top=True,
        labelsize=14,
        length=7,
        width=1.2,
    )
    ax_sep.tick_params(
        axis="both",
        which="minor",
        direction="in",
        right=True,
        top=True,
        length=3.5,
        width=1,
    )
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    if out_fig is None:
        model_str = f"_model{best_model_id}" if best_model_id is not None else ""
        lim_str = "_with_limits" if with_limits else "_without_limits"
        base_name = f"figures/absolute_parallel_perp_fit_plot{model_str}{lim_str}"
        if show_limits == with_limits:
            out_fig = f"{base_name}.png"
        elif show_limits:
            out_fig = f"{base_name}_with_arrows.png"
        else:
            out_fig = f"{base_name}_no_arrows.png"
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig_sep)
    print(f"Saved separation plot to {out_fig}")
    return out_fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Parallel and Perpendicular Separation"
    )
    parser.add_argument(
        "--with-limits",
        action="store_true",
        help="Use MCMC run data with limits",
    )
    parser.add_argument(
        "--model-id",
        type=int,
        default=None,
        choices=[1, 2, 3, 4, 5, 6],
        help="Model ID to plot (default: None, auto-detected from data file)",
    )
    parser.add_argument(
        "--show-limits",
        action="store_true",
        default=None,
        help="Explicitly show upper limits arrows on the plot",
    )
    parser.add_argument(
        "--hide-limits",
        action="store_true",
        default=False,
        help="Explicitly hide upper limits arrows on the plot",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to .npz data file",
    )
    parser.add_argument(
        "--out-fig",
        type=str,
        default=None,
        help="Custom output figure file path",
    )
    args = parser.parse_args()

    if args.hide_limits:
        show_limits = False
    elif args.show_limits is not None:
        show_limits = args.show_limits
    else:
        show_limits = args.with_limits

    plot_absolute_separation(
        with_limits=args.with_limits,
        show_limits=show_limits,
        model_id=args.model_id,
        data_file=args.data_file,
        out_fig=args.out_fig,
    )
