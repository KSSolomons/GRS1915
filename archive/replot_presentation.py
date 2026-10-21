import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Load data
data_file = "data/absolute_kinematics_plot_data.npz"
data = np.load(data_file)
flat_samples = data["flat_samples"]
best_model_id = int(data["best_model_id"])
p_Tej = data["p_Tej"]
best_model_res_x = data["best_model_res_x"]
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


def unpack_params_abs(sample, model_id):
    if model_id == 4:
        return (
            sample[0],
            sample[1],
            0.0,
            sample[2],
            0.0,
            sample[3],
            0.0,
            sample[4],
            0.0,
            sample[5],
            sample[6],
            sample[7],
        )
    elif model_id == 2:
        return (
            sample[0],
            sample[1],
            0.0,
            sample[2],
            0.0,
            sample[3],
            sample[4],
            sample[5],
            sample[6],
            sample[7],
            sample[8],
            sample[9],
        )
    elif model_id in [3, 5]:
        return (
            sample[0],
            sample[1],
            sample[2],
            sample[3],
            sample[4],
            sample[5],
            0.0,
            sample[6],
            0.0,
            sample[7],
            sample[8],
            sample[9],
        )
    else:
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


# Recompute best fit params
(
    med_t_ej,
    med_v0_n,
    med_a_n,
    med_v0_s,
    med_a_s,
    med_pa_n,
    med_w_n,
    med_pa_s,
    med_w_s,
    med_theta_cone,
    xc_fit,
    yc_fit,
) = unpack_params_abs(best_model_res_x, best_model_id)

avg_pa_rad = np.radians((med_pa_n + med_pa_s) / 2.0)
std_pa_rad = np.radians(med_theta_cone)

# --- Compute Derived Parameters for Text Box ---
v0_n_chain = flat_samples[:, 1]
v0_s_idx = 2 if best_model_id in [2, 4] else 3
v0_s_chain = flat_samples[:, v0_s_idx]
mu_app_chain = np.abs(v0_s_chain)
mu_rec_chain = np.abs(v0_n_chain)

beta_cos_theta_chain = (mu_app_chain - mu_rec_chain) / (mu_app_chain + mu_rec_chain)
D_kpc_chain = np.random.normal(9.4, 1.0, size=len(mu_app_chain))
D_m_chain = D_kpc_chain * 3.086e19
c = 2.99792458e8

mu_app_rad_s = mu_app_chain * (np.pi / (180.0 * 3600.0)) / 86400.0
beta_sin_theta_chain = mu_app_rad_s * (1.0 - beta_cos_theta_chain) * D_m_chain / c

beta_chain = np.sqrt(beta_cos_theta_chain**2 + beta_sin_theta_chain**2)
theta_deg_chain = np.degrees(np.arctan2(beta_sin_theta_chain, beta_cos_theta_chain))


def compute_hdi(samples, credible_mass=0.6827):
    sorted_samples = np.sort(samples)
    window_size = int(np.ceil(credible_mass * len(sorted_samples)))
    if window_size == 0 or window_size > len(sorted_samples):
        return [sorted_samples[0], np.median(samples), sorted_samples[-1]]
    window_widths = (
        sorted_samples[window_size - 1 :] - sorted_samples[: -window_size + 1]
    )
    min_idx = np.argmin(window_widths)
    return [
        sorted_samples[min_idx],
        np.median(samples),
        sorted_samples[min_idx + window_size - 1],
    ]


p_beta = compute_hdi(beta_chain)
p_theta = compute_hdi(theta_deg_chain)
p_pa = compute_hdi(
    np.radians(flat_samples[:, 5] if best_model_id in [3, 4, 5] else flat_samples[:, 3])
)  # approximate for box

# ---------------------------------------------------------
# AESTHETICS (Elegant Academic)
# ---------------------------------------------------------
plt.style.use("default")
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["axes.linewidth"] = 1.2

COLOR_N = "#008080"  # Elegant Teal
COLOR_S = "#E65100"  # Deep Orange
COLOR_CORE = "black"
COLOR_MODEL = "#424242"  # Dark gray

# ---------------------------------------------------------
# 1. RA/DEC PLOT (Presentation)
# ---------------------------------------------------------
fig = plt.figure(figsize=(7.5, 8))
ax = fig.add_subplot(111)

mjd_min = min(np.min(n_mjd), np.min(s_mjd), np.min(c_mjd)) - 60000
mjd_max = max(np.max(n_mjd), np.max(s_mjd), np.max(c_mjd)) - 60000
norm = Normalize(vmin=mjd_min, vmax=mjd_max)

cmap_unified = plt.cm.viridis

L = 7.0
x_line = np.array([L * np.sin(avg_pa_rad), -L * np.sin(avg_pa_rad)])
y_line = np.array([L * np.cos(avg_pa_rad), -L * np.cos(avg_pa_rad)])
ax.plot(x_line, y_line, color="black", linestyle="--", lw=1.5, zorder=1.5, alpha=0.8)

pa_hi = avg_pa_rad + std_pa_rad
pa_lo = avg_pa_rad - std_pa_rad

ax.fill(
    [0, L * np.sin(pa_lo), L * np.sin(pa_hi)],
    [0, L * np.cos(pa_lo), L * np.cos(pa_hi)],
    color="lightgray",
    alpha=0.4,
    zorder=1,
    edgecolor="none",
)
ax.fill(
    [0, -L * np.sin(pa_lo), -L * np.sin(pa_hi)],
    [0, -L * np.cos(pa_lo), -L * np.cos(pa_hi)],
    color="lightgray",
    alpha=0.4,
    zorder=1,
    edgecolor="none",
)

ax.scatter(
    0,
    0,
    marker="*",
    s=250,
    color="gold",
    edgecolors="black",
    linewidth=1.0,
    zorder=5,
    label="Core",
)

ax.plot(n_da, n_dd, color="gray", linestyle="-", linewidth=1.0, alpha=0.6, zorder=3)
ax.plot(s_da, s_dd, color="gray", linestyle="-", linewidth=1.0, alpha=0.6, zorder=3)

ax.errorbar(
    n_da,
    n_dd,
    xerr=n_da_err,
    yerr=n_dd_err,
    fmt="none",
    ecolor="gray",
    capsize=0,
    zorder=2,
    alpha=0.5,
    elinewidth=1.0,
)
ax.errorbar(
    s_da,
    s_dd,
    xerr=s_da_err,
    yerr=s_dd_err,
    fmt="none",
    ecolor="gray",
    capsize=0,
    zorder=2,
    alpha=0.5,
    elinewidth=1.0,
)

scatter_north = ax.scatter(
    n_da,
    n_dd,
    c=n_mjd - 60000,
    cmap=cmap_unified,
    norm=norm,
    marker="o",
    edgecolors="black",
    linewidth=0.5,
    s=70,
    zorder=4,
    label="Northern Lobe",
)
scatter_south = ax.scatter(
    s_da,
    s_dd,
    c=s_mjd - 60000,
    cmap=cmap_unified,
    norm=norm,
    marker="s",
    edgecolors="black",
    linewidth=0.5,
    s=60,
    zorder=4,
    label="Southern Lobe",
)

ax.set_xlim(-5.0, 5.0)
ax.set_ylim(-5.0, 5.0)
ax.set_xlabel(r"$\Delta\alpha$ (arcsec)", fontsize=16)
ax.set_ylabel(r"$\Delta\delta$ (arcsec)", fontsize=16)
ax.invert_xaxis()

ax.minorticks_on()
ax.tick_params(
    axis="both",
    which="major",
    direction="in",
    top=True,
    right=True,
    labelsize=14,
    length=6,
    width=1.2,
)
ax.tick_params(
    axis="both", which="minor", direction="in", top=True, right=True, length=3, width=1
)

cbar = fig.colorbar(
    ScalarMappable(norm=norm, cmap=cmap_unified),
    ax=ax,
    orientation="vertical",
    pad=0.04,
    aspect=40,
)
cbar.set_label("Time (MJD - 60000)", fontsize=14, rotation=270, labelpad=20)
cbar.ax.tick_params(labelsize=12)

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles=[handles[-3], handles[-2], handles[-1]],
    labels=["Core", "Receding (NW)", "Approaching (SE)"],
    loc="upper left",
    fontsize=15,
    frameon=True,
    edgecolor="black",
)

plt.tight_layout()
plt.savefig("figures/presentation_ra_dec.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------
# 2. SEPARATION PLOT (Presentation)
# ---------------------------------------------------------
fig_sep, ax_sep = plt.subplots(figsize=(10, 6))

t_start = p_Tej[1] - 100
t_end = max(np.max(n_mjd), np.max(s_mjd), np.max(c_mjd)) + 30
t_smooth = np.linspace(t_start, t_end, 500)

models_n = []
models_s = []

inds = np.random.randint(len(flat_samples), size=3000)
for i in inds:
    (
        t_ej_s,
        v0_n_s,
        a_n_s,
        v0_s_s,
        a_s_s,
        pa_0_n_s,
        omega_n_s,
        pa_0_s_s,
        omega_s_s,
        _,
        _,
        _,
    ) = unpack_params_abs(flat_samples[i], best_model_id)
    dt = t_smooth - t_ej_s
    m_n = v0_n_s * dt + 0.5 * a_n_s * dt**2
    m_n[dt < 0] = 0.0
    models_n.append(m_n)

    m_s = v0_s_s * dt + 0.5 * a_s_s * dt**2
    m_s[dt < 0] = 0.0
    models_s.append(m_s)

q_n = np.percentile(np.array(models_n), [16, 50, 84], axis=0)
q_s = np.percentile(np.array(models_s), [16, 50, 84], axis=0)

t_plot = t_smooth[t_smooth >= p_Tej[1]]
mask_plot = t_smooth >= p_Tej[1]

t_plot_line = np.concatenate(([p_Tej[1]], t_plot))
med_n_line = np.concatenate(([0.0], q_n[1][mask_plot]))
med_s_line = np.concatenate(([0.0], q_s[1][mask_plot]))

ax_sep.plot(
    t_plot_line - 60000,
    med_n_line,
    color="black",
    linestyle="-",
    linewidth=2.0,
    zorder=3,
    label="Model Median",
)
ax_sep.fill_between(
    t_smooth - 60000,
    q_n[0],
    q_n[2],
    color="lightgray",
    alpha=0.6,
    zorder=2,
    label="Model 68% HDI",
)
ax_sep.plot(
    t_plot_line - 60000, med_s_line, color="black", linestyle="-", linewidth=2.0, zorder=3
)
ax_sep.fill_between(
    t_smooth - 60000, q_s[0], q_s[2], color="lightgray", alpha=0.6, zorder=2
)

th_n_opt = np.radians(med_pa_n + med_w_n * (n_mjd - med_t_ej))
sin_n_opt, cos_n_opt = np.sin(th_n_opt), np.cos(th_n_opt)
n_par_opt = n_da * sin_n_opt + n_dd * cos_n_opt
n_perp_opt = n_da * cos_n_opt - n_dd * sin_n_opt
n_par_err_opt = np.sqrt((n_da_err * sin_n_opt) ** 2 + (n_dd_err * cos_n_opt) ** 2)
n_perp_err_opt = np.sqrt((n_da_err * cos_n_opt) ** 2 + (n_dd_err * sin_n_opt) ** 2)

th_s_opt = np.radians(med_pa_s + med_w_s * (s_mjd - med_t_ej))
sin_s_opt, cos_s_opt = np.sin(th_s_opt), np.cos(th_s_opt)
s_par_opt = s_da * sin_s_opt + s_dd * cos_s_opt
s_perp_opt = s_da * cos_s_opt - s_dd * sin_s_opt
s_par_err_opt = np.sqrt((s_da_err * sin_s_opt) ** 2 + (s_dd_err * cos_s_opt) ** 2)
s_perp_err_opt = np.sqrt((s_da_err * cos_s_opt) ** 2 + (s_dd_err * sin_s_opt) ** 2)

ax_sep.errorbar(
    n_mjd - 60000,
    n_perp_opt,
    yerr=n_perp_err_opt,
    fmt="x",
    color="darkgray",
    markersize=6,
    zorder=1,
    capsize=0,
    label=r"Transverse ($\Delta\theta_\perp^N$)",
)
ax_sep.errorbar(
    s_mjd - 60000,
    s_perp_opt,
    yerr=s_perp_err_opt,
    fmt="+",
    color="gray",
    markersize=7,
    zorder=1,
    capsize=0,
    label=r"Transverse ($\Delta\theta_\perp^S$)",
)

ax_sep.plot(
    n_mjd - 60000,
    n_par_opt,
    color=COLOR_N,
    linestyle="-",
    linewidth=1.5,
    alpha=0.5,
    zorder=3,
)
ax_sep.plot(
    s_mjd - 60000,
    s_par_opt,
    color=COLOR_S,
    linestyle="-",
    linewidth=1.5,
    alpha=0.5,
    zorder=3,
)

ax_sep.errorbar(
    n_mjd - 60000,
    n_par_opt,
    yerr=n_par_err_opt,
    fmt="o",
    color=COLOR_N,
    markeredgecolor="black",
    markeredgewidth=1.0,
    markersize=9,
    zorder=4,
    ecolor="black",
    capsize=0,
    elinewidth=1.2,
    label=r"Receding ($\Delta\theta_\parallel^N$)",
)

ax_sep.errorbar(
    s_mjd - 60000,
    s_par_opt,
    yerr=s_par_err_opt,
    fmt="s",
    color=COLOR_S,
    markeredgecolor="black",
    markeredgewidth=1.0,
    markersize=8,
    zorder=4,
    ecolor="black",
    capsize=0,
    elinewidth=1.2,
    label=r"Approaching ($\Delta\theta_\parallel^S$)",
)

ax_sep.axhline(0, color="gray", linestyle=":", linewidth=1.5, zorder=0)



ax_sep.set_ylabel(r"Separation ($^{\prime\prime}$)", fontsize=16)
ax_sep.set_xlabel("Time (MJD - 60000)", fontsize=16)
ax_sep.set_xlim(t_start - 60000, t_end - 60000)
ax_sep.set_ylim(-6.2, 5.0)

ax_sep.minorticks_on()
ax_sep.tick_params(
    axis="both",
    which="major",
    direction="in",
    right=True,
    top=True,
    labelsize=14,
    length=6,
    width=1.2,
)
ax_sep.tick_params(
    axis="both", which="minor", direction="in", right=True, top=True, length=3, width=1
)

handles, labels = ax_sep.get_legend_handles_labels()
unique = [
    (h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]
]
ax_sep.legend(
    *zip(*unique),
    loc="lower left",
    ncol=3,
    fontsize=13,
    frameon=True,
    edgecolor="black"
)

plt.tight_layout()
plt.savefig("figures/presentation_separation.png", dpi=300)
plt.close(fig_sep)

print("Generated presentation plots!")
