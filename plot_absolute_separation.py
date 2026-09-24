import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.time import Time

def unpack_params_abs(sample, model_id):
    if model_id == 4:
        return sample[0], sample[1], 0.0, sample[2], 0.0, sample[3], 0.0, sample[4], 0.0, sample[5], sample[6], sample[7]
    elif model_id == 2:
        return sample[0], sample[1], 0.0, sample[2], 0.0, sample[3], sample[4], sample[5], sample[6], sample[7], sample[8], sample[9]
    elif model_id == 3:
        return sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], 0.0, sample[6], 0.0, sample[7], sample[8], sample[9]
    else:
        return sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], sample[7], sample[8], sample[9], sample[10], sample[11]

# Load data
data_file = '/media/kyle/kyle_phd/GRS1915/data/absolute_kinematics_plot_data.npz'
if not os.path.exists(data_file):
    print(f"Error: {data_file} not found. Please run pipeline_absolute_kinematics.py first to generate the data.")
    exit(1)

data = np.load(data_file)
flat_samples = data['flat_samples']
best_model_id = int(data['best_model_id'])
p_Tej = data['p_Tej']
best_model_res_x = data['best_model_res_x']
n_mjd = data['n_mjd']
n_da = data['n_da']
n_dd = data['n_dd']
n_da_err = data['n_da_err']
n_dd_err = data['n_dd_err']
s_mjd = data['s_mjd']
s_da = data['s_da']
s_dd = data['s_dd']
s_da_err = data['s_da_err']
s_dd_err = data['s_dd_err']
c_mjd = data['c_mjd']

print("Generating Parallel and Perpendicular Separation Plot...")

fig_sep, ax_sep = plt.subplots(figsize=(10, 6))
ax_sep.axhline(0, color='grey', linestyle='-', linewidth=1.5, alpha=0.5)

t_start = p_Tej[1] - 30
t_end = max(np.max(n_mjd), np.max(s_mjd), np.max(c_mjd)) + 30
t_smooth = np.linspace(t_start, t_end, 500)

models_n = []
models_s = []

inds = np.random.randint(len(flat_samples), size=5000)
for i in inds:
    t_ej_s, v0_n_s, a_n_s, v0_s_s, a_s_s, pa_0_n_s, omega_n_s, pa_0_s_s, omega_s_s, _, _, _ = unpack_params_abs(flat_samples[i], best_model_id)
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
dt_plot = t_plot - p_Tej[1]
med_t_ej, med_v0_n, med_a_n, med_v0_s, med_a_s, med_pa_n, med_w_n, med_pa_s, med_w_s, _, _, _ = unpack_params_abs(best_model_res_x, best_model_id)

med_n = med_v0_n * dt_plot + 0.5 * med_a_n * dt_plot**2
med_s = med_v0_s * dt_plot + 0.5 * med_a_s * dt_plot**2

ax_sep.plot(t_plot, med_n, color='orange', linestyle='-', linewidth=1.5, zorder=1.5, label='Model Posterior Median')
ax_sep.fill_between(t_smooth, q_n[0], q_n[2], color='navajowhite', alpha=0.5, zorder=1.4, label='Model Posterior 68% HDI')

ax_sep.plot(t_plot, med_s, color='orange', linestyle='-', linewidth=1.5, zorder=1.5)
ax_sep.fill_between(t_smooth, q_s[0], q_s[2], color='navajowhite', alpha=0.5, zorder=1.4)

th_n_opt = np.radians(med_pa_n + med_w_n * (n_mjd - med_t_ej))
sin_n_opt, cos_n_opt = np.sin(th_n_opt), np.cos(th_n_opt)
n_par_opt = n_da * sin_n_opt + n_dd * cos_n_opt
n_perp_opt = n_da * cos_n_opt - n_dd * sin_n_opt
n_par_err_opt = np.sqrt((n_da_err * sin_n_opt)**2 + (n_dd_err * cos_n_opt)**2)
n_perp_err_opt = np.sqrt((n_da_err * cos_n_opt)**2 + (n_dd_err * sin_n_opt)**2)

th_s_opt = np.radians(med_pa_s + med_w_s * (s_mjd - med_t_ej))
sin_s_opt, cos_s_opt = np.sin(th_s_opt), np.cos(th_s_opt)
s_par_opt = s_da * sin_s_opt + s_dd * cos_s_opt
s_perp_opt = s_da * cos_s_opt - s_dd * sin_s_opt
s_par_err_opt = np.sqrt((s_da_err * sin_s_opt)**2 + (s_dd_err * cos_s_opt)**2)
s_perp_err_opt = np.sqrt((s_da_err * cos_s_opt)**2 + (s_dd_err * sin_s_opt)**2)

ax_sep.errorbar(n_mjd, n_par_opt, yerr=n_par_err_opt, fmt='^', color='dodgerblue', markeredgecolor='black', markersize=6, zorder=2, ecolor='black', capsize=3, elinewidth=1.5, label=r'$\theta_\parallel^N$')
ax_sep.errorbar(s_mjd, s_par_opt, yerr=s_par_err_opt, fmt='v', color='red', markeredgecolor='black', markersize=6, zorder=2, ecolor='black', capsize=3, elinewidth=1.5, label=r'$\theta_\parallel^S$')

ax_sep.errorbar(n_mjd, n_perp_opt, yerr=n_perp_err_opt, fmt='^', color='lightgray', markeredgecolor='lightgray', markersize=6, zorder=1, ecolor='lightgray', capsize=3, elinewidth=1.5, label=r'$\theta_\perp^N$')
ax_sep.errorbar(s_mjd, s_perp_opt, yerr=s_perp_err_opt, fmt='v', color='lightgray', markeredgecolor='lightgray', markersize=6, zorder=1, ecolor='lightgray', capsize=3, elinewidth=1.5, label=r'$\theta_\perp^S$')

ax2 = ax_sep.twiny()
t_start_date = Time(t_start, format='mjd').datetime
t_end_date = Time(t_end, format='mjd').datetime
ax2.set_xlim([t_start, t_end])
times_smooth_dt = Time(t_smooth, format='mjd').datetime
ax2.plot(times_smooth_dt, np.zeros_like(t_smooth), alpha=0)
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax2.tick_params(axis='x', rotation=45, labelsize=12)

ax_sep.set_ylabel(r'Separation ($^{\prime\prime}$)', fontsize=14)
ax_sep.set_xlabel('Time (MJD)', fontsize=14)
ax_sep.set_xlim(t_start, t_end)
ax_sep.set_ylim(-5, 5)

handles, labels = ax_sep.get_legend_handles_labels()
order = [4, 2, 0, 5, 3, 1] 
ax_sep.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=12)
ax_sep.minorticks_on()
ax_sep.tick_params(axis='both', which='major', direction='in', right=True, labelsize=12, length=6, width=1.2)
ax_sep.tick_params(axis='both', which='minor', direction='in', right=True, length=3, width=1)
plt.tight_layout()
os.makedirs('/media/kyle/kyle_phd/GRS1915/figures', exist_ok=True)
plt.savefig('/media/kyle/kyle_phd/GRS1915/figures/absolute_parallel_perp_fit_plot.png', dpi=300, bbox_inches='tight')
plt.close(fig_sep)
print("Saved separation plot to /media/kyle/kyle_phd/GRS1915/figures/absolute_parallel_perp_fit_plot.png")
