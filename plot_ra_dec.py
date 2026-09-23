import os
import glob
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator

# Set typography styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# 1. Load Data
dir_path = '/media/kyle/kyle_phd/GRS1915/S-band-fits'
files = glob.glob(os.path.join(dir_path, '*2D_Fitting*.txt'))

core_ras_deg = []
core_decs_deg = []
core_ra_errs_sec = []
core_dec_errs_arcsec = []
epochs_data = []

for f in sorted(files):
    filename = os.path.basename(f)
    m = re.search(r'img_(\d+)', filename)
    if not m: continue
    ts = int(m.group(1))
    mjd = ts / 86400.0 + 40587.0
    
    with open(f, 'r') as fp:
        lines = fp.readlines()
    
    components = []
    curr_comp = None
    for line in lines:
        if line.startswith('Component #'):
            curr_comp = {'name': line.strip()}
            components.append(curr_comp)
        elif 'Center X' in line and curr_comp is not None:
            m_ra = re.search(r'=\s*(\d+):(\d+):(\d+\.\d+)\s*±\s*(\d+\.\d+(?:[eE][+-]?\d+)?)', line)
            if m_ra: 
                h, m_time, s, err_s = float(m_ra.group(1)), float(m_ra.group(2)), float(m_ra.group(3)), float(m_ra.group(4))
                curr_comp['ra_deg'] = (h + m_time/60.0 + s/3600.0) * 15.0
                curr_comp['ra_err'] = err_s
        elif 'Center Y' in line and curr_comp is not None:
            m_dec = re.search(r'=\s*([+-]?\d+):(\d+):(\d+\.\d+)\s*±\s*(\d+\.\d+(?:[eE][+-]?\d+)?)', line)
            if m_dec:
                d, m_arc, s_arc, err_arcsec = float(m_dec.group(1)), float(m_dec.group(2)), float(m_dec.group(3)), float(m_dec.group(4))
                sign = -1 if '-' in m_dec.group(1) else 1
                curr_comp['dec_deg'] = sign * (abs(d) + m_arc/60.0 + s_arc/3600.0)
                curr_comp['dec_err'] = err_arcsec
            
    epoch_comps = []
    epoch_core_ra = None
    epoch_core_dec = None
    epoch_core_ra_err = None
    epoch_core_dec_err = None
    
    for c in components:
        ra_deg = c.get('ra_deg')
        dec_deg = c.get('dec_deg')
        ra_err = c.get('ra_err')
        dec_err = c.get('dec_err')
        if ra_deg is not None and dec_deg is not None:
            epoch_comps.append({'ra_deg': ra_deg, 'dec_deg': dec_deg, 'ra_err': ra_err, 'dec_err': dec_err})
            if 288.7979 < ra_deg < 288.7982:
                epoch_core_ra = ra_deg
                epoch_core_dec = dec_deg
                epoch_core_ra_err = ra_err
                epoch_core_dec_err = dec_err
                core_ras_deg.append(ra_deg)
                core_decs_deg.append(dec_deg)
                core_ra_errs_sec.append(ra_err)
                core_dec_errs_arcsec.append(dec_err)
                
    epochs_data.append({
        'mjd': mjd, 
        'comps': epoch_comps, 
        'core_ra': epoch_core_ra, 
        'core_dec': epoch_core_dec,
        'core_ra_err': epoch_core_ra_err,
        'core_dec_err': epoch_core_dec_err
    })

act_ra_deg = (19 + 15/60.0 + 11.558/3600.0) * 15.0
act_dec_deg = 10 + 56/60.0 + 44.905/3600.0

north_pts = []
south_pts = []
core_pts = []

# Sort epochs by time and discard first two early unresolved epochs
epochs_data.sort(key=lambda x: x['mjd'])
epochs_data = epochs_data[2:]

for epoch in epochs_data:
    mjd = epoch['mjd']
    ref_ra = act_ra_deg
    ref_dec = act_dec_deg
    cos_dec = np.cos(np.radians(ref_dec))
    
    for c in epoch['comps']:
        ra_deg = c['ra_deg']
        dec_deg = c['dec_deg']
        ra_err = c['ra_err']
        dec_err = c['dec_err']
        
        d_alpha = (ra_deg - ref_ra) * 3600.0 * cos_dec
        d_delta = (dec_deg - ref_dec) * 3600.0
        
        sigma_alpha_lobe = ra_err * 15.0 * cos_dec
        sigma_delta_lobe = dec_err
        
        if 288.7979 < ra_deg < 288.7982:
            # Filter the outlier core point as in kinematics script
            if np.abs(d_alpha - -0.317) < 0.01 and np.abs(d_delta - -0.601) < 0.01:
                continue
            core_pts.append((mjd, d_alpha, d_delta, sigma_alpha_lobe, sigma_delta_lobe))
            continue 
        
        if d_delta > 0:
            north_pts.append((mjd, d_alpha, d_delta, sigma_alpha_lobe, sigma_delta_lobe))
        else:
            south_pts.append((mjd, d_alpha, d_delta, sigma_alpha_lobe, sigma_delta_lobe))

north_pts.sort(key=lambda x: x[0])
south_pts.sort(key=lambda x: x[0])

n_mjd = np.array([p[0] for p in north_pts])
n_da = np.array([p[1] for p in north_pts])
n_dd = np.array([p[2] for p in north_pts])
n_da_err = np.array([p[3] for p in north_pts])
n_dd_err = np.array([p[4] for p in north_pts])

s_mjd = np.array([p[0] for p in south_pts])
s_da = np.array([p[1] for p in south_pts])
s_dd = np.array([p[2] for p in south_pts])
s_da_err = np.array([p[3] for p in south_pts])
s_dd_err = np.array([p[4] for p in south_pts])

c_mjd = np.array([p[0] for p in core_pts])
c_da = np.array([p[1] for p in core_pts])
c_dd = np.array([p[2] for p in core_pts])
c_da_err = np.array([p[3] for p in core_pts])
c_dd_err = np.array([p[4] for p in core_pts])

# -----------------------------------------------------------------------------
# 2. LOAD MCMC POSTERIOR OR CALCULATE SINGLE WEIGHTED AVERAGE JET AXIS
# -----------------------------------------------------------------------------
best_fit_file = '/media/kyle/kyle_phd/GRS1915/data/best_fit_params.txt'
if os.path.exists(best_fit_file):
    params = {}
    with open(best_fit_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                params[parts[0]] = float(parts[1])
                
    weighted_mean_pa_deg = (params['pa_n_deg'] + params['pa_s_deg']) / 2.0
    weighted_std_deg = params['theta_cone_deg']
    
    avg_pa_rad = np.radians(weighted_mean_pa_deg)
    std_pa_rad = np.radians(weighted_std_deg)
    
    print(f"Using MCMC Jet Axis: {weighted_mean_pa_deg:.2f}° with Conical Scatter: ±{weighted_std_deg:.2f}°")
else:
    pa_n_all = np.degrees(np.arctan2(n_da, n_dd))
    pa_n_err = np.degrees(np.sqrt((n_dd * n_da_err)**2 + (n_da * n_dd_err)**2) / (n_da**2 + n_dd**2))

    pa_s_all = np.degrees(np.arctan2(-s_da, -s_dd))
    pa_s_err = np.degrees(np.sqrt((s_dd * s_da_err)**2 + (s_da * s_dd_err)**2) / (s_da**2 + s_dd**2))

    all_pas = np.concatenate([pa_n_all, pa_s_all])
    all_errs = np.concatenate([pa_n_err, pa_s_err])

    weights = 1.0 / (all_errs**2)
    V1 = np.sum(weights)
    V2 = np.sum(weights**2)
    weighted_mean_pa_deg = np.sum(weights * all_pas) / V1

    weighted_variance = (V1 / (V1**2 - V2)) * np.sum(weights * (all_pas - weighted_mean_pa_deg)**2)
    weighted_std_deg = np.sqrt(weighted_variance)

    avg_pa_rad = np.radians(weighted_mean_pa_deg)
    std_pa_rad = np.radians(weighted_std_deg)

    print(f"Weighted Mean Jet Axis: {weighted_mean_pa_deg:.2f}° ± {weighted_std_deg:.2f}°")

# -----------------------------------------------------------------------------
# 3. PLOTTING
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=( 6, 9))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 0.06], height_ratios=[1, 1], wspace=0.08, hspace=0.15)

ax = fig.add_subplot(gs[:, 0])
cax_north = fig.add_subplot(gs[0, 1])
cax_south = fig.add_subplot(gs[1, 1])

# Color normalization (MJD - 60000)
mjd_min = min(np.min(n_mjd), np.min(s_mjd), np.min(c_mjd) if len(c_mjd) > 0 else np.inf) - 60000
mjd_max = max(np.max(n_mjd), np.max(s_mjd), np.max(c_mjd) if len(c_mjd) > 0 else -np.inf) - 60000
norm = Normalize(vmin=mjd_min, vmax=mjd_max)

cmap_north = plt.cm.winter_r
cmap_south = plt.cm.autumn_r
cmap_core = plt.cm.viridis

# A. Single Average Jet Axis Line
L = 7.0
x_line = np.array([L * np.sin(avg_pa_rad), -L * np.sin(avg_pa_rad)])
y_line = np.array([L * np.cos(avg_pa_rad), -L * np.cos(avg_pa_rad)])
ax.plot(x_line, y_line, color='black', linestyle='--', lw=1.8, zorder=1.5)

# B. Conical Scatter Envelope (Bow-Tie)
pa_hi = avg_pa_rad + std_pa_rad
pa_lo = avg_pa_rad - std_pa_rad

# Northern Cone
ax.fill([0, L * np.sin(pa_lo), L * np.sin(pa_hi)], 
        [0, L * np.cos(pa_lo), L * np.cos(pa_hi)], 
        color='gray', alpha=0.22, zorder=1, edgecolor='none')

# Southern Cone
ax.fill([0, -L * np.sin(pa_lo), -L * np.sin(pa_hi)], 
        [0, -L * np.cos(pa_lo), -L * np.cos(pa_hi)], 
        color='gray', alpha=0.22, zorder=1, edgecolor='none')

# C. Central Core Marker
ax.scatter(0, 0, marker='*', s=180, color='gold', edgecolors='black', linewidth=1.2, zorder=4)

# D. Observed Data Points
ax.errorbar(n_da, n_dd, xerr=n_da_err, yerr=n_dd_err, fmt='none', ecolor='gray', capsize=0, zorder=2, alpha=0.6, elinewidth=1.2)
ax.errorbar(s_da, s_dd, xerr=s_da_err, yerr=s_dd_err, fmt='none', ecolor='gray', capsize=0, zorder=2, alpha=0.6, elinewidth=1.2)
if len(c_da) > 0:
    ax.errorbar(c_da, c_dd, xerr=c_da_err, yerr=c_dd_err, fmt='none', ecolor='gray', capsize=0, zorder=2, alpha=0.6, elinewidth=1.2)


scatter_north = ax.scatter(n_da, n_dd, c=n_mjd - 60000, cmap=cmap_north, norm=norm, 
                           marker='^', edgecolors='black', s=60, zorder=3, label=r'$\Delta\theta^N$ Data')
scatter_south = ax.scatter(s_da, s_dd, c=s_mjd - 60000, cmap=cmap_south, norm=norm, 
                           marker='v', edgecolors='black', s=60, zorder=3, label=r'$\Delta\theta^S$ Data')
if len(c_da) > 0:
    scatter_core = ax.scatter(c_da, c_dd, c=c_mjd - 60000, cmap=cmap_core, norm=norm,
                              marker='o', edgecolors='black', s=60, zorder=3, label=r'Core Data')

# Limits and Labels
ax.set_xlim(-5.0, 5.0)
ax.set_ylim(-5.0, 5.0)
ax.set_xlabel(r'$\Delta\alpha$ ($^{\prime\prime}$)', fontsize=22)
ax.set_ylabel(r'$\Delta\delta$ ($^{\prime\prime}$)', fontsize=22)

# Grid and Ticks
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(2))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
ax.minorticks_on()
ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, labelsize=16, length=7, width=1.4)
ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True, length=3.5, width=1.0)
ax.invert_xaxis()  # Standard astronomical convention (East on left)
ax.grid(True, linestyle=':', alpha=0.5)

# Legend
#ax.legend(loc='best', fontsize=10, framealpha=0.9, edgecolor='gray')

# Colorbars
cb_north = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap_north), cax=cax_north)
cb_north.set_label('MJD - 60000 (NW)', fontsize=14)
cb_north.ax.tick_params(labelsize=12)

cb_south = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap_south), cax=cax_south)
cb_south.set_label('MJD - 60000 (SE)', fontsize=14)
cb_south.ax.tick_params(labelsize=12)
cb_south.ax.invert_yaxis()
plt.tight_layout()
plt.savefig('/media/kyle/kyle_phd/GRS1915/figures/ra_dec_plot.png', dpi=300, bbox_inches='tight')
print("Successfully generated and saved updated /media/kyle/kyle_phd/GRS1915/ra_dec_plot.png")