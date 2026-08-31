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

# Use LaTeX style for fonts if needed, or just normal math mode
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

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

mean_core_ra = np.mean(core_ras_deg)
mean_core_dec = np.mean(core_decs_deg)
mean_core_ra_err = np.sqrt(np.sum(np.array(core_ra_errs_sec)**2)) / len(core_ra_errs_sec)
mean_core_dec_err = np.sqrt(np.sum(np.array(core_dec_errs_arcsec)**2)) / len(core_dec_errs_arcsec)

north_pts = []
south_pts = []
core_pts = []

# Sort epochs by time and remove the first two earliest observations
epochs_data.sort(key=lambda x: x['mjd'])
epochs_data = epochs_data[2:]

for epoch in epochs_data:
    mjd = epoch['mjd']
    # Revert to using the global mean core position as the fixed (0,0) reference
    ref_ra = mean_core_ra
    ref_dec = mean_core_dec
    ref_ra_err = mean_core_ra_err
    ref_dec_err = mean_core_dec_err
    
    # Calculate exact cosine factor dynamically for this epoch's reference declination
    cos_dec = np.cos(np.radians(ref_dec))
    
    for c in epoch['comps']:
        ra_deg = c['ra_deg']
        dec_deg = c['dec_deg']
        ra_err = c['ra_err']
        dec_err = c['dec_err']
        
        # Skip plotting the core itself? No, we will plot it now.
        if 288.7979 < ra_deg < 288.7982:
            d_alpha = (ra_deg - ref_ra) * 3600.0 * cos_dec
            d_delta = (dec_deg - ref_dec) * 3600.0
            sigma_alpha_lobe = ra_err * 15.0 * cos_dec
            sigma_delta_lobe = dec_err
            core_pts.append((mjd, d_alpha, d_delta, sigma_alpha_lobe, sigma_delta_lobe))
            continue 
        
        # Convert degrees to arcseconds
        # RA offset uses the dynamic cosine projection factor
        d_alpha = (ra_deg - ref_ra) * 3600.0 * cos_dec
        d_delta = (dec_deg - ref_dec) * 3600.0
        
        sigma_alpha_lobe = ra_err * 15.0 * cos_dec
        sigma_alpha_core = ref_ra_err * 15.0 * cos_dec
        sigma_d_alpha = np.sqrt(sigma_alpha_lobe**2 + sigma_alpha_core**2)
        
        sigma_delta_lobe = dec_err
        sigma_delta_core = ref_dec_err
        sigma_d_delta = np.sqrt(sigma_delta_lobe**2 + sigma_delta_core**2)
        
        if d_delta > 0:
            north_pts.append((mjd, d_alpha, d_delta, sigma_d_alpha, sigma_d_delta))
        else:
            south_pts.append((mjd, d_alpha, d_delta, sigma_d_alpha, sigma_d_delta))

north_pts.sort(key=lambda x: x[0])
south_pts.sort(key=lambda x: x[0])

# Unpack
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

# Setup figure
fig = plt.figure(figsize=(4.5, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 0.05], height_ratios=[1, 1], wspace=0.05, hspace=0)

ax = fig.add_subplot(gs[:, 0])
cax_north = fig.add_subplot(gs[0, 1])
cax_south = fig.add_subplot(gs[1, 1])

# Color normalization
mjd_min = min(np.min(n_mjd), np.min(s_mjd))
mjd_max = max(np.max(n_mjd), np.max(s_mjd))
norm = Normalize(vmin=mjd_min, vmax=mjd_max)

cmap_north = plt.cm.winter
cmap_south = plt.cm.autumn

# Plot lines
ax.plot(n_da, n_dd, color='black', lw=0.5, zorder=1)
ax.plot(s_da, s_dd, color='black', lw=0.5, zorder=1)

# Plot scatter
scatter_north = ax.scatter(n_da, n_dd, c=n_mjd, cmap=cmap_north, norm=norm, 
                           marker='^', edgecolors='black', s=40, zorder=2)
scatter_south = ax.scatter(s_da, s_dd, c=s_mjd, cmap=cmap_south, norm=norm, 
                           marker='v', edgecolors='black', s=40, zorder=2)
#scatter_core = ax.scatter(c_da, c_dd, color='black', 
 #                         marker='o', edgecolors='white', s=60, zorder=3, label='Core')

ax.errorbar(n_da, n_dd, xerr=n_da_err, yerr=n_dd_err, fmt='none', ecolor='gray', capsize=0, zorder=1.5, alpha=0.5)
ax.errorbar(s_da, s_dd, xerr=s_da_err, yerr=s_dd_err, fmt='none', ecolor='gray', capsize=0, zorder=1.5, alpha=0.5)
#ax.errorbar(c_da, c_dd, xerr=c_da_err, yerr=c_dd_err, fmt='none', ecolor='gray', capsize=0, zorder=1.5, alpha=0.5)

# Calculate and plot average jet axis and its uncertainty using formally propagated errors
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

# Calculate unbiased weighted standard deviation to show the physical spread of the points
weighted_variance = (V1 / (V1**2 - V2)) * np.sum(weights * (all_pas - weighted_mean_pa_deg)**2)
weighted_std_deg = np.sqrt(weighted_variance)

if weighted_mean_pa_deg < 0:
    weighted_mean_pa_deg += 180.0

avg_pa = np.radians(weighted_mean_pa_deg)
std_pa = np.radians(weighted_std_deg)

L = 7.0
x_line = np.array([-L * np.sin(avg_pa), L * np.sin(avg_pa)])
y_line = np.array([-L * np.cos(avg_pa), L * np.cos(avg_pa)])
ax.plot(x_line, y_line, color='gray', linestyle='--', lw=1.5, zorder=0, alpha=0.8, label=f'Avg Axis ({weighted_mean_pa_deg:.1f}° ± {weighted_std_deg:.1f}°)')

# Create a "bow-tie" polygon for the uncertainty wedge
pa_min = avg_pa - std_pa
pa_max = avg_pa + std_pa
# North wedge
ax.fill([0, L * np.sin(pa_min), L * np.sin(pa_max)], 
        [0, L * np.cos(pa_min), L * np.cos(pa_max)], 
        color='gray', alpha=0.2, zorder=0, edgecolor='none')
# South wedge
ax.fill([0, -L * np.sin(pa_min), -L * np.sin(pa_max)], 
        [0, -L * np.cos(pa_min), -L * np.cos(pa_max)], 
        color='gray', alpha=0.2, zorder=0, edgecolor='none')

# ax.legend(loc='lower left', fontsize=18)

# Axis limits and labels
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_xlabel(r'$\Delta\alpha (^{\prime\prime})$', fontsize=24)
ax.set_ylabel(r'$\Delta\delta (^{\prime\prime})$', fontsize=24)

# Tick formatting
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(1))
ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=18, length=6)
ax.tick_params(which='minor', length=3)
# The reference image has -5 on the right, 5 on the left
ax.invert_xaxis()

# Colorbars
cb_north = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap_north), cax=cax_north)
cb_north.set_label('Time (NW lobe)', fontsize=18)
cb_north.ax.tick_params(labelsize=14)

cb_south = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap_south), cax=cax_south)
cb_south.set_label('Time (SE lobe)', fontsize=18)
cb_south.ax.tick_params(labelsize=14)

ax.grid(True, linestyle='--', alpha=0.6)
plt.savefig('/media/kyle/kyle_phd/GRS1915/ra_dec_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved successfully.")