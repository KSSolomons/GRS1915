import os
import glob
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Publication styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

dir_path = '/media/kyle/kyle_phd/GRS1915/S-band-fits'
files = glob.glob(os.path.join(dir_path, '*2D_Fitting*.txt'))

# Actual catalog position: 19h 15m 11.558s, +10d 56m 44.905s
act_ra_deg = (19 + 15/60.0 + 11.558/3600.0) * 15.0
act_dec_deg = 10 + 56/60.0 + 44.905/3600.0
cos_dec = np.cos(np.radians(act_dec_deg))

records = []

for f in sorted(files):
    filename = os.path.basename(f)
    m = re.search(r'img_(\d+)', filename)
    if not m: continue
    ts = int(m.group(1))
    mjd = ts / 86400.0 + 40587.0
    
    with open(f) as fp: lines = fp.readlines()
    curr_comp = None
    for line in lines:
        if line.startswith('Component #'):
            curr_comp = {'name': line.strip()}
        elif 'Center X' in line and curr_comp is not None:
            m_ra = re.search(r'=\s*(\d+):(\d+):(\d+\.\d+)\s*±\s*(\S+)', line)
            if m_ra:
                h, mt, s = float(m_ra.group(1)), float(m_ra.group(2)), float(m_ra.group(3))
                curr_comp['ra_deg'] = (h + mt/60.0 + s/3600.0) * 15.0
                curr_comp['ra_err_s'] = float(m_ra.group(4))
        elif 'Center Y' in line and curr_comp is not None:
            m_dec = re.search(r'=\s*([+-]?\d+):(\d+):(\d+\.\d+)\s*±\s*(\S+)', line)
            if m_dec:
                d, mt, s = float(m_dec.group(1)), float(m_dec.group(2)), float(m_dec.group(3))
                sign = -1 if '-' in m_dec.group(1) else 1
                curr_comp['dec_deg'] = sign * (abs(d) + mt/60.0 + s/3600.0)
                curr_comp['dec_err_arcsec'] = float(m_dec.group(4))
                
                if 288.7979 < curr_comp['ra_deg'] < 288.7982:
                    d_ra = (curr_comp['ra_deg'] - act_ra_deg) * 3600.0 * cos_dec
                    d_dec = (curr_comp['dec_deg'] - act_dec_deg) * 3600.0
                    ra_err = curr_comp['ra_err_s'] * 15.0 * cos_dec
                    dec_err = curr_comp['dec_err_arcsec']
                    records.append({
                        'mjd': mjd,
                        'dra': d_ra,
                        'ddec': d_dec,
                        'dra_err': ra_err,
                        'ddec_err': dec_err
                    })

mjds = np.array([r['mjd'] for r in records])
dras = np.array([r['dra'] for r in records])
ddecs = np.array([r['ddec'] for r in records])
dra_errs = np.array([r['dra_err'] for r in records])
ddec_errs = np.array([r['ddec_err'] for r in records])

mean_dra = np.mean(dras)
std_dra = np.std(dras)
mean_ddec = np.mean(ddecs)
std_ddec = np.std(ddecs)

# Weighted means
w_ra = 1.0 / dra_errs**2
w_dec = 1.0 / ddec_errs**2
w_mean_dra = np.sum(dras * w_ra) / np.sum(w_ra)
w_mean_ddec = np.sum(ddecs * w_dec) / np.sum(w_dec)
w_err_dra = 1.0 / np.sqrt(np.sum(w_ra))
w_err_ddec = 1.0 / np.sqrt(np.sum(w_dec))

# Figure creation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

# ----------------- PANEL 1: 2D SKY PLANE OFFSET -----------------
ax1.axhline(0, color='grey', linestyle=':', linewidth=1.2)
ax1.axvline(0, color='grey', linestyle=':', linewidth=1.2)

# Scatter with colormap by MJD
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=np.min(mjds), vmax=np.max(mjds))

for i in range(len(mjds)):
    color = cmap(norm(mjds[i]))
    ax1.errorbar(dras[i], ddecs[i], xerr=dra_errs[i], yerr=ddec_errs[i],
                 fmt='o', color=color, ecolor='darkgrey', elinewidth=1.2,
                 capsize=2, markersize=8, markeredgecolor='black', zorder=3)

# Actual core marker at (0, 0)
ax1.scatter([0], [0], marker='*', s=350, color='crimson', edgecolors='black',
            linewidths=1.5, zorder=5, label=r'Catalog Core ($19^{\mathrm{h}}15^{\mathrm{m}}11.558^{\mathrm{s}}, +10^\circ56^{\prime}44.905^{\prime\prime}$)')

# Mean fitted core marker
ax1.errorbar(w_mean_dra, w_mean_ddec, xerr=w_err_dra, yerr=w_err_ddec,
             fmt='s', color='gold', markersize=12, markeredgecolor='black',
             linewidth=2, capsize=4, zorder=6,
             label=f'Weighted Mean Offset\n($\\Delta\\alpha^* = {w_mean_dra:.3f}^{{\\prime\\prime}},\\ \\Delta\\delta = {w_mean_ddec:.3f}^{{\\prime\\prime}}$)')

# Systematic offset vector arrow
ax1.annotate('', xy=(w_mean_dra, w_mean_ddec), xytext=(0, 0),
             arrowprops=dict(arrowstyle="-|>", color='black', lw=2, mutation_scale=18, linestyle='--'))

ax1.set_xlabel(r'$\Delta \alpha \cos \delta$ (arcsec)', fontsize=16)
ax1.set_ylabel(r'$\Delta \delta$ (arcsec)', fontsize=16)
ax1.set_title('Sky-Plane Offset: Measured vs. Catalog Core', fontsize=17, pad=12)
ax1.set_xlim(-0.75, 0.25)
ax1.set_ylim(-0.85, 0.15)
ax1.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.2)
ax1.tick_params(axis='both', which='minor', length=3, width=1)
ax1.minorticks_on()
ax1.legend(loc='upper left', fontsize=12, frameon=True, edgecolor='darkgrey')
ax1.set_aspect('equal')

# Colorbar for Panel 1
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label('Observation Time (MJD)', fontsize=13)
cbar.ax.tick_params(labelsize=11)

# ----------------- PANEL 2: OFFSET VS TIME (MJD) -----------------
ax2.axhline(0, color='grey', linestyle=':', linewidth=1.2)

# Shaded bands for mean values
ax2.axhspan(w_mean_dra - std_dra, w_mean_dra + std_dra, color='dodgerblue', alpha=0.15)
ax2.axhline(w_mean_dra, color='dodgerblue', linestyle='--', linewidth=1.5,
            label=f'Mean $\\Delta\\alpha^* = {w_mean_dra:.3f} \\pm {std_dra:.3f}^{{\\prime\\prime}}$')

ax2.axhspan(w_mean_ddec - std_ddec, w_mean_ddec + std_ddec, color='crimson', alpha=0.15)
ax2.axhline(w_mean_ddec, color='crimson', linestyle='--', linewidth=1.5,
            label=f'Mean $\\Delta\\delta = {w_mean_ddec:.3f} \\pm {std_ddec:.3f}^{{\\prime\\prime}}$')

# Data points
ax2.errorbar(mjds, dras, yerr=dra_errs, fmt='o-', color='dodgerblue',
             markeredgecolor='black', markersize=8, capsize=3, elinewidth=1.2,
             linewidth=1.2, label=r'$\Delta \alpha \cos \delta$ Offset', zorder=3)

ax2.errorbar(mjds, ddecs, yerr=ddec_errs, fmt='s-', color='crimson',
             markeredgecolor='black', markersize=8, capsize=3, elinewidth=1.2,
             linewidth=1.2, label=r'$\Delta \delta$ Offset', zorder=3)

ax2.set_xlabel('Time (MJD)', fontsize=16)
ax2.set_ylabel('Offset from Catalog Core (arcsec)', fontsize=16)
ax2.set_title('Core Positional Offset Stability Over Time', fontsize=17, pad=12)
ax2.set_ylim(-0.85, 0.15)
ax2.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.2)
ax2.tick_params(axis='both', which='minor', length=3, width=1)
ax2.minorticks_on()
ax2.legend(loc='lower left', fontsize=12, frameon=True, edgecolor='darkgrey', ncol=2)

plt.tight_layout()
out_path = '/media/kyle/kyle_phd/GRS1915/figures/core_offset_plot.png'
plt.savefig(out_path, bbox_inches='tight')
plt.close(fig)
print(f"Saved plot to {out_path}")
