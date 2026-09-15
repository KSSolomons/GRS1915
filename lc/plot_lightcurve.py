import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Typography for paper-ready plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

data_file = '/media/kyle/kyle_phd/GRS1915/lc/MeerKAT_GRS1915+105_all_NEW.txt'

mjds = []
fluxes_mJy = []
errs_mJy = []

with open(data_file, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split()
        if len(parts) >= 6:
            try:
                mjd = float(parts[1])
                flux_jy = float(parts[2])
                err_jy = float(parts[5]) # Err_Total
                
                mjds.append(mjd)
                fluxes_mJy.append(flux_jy * 1000.0)
                errs_mJy.append(err_jy * 1000.0)
            except ValueError:
                continue

mjds = np.array(mjds)
fluxes_mJy = np.array(fluxes_mJy)
errs_mJy = np.array(errs_mJy)

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

ax.errorbar(mjds, fluxes_mJy, yerr=errs_mJy, fmt='o', color='black', 
            markersize=6, capsize=3, elinewidth=1.2, markeredgecolor='black',
            markerfacecolor='dodgerblue', zorder=4, label='MeerKAT L-band (1.28 GHz)')

flares = [
    ('Flare I', 60690.4, 1.0, 'tab:red'),
    ('Flare II', 60700.0, 1.0, 'tab:orange'),
    ('Flare III', 60711.3, 1.0, 'tab:green'),
    ('Flare IV', 60722.4, 1.0, 'tab:purple')
]

for name, mjd, err, color in flares:
    ax.axvline(x=mjd, color=color, linestyle='--', zorder=1, label=name)
    ax.axvspan(mjd - err, mjd + err, color=color, alpha=0.2, zorder=0)

# Aesthetics
ax.set_xlabel('Time (MJD)', fontsize=18)
ax.set_ylabel('Flux Density (mJy)', fontsize=18)
ax.set_yscale('log')
ax.set_xlim(left=60000)

ax.tick_params(axis='both', which='major', labelsize=14, length=7, width=1.2)
ax.tick_params(axis='both', which='minor', labelsize=14, length=4, width=1)
ax.minorticks_on()

ax.legend(loc='best', fontsize=16, frameon=True, edgecolor='black')

plt.tight_layout()
output_path = '/media/kyle/kyle_phd/GRS1915/lc/MeerKAT_GRS1915_Lightcurve.png'
plt.savefig(output_path, bbox_inches='tight')
plt.close(fig)

print(f"Plot saved to {output_path}")
