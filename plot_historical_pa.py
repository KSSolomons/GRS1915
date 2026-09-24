import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

# Data transcribed from Table 3 with Telescope
data = [
    # Date (YYYY/MM/DD), PA, PA_err, Telescope
    ("1994/01/29", 159, 8, "VLA"),
    ("1994/02/19", 157, 6, "VLA"),
    ("1994/03/19", 149, 4, "VLA"),
    ("1994/04/21", 147, 6, "VLA"),
    ("1995/08/10", 140, 10, "VLA"),
    ("1997/10/29", 142, 2, "MERLIN"),
    ("1997/10/23", 157, 2, "VLBA"),
    ("1997/10/28", 143, 4, "VLBA"),
    ("1997/10/31", 133, 5, "VLBA"),
    ("1998/04/11", 155, 5, "VLBA"),
    ("1998/04/29", 148, 4, "VLBA"),
    ("1998/05/02", 145, 6, "VLBA"),
    ("1998/05/31", 160, 10, "EVN"),
    ("1999/12/28", 145, 10, "MERLIN"),
    ("1999/12/28", 180, 10, "MERLIN"),
    ("2001/03/21", 143, 9, "MERLIN"),
    ("2001/03/27", 143, 9, "MERLIN"),
    ("2001/03/31", 143, 9, "MERLIN"),
    ("2001/07/16", 143, 9, "MERLIN"),
    ("2003/04/19", 155, 5, "VLBA"),
    ("2006/01/17", 147, 3, "VLA"),
    ("2006/02/20", 147, 2, "VLA"),
    ("2006/02/28", 141, 1, "VLBA"),
    ("2006/03/04", 142, 1, "VLBA"),
    ("2006/03/06", 145, 2, "VLBA"),
    ("2006/03/09", 151, 2, "VLBA"),
    ("2006/04/20", 140, 2, "EVN"),
    ("2013/05/24", 130, 1, "VLBA"),
    ("2015/03/05", 140, 8, "VLA"),
    ("2023/04/11", 188, 3, "EAVN"),
    ("2023/05/23", 138, 9, "VLA"),
    ("2023/09/30", 174, 1, "VLA"),
    ("2024/05/25", 152, 1, "VLA"),
    ("2024/06/23", 149, 1, "VLA"),
    ("2025/01/18", 118, 7, "EAVN"),
    ("2025/01/25", 131, 6, "EAVN"),
    ("2025/02/24", 152, 2, "EAVN"),
]

def parse_date(date_str):
    parts = date_str.split('/')
    year = int(parts[0])
    month = int(parts[1])
    day = int(parts[2])
    dt = datetime.datetime(year, month, day)
    return year + (dt.timetuple().tm_yday - 1) / 365.25

parsed_data = {}
for date_str, pa, pa_err, telescope in data:
    dec_year = parse_date(date_str)
    if telescope not in parsed_data:
        parsed_data[telescope] = {'dates': [], 'pas': [], 'errs': []}
    parsed_data[telescope]['dates'].append(dec_year)
    parsed_data[telescope]['pas'].append(pa)
    parsed_data[telescope]['errs'].append(pa_err)

# Get our overall average PA
pa_table = '/media/kyle/kyle_phd/GRS1915/data/pa_evolution_table.txt'
our_avg_pa = None
our_avg_err = None
our_mean_date = None

if os.path.exists(pa_table):
    pas, errs, dates = [], [], []
    with open(pa_table, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 6 and parts[5] != 'nan':
                mjd = float(parts[0])
                avg_pa = float(parts[5])
                dt = datetime.datetime.fromtimestamp((mjd - 40587.0) * 86400.0)
                year = dt.year + (dt.timetuple().tm_yday - 1) / 365.25
                astronomical_pa = avg_pa + 180 if avg_pa < 0 else avg_pa
                if astronomical_pa > 180: astronomical_pa -= 180
                
                pas.append(astronomical_pa)
                errs.append(2.0)
                dates.append(year)
                
    if len(pas) > 0:
        pas = np.array(pas)
        errs = np.array(errs)
        weights = 1.0 / (errs**2)
        our_avg_pa = np.average(pas, weights=weights)
        our_avg_err = np.std(pas) # Use standard deviation to show the scatter
        our_mean_date = np.mean(dates)

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 1.2]})

colors = {
    "VLA": "blue",
    "MERLIN": "green",
    "VLBA": "orange",
    "EVN": "purple",
    "EAVN": "cyan"
}
markers = {
    "VLA": "o",
    "MERLIN": "s",
    "VLBA": "^",
    "EVN": "D",
    "EAVN": "v"
}

# --- Top Subplot (Full History) ---
for telescope, d in parsed_data.items():
    ax.errorbar(d['dates'], d['pas'], yerr=d['errs'], fmt=markers.get(telescope, 'o'), 
                label=telescope, capsize=3, color=colors.get(telescope, 'gray'), alpha=0.8, markersize=8)

# Plot our single average data point
if our_avg_pa is not None:
    ax.errorbar(our_mean_date, our_avg_pa, yerr=our_avg_err, fmt='*', 
                color='red', markersize=16, label="This Work", capsize=4, markeredgecolor='black')

# Add the pre-2019 mean line (removed from legend)
ax.axhline(147, color='grey', linestyle='--', linewidth=1.5, zorder=0)
ax.axhspan(147 - 8, 147 + 8, color='gray', alpha=0.2, zorder=0)

# Draw a box showing the zoomed region
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch

rect = patches.Rectangle((2022.5, 110), 3.5, 85, linewidth=1.5, edgecolor='blue', facecolor='none', linestyle=':')
ax.add_patch(rect)

ax.set_ylabel('Position Angle (Degrees)', fontsize=18)
ax.tick_params(axis='both', labelsize=14)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='best', fontsize=16, framealpha=0.9, ncol=2)

# --- Bottom Subplot (Zoomed Post-2019) ---
for telescope, d in parsed_data.items():
    ax2.errorbar(d['dates'], d['pas'], yerr=d['errs'], fmt=markers.get(telescope, 'o'), 
                capsize=3, color=colors.get(telescope, 'gray'), alpha=0.8, markersize=10)

if our_avg_pa is not None:
    ax2.errorbar(our_mean_date, our_avg_pa, yerr=our_avg_err, fmt='*', 
                color='red', markersize=20, capsize=4, markeredgecolor='black')

ax2.axhline(147, color='grey', linestyle='--', linewidth=1.5, zorder=0)
ax2.axhspan(147 - 8, 147 + 8, color='gray', alpha=0.2, zorder=0)

ax2.set_xlim(2022.5, 2026.0)
ax2.set_ylim(110, 195)
ax2.set_xlabel('Year', fontsize=18)
ax2.set_ylabel('Position Angle (Degrees)', fontsize=18)
ax2.tick_params(axis='both', labelsize=14)
ax2.grid(True, linestyle='--', alpha=0.6)

# Draw connecting lines from the box in ax to the corners of ax2
con1 = ConnectionPatch(xyA=(2022.5, 110), xyB=(2022.5, 195), coordsA="data", coordsB="data",
                       axesA=ax, axesB=ax2, color="blue", linestyle=":", linewidth=1.5)
con2 = ConnectionPatch(xyA=(2026.0, 110), xyB=(2026.0, 195), coordsA="data", coordsB="data",
                       axesA=ax, axesB=ax2, color="blue", linestyle=":", linewidth=1.5)
ax2.add_artist(con1)
ax2.add_artist(con2)

plt.subplots_adjust(hspace=0.15, left=0.12, right=0.95, top=0.95, bottom=0.08)

plt.savefig('/media/kyle/kyle_phd/GRS1915/figures/historical_pa_plot.png', dpi=300)
print("Saved to /media/kyle/kyle_phd/GRS1915/figures/historical_pa_plot.png")
