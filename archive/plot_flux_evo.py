import os
import glob
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime

dir_path = '/media/kyle/kyle_phd/GRS1915/S-band-fits'
files = glob.glob(os.path.join(dir_path, '*2D_Fitting*.txt'))

# Explicit mapping from timestamp to block number as per the text file
block_mapping = {
    1736575328: "B1",
    1738471273: "B2",
    1740883517: "B3",
    1742780174: "B4",
    1745709012: "B5",
    1749586570: "B6",
    1751137270: "B7",
    1753549996: "B8",
    1756479610: "B9",
    1759581193: "B10",
    1761391938: "B11",
    1764664923: "B12",
    1675587197: "B13",
    1706590573: "B14",
    1718044512: "B15",
    1725115877: "B16",
    1780867979: "B17",
}

points_core = []
points_nw = []
points_se = []

for f in sorted(files):
    filename = os.path.basename(f)
    m = re.search(r'img_(\d+)', filename)
    if not m:
        continue
    ts = int(m.group(1))
    
    dt = datetime.datetime.fromtimestamp(ts)
    block_label = block_mapping.get(ts, "")
    
    if block_label not in [f"B{i}" for i in range(1, 13)]:
        continue
    
    with open(f, 'r') as fp:
        lines = fp.readlines()
    
    components = []
    curr_comp = None
    for line in lines:
        if line.startswith('Component #'):
            curr_comp = {'name': line.strip()}
            components.append(curr_comp)
        elif 'Center X' in line and curr_comp is not None:
            m_ra = re.search(r'=\s*\d+:\d+:(\d+\.\d+)', line)
            if m_ra:
                curr_comp['ra'] = float(m_ra.group(1))
        elif 'Center Y' in line and curr_comp is not None:
            m_dec = re.search(r'=\s*\d+:\d+:(\d+\.\d+)', line)
            if m_dec:
                curr_comp['dec'] = float(m_dec.group(1))
        elif 'Integrated flux' in line and curr_comp is not None:
            m_flux = re.search(r'=\s*([\d\.eE+-]+)\s*±\s*([\d\.eE+-]+)', line)
            if m_flux:
                curr_comp['flux'] = float(m_flux.group(1))
                curr_comp['flux_err'] = float(m_flux.group(2))
    
    for c in components:
        ra = c.get('ra')
        flux = c.get('flux')
        flux_err = c.get('flux_err', 0.0)
        
        if ra is None or flux is None:
            continue
            
        pt = (dt, flux, flux_err, block_label)
        if ra < 11.51:
            if block_label not in ["B1", "B2", "B3", "B4"]:
                points_nw.append(pt)
        elif ra > 11.57:
            if block_label not in ["B1", "B2", "B3", "B4"]:
                points_se.append(pt)
        else:
            points_core.append(pt)

plt.figure(figsize=(12, 7))

for pts, label, color, fmt in [(points_core, 'Core', 'black', 'o-'), 
                               (points_nw, 'NW Lobe', 'blue', 's-'), 
                               (points_se, 'SE Lobe', 'red', '^-')]:
    if pts:
        times = [p[0] for p in pts]
        fluxes = [p[1] for p in pts]
        errs = [p[2] for p in pts]
        labels = [p[3] for p in pts]
        
        plt.errorbar(times, fluxes, yerr=errs, fmt=fmt, label=label, color=color, capsize=3)
        
        for i, txt in enumerate(labels):
            # add text slightly offset from the point
            plt.annotate(txt, (times[i], fluxes[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color=color)

plt.yscale('log')
plt.xlabel('Date')
plt.ylabel('Flux (Jy)')
plt.title('Time vs Flux Evolution of GRS 1915+105 Components')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('/home/kyle/.gemini/antigravity-ide/brain/2aaabcc4-d0a7-4471-82a3-349f01ce5c1b/flux_evolution.png', dpi=300)
plt.savefig('/media/kyle/kyle_phd/GRS1915/figures/flux_evolution.png', dpi=300)
print("Plot saved.")
