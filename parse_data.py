import os
import glob
import re

dir_path = '/media/kyle/kyle_phd/GRS1915/S-band-fits'
files = glob.glob(os.path.join(dir_path, '*2D_Fitting*.txt'))

for f in sorted(files):
    filename = os.path.basename(f)
    m = re.search(r'img_(\d+)', filename)
    if not m:
        continue
    ts = int(m.group(1))
    
    with open(f, 'r') as fp:
        lines = fp.readlines()
    
    components = []
    curr_comp = None
    for line in lines:
        if line.startswith('Component #'):
            curr_comp = {'name': line.strip()}
            components.append(curr_comp)
        elif 'Center X' in line and curr_comp is not None:
            # Center X        = 19:15:11.5519254946 ± 0.000089 (s)
            m_ra = re.search(r'=\s*\d+:\d+:(\d+\.\d+)', line)
            if m_ra:
                curr_comp['ra'] = float(m_ra.group(1))
        elif 'Center Y' in line and curr_comp is not None:
            # Center Y        = 10:56:44.5670117900 ± 0.001389 (arcsec)
            m_dec = re.search(r'=\s*\d+:\d+:(\d+\.\d+)', line)
            if m_dec:
                curr_comp['dec'] = float(m_dec.group(1))
        elif 'Integrated flux' in line and curr_comp is not None:
            # Integrated flux = 0.049220 ± 0.000064 (Jy)
            m_flux = re.search(r'=\s*([\d\.eE+-]+)\s*±\s*([\d\.eE+-]+)', line)
            if m_flux:
                curr_comp['flux'] = float(m_flux.group(1))
                curr_comp['flux_err'] = float(m_flux.group(2))
    
    print(f"Time: {ts}")
    for c in components:
        print(f"  RA: {c.get('ra')}, DEC: {c.get('dec')}, Flux: {c.get('flux')}")
