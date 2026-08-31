import numpy as np
import os, glob, re
dir_path = '/media/kyle/kyle_phd/GRS1915/S-band-fits'
files = glob.glob(os.path.join(dir_path, '*2D_Fitting*.txt'))
core_ras_deg = []
core_decs_deg = []
for f in sorted(files):
    with open(f, 'r') as fp: lines = fp.readlines()
    curr_comp = None
    for line in lines:
        if line.startswith('Component #'): curr_comp = {}
        elif 'Center X' in line and curr_comp is not None:
            m_ra = re.search(r'=\s*(\d+):(\d+):(\d+\.\d+)', line)
            if m_ra: 
                h, m_time, s = float(m_ra.group(1)), float(m_ra.group(2)), float(m_ra.group(3))
                curr_comp['ra_deg'] = (h + m_time/60.0 + s/3600.0) * 15.0
        elif 'Center Y' in line and curr_comp is not None:
            m_dec = re.search(r'=\s*([+-]?\d+):(\d+):(\d+\.\d+)', line)
            if m_dec:
                d, m_arc, s_arc = float(m_dec.group(1)), float(m_dec.group(2)), float(m_dec.group(3))
                sign = -1 if '-' in m_dec.group(1) else 1
                curr_comp['dec_deg'] = sign * (abs(d) + m_arc/60.0 + s_arc/3600.0)
                if 'ra_deg' in curr_comp and 288.7979 < curr_comp['ra_deg'] < 288.7982:
                    core_ras_deg.append(curr_comp['ra_deg'])
                    core_decs_deg.append(curr_comp['dec_deg'])
print("Mean RA:", np.mean(core_ras_deg))
print("Mean DEC:", np.mean(core_decs_deg))
