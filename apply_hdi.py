import re

with open('/media/kyle/kyle_phd/GRS1915/plot_sep_vs_time_poly.py', 'r') as f:
    content = f.read()

hdi_functions = """
def compute_hdi(samples, credible_mass=0.6827):
    # Computes 1D HDI
    sorted_samples = np.sort(samples)
    window_size = int(np.ceil(credible_mass * len(sorted_samples)))
    if window_size == 0 or window_size > len(sorted_samples):
        return [sorted_samples[0], np.median(samples), sorted_samples[-1]]
    window_widths = sorted_samples[window_size-1:] - sorted_samples[:-window_size+1]
    min_idx = np.argmin(window_widths)
    hdi_lower = sorted_samples[min_idx]
    hdi_upper = sorted_samples[min_idx + window_size - 1]
    return [hdi_lower, np.median(samples), hdi_upper]

def compute_hdi_2d(samples_2d, credible_mass=0.6827):
    # Computes 2D HDI along axis=0
    n_samples, n_points = samples_2d.shape
    window_size = int(np.ceil(credible_mass * n_samples))
    sorted_samples = np.sort(samples_2d, axis=0)
    window_widths = sorted_samples[window_size-1:, :] - sorted_samples[:-window_size+1, :]
    min_idx = np.argmin(window_widths, axis=0)
    point_indices = np.arange(n_points)
    hdi_lower = sorted_samples[min_idx, point_indices]
    hdi_upper = sorted_samples[min_idx + window_size - 1, point_indices]
    medians = np.median(samples_2d, axis=0)
    return np.array([hdi_lower, medians, hdi_upper])
"""

import_end = content.find('import corner\n') + len('import corner\n')
content = content[:import_end] + "\n" + hdi_functions + content[import_end:]

content = re.sub(r'np\.percentile\((.*?),\s*\[16,\s*50,\s*84\],\s*axis=0\)', r'compute_hdi_2d(\1)', content)
content = re.sub(r'np\.percentile\((.*?),\s*\[16,\s*50,\s*84\]\)', r'compute_hdi(\1)', content)

with open('/media/kyle/kyle_phd/GRS1915/plot_sep_vs_time_poly.py', 'w') as f:
    f.write(content)
