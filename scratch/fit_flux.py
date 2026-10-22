import os, glob, re, datetime, sys
import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from epoch_limits import EJECTION_CAMPAIGN_START

def exp_model(t, a, b):
    return a * np.exp(b * t)

act_ra_deg = (19 + 15 / 60.0 + 11.558 / 3600.0) * 15.0
act_dec_deg = 10 + 56 / 60.0 + 44.905 / 3600.0
cos_dec = np.cos(np.radians(act_dec_deg))

dir_path = "S-band-fits"
files = sorted(glob.glob(os.path.join(dir_path, "*2D_Fitting*.txt")))

core_data, n_data, s_data = [], [], []

for f in files:
    filename = os.path.basename(f)
    m = re.search(r"img_(\d+)", filename)
    if not m: continue
    ts = int(m.group(1))
    mjd = ts / 86400.0 + 40587.0

    with open(f, "r") as fp:
        lines = fp.readlines()

    comps = []
    curr_comp = None
    for line in lines:
        if line.startswith("Component #"):
            curr_comp = {"name": line.strip()}
            comps.append(curr_comp)
        elif "Center X" in line and curr_comp is not None:
            m_ra = re.search(r"=\s*(\d+):(\d+):(\d+\.\d+)\s*±\s*(\S+)", line)
            if m_ra:
                h, m_time, s = float(m_ra.group(1)), float(m_ra.group(2)), float(m_ra.group(3))
                curr_comp["ra_deg"] = (h + m_time / 60.0 + s / 3600.0) * 15.0
        elif "Center Y" in line and curr_comp is not None:
            m_dec = re.search(r"=\s*([+-]?\d+):(\d+):(\d+\.\d+)\s*±\s*(\S+)", line)
            if m_dec:
                d, m_arc, s_arc = float(m_dec.group(1)), float(m_dec.group(2)), float(m_dec.group(3))
                sign = -1 if "-" in m_dec.group(1) else 1
                curr_comp["dec_deg"] = sign * (abs(d) + m_arc / 60.0 + s_arc / 3600.0)
        elif "Integrated flux" in line and curr_comp is not None:
            m_flux = re.search(r"=\s*([\d\.eE+-]+)\s*±\s*([\d\.eE+-]+)", line)
            if m_flux:
                curr_comp["flux"] = float(m_flux.group(1))
                curr_comp["flux_err"] = float(m_flux.group(2))
                
    for c in comps:
        if "ra_deg" not in c or "dec_deg" not in c:
            continue
        
        dx = (c["ra_deg"] - act_ra_deg) * 3600.0 * cos_dec
        dy = (c["dec_deg"] - act_dec_deg) * 3600.0
        flux = c.get("flux", 0.0)
        flux_err = c.get("flux_err", 0.0)
        
        if 288.7979 < c["ra_deg"] < 288.7982:
            pass # core
        elif mjd < EJECTION_CAMPAIGN_START:
            pass # pre-2025 components belong to older ejecta
        elif dy > 0:
            if ts != 1764664923:
                n_data.append([mjd, flux, flux_err])
        else:
            if ts != 1764664923:
                s_data.append([mjd, flux, flux_err])

n_data = np.array(n_data)
s_data = np.array(s_data)

# Epochs 1-4 (S-band capture-block IDs are unix timestamps, so MJD follows directly)
target_ids = [1736575328, 1738471273, 1740883517, 1742780174]
target_mjd = np.array(target_ids) / 86400.0 + 40587.0
t0_mjd = target_mjd[0]

out = ["Extrapolated lobe flux densities for Epochs 1-4",
       "Exponential fade S(t) = a exp(-(t - t0)/tau_fade), t0 = Epoch 1, fitted to Epochs 5-11 only",
       "(Epoch 12 jet components and pre-2025 components excluded)", ""]
for label, dataset in [("NW Lobe", n_data), ("SE Lobe", s_data)]:
    t_data = dataset[:, 0]
    f_data = dataset[:, 1]
    err_data = dataset[:, 2]

    valid = f_data > 0
    t_data = t_data[valid]
    f_data = f_data[valid]
    err_data = err_data[valid]

    t_days = t_data - t0_mjd
    target_days = target_mjd - t0_mjd

    popt, pcov = curve_fit(exp_model, t_days, f_data, sigma=err_data, absolute_sigma=True, p0=[f_data[0], -0.01])
    perr = np.sqrt(np.diag(pcov))
    f_extrap = exp_model(target_days, *popt)
    tau_fade, tau_err = -1.0 / popt[1], perr[1] / popt[1] ** 2
    chi2_nu = np.sum(((f_data - exp_model(t_days, *popt)) / err_data) ** 2) / (len(f_data) - 2)

    out.append(f"--- {label} (epochs used: MJD {', '.join(f'{t:.0f}' for t in t_data)}) ---")
    out.append(f"tau_fade = {tau_fade:.1f} +/- {tau_err:.1f} d   (formal error; reduced chi2 = {chi2_nu:.2f})")
    for i, (m, f) in enumerate(zip(target_mjd, f_extrap)):
        out.append(f"Epoch {i + 1} (MJD {m:.2f}): {f * 1e3:.2f} mJy")
    out.append("")

print("\n".join(out))
with open("data/extrapolated_flux_measurements_ep5-11.txt", "w") as fp:
    fp.write("\n".join(out) + "\n")
