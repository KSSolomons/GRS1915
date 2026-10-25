"""
Full Suite Runner, Compiler, and Cleanup for GRS 1915+105 Absolute Kinematics Models.

Executes all 4 models (Ballistic Free Tej, Ballistic Anchored Tej, Piecewise Decel Free Tej,
Piecewise Decel Anchored Tej) with and without observational upper limits (8 configurations total,
configurable MCMC steps).

Once finished:
- Generates the multi-panel model comparison summary figure (figures/kinematics/model_comparison_parameter_summary.png).
- Compiles the full kinematic results into data/absolute_kinematics/kinematics_comparison.txt.
- Cleans up obsolete/redundant files (stale model 5 files, test output files, old logs).
"""

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

PYTHON_EXEC = sys.executable
SCRIPT = os.path.abspath("src/pipeline_absolute_kinematics_with_limits.py")
PLOT_SCRIPT = os.path.abspath("src/plot_model_comparison_table.py")
CWD = os.path.abspath(".")

MODEL_DESCRIPTIONS = {
    1: "Model 1 (Ballistic Free Tej)",
    2: "Model 2 (Ballistic Anchored Tej)",
    3: "Model 3 (Piecewise Decel Free Tej)",
    4: "Model 4 (Piecewise Decel Anchored Tej)",
    5: "Model 5 (Smooth Decel Free Tej)",
    6: "Model 6 (Smooth Decel Anchored Tej)",
}


def build_suites(models, limits_mode):
    suites = {}
    if limits_mode in ["all", "with"]:
        suites["With Limits"] = [
            {"model": m, "limits": True, "label": f"{MODEL_DESCRIPTIONS[m]} [WITH Limits]"}
            for m in models
        ]
    if limits_mode in ["all", "without"]:
        suites["Without Limits"] = [
            {"model": m, "limits": False, "label": f"{MODEL_DESCRIPTIONS[m]} [WITHOUT Limits]"}
            for m in models
        ]
    return suites


def run_worker_suite(suite_name, runs, n_steps=10000, anchor_date=60650.0, anchor_std=100.0):
    results = []
    print(f"--> Starting suite: {suite_name} ({len(runs)} models)...", flush=True)
    suite_start = time.time()
    for i, r in enumerate(runs, 1):
        lim_flag = "--include-limits" if r["limits"] else "--no-include-limits"
        cmd = [
            PYTHON_EXEC,
            SCRIPT,
            "--model-id", str(r["model"]),
            lim_flag,
            "--n-steps", str(n_steps),
            "--anchor-date", str(anchor_date),
        ]
        if anchor_std is not None and anchor_std > 0:
            cmd.extend(["--anchor-std", str(anchor_std)])

        print(f"[{suite_name}] ({i}/{len(runs)}) Running {r['label']} ({n_steps:,} steps)...", flush=True)
        t0 = time.time()
        res = subprocess.run(cmd, cwd=CWD, capture_output=True, text=True)
        dt = time.time() - t0
        if res.returncode == 0:
            print(f"[{suite_name}] ({i}/{len(runs)}) DONE: {r['label']} finished in {dt:.1f}s ({dt/60:.1f} min)", flush=True)
            results.append((r, dt, True, ""))
        else:
            print(f"[{suite_name}] ({i}/{len(runs)}) FAILED: {r['label']} after {dt:.1f}s:\n{res.stderr}\n{res.stdout}", flush=True)
            results.append((r, dt, False, res.stderr + "\n" + res.stdout))
            raise RuntimeError(f"Run failed for {r['label']}")
    suite_dt = time.time() - suite_start
    print(f"--> Finished suite: {suite_name} in {suite_dt:.1f}s ({suite_dt/60:.1f} min)", flush=True)
    return results


def parse_report_file(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        text = f.read()

    def get_val_err(pattern):
        match = re.search(pattern, text)
        if match:
            return float(match.group(1)), float(match.group(2)), float(match.group(3))
        return None, None, None

    data = {}
    m_pref = re.search(r"Statistically Preferred Model\s*:\s*(.+)", text)
    data["pref_model"] = m_pref.group(1).strip() if m_pref else "Unknown"

    m_bic = re.search(r"Optimized BIC\s*:\s*([\d\.]+)", text)
    data["opt_bic"] = float(m_bic.group(1)) if m_bic else None

    # Ejection parameters
    m_launch_free = re.search(r"Extrapolated Launch Time \(T_ej\):\s*MJD\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)([^\n]*)", text)
    if m_launch_free:
        extra_str = m_launch_free.group(4)
        is_anch = "Anchored" in extra_str
        data["Tej"] = (float(m_launch_free.group(1)), float(m_launch_free.group(2)), float(m_launch_free.group(3)), is_anch)
    else:
        data["Tej"] = (None, None, None, False)

    data["xc"] = get_val_err(r"Core Offset RA \(x_c\)\s*:\s*([+-]?[\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")
    data["yc"] = get_val_err(r"Core Offset Dec \(y_c\)\s*:\s*([+-]?[\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")

    # Proper motions
    data["mu_r"] = get_val_err(r"Receding Lobe \(North\) Speed \(mu_r\):\s*([\d\.eE\+-]+)\s*\+\s*([\d\.eE\+-]+)\s*-\s*([\d\.eE\+-]+)")
    data["mu_a"] = get_val_err(r"Approaching Lobe \(South\) Speed \(mu_a\):\s*([\d\.eE\+-]+)\s*\+\s*([\d\.eE\+-]+)\s*-\s*([\d\.eE\+-]+)")
    data["beta_app_r"] = get_val_err(r"Receding Apparent Speed \(beta_app,r\)\s*:\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")
    data["beta_app_a"] = get_val_err(r"Approaching Apparent Speed \(beta_app,a\)\s*:\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")

    data["pa_avg"] = get_val_err(r"Average Position Angle \(PA_avg\)\s*:\s*([+-]?[\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")
    data["pa_eon"] = get_val_err(r"Standard Astronomical PA_avg \(EoN\):\s*([+-]?[\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")
    data["theta_scatter"] = get_val_err(r"Scatter Angle \(theta_scatter\)\s*:\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")

    # Piecewise breaks if present
    data["tbn"] = get_val_err(r"Receding Break Time \(t_break,r\)\s*:\s*MJD\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")
    if data["tbn"][0] is None:
        data["tbn"] = get_val_err(r"Receding Break Time \(t_b,n\)\s*:\s*MJD\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")

    data["tbs"] = get_val_err(r"Approaching Break Time \(t_break,a\)\s*:\s*MJD\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")
    if data["tbs"][0] is None:
        data["tbs"] = get_val_err(r"Approaching Break Time \(t_b,s\)\s*:\s*MJD\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")

    data["v1n"] = get_val_err(r"Initial Receding Speed \(mu_1,r\)\s*:\s*([\d\.eE\+-]+)\s*\+\s*([\d\.eE\+-]+)\s*-\s*([\d\.eE\+-]+)")
    data["v1s"] = get_val_err(r"Initial Approaching Speed \(mu_1,a\)\s*:\s*([\d\.eE\+-]+)\s*\+\s*([\d\.eE\+-]+)\s*-\s*([\d\.eE\+-]+)")
    data["beta_app_1r"] = get_val_err(r"Initial Receding Apparent Speed \(beta_app,1r\)\s*:\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")
    data["beta_app_1a"] = get_val_err(r"Initial Approaching Apparent Speed \(beta_app,1a\)\s*:\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")

    data["v2n"] = get_val_err(r"Post-Break Receding Speed \(mu_2,r\)\s*:\s*([\d\.eE\+-]+)\s*\+\s*([\d\.eE\+-]+)\s*-\s*([\d\.eE\+-]+)")
    data["v2s"] = get_val_err(r"Post-Break Approaching Speed \(mu_2,a\)\s*:\s*([\d\.eE\+-]+)\s*\+\s*([\d\.eE\+-]+)\s*-\s*([\d\.eE\+-]+)")
    data["beta_app_2r"] = get_val_err(r"Post-Break Receding Apparent Speed \(beta_app,2r\)\s*:\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")
    data["beta_app_2a"] = get_val_err(r"Post-Break Approaching Apparent Speed \(beta_app,2a\)\s*:\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")

    # Relativistic kinematics
    data["bct"] = get_val_err(r"Derived beta \* cos\(theta\)\s*:\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")
    data["beta"] = get_val_err(r"Intrinsic Jet Speed \(beta = v/c\)\s*:\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")
    data["gamma"] = get_val_err(r"Lorentz Factor \(Gamma\)\s*:\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")
    data["proper_4v"] = get_val_err(r"Proper 4-Velocity \(Gamma \* Beta\)\s*:\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")
    data["theta"] = get_val_err(r"Jet Inclination Angle \(theta\)\s*:\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")
    data["delta_a"] = get_val_err(r"Approaching Doppler Factor \(delta_a\):\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")
    data["delta_r"] = get_val_err(r"Receding Doppler Factor \(delta_r\)\s*:\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")
    data["dratio"] = get_val_err(r"Kinematic Doppler Factor Ratio\s*:\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")
    data["Dmax"] = get_val_err(r"Maximum Physical Distance \(D_max\)\s*:\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")

    # Smooth deceleration parameters if present
    data["rmax_r"] = get_val_err(r"Receding (?:Terminal )?Stall (?:Distance|Radius) \(r_max,r\)\s*:\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")
    data["rmax_a"] = get_val_err(r"Approaching (?:Terminal )?Stall (?:Distance|Radius) \(r_max,a\)\s*:\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")
    data["tau_r"] = get_val_err(r"Receding (?:Decel|Relaxation) Timescale \(tau_r\)\s*:\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")
    data["tau_a"] = get_val_err(r"Approaching (?:Decel|Relaxation) Timescale \(tau_a\)\s*:\s*([\d\.]+)\s*\+\s*([\d\.]+)\s*-\s*([\d\.]+)")
    data["mu_end_r"] = get_val_err(r"Final Epoch Receding Speed \(mu_end,r\)\s*:\s*([\d\.eE\+-]+)\s*\+\s*([\d\.eE\+-]+)\s*-\s*([\d\.eE\+-]+)")
    data["mu_end_a"] = get_val_err(r"Final Epoch Approaching Speed \(mu_end,a\)\s*:\s*([\d\.eE\+-]+)\s*\+\s*([\d\.eE\+-]+)\s*-\s*([\d\.eE\+-]+)")

    # BIC table extraction
    bic_table = []
    in_table = False
    for line in text.splitlines():
        if "Model Description" in line and "Delta BIC" in line:
            in_table = True
            continue
        if in_table and line.startswith("---"):
            continue
        if in_table and line.startswith("-->"):
            in_table = False
            break
        if in_table and line.strip():
            bic_table.append(line.strip())
    data["bic_table"] = bic_table

    return data


def compile_comparison_file(anchor_date=60650.0, anchor_std=100.0):
    k_anch_str = f"k={8 if anchor_std else 7}"
    k_pw_anch_str = f"k={12 if anchor_std else 11}"
    k_smooth_anch_str = f"k={10 if anchor_std else 9}"
    models_info = [
        (1, "Model 1: Ballistic (Free Tej)", "k=8"),
        (2, f"Model 2: Ballistic (Anchored Tej={anchor_date:.0f}±{anchor_std:.0f})", k_anch_str),
        (3, "Model 3: Piecewise Deceleration (Free Tej)", "k=12"),
        (4, f"Model 4: Piecewise Deceleration (Anchored Tej={anchor_date:.0f}±{anchor_std:.0f})", k_pw_anch_str),
        (5, "Model 5: Smooth Continuous Deceleration (Free Tej)", "k=10"),
        (6, f"Model 6: Smooth Continuous Deceleration (Anchored Tej={anchor_date:.0f}±{anchor_std:.0f})", k_smooth_anch_str),
    ]

    reports = {}
    for m_id, name, k_str in models_info:
        f_lim = f"data/absolute_kinematics/kinematic_report_model{m_id}_with_limits.txt"
        f_nolim = f"data/absolute_kinematics/kinematic_report_model{m_id}_without_limits.txt"
        reports[(m_id, True)] = parse_report_file(f_lim)
        reports[(m_id, False)] = parse_report_file(f_nolim)

    lines = []
    lines.append("=" * 96)
    lines.append(f"{'GRS 1915+105 COMPREHENSIVE KINEMATIC MODEL COMPARISON SUITE':^96}")
    lines.append(f"{'(With Observational Limits vs. Without Limits)':^96}")
    lines.append("=" * 96)
    lines.append("")

    # Model Selection Suite Table
    lines.append("1. QUANTITATIVE MODEL SELECTION SUITE (BIC / AIC COMPARISONS)")
    lines.append("-" * 96)
    rep_lim_longest = max([r for (m, lim), r in reports.items() if lim and r and "bic_table" in r], key=lambda x: len(x["bic_table"]), default=None)
    if rep_lim_longest:
        lines.append("WITH OBSERVATIONAL UPPER LIMITS:")
        for bl in rep_lim_longest["bic_table"]:
            lines.append("  " + bl)
        lines.append("")

    rep_nolim_longest = max([r for (m, lim), r in reports.items() if not lim and r and "bic_table" in r], key=lambda x: len(x["bic_table"]), default=None)
    if rep_nolim_longest:
        lines.append("WITHOUT OBSERVATIONAL UPPER LIMITS:")
        for bl in rep_nolim_longest["bic_table"]:
            lines.append("  " + bl)
        lines.append("")

    lines.append("=" * 96)
    lines.append("2. COMPLETE PARAMETER COMPARISON: WITH LIMITS VS. WITHOUT LIMITS")
    lines.append("=" * 96)

    for m_id, m_name, k_str in models_info:
        r_lim = reports.get((m_id, True))
        r_nolim = reports.get((m_id, False))
        if not r_lim and not r_nolim:
            continue

        lines.append(f"\n>>> {m_name} ({k_str})")
        lines.append("-" * 96)
        header = f"{'PARAMETER':<40} {'WITHOUT LIMITS':<28} {'WITH LIMITS':<28}"
        lines.append(header)
        lines.append("-" * 96)

        def fmt_str(tup, unit="", scale=1.0, is_date=False, is_anchored=False):
            if not tup or tup[0] is None:
                return "N/A"
            if is_date:
                if is_anchored:
                    return f"MJD {tup[0]:.2f} (Anchored)"
                return f"MJD {tup[0]:.2f} (+{tup[1]:.2f}/-{tup[2]:.2f})"
            v, ep, em = tup[0] * scale, tup[1] * scale, tup[2] * scale
            u_str = f" {unit}" if unit else ""
            return f"{v:.4f} (+{ep:.4f}/-{em:.4f}){u_str}"

        # Launch time
        t_nolim = fmt_str(r_nolim["Tej"][:3], is_date=True, is_anchored=r_nolim["Tej"][3]) if r_nolim else "N/A"
        t_lim = fmt_str(r_lim["Tej"][:3], is_date=True, is_anchored=r_lim["Tej"][3]) if r_lim else "N/A"
        lines.append(f"{'Launch Time (T_ej)':<40} {t_nolim:<28} {t_lim:<28}")

        # Core offsets
        xc_nolim = fmt_str(r_nolim["xc"], unit="arcsec") if r_nolim else "N/A"
        xc_lim = fmt_str(r_lim["xc"], unit="arcsec") if r_lim else "N/A"
        lines.append(f"{'Core RA Offset (x_c)':<40} {xc_nolim:<28} {xc_lim:<28}")

        yc_nolim = fmt_str(r_nolim["yc"], unit="arcsec") if r_nolim else "N/A"
        yc_lim = fmt_str(r_lim["yc"], unit="arcsec") if r_lim else "N/A"
        lines.append(f"{'Core Dec Offset (y_c)':<40} {yc_nolim:<28} {yc_lim:<28}")

        # Proper motions
        mur_nolim = fmt_str(r_nolim["mu_r"], unit="mas/d", scale=1e3) if r_nolim else "N/A"
        mur_lim = fmt_str(r_lim["mu_r"], unit="mas/d", scale=1e3) if r_lim else "N/A"
        lines.append(f"{'North Receding Speed (mu_r)':<40} {mur_nolim:<28} {mur_lim:<28}")

        b_app_r_nolim = fmt_str(r_nolim["beta_app_r"], unit="c") if r_nolim else "N/A"
        b_app_r_lim = fmt_str(r_lim["beta_app_r"], unit="c") if r_lim else "N/A"
        if b_app_r_nolim != "N/A" or b_app_r_lim != "N/A":
            lines.append(f"{'North Receding Apparent Speed (beta_app,r)':<40} {b_app_r_nolim:<28} {b_app_r_lim:<28}")

        mua_nolim = fmt_str(r_nolim["mu_a"], unit="mas/d", scale=1e3) if r_nolim else "N/A"
        mua_lim = fmt_str(r_lim["mu_a"], unit="mas/d", scale=1e3) if r_lim else "N/A"
        lines.append(f"{'South Approaching Speed (mu_a)':<40} {mua_nolim:<28} {mua_lim:<28}")

        b_app_a_nolim = fmt_str(r_nolim["beta_app_a"], unit="c") if r_nolim else "N/A"
        b_app_a_lim = fmt_str(r_lim["beta_app_a"], unit="c") if r_lim else "N/A"
        if b_app_a_nolim != "N/A" or b_app_a_lim != "N/A":
            lines.append(f"{'South Approaching Apparent Speed (beta_app,a)':<40} {b_app_a_nolim:<28} {b_app_a_lim:<28}")

        pa_nolim = fmt_str(r_nolim["pa_eon"], unit="deg") if r_nolim else "N/A"
        pa_lim = fmt_str(r_lim["pa_eon"], unit="deg") if r_lim else "N/A"
        lines.append(f"{'Average Position Angle (EoN)':<40} {pa_nolim:<28} {pa_lim:<28}")

        sc_nolim = fmt_str(r_nolim["theta_scatter"], unit="deg") if r_nolim else "N/A"
        sc_lim = fmt_str(r_lim["theta_scatter"], unit="deg") if r_lim else "N/A"
        lines.append(f"{'Opening / Scatter Angle (theta_sc)':<40} {sc_nolim:<28} {sc_lim:<28}")

        # Piecewise parameters if applicable
        if (r_lim and r_lim["tbn"][0] is not None) or (r_nolim and r_nolim["tbn"][0] is not None):
            lines.append("-" * 96)
            tbn_nolim = fmt_str(r_nolim["tbn"], is_date=True) if r_nolim else "N/A"
            tbn_lim = fmt_str(r_lim["tbn"], is_date=True) if r_lim else "N/A"
            lines.append(f"{'Receding Break Time (t_break,r)':<40} {tbn_nolim:<28} {tbn_lim:<28}")

            tbs_nolim = fmt_str(r_nolim["tbs"], is_date=True) if r_nolim else "N/A"
            tbs_lim = fmt_str(r_lim["tbs"], is_date=True) if r_lim else "N/A"
            lines.append(f"{'Approaching Break Time (t_break,a)':<40} {tbs_nolim:<28} {tbs_lim:<28}")

            v1n_nolim = fmt_str(r_nolim["v1n"], unit="mas/d", scale=1e3) if r_nolim else "N/A"
            v1n_lim = fmt_str(r_lim["v1n"], unit="mas/d", scale=1e3) if r_lim else "N/A"
            lines.append(f"{'Initial Receding Speed (mu_1,r)':<40} {v1n_nolim:<28} {v1n_lim:<28}")

            b1r_nolim = fmt_str(r_nolim["beta_app_1r"], unit="c") if r_nolim else "N/A"
            b1r_lim = fmt_str(r_lim["beta_app_1r"], unit="c") if r_lim else "N/A"
            lines.append(f"{'Initial Receding Apparent (beta_app,1r)':<40} {b1r_nolim:<28} {b1r_lim:<28}")

            v1s_nolim = fmt_str(r_nolim["v1s"], unit="mas/d", scale=1e3) if r_nolim else "N/A"
            v1s_lim = fmt_str(r_lim["v1s"], unit="mas/d", scale=1e3) if r_lim else "N/A"
            lines.append(f"{'Initial Approaching Speed (mu_1,a)':<40} {v1s_nolim:<28} {v1s_lim:<28}")

            b1a_nolim = fmt_str(r_nolim["beta_app_1a"], unit="c") if r_nolim else "N/A"
            b1a_lim = fmt_str(r_lim["beta_app_1a"], unit="c") if r_lim else "N/A"
            lines.append(f"{'Initial Approaching Apparent (beta_app,1a)':<40} {b1a_nolim:<28} {b1a_lim:<28}")

            v2n_nolim = fmt_str(r_nolim["v2n"], unit="mas/d", scale=1e3) if r_nolim else "N/A"
            v2n_lim = fmt_str(r_lim["v2n"], unit="mas/d", scale=1e3) if r_lim else "N/A"
            lines.append(f"{'Post-Break Receding Speed (mu_2,r)':<40} {v2n_nolim:<28} {v2n_lim:<28}")

            v2s_nolim = fmt_str(r_nolim["v2s"], unit="mas/d", scale=1e3) if r_nolim else "N/A"
            v2s_lim = fmt_str(r_lim["v2s"], unit="mas/d", scale=1e3) if r_lim else "N/A"
            lines.append(f"{'Post-Break Approaching Speed (mu_2,a)':<40} {v2s_nolim:<28} {v2s_lim:<28}")

        # Smooth Deceleration parameters if applicable
        if (r_lim and r_lim.get("rmax_r") and r_lim["rmax_r"][0] is not None) or (r_nolim and r_nolim.get("rmax_r") and r_nolim["rmax_r"][0] is not None):
            lines.append("-" * 96)
            rmax_r_nolim = fmt_str(r_nolim.get("rmax_r"), unit="arcsec") if r_nolim else "N/A"
            rmax_r_lim = fmt_str(r_lim.get("rmax_r"), unit="arcsec") if r_lim else "N/A"
            lines.append(f"{'Receding Stall Radius (r_max,r)':<40} {rmax_r_nolim:<28} {rmax_r_lim:<28}")

            rmax_a_nolim = fmt_str(r_nolim.get("rmax_a"), unit="arcsec") if r_nolim else "N/A"
            rmax_a_lim = fmt_str(r_lim.get("rmax_a"), unit="arcsec") if r_lim else "N/A"
            lines.append(f"{'Approaching Stall Radius (r_max,a)':<40} {rmax_a_nolim:<28} {rmax_a_lim:<28}")

            tau_r_nolim = fmt_str(r_nolim.get("tau_r"), unit="days") if r_nolim else "N/A"
            tau_r_lim = fmt_str(r_lim.get("tau_r"), unit="days") if r_lim else "N/A"
            lines.append(f"{'Receding Decel Timescale (tau_r)':<40} {tau_r_nolim:<28} {tau_r_lim:<28}")

            tau_a_nolim = fmt_str(r_nolim.get("tau_a"), unit="days") if r_nolim else "N/A"
            tau_a_lim = fmt_str(r_lim.get("tau_a"), unit="days") if r_lim else "N/A"
            lines.append(f"{'Approaching Decel Timescale (tau_a)':<40} {tau_a_nolim:<28} {tau_a_lim:<28}")

            mu_end_r_nolim = fmt_str(r_nolim.get("mu_end_r"), unit="mas/d", scale=1e3) if r_nolim else "N/A"
            mu_end_r_lim = fmt_str(r_lim.get("mu_end_r"), unit="mas/d", scale=1e3) if r_lim else "N/A"
            lines.append(f"{'Final Receding Speed (mu_end,r)':<40} {mu_end_r_nolim:<28} {mu_end_r_lim:<28}")

            mu_end_a_nolim = fmt_str(r_nolim.get("mu_end_a"), unit="mas/d", scale=1e3) if r_nolim else "N/A"
            mu_end_a_lim = fmt_str(r_lim.get("mu_end_a"), unit="mas/d", scale=1e3) if r_lim else "N/A"
            lines.append(f"{'Final Approaching Speed (mu_end,a)':<40} {mu_end_a_nolim:<28} {mu_end_a_lim:<28}")

        # Relativistic Kinematics
        lines.append("-" * 96)
        bct_nolim = fmt_str(r_nolim["bct"]) if r_nolim else "N/A"
        bct_lim = fmt_str(r_lim["bct"]) if r_lim else "N/A"
        lines.append(f"{'Derived beta * cos(theta)':<40} {bct_nolim:<28} {bct_lim:<28}")

        beta_nolim = fmt_str(r_nolim["beta"], unit="c") if r_nolim else "N/A"
        beta_lim = fmt_str(r_lim["beta"], unit="c") if r_lim else "N/A"
        lines.append(f"{'Intrinsic Jet Speed (beta = v/c)':<40} {beta_nolim:<28} {beta_lim:<28}")

        gam_nolim = fmt_str(r_nolim["gamma"]) if r_nolim else "N/A"
        gam_lim = fmt_str(r_lim["gamma"]) if r_lim else "N/A"
        lines.append(f"{'Lorentz Factor (Gamma)':<40} {gam_nolim:<28} {gam_lim:<28}")

        gbeta_nolim = fmt_str(r_nolim["proper_4v"]) if r_nolim else "N/A"
        gbeta_lim = fmt_str(r_lim["proper_4v"]) if r_lim else "N/A"
        lines.append(f"{'Proper 4-Velocity (Gamma * Beta)':<40} {gbeta_nolim:<28} {gbeta_lim:<28}")

        th_nolim = fmt_str(r_nolim["theta"], unit="deg") if r_nolim else "N/A"
        th_lim = fmt_str(r_lim["theta"], unit="deg") if r_lim else "N/A"
        lines.append(f"{'Jet Inclination Angle (theta)':<40} {th_nolim:<28} {th_lim:<28}")

        da_nolim = fmt_str(r_nolim["delta_a"]) if r_nolim else "N/A"
        da_lim = fmt_str(r_lim["delta_a"]) if r_lim else "N/A"
        lines.append(f"{'Approaching Doppler Factor (delta_a)':<40} {da_nolim:<28} {da_lim:<28}")

        dr_nolim = fmt_str(r_nolim["delta_r"]) if r_nolim else "N/A"
        dr_lim = fmt_str(r_lim["delta_r"]) if r_lim else "N/A"
        lines.append(f"{'Receding Doppler Factor (delta_r)':<40} {dr_nolim:<28} {dr_lim:<28}")

        drat_nolim = fmt_str(r_nolim["dratio"]) if r_nolim else "N/A"
        drat_lim = fmt_str(r_lim["dratio"]) if r_lim else "N/A"
        lines.append(f"{'Kinematic Doppler Ratio (da/dr)':<40} {drat_nolim:<28} {drat_lim:<28}")

        dmax_nolim = fmt_str(r_nolim["Dmax"], unit="kpc") if r_nolim else "N/A"
        dmax_lim = fmt_str(r_lim["Dmax"], unit="kpc") if r_lim else "N/A"
        lines.append(f"{'Maximum Physical Distance (D_max)':<40} {dmax_nolim:<28} {dmax_lim:<28}")

        bic_nolim = f"{r_nolim['opt_bic']:.2f}" if r_nolim and r_nolim["opt_bic"] is not None else "N/A"
        bic_lim = f"{r_lim['opt_bic']:.2f}" if r_lim and r_lim["opt_bic"] is not None else "N/A"
        lines.append(f"{'Adopted Model BIC':<40} {bic_nolim:<28} {bic_lim:<28}")

    lines.append("\n" + "=" * 96)
    lines.append("3. SYNTHESIS OF SCIENTIFIC TAKEAWAYS")
    lines.append("=" * 96)
    lines.append("1. Statistical Hierarchy & Physical Realism:")
    lines.append("   - Ballistic models (Models 1 & 2) assume constant expansion across all epochs.")
    lines.append("   - Piecewise deceleration models (Models 3 & 4) capture the observed sharp break")
    lines.append("     around MJD ~60805-60820, dropping from relativistic launch speeds to ~5-7 mas/day.")
    lines.append("2. Superluminal Apparent Motion:")
    lines.append("   - Unconstrained distance propagation (D = 9.4 ± 1.0 kpc) yields apparent expansion")
    lines.append("     beta_app > 1.0c for the initial approaching lobe in piecewise models.")
    lines.append("3. Impact of Observational Upper Limits:")
    lines.append("   - Non-detections in early blocks (MJDs 60736 and 60758) penalize models that")
    lines.append("     predict early knot separation before observed resolution.")
    lines.append("=" * 96)

    compiled_text = "\n".join(lines) + "\n"
    out_file = "data/absolute_kinematics/kinematics_comparison.txt"
    with open(out_file, "w") as f:
        f.write(compiled_text)
    print(f"\nSuccessfully compiled results to {out_file} ({len(compiled_text)} bytes)")
    return compiled_text


def remove_redundant_files():
    redundant_patterns = [
        "output_test_0.01.txt",
        "output_test_0.1.txt",
        "output_test_0.5.txt",
        "scratch/temp.py",
        "scratch/run_all_models.py",
        "figures/kinematics/pa_wobble_explanation.pdf",
        "data/run_with_limits.log",
        "data/run_without_limits.log",
    ]

    removed = []
    skipped = []
    for rel_path in redundant_patterns:
        abs_path = os.path.join(CWD, rel_path)
        if os.path.exists(abs_path):
            try:
                os.remove(abs_path)
                removed.append(rel_path)
            except Exception as e:
                print(f"Error removing {rel_path}: {e}")
        else:
            skipped.append(rel_path)

    print("\n" + "=" * 70)
    print("CLEANUP SUMMARY")
    print("=" * 70)
    print(f"Total files removed: {len(removed)}")
    for r in removed:
        print(f"  [-] Removed: {r}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Full Suite Runner & Compiler for GRS 1915+105 Kinematic Models")
    parser.add_argument("--n-steps", type=int, default=10000, help="Number of MCMC steps per configuration (default: 10000)")
    parser.add_argument("--anchor-date", type=float, default=60650.0, help="Anchored launch epoch (MJD) for Models 2 & 4 (default: 60650.0)")
    parser.add_argument("--anchor-std", type=float, default=100.0, help="Gaussian prior std (days) for anchor date in Models 2 & 4 (default: 100.0)")
    parser.add_argument("--models", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6], choices=[1, 2, 3, 4, 5, 6], help="Subset of models to run (default: 1 2 3 4 5 6)")
    parser.add_argument("--limits-mode", choices=["all", "with", "without"], default="all", help="Whether to run with limits, without limits, or all (default: all)")
    parser.add_argument("--max-workers", type=int, default=2, help="Max parallel worker processes (default: 2)")
    parser.add_argument("--compile-only", action="store_true", help="Skip MCMC sampling and only compile tables/figures from existing reports")
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True, help="Clean up obsolete/redundant files after execution (default: True)")
    args = parser.parse_args()

    print("=" * 80)
    print(f"{'GRS 1915+105 ABSOLUTE KINEMATICS MODEL SUITE':^80}")
    print("=" * 80)
    print(f"Selected Models : {args.models}")
    print(f"Limits Mode     : {args.limits_mode}")
    print(f"MCMC Steps      : {args.n_steps:,}")
    print(f"Anchor Date     : MJD {args.anchor_date:.1f} (std = {args.anchor_std})")
    print(f"Max Workers     : {args.max_workers}")
    print("=" * 80)

    start_time = time.time()

    if not args.compile_only:
        suites = build_suites(args.models, args.limits_mode)
        total_runs = sum(len(v) for v in suites.values())
        print(f"\nLaunching {total_runs} MCMC runs across {len(suites)} suite(s)...", flush=True)

        if args.max_workers > 1 and len(suites) > 1:
            with ProcessPoolExecutor(max_workers=min(args.max_workers, len(suites))) as executor:
                futures = {
                    executor.submit(
                        run_worker_suite,
                        s_name,
                        s_runs,
                        n_steps=args.n_steps,
                        anchor_date=args.anchor_date,
                        anchor_std=args.anchor_std,
                    ): s_name
                    for s_name, s_runs in suites.items()
                }
                for f in futures:
                    f.result()
        else:
            for s_name, s_runs in suites.items():
                run_worker_suite(
                    s_name,
                    s_runs,
                    n_steps=args.n_steps,
                    anchor_date=args.anchor_date,
                    anchor_std=args.anchor_std,
                )

        total_mcmc_time = time.time() - start_time
        print(f"\nAll MCMC runs finished successfully in {total_mcmc_time:.1f}s ({total_mcmc_time/60:.1f} min)!", flush=True)

    # Step 2: Regenerate summary figure
    print("\nRegenerating model comparison summary figure...", flush=True)
    res_plot = subprocess.run([PYTHON_EXEC, PLOT_SCRIPT], cwd=CWD, capture_output=True, text=True)
    if res_plot.returncode == 0:
        print(res_plot.stdout.strip())
    else:
        print(f"Warning: plot_model_comparison_table.py had error: {res_plot.stderr}")

    # Step 3: Compile results into kinematics_comparison.txt
    print("\nCompiling full comparison table...", flush=True)
    compile_comparison_file(anchor_date=args.anchor_date, anchor_std=args.anchor_std)

    # Step 4: Clean up redundant files
    if args.clean:
        print("\nCleaning up redundant files...", flush=True)
        remove_redundant_files()

    # Step 5: Synchronize plain-text scripts to scripts_txt/
    try:
        from sync_txt_scripts import sync_all_scripts
        print("\nSynchronizing plain-text scripts to scripts_txt/...", flush=True)
        sync_all_scripts(verbose=True)
    except Exception as e:
        print(f"Notice: scripts_txt sync skipped ({e})")

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"MODEL SUITE EXECUTION & COMPILATION COMPLETE IN {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 80)


if __name__ == "__main__":
    main()
