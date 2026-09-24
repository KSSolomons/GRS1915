# GRS 1915+105 Kinematic Analysis Pipelines

This directory contains the primary python pipelines for modeling and plotting the astrometric and flux data of the GRS 1915+105 microquasar jets.

## 1. `pipeline_absolute_kinematics.py`
This script models the jet kinematics based on the **absolute coordinates** (RA and Dec) of the components relative to a fixed catalog position on the sky. 

**Key Features:**
- Calculates the global astrometric offset of the central core.
- Fits ballistic/polynomial kinematic models (with MCMC sampling) directly to the absolute RA/Dec coordinates.
- Plots the 2D absolute tracks of the ejecta on the sky (`figures/ra_dec_plot.png`).
- Analyzes and fits the **flux evolution** of the core and lobes over time, producing extrapolated predictions for early epochs (`figures/flux_evolution_with_extrap.png`).

## 2. `pipeline_relative_kinematics.py`
This script models the jet kinematics based on the **relative angular separation** between the moving lobes and the observed core position at each individual epoch.

**Key Features:**
- Ignores global astrometric shifts between observing epochs, making it more robust against epoch-to-epoch absolute referencing errors.
- Specifically projects the separations into parallel and perpendicular components along the jet axis.
- Tracks and models the position angle (PA) evolution over time, allowing for jet precession.
- Produces the standard separation-vs-time plots (`figures/parallel_perp_fit_plot.png`) and PA evolution plots (`figures/pa_evolution_plot.png`).

## Archive
Older, standalone plotting scripts (`plot_ra_dec.py`, `plot_flux_evo.py`, and `plot_core_offset.py`) have been moved into the `archive/` directory, as their functionality has been fully integrated into the two main pipelines above.
