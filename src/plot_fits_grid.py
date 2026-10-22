import glob
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from datetime import datetime
import numpy as np
from matplotlib.patches import Ellipse

fits_dir = "/media/kyle/kyle_phd/GRS1915/FITS"
fits_files = glob.glob(os.path.join(fits_dir, "*_pcalmask-MFS-image.pbcor.fits"))

data_list = []
for f in fits_files:
    try:
        hdul = fits.open(f)
        hdr = hdul[0].header
        date_obs = hdr.get('DATE-OBS', '2000-01-01T00:00:00')
        dt = datetime.strptime(date_obs.split('.')[0], "%Y-%m-%dT%H:%M:%S")
        
        # WCS for 2D image
        wcs = WCS(hdr).celestial
        data_list.append({'file': f, 'date': dt, 'hdr': hdr, 'data': hdul[0].data, 'wcs': wcs})
    except Exception as e:
        print(f"Failed to read {f}: {e}")

data_list.sort(key=lambda x: x['date'])

# Zoom axis: tight crop centered on core
cutout_size = 60 # pixels

# Determine unified reference center sky position from epoch 0 core
d0 = data_list[0]['data']
while d0.ndim > 2:
    d0 = d0[0]
cy0, cx0 = d0.shape[0]//2, d0.shape[1]//2
s0 = d0[cy0-200:cy0+200, cx0-200:cx0+200]
my0, mx0 = np.unravel_index(np.nanargmax(s0), s0.shape)
ref_sky = data_list[0]['wcs'].pixel_to_world(cx0 - 200 + mx0, cy0 - 200 + my0)

# Make cutouts centered on the exact same celestial position
for item in data_list:
    d = item['data']
    while d.ndim > 2:
        d = d[0]
        
    cutout = Cutout2D(d, ref_sky, cutout_size, wcs=item['wcs'])
    
    item['cutout'] = cutout.data * 1000 # mJy/beam
    item['cutout_wcs'] = cutout.wcs


from mpl_toolkits.axes_grid1 import make_axes_locatable

# Grid layout
ncols = 4
nrows = 3

fig = plt.figure(figsize=(3.5 * ncols, 3.5 * nrows), facecolor='white')
plt.subplots_adjust(wspace=0.08, hspace=0.0)

axes = []
for i in range(12):
    if i < len(data_list):
        item = data_list[i]
        cutout_wcs = item['cutout_wcs']
        ax = fig.add_subplot(nrows, ncols, i+1, projection=cutout_wcs, facecolor='black')
        
        cutout = item['cutout']
        date_str = item['date'].strftime('%Y-%m-%d')
        
        # Local scaling for each epoch
        if i == 4:
            vmax = np.nanmax(cutout) * 0.3
        else:
            vmax = np.nanmax(cutout) * 0.95
        vmin = -vmax * 0.05
        
        im = ax.imshow(cutout, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
        
        ax.text(0.03, 0.03, date_str, transform=ax.transAxes, color='white', fontsize=14, fontweight='bold', ha='left', va='bottom')
        
        bmaj = item['hdr'].get('BMAJ', 0) * 3600
        bmin = item['hdr'].get('BMIN', 0) * 3600
        bpa = item['hdr'].get('BPA', 0)
        cdelt = abs(item['hdr'].get('CDELT1', 0.000138888)) * 3600
        w = bmaj / cdelt
        h = bmin / cdelt
        
        beam_x = cutout_size - 6
        beam_y = 6
        
        ellipse = Ellipse((beam_x, beam_y), width=w, height=h, angle=90+bpa,
                          edgecolor='white', facecolor='white', alpha=0.8, lw=1.5)
        ax.add_patch(ellipse)
        
        # Individual colorbar on the right
        # WCSAxes doesn't play perfectly with make_axes_locatable sometimes, but it usually works if we're careful.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.02, axes_class=plt.Axes)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        cbar.ax.tick_params(labelsize=9, colors='black', pad=1)
        for lbl in cbar.ax.yaxis.get_ticklabels():
            lbl.set_rotation(90)
            lbl.set_verticalalignment('center')
        
        for spine in cbar.ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.0)
        
        # RA and DEC labels
        ra = ax.coords[0]
        dec = ax.coords[1]
        
        ra.axislabels.set_visible(False)
        dec.axislabels.set_visible(False)
        
        ra.set_major_formatter('hh:mm:ss')
        dec.set_major_formatter('dd:mm:ss')
        
        ra.set_ticklabel(color='black', size=13)
        dec.set_ticklabel(color='black', size=13)
        
        ra.set_ticks(number=4, color='black')
        dec.set_ticks(number=4, color='black')
        
        ra.set_ticks_position('bt')
        dec.set_ticks_position('lr')
        ra.display_minor_ticks(False)
        dec.display_minor_ticks(False)
        
        # Only show numerical labels on outer edges, but keep ticks visible everywhere
        if i % ncols != 0:
            dec.set_ticklabel_visible(False)
        else:
            dec.set_ticklabel_visible(True)
        
        if i < (nrows - 1) * ncols:
            ra.set_ticklabel_visible(False)
        else:
            ra.set_ticklabel_visible(True)
            
        ra.set_ticks_visible(True)
        dec.set_ticks_visible(True)
            
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.0)
            
        axes.append(ax)
    else:
        ax = fig.add_subplot(nrows, ncols, i+1, facecolor='black')
        ax.axis('off')

fig.supxlabel('Right Ascension (J2000)', color='black', fontsize=18, y=0.04)
fig.supylabel('Declination (J2000)', color='black', fontsize=18, x=0.04)

out_png = os.path.join(fits_dir, "kinematics_grid.png")
plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved grid to {out_png}")
