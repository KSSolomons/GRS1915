import glob
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

fits_dir = "/media/kyle/kyle_phd/GRS1915/FITS"
fits_files = glob.glob(os.path.join(fits_dir, "*.fits"))

for fits_file in fits_files:
    try:
        print(f"Processing {fits_file}...")
        hdul = fits.open(fits_file)
        data = hdul[0].data
        header = hdul[0].header
        
        # Squeeze out degenerate dimensions if necessary (e.g. Stokes, Freq)
        while data.ndim > 2:
            data = data[0]
            
        # Try to parse WCS
        wcs = WCS(header).celestial
        
        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(projection=wcs)
        
        # Calculate vmin and vmax based on standard deviation to make the plot look good
        mean_val = np.nanmean(data)
        std_val = np.nanstd(data)
        vmin = mean_val - 1 * std_val
        vmax = mean_val + 5 * std_val
        
        im = ax.imshow(data, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
        
        ax.set_xlabel('Right Ascension')
        ax.set_ylabel('Declination')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Flux Density (Jy/beam)')
        
        # Add title
        basename = os.path.basename(fits_file)
        plt.title(basename)
        
        out_png = fits_file.replace('.fits', '.png')
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out_png}")
        hdul.close()
    except Exception as e:
        print(f"Error processing {fits_file}: {e}")
