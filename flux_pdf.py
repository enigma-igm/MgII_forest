from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('/Users/suksientie/codes/enigma')
from enigma.reion_forest import utils
from pypeit import utils as putils
sys.path.append('/Users/suksientie/Research/data_redux')
from scripts import rdx_utils

fitsfile_list = ['/Users/suksientie/Research/data_redux/mgii_stack_fits/J0313-1806_stacked_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/mgii_stack_fits/J1342+0928_stacked_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/mgii_stack_fits/J0252-0503_stacked_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/2010_done/Redux/J0038-1527_201024_done/J0038-1527_coadd_tellcorr.fits']
qso_zlist = [7.6, 7.54, 7.0, 7.0]
nbins, oneminf_min, oneminf_max = 101, 1e-5, 1.0  # gives d(oneminf) = 0.01
one_minF_thresh = 0.2

all_transmission = []
all_std = []
for i, fitsfile in enumerate(fitsfile_list):
    wave, flux, ivar, std, mask, cont_flux, norm_std = rdx_utils.continuum_normalize(fitsfile, qso_zlist[i])

    x_mask = wave[mask] <= (2800 * (1 + qso_zlist[i]))
    all_transmission.extend(cont_flux[mask][x_mask])
    all_std.extend(norm_std[mask][x_mask])

    flux_bins, flux_pdf_tot = utils.pdf_calc(1.0 - cont_flux[mask][x_mask], oneminf_min, oneminf_max, nbins)
    plt.plot(flux_bins, flux_pdf_tot, drawstyle='steps-mid', alpha=0.5, label=fitsfile.split('/')[-1].split('_')[0])

all_transmission = np.array(all_transmission)
all_std = np.array(all_std)
SNR = 10.0
gaussian_noise = np.random.normal(0.0, 1.0 / SNR, all_transmission.shape)

flux_bins, flux_pdf_tot = utils.pdf_calc(1.0 - all_transmission, oneminf_min, oneminf_max, nbins)
#_, flux_pdf_noise = utils.pdf_calc(all_std, oneminf_min, oneminf_max, nbins)
_, flux_pdf_gnoise = utils.pdf_calc(gaussian_noise, oneminf_min, oneminf_max, nbins)

plt.plot(flux_bins, flux_pdf_tot, drawstyle='steps-mid', color='k', alpha=1.0, lw=2, label='all spectra')
#plt.plot(flux_bins, flux_pdf_noise, drawstyle='steps-mid', color='y', alpha=1.0, lw=2, label='noise')
#plt.plot(flux_bins, flux_pdf_gnoise, drawstyle='steps-mid', color='y', alpha=1.0, lw=2, label='gaussian noise')
plt.axvline(one_minF_thresh, color='k', ls='--', lw=2)

plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('1$-$F')
plt.ylabel('PDF')
plt.tight_layout()
plt.show()
