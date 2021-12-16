from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('/Users/suksientie/codes/enigma')
from enigma.reion_forest import utils
from pypeit import utils as putils
sys.path.append('/Users/suksientie/Research/data_redux')
from scripts import rdx_utils
import mutils

# old
#fitsfile_list = ['/Users/suksientie/Research/data_redux/mgii_stack_fits/J0313-1806_stacked_coadd_tellcorr.fits', \
#                 '/Users/suksientie/Research/data_redux/mgii_stack_fits/J1342+0928_stacked_coadd_tellcorr.fits', \
#                 '/Users/suksientie/Research/data_redux/mgii_stack_fits/J0252-0503_stacked_coadd_tellcorr.fits', \
#                 '/Users/suksientie/Research/data_redux/2010_done/Redux/J0038-1527_201024_done/J0038-1527_coadd_tellcorr.fits']

fitsfile_list = ['/Users/suksientie/Research/data_redux/wavegrid_vel/J0313-1806/vel123_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/wavegrid_vel/J1342+0928/vel123_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/wavegrid_vel/J0252-0503/vel12_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/wavegrid_vel/J0038-1527/vel1_tellcorr_pad.fits']
qso_namelist =['J0313-1806', 'J1342+0928', 'J0252-0503', 'J0038-1527']

qso_zlist = [7.6, 7.54, 7.0, 7.0]
everyn_break_list = [20, 20, 20, 20] # b-spline fitting

#fitsfile_list = [fitsfile_list[3]]
#everyn_break_list = [20] # b-spline fitting

###### mask and PDF settings ######
nbins, oneminf_min, oneminf_max = 101, 1e-5, 1.0  # gives d(oneminf) = 0.01
one_minF_thresh = 0.3
seed = 4101877
if seed != None:
    rand = np.random.RandomState(seed)
else:
    rand = np.random.RandomState()

all_transmission = []
all_std = []
all_std_cent = []
noise = []
"""
for i, fitsfile in enumerate(fitsfile_list):
    wave, flux, ivar, std, mask, cont_flux, norm_std = rdx_utils.continuum_normalize(fitsfile, qso_zlist[i])
    wave, flux, ivar, mask, std = mutils.extract_data(fitsfile)
    
    x_mask = wave[mask] <= (2800 * (1 + qso_zlist[i]))
    all_transmission.extend(cont_flux[mask][x_mask])
    all_std.extend(norm_std[mask][x_mask])

    flux_bins, flux_pdf_tot = utils.pdf_calc(1.0 - cont_flux[mask][x_mask], oneminf_min, oneminf_max, nbins)
    plt.plot(flux_bins, flux_pdf_tot, drawstyle='steps-mid', alpha=0.5, label=fitsfile.split('/')[-1].split('_')[0])

    temp_std = []
    for elem in norm_std[mask][x_mask]:
        sign = 1 if rand.random() < 0.5 else -1
        temp_std.append(elem * sign)
    all_std_cent.extend(temp_std)
"""

for i, fitsfile in enumerate(fitsfile_list):
    wave, flux, ivar, mask, std, fluxfit, outmask, sset = mutils.extract_and_norm(fitsfile, everyn_break_list[i])

    good_wave, good_flux, good_std = wave[outmask], flux[outmask], std[outmask]
    norm_good_flux = good_flux / fluxfit
    norm_good_std = good_std / fluxfit

    all_transmission.extend(norm_good_flux)
    all_std.extend(norm_good_std)

    flux_bins, flux_pdf_tot = utils.pdf_calc(1.0 - norm_good_flux, oneminf_min, oneminf_max, nbins)
    plt.plot(flux_bins, flux_pdf_tot, drawstyle='steps-mid', alpha=0.5, label=qso_namelist[i])

    temp = []
    for i_std in norm_good_std:
        temp.append(rand.normal(0, i_std))
    noise.extend(temp)

all_transmission = np.array(all_transmission)
all_std = np.array(all_std)
all_std_cent = np.array(all_std_cent)
noise = np.array(noise)

SNR = 10.0
gaussian_noise = rand.normal(0.0, 1.0 / SNR, all_transmission.shape)

flux_bins, flux_pdf_tot = utils.pdf_calc(1.0 - all_transmission, oneminf_min, oneminf_max, nbins)
#_, flux_pdf_noise = utils.pdf_calc(all_std_cent, oneminf_min, oneminf_max, nbins)
_, flux_pdf_noise = utils.pdf_calc(noise, oneminf_min, oneminf_max, nbins)
_, flux_pdf_gnoise = utils.pdf_calc(gaussian_noise, oneminf_min, oneminf_max, nbins)

plt.plot(flux_bins, flux_pdf_tot, drawstyle='steps-mid', color='k', alpha=1.0, lw=2, label='all spectra')
plt.plot(flux_bins, flux_pdf_noise, drawstyle='steps-mid', color='orange', alpha=1.0, lw=2, label='all noise')
#plt.plot(flux_bins, flux_pdf_gnoise, drawstyle='steps-mid', color='y', alpha=1.0, lw=2, label='gaussian noise')
plt.axvline(one_minF_thresh, color='k', ls='--', lw=2)
plt.axvspan(one_minF_thresh, oneminf_max, facecolor = 'k', alpha = 0.2, label='masked')

plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('1$-$F')
plt.ylabel('PDF')
plt.tight_layout()
plt.show()
