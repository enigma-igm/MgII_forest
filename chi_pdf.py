from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('/Users/suksientie/codes/enigma')
sys.path.append('/Users/suksientie/Research/data_redux')
sys.path.append('/Users/suksientie/Research/CIV_forest')
from enigma.reion_forest.mgii_find import MgiiFinder
from enigma.reion_forest import utils
from pypeit import utils as putils
import misc # from CIV_forest
from scripts import rdx_utils
import mutils

mosfire_res = 3610 # K-band for 0.7" slit (https://www2.keck.hawaii.edu/inst/mosfire/grating.html)
fwhm = misc.convert_resolution(mosfire_res).value
print(fwhm)

fitsfile_list = ['/Users/suksientie/Research/data_redux/mgii_stack_fits/J0313-1806_stacked_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/mgii_stack_fits/J1342+0928_stacked_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/mgii_stack_fits/J0252-0503_stacked_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/2010_done/Redux/J0038-1527_201024_done/J0038-1527_coadd_tellcorr.fits']
qso_zlist = [7.6, 7.54, 7.0, 7.0]

wave_zeropoint_value = (1 + 6) * 2800
vel_zeropoint = True

# chi PDF
signif_thresh = 4.0
signif_mask_dv = 300.0
signif_mask_nsigma = 8 # not masking
one_minF_thresh = 0.2 # not masking
nbins = 81
sig_min = 1e-2
sig_max = 100.0

all_signif = []
all_std = []
all_high_signif_mask = []

for i, fitsfile in enumerate(fitsfile_list):
    wave, flux, ivar, std, mask, cont_flux, norm_std = rdx_utils.continuum_normalize(fitsfile, qso_zlist[i])

    vel = mutils.obswave_to_vel(wave[mask], vel_zeropoint=vel_zeropoint, wave_zeropoint_value=wave_zeropoint_value)
    x_mask = wave[mask] <= (2800 * (1 + qso_zlist[i]))

    vel = vel[x_mask]
    cont_flux = cont_flux[mask][x_mask]
    ivar = ivar[mask][x_mask]
    norm_std = norm_std[mask][x_mask]

    cont_flux = cont_flux.reshape((1, len(cont_flux)))
    ivar = ivar.reshape((1, len(ivar)))
    norm_std = norm_std.reshape((1, len(norm_std)))

    mgii_tot = MgiiFinder(vel, cont_flux, ivar, fwhm, signif_thresh, signif_mask_nsigma=signif_mask_nsigma,
                                signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh)
    sig_bins, sig_pdf_tot = utils.pdf_calc(mgii_tot.signif, sig_min, sig_max, nbins)
    plt.plot(sig_bins, sig_pdf_tot, drawstyle='steps-mid', alpha=0.5, label=fitsfile.split('/')[-1].split('_')[0])

    high_signif_mask = mgii_tot.signif[0] >= 10
    all_high_signif_mask.append(high_signif_mask)

    # placeholder for now
    norm_std = np.random.normal(0.0, 1.0 / 10, vel.shape)
    norm_std = norm_std.reshape((1, len(norm_std)))
    mgii_noise = MgiiFinder(vel, 1.0 + norm_std, ivar, fwhm, signif_thresh, signif_mask_nsigma=signif_mask_nsigma,
                                signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh)

    all_signif.extend(mgii_tot.signif[0])
    all_std.extend(mgii_noise.signif[0])

all_signif = np.array(all_signif)
all_std = np.array(all_std)

sig_bins, sig_pdf_tot = utils.pdf_calc(all_signif, sig_min, sig_max, nbins)
_, sig_pdf_noise = utils.pdf_calc(all_std, sig_min, sig_max, nbins)

plt.plot(sig_bins, sig_pdf_tot, drawstyle='steps-mid', color='k', alpha=1.0, lw=2, label='all spectra')
#plt.plot(sig_bins, sig_pdf_noise, drawstyle='steps-mid', color='y', alpha=1.0, lw=2, label='noise')
plt.axvline(signif_mask_nsigma, color='k', ls='--', lw=2)

plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\chi$')
plt.ylabel('PDF')
plt.tight_layout()
plt.show()
