from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('/Users/suksientie/codes/enigma')
sys.path.append('/Users/suksientie/Research/data_redux')
sys.path.append('/Users/suksientie/Research/CIV_forest')
from enigma.reion_forest.mgii_find import MgiiFinder
from enigma.reion_forest import utils as reion_utils
import misc # from CIV_forest
from scripts import rdx_utils
import mutils

"""
fitsfile_list = ['/Users/suksientie/Research/data_redux/mgii_stack_fits/J0313-1806_stacked_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/mgii_stack_fits/J1342+0928_stacked_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/mgii_stack_fits/J0252-0503_stacked_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/2010_done/Redux/J0038-1527_201024_done/J0038-1527_coadd_tellcorr.fits']
qso_zlist = [7.6, 7.54, 7.0, 7.0]
"""

#fitsfile = '/Users/suksientie/Research/data_redux/mgii_stack_fits/J0313-1806_stacked_coadd_tellcorr.fits'
#qso_z = 7.6

#vel_zeropoint = True
#wave_zeropoint_value = (1 + 6) * 2800 # setting v=0 km/s at z=6 if vel_zeropoint=True
vel_zeropoint = False
wave_zeropoint_value = None
everyn_break = 20

########## masks settings ##########
signif_thresh = 4.0
signif_mask_dv = 300.0
signif_mask_nsigma = 7 #8 # masking
one_minF_thresh = 0.2 #0.3 #0.2 # masking
nbins = 81
sig_min = 1e-2
sig_max = 100.0

########## 2PCF settings ##########
vmin_corr = 10
vmax_corr = 2000
dv_corr = 100  # slightly larger than fwhm
dv_corr = 60
mosfire_res = 3610 # K-band for 0.7" slit (https://www2.keck.hawaii.edu/inst/mosfire/grating.html)
fwhm = round(misc.convert_resolution(mosfire_res).value) # 83 km/s

def onespec(fitsfile, qso_z=None, plot=False, shuffle=False):
    wave, flux, ivar, mask, std, fluxfit, outmask, sset = mutils.extract_and_norm(fitsfile, everyn_break)
    good_wave, good_flux, good_std, good_ivar = wave[outmask], flux[outmask], std[outmask], ivar[outmask]
    norm_good_flux = good_flux / fluxfit
    #vel = mutils.obswave_to_vel(good_wave, vel_zeropoint=vel_zeropoint, wave_zeropoint_value=wave_zeropoint_value)
    vel = mutils.obswave_to_vel_2(good_wave)

    if shuffle:
        np.random.shuffle(norm_good_flux)
        np.random.shuffle(good_ivar)

    if qso_z != None:
        x_mask = good_wave <= (2800 * (1 + qso_z))
        vel = vel[x_mask]
        good_wave = good_wave[x_mask]
        norm_good_flux = norm_good_flux[x_mask]
        good_std = good_std[x_mask]
        good_ivar = good_ivar[x_mask]

    # reshaping to be compatible with MgiiFinder
    norm_good_flux = norm_good_flux.reshape((1, len(norm_good_flux)))
    good_ivar = good_ivar.reshape((1, len(good_ivar)))

    mgii_tot = MgiiFinder(vel, norm_good_flux, good_ivar, fwhm, signif_thresh, signif_mask_nsigma=signif_mask_nsigma,
                          signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh)

    # not masked
    meanflux_tot = np.mean(norm_good_flux)
    deltaf_tot = (norm_good_flux - meanflux_tot) / meanflux_tot
    vel_mid, xi_tot, npix_tot, _ = reion_utils.compute_xi(deltaf_tot, vel, vmin_corr, vmax_corr, dv_corr)
    xi_mean_tot = np.mean(xi_tot, axis=0)  # 2PCF from all the skewers

    # masked
    meanflux_tot_mask = np.mean(norm_good_flux[mgii_tot.fit_gpm])
    deltaf_tot_mask = (norm_good_flux - meanflux_tot_mask) / meanflux_tot_mask
    vel_mid, xi_tot_mask, npix_tot_chimask, _ = reion_utils.compute_xi(deltaf_tot_mask, vel, vmin_corr, vmax_corr,
                                                                       dv_corr, gpm=mgii_tot.fit_gpm)
    xi_mean_tot_mask = np.mean(xi_tot_mask, axis=0)

    # fractional path used
    fraction_used = np.sum(mgii_tot.fit_gpm) / mgii_tot.fit_gpm.size
    print("fraction pixels used", fraction_used)

    if plot:
        plt.plot(vel_mid, xi_mean_tot, linewidth=1.5, c='tab:gray', label='unmasked')
        plt.plot(vel_mid, xi_mean_tot_mask, linewidth=1.5, c='tab:orange', label='masked')

        plt.legend(fontsize=15)
        plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
        plt.ylabel(r'$\xi(\Delta v)$ $[10^{-5}]$', fontsize=18)
        vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
        print("vel doublet at", vel_doublet.value)
        plt.axvline(vel_doublet.value, color='red', linestyle=':', linewidth=1.5, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)

        plt.show()

    return vel, norm_good_flux, good_ivar, vel_mid, xi_tot, xi_tot_mask

def allspec(fitsfile_list, qso_zlist, plot=False, shuffle=False):
    xi_unmask_all = []
    xi_mask_all = []
    for ifile, fitsfile in enumerate(fitsfile_list):
        v, f, ivar, vel_mid, xi_unmask, xi_mask = onespec(fitsfile, qso_z=qso_zlist[ifile], shuffle=shuffle)
        xi_unmask_all.append(xi_unmask[0])
        xi_mask_all.append(xi_mask[0])

    xi_unmask_all = np.array(xi_unmask_all)
    xi_mask_all = np.array(xi_mask_all)

    xi_mean_unmask = np.mean(xi_unmask_all, axis=0)
    xi_mean_mask = np.mean(xi_mask_all, axis=0)
    xi_std_mask = np.std(xi_mask_all, axis=0)

    if plot:
        for xi in xi_mask_all:
            plt.plot(vel_mid, xi, linewidth=0.7, c='tab:orange', alpha=0.7)

        plt.errorbar(vel_mid, xi_mean_mask, yerr=xi_std_mask/np.sqrt(4-1), marker='o', c='tab:orange', ecolor='tab:orange', capthick=1.5, capsize=2, \
                     mec='none', label='masked', zorder=20)
        plt.plot(vel_mid, xi_mean_unmask, linewidth=1.5, c='tab:gray', label='unmasked')

        plt.legend(fontsize=15, loc=4)
        plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
        plt.ylabel(r'$\xi(\Delta v)$', fontsize=18)
        vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
        print("vel doublet at", vel_doublet.value)
        plt.axvline(vel_doublet.value, color='green', linestyle=':', linewidth=1.5, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)

        plt.show()

    return vel_mid, xi_mean_unmask, xi_mean_mask

def allspec_shuffle(fitsfile_list, qso_zlist, nshuffle, seed):
    xi_mean_unmask_all = []
    xi_mean_mask_all = []

    rand = np.random.RandomState(seed)
    for i in range(nshuffle):
        vel_mid, xi_mean_unmask, xi_mean_mask = allspec(fitsfile_list, qso_zlist, shuffle=True)
        xi_mean_unmask_all.append(xi_mean_unmask)
        xi_mean_mask_all.append(xi_mean_mask)

    return vel_mid, xi_mean_unmask_all, xi_mean_mask_all

def onespec_wave(fitsfile, qso_z=None, plot=False):
    # compute CF in wavelength unit
    wave, flux, ivar, mask, std, fluxfit, outmask, sset = mutils.extract_and_norm(fitsfile, everyn_break)
    good_wave, good_flux, good_std, good_ivar = wave[outmask], flux[outmask], std[outmask], ivar[outmask]
    norm_good_flux = good_flux / fluxfit
    vel = mutils.obswave_to_vel(good_wave, vel_zeropoint=vel_zeropoint, wave_zeropoint_value=wave_zeropoint_value)

    if qso_z != None:
        x_mask = good_wave <= (2800 * (1 + qso_z))
        vel = vel[x_mask]
        good_wave = good_wave[x_mask]
        norm_good_flux = norm_good_flux[x_mask]
        good_std = good_std[x_mask]
        good_ivar = good_ivar[x_mask]

    # reshaping to be compatible with MgiiFinder
    norm_good_flux = norm_good_flux.reshape((1, len(norm_good_flux)))
    good_ivar = good_ivar.reshape((1, len(good_ivar)))

    mgii_tot = MgiiFinder(vel, norm_good_flux, good_ivar, fwhm, signif_thresh, signif_mask_nsigma=signif_mask_nsigma,
                          signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh)

    # not masked
    meanflux_tot = np.mean(norm_good_flux)
    deltaf_tot = (norm_good_flux - meanflux_tot) / meanflux_tot
    wave_mid, xi_tot, npix_tot, _ = reion_utils.compute_xi(deltaf_tot, good_wave, 2, 200, 4)
    xi_mean_tot = np.mean(xi_tot, axis=0)  # 2PCF from all the skewers

    # masked
    meanflux_tot_mask = np.mean(norm_good_flux[mgii_tot.fit_gpm])
    deltaf_tot_mask = (norm_good_flux - meanflux_tot_mask) / meanflux_tot_mask
    wave_mid, xi_tot_mask, npix_tot_chimask, _ = reion_utils.compute_xi(deltaf_tot_mask, good_wave, 2, 200, 4, gpm=mgii_tot.fit_gpm)
    xi_mean_tot_mask = np.mean(xi_tot_mask, axis=0)

    # fractional path used
    fraction_used = np.sum(mgii_tot.fit_gpm) / mgii_tot.fit_gpm.size
    print("fraction pixels used", fraction_used)

    if plot:
        plt.plot(wave_mid, xi_mean_tot, linewidth=1.5, c='tab:gray', label='unmasked')
        plt.plot(wave_mid, xi_mean_tot_mask, linewidth=1.5, c='tab:orange', label='masked')

        plt.legend(fontsize=15)
        plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
        plt.ylabel(r'$\xi(\Delta v)$', fontsize=18)
        vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
        print("vel doublet at", vel_doublet.value)
        #plt.axvline(vel_doublet.value, color='red', linestyle=':', linewidth=1.5, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)

        plt.show()

    return vel, norm_good_flux, good_ivar, wave_mid, xi_tot, xi_tot_mask

######### old scripts #########
def onespec_old(fitsfile, qso_z):
    wave, flux, ivar, std, mask, cont_flux, norm_std = rdx_utils.continuum_normalize(fitsfile, qso_z)
    vel = mutils.obswave_to_vel(wave[mask], vel_zeropoint=vel_zeropoint, wave_zeropoint_value=wave_zeropoint_value)
    x_mask = wave[mask] <= (2800 * (1 + qso_z))

    # masking bad pixels
    vel = vel[x_mask]
    cont_flux = cont_flux[mask][x_mask]
    ivar = ivar[mask][x_mask]
    norm_std = norm_std[mask][x_mask]

    # reshaping to be compatible with MgiiFinder
    cont_flux = cont_flux.reshape((1, len(cont_flux)))
    ivar = ivar.reshape((1, len(ivar)))
    norm_std = norm_std.reshape((1, len(norm_std)))

    return vel, cont_flux, ivar

def compute(vel, cont_flux, ivar):
    mgii_tot = MgiiFinder(vel, cont_flux, ivar, fwhm, signif_thresh, signif_mask_nsigma=signif_mask_nsigma, signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh)

    ########## 2PCF ##########
    vmin_corr = 10
    vmax_corr = 2000
    dv_corr = 100 # slightly larger than fwhm

    # not masked
    meanflux_tot = np.mean(cont_flux)
    deltaf_tot = (cont_flux - meanflux_tot) / meanflux_tot
    vel_mid, xi_tot, npix_tot, _ = reion_utils.compute_xi(deltaf_tot, vel, vmin_corr, vmax_corr, dv_corr)
    xi_mean_tot = np.mean(xi_tot, axis=0) # 2PCF from all the skewers

    # masked
    meanflux_tot_mask = np.mean(cont_flux[mgii_tot.fit_gpm])
    deltaf_tot_mask = (cont_flux - meanflux_tot_mask) / meanflux_tot_mask
    vel_mid, xi_tot_mask, npix_tot_chimask, _ = reion_utils.compute_xi(deltaf_tot_mask, vel, vmin_corr, vmax_corr, dv_corr, gpm=mgii_tot.fit_gpm)
    xi_mean_tot_mask = np.mean(xi_tot_mask, axis=0)

    # fractional path used
    fraction_used = np.sum(mgii_tot.fit_gpm)/mgii_tot.fit_gpm.size
    print("fraction pixels used", fraction_used)

    plt.plot(vel_mid, xi_mean_tot, linewidth=1.5, c='tab:gray', label='unmasked')
    plt.plot(vel_mid, xi_mean_tot_mask, linewidth=1.5, c='tab:orange', label='masked')

    plt.legend(fontsize=15)
    plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
    plt.ylabel(r'$\xi(\Delta v)$ $[10^{-5}]$', fontsize=18)
    vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
    print("vel doublet at", vel_doublet.value)
    plt.axvline(vel_doublet.value, color='red', linestyle=':', linewidth=1.5, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)

    plt.show()

    return vel_mid, xi_tot_mask