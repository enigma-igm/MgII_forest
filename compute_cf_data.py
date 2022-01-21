'''
Functions here:
    - onespec
    - allspec
'''

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
fitsfile_list = ['/Users/suksientie/Research/data_redux/wavegrid_vel/J0313-1806/vel123_coadd_tellcorr.fits', \
                     '/Users/suksientie/Research/data_redux/wavegrid_vel/J1342+0928/vel123_coadd_tellcorr.fits', \
                     '/Users/suksientie/Research/data_redux/wavegrid_vel/J0252-0503/vel12_coadd_tellcorr.fits', \
                     '/Users/suksientie/Research/data_redux/wavegrid_vel/J0038-1527/vel1_tellcorr_pad.fits']
qso_namelist = ['J0313-1806', 'J1342+0928', 'J0252-0503', 'J0038-1527']
qso_zlist = [7.6, 7.54, 7.0, 7.0]
"""
everyn_break = 20

########## masks settings ##########
signif_thresh = 4.0
signif_mask_dv = 300.0
signif_mask_nsigma = 8 # 8 (used in chi_pdf.py)
one_minF_thresh = 0.3 # 0.3 (used in flux.py)
nbins = 81
sig_min = 1e-2
sig_max = 100.0

########## 2PCF settings ##########
mosfire_res = 3610 # K-band for 0.7" slit (https://www2.keck.hawaii.edu/inst/mosfire/grating.html)
fwhm = round(misc.convert_resolution(mosfire_res).value) # 83 km/s
vmin_corr = 10
vmax_corr = 2000
dv_corr = 100  # slightly larger than fwhm

def onespec(fitsfile, qso_z=None, plot=False, shuffle=False, seed=None):
    # compute the CF for one QSO spectrum

    # extract and continuum normalize
    # converting Angstrom to km/s
    wave, flux, ivar, mask, std, fluxfit, outmask, sset = mutils.extract_and_norm(fitsfile, everyn_break)
    good_wave, good_flux, good_std, good_ivar = wave[outmask], flux[outmask], std[outmask], ivar[outmask]
    norm_good_flux = good_flux / fluxfit
    norm_good_std = good_std / fluxfit
    vel = mutils.obswave_to_vel_2(good_wave)

    rand = np.random.RandomState(seed) if seed != None else np.random.RandomState()

    # if want to shuffle
    if shuffle:
        rand.shuffle(norm_good_flux)
        rand.shuffle(good_ivar)

    # if qso_z provided, then also mask out region redward of the QSO redshift
    if qso_z != None:
        redshift_mask = good_wave <= (2800 * (1 + qso_z))
        vel = vel[redshift_mask]
        norm_good_flux = norm_good_flux[redshift_mask]
        norm_good_std = norm_good_std[redshift_mask]
        good_ivar = good_ivar[redshift_mask]

    # reshaping arrays to be compatible with MgiiFinder
    norm_good_flux = norm_good_flux.reshape((1, len(norm_good_flux)))
    good_ivar = good_ivar.reshape((1, len(good_ivar)))
    norm_good_std = norm_good_std.reshape((1, len(norm_good_std)))

    mgii_tot = MgiiFinder(vel, norm_good_flux, good_ivar, fwhm, signif_thresh, signif_mask_nsigma=signif_mask_nsigma,
                          signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh)

    ### CF from not masking
    meanflux_tot = np.mean(norm_good_flux)
    deltaf_tot = (norm_good_flux - meanflux_tot) / meanflux_tot
    vel_mid, xi_tot, npix_tot, _ = reion_utils.compute_xi(deltaf_tot, vel, vmin_corr, vmax_corr, dv_corr)
    xi_mean_tot = np.mean(xi_tot, axis=0)  # not really averaging here since it's one spectrum (i.e. xi_mean_tot = xi_tot)

    ### CF from masking
    meanflux_tot_mask = np.mean(norm_good_flux[mgii_tot.fit_gpm])
    deltaf_tot_mask = (norm_good_flux - meanflux_tot_mask) / meanflux_tot_mask
    vel_mid, xi_tot_mask, npix_tot_chimask, _ = reion_utils.compute_xi(deltaf_tot_mask, vel, vmin_corr, vmax_corr,
                                                                       dv_corr, gpm=mgii_tot.fit_gpm)
    xi_mean_tot_mask = np.mean(xi_tot_mask, axis=0) # again not really averaging here since we only have 1 spectrum

    print("MEAN FLUX", meanflux_tot, meanflux_tot_mask)
    print("mean(DELTA FLUX)", np.mean(deltaf_tot), np.mean(deltaf_tot_mask))

    # fractional path used
    fraction_used = np.sum(mgii_tot.fit_gpm) / mgii_tot.fit_gpm.size
    print("::::: fraction pixels used :::::", fraction_used)

    ### now dealing with CF from pure noise alone (not masking)
    #fnoise = rand.normal(0, norm_good_std)
    fnoise = []
    n_real = 500
    for i in range(n_real): # generate n_real of the noise vector
        fnoise.append(rand.normal(0, norm_good_std[0])) # taking the 0-th index because norm_good_std has been reshaped above
    fnoise = np.array(fnoise)

    #meanflux_tot = np.mean(fnoise)
    #deltaf_tot = (fnoise - meanflux_tot) / meanflux_tot
    deltaf_tot = fnoise / meanflux_tot
    vel_mid, xi_noise, npix_tot, _ = reion_utils.compute_xi(deltaf_tot, vel, vmin_corr, vmax_corr, dv_corr)

    ### noise CF with masking
    #meanflux_tot_mask = np.mean(fnoise[mgii_tot.fit_gpm])
    #deltaf_tot_mask = (fnoise - meanflux_tot_mask) / meanflux_tot_mask
    deltaf_tot_mask = fnoise / meanflux_tot_mask
    vel_mid, xi_noise_masked, npix_tot, _ = reion_utils.compute_xi(deltaf_tot_mask, vel, vmin_corr, vmax_corr, dv_corr, gpm=mgii_tot.fit_gpm)

    print("mean(DELTA FLUX)", np.mean(deltaf_tot), np.mean(deltaf_tot_mask))

    if plot:
        # plot with no masking
        plt.figure()
        for i in range(n_real):
            plt.plot(vel_mid, xi_noise[i], linewidth=1., c='tab:gray', ls='--', alpha=0.1)

        plt.plot(vel_mid, xi_mean_tot, linewidth=1.5, label='data unmasked')
        plt.legend(fontsize=15)
        plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
        plt.ylabel(r'$\xi(\Delta v)$', fontsize=18)
        vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
        plt.axvline(vel_doublet.value, color='red', linestyle=':', linewidth=1.5, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)

        # plot with masking
        plt.figure()
        for i in range(n_real):
            plt.plot(vel_mid, xi_noise_masked[i], linewidth=1., c='tab:gray', ls='--', alpha=0.1)
        plt.plot(vel_mid, xi_mean_tot_mask, linewidth=1.5, label='data masked')

        plt.legend(fontsize=15)
        plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
        plt.ylabel(r'$\xi(\Delta v)$', fontsize=18)
        vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
        plt.axvline(vel_doublet.value, color='red', linestyle=':', linewidth=1.5, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)

        plt.show()

    return vel, norm_good_flux, good_ivar, vel_mid, xi_tot, xi_tot_mask, xi_noise, xi_noise_masked

def allspec(fitsfile_list, qso_zlist, plot=False, shuffle=False, seed_list=[None, None, None, None]):
    # running onespec() for all the 4 QSOs

    xi_unmask_all = []
    xi_mask_all = []
    xi_noise_unmask_all = []
    xi_noise_mask_all = []

    for ifile, fitsfile in enumerate(fitsfile_list):
        v, f, ivar, vel_mid, xi_unmask, xi_mask, xi_noise, xi_noise_masked = onespec(fitsfile, qso_z=qso_zlist[ifile], shuffle=shuffle, seed=seed_list[ifile])
        xi_unmask_all.append(xi_unmask[0])
        xi_mask_all.append(xi_mask[0])
        #xi_noise_unmask_all.append(xi_noise[0])
        #xi_noise_mask_all.append(xi_noise_masked[0])
        xi_noise_unmask_all.append(xi_noise)
        xi_noise_mask_all.append(xi_noise_masked)

    ### un-masked quantities
    # data
    xi_unmask_all = np.array(xi_unmask_all)
    xi_mean_unmask = np.mean(xi_unmask_all, axis=0)
    xi_std_unmask = np.std(xi_unmask_all, axis=0)

    # noise
    xi_noise_unmask_all = np.array(xi_noise_unmask_all)
    #xi_mean_noise_unmask = np.mean(xi_noise_unmask_all, axis=0)

    ### masked quantities
    # data
    xi_mask_all = np.array(xi_mask_all)
    xi_mean_mask = np.mean(xi_mask_all, axis=0)
    xi_std_mask = np.std(xi_mask_all, axis=0)

    # noise
    xi_noise_mask_all = np.array(xi_noise_mask_all)
    #xi_mean_noise_mask = np.mean(xi_noise_mask_all, axis=0)

    if plot:
        plt.figure()
        for i in range(4):
            for xi in xi_noise_unmask_all[i]:
                #plt.plot(vel_mid, xi, linewidth=0.5, c='tab:gray', alpha=0.1)
                plt.plot(vel_mid, xi, linewidth=0.5, alpha=0.1)

        for xi in xi_unmask_all:
            plt.plot(vel_mid, xi, linewidth=0.7, c='tab:orange', alpha=0.5)

        plt.errorbar(vel_mid, xi_mean_unmask, yerr=xi_std_unmask/np.sqrt(4.), marker='o', c='tab:orange', ecolor='tab:orange', capthick=1.5, capsize=2, \
                     mec='none', label='data, unmasked', zorder=20)
        #plt.plot(vel_mid, xi_mean_noise_unmask, linewidth=1.5, c='tab:gray', label='noise, unmasked')

        plt.legend(fontsize=15, loc=4)
        plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
        plt.ylabel(r'$\xi(\Delta v)$', fontsize=18)
        vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
        print("vel doublet at", vel_doublet.value)
        plt.axvline(vel_doublet.value, color='green', linestyle=':', linewidth=1.5, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
        plt.title('vmin_corr = %0.1f, vmax_corr = %0.1f, dv_corr = %0.1f' % (vmin_corr, vmax_corr, dv_corr), fontsize=15)

        plt.figure()
        for i in range(4):
            for xi in xi_noise_mask_all[i]:
                plt.plot(vel_mid, xi, linewidth=0.5, alpha=0.1)

        for xi in xi_mask_all:
            plt.plot(vel_mid, xi, linewidth=0.7, c='tab:orange', alpha=0.5)
        plt.errorbar(vel_mid, xi_mean_mask, yerr=xi_std_mask / np.sqrt(4.), marker='o', c='tab:orange', ecolor='tab:orange', capthick=1.5, capsize=2, \
                     mec='none', label='data, masked', zorder=20)
        #plt.plot(vel_mid, xi_mean_noise_mask, linewidth=1.5, c='tab:gray', label='noise, masked')

        plt.legend(fontsize=15, loc=4)
        plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
        plt.ylabel(r'$\xi(\Delta v)$', fontsize=18)
        vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
        print("vel doublet at", vel_doublet.value)
        plt.axvline(vel_doublet.value, color='green', linestyle=':', linewidth=1.5,
                    label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
        plt.title('vmin_corr = %0.1f, vmax_corr = %0.1f, dv_corr = %0.1f' % (vmin_corr, vmax_corr, dv_corr),
                  fontsize=15)

        plt.show()

    return vel_mid, xi_mean_unmask, xi_mean_mask

######################## old/unused/misc scripts ########################
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

def onespec_wave(fitsfile, qso_z=None, plot=False):
    # compute CF in wavelength unit (as opposed to dv unit)
    wave, flux, ivar, mask, std, fluxfit, outmask, sset = mutils.extract_and_norm(fitsfile, everyn_break)
    good_wave, good_flux, good_std, good_ivar = wave[outmask], flux[outmask], std[outmask], ivar[outmask]
    norm_good_flux = good_flux / fluxfit
    vel = mutils.obswave_to_vel(good_wave, vel_zeropoint=vel_zeropoint, wave_zeropoint_value=wave_zeropoint_value)

    if qso_z != None:
        redshift_mask = good_wave <= (2800 * (1 + qso_z))
        vel = vel[redshift_mask]
        good_wave = good_wave[redshift_mask]
        norm_good_flux = norm_good_flux[redshift_mask]
        good_std = good_std[redshift_mask]
        good_ivar = good_ivar[redshift_mask]

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

def allspec_shuffle(fitsfile_list, qso_zlist, nshuffle, seed):
    xi_mean_unmask_all = []
    xi_mean_mask_all = []

    rand = np.random.RandomState(seed)
    for i in range(nshuffle):
        vel_mid, xi_mean_unmask, xi_mean_mask = allspec(fitsfile_list, qso_zlist, shuffle=True)
        xi_mean_unmask_all.append(xi_mean_unmask)
        xi_mean_mask_all.append(xi_mean_mask)

    return vel_mid, xi_mean_unmask_all, xi_mean_mask_all

def allspec_noise(fitsfile_list, qso_zlist, everyn_break_list):
    # in progress, not masking CGM
    xi_allspec = []
    for iqso, fitsfile in enumerate(fitsfile_list):
        wave, flux, ivar, mask, std, fluxfit, outmask, sset = mutils.extract_and_norm(fitsfile, everyn_break_list[iqso])
        vel_data = mutils.obswave_to_vel_2(wave)

        redshift_mask = wave <= (2800 * (1 + qso_zlist[iqso]))  # removing spectral region beyond qso redshift
        master_mask = redshift_mask * outmask

        # masked arrays
        good_wave = wave[master_mask]
        good_flux = flux[master_mask]
        good_ivar = ivar[master_mask]
        good_std = std[master_mask]
        fluxfit_redshift = fluxfit[wave[outmask] <= (2800 * (1 + qso_zlist[iqso]))]
        norm_good_flux = good_flux / fluxfit_redshift
        norm_good_std = good_std / fluxfit_redshift
        good_vel_data = mutils.obswave_to_vel_2(good_wave)

        #fnoise_all = np.random.normal(0, norm_good_std)
        fnoise_all = np.random.normal(0, 1 / 100, norm_good_std.shape)
        meanflux_tot = np.mean(fnoise_all)
        deltaf_tot = (fnoise_all - meanflux_tot) / meanflux_tot
        vel_mid, xi_tot, npix_tot, _ = reion_utils.compute_xi(deltaf_tot, good_vel_data, vmin_corr, vmax_corr, dv_corr)
        xi_allspec.append(xi_tot[0])

    return vel_mid, xi_allspec

"""
datapath = '/Users/suksientie/Research/data_redux/'
#datapath = '/mnt/quasar/sstie/MgII_forest/z75/'

#fitsfile = '/Users/suksientie/Research/data_redux/mgii_stack_fits/J0313-1806_stacked_coadd_tellcorr.fits'
#qso_z = 7.6

#vel_zeropoint = True
#wave_zeropoint_value = (1 + 6) * 2800 # setting v=0 km/s at z=6 if vel_zeropoint=True
vel_zeropoint = False
wave_zeropoint_value = None
everyn_break = 20
"""
