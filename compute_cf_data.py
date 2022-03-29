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
#import misc # from CIV_forest
#from scripts import rdx_utils
import mutils

"""
fitsfile_list = ['/Users/suksientie/Research/data_redux/wavegrid_vel/J0313-1806/vel1234_coadd_tellcorr.fits', \
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
#fwhm = round(misc.convert_resolution(mosfire_res).value) # 83 km/s
fwhm = 90 # what is used in compute_model_grid.py

vmin_corr = 10
vmax_corr = 2000
dv_corr = 100  # slightly larger than fwhm
corr_all = [0.689, 0.640, 0.616, 0.583] # values used by compute_model_grid_new.py

def onespec(fitsfile, qso_z=None, shuffle=False, seed=None, plot=False, std_corr=1.0):
    # compute the CF for one QSO spectrum

    # extract and continuum normalize
    # converting Angstrom to km/s
    wave, flux, ivar, mask, std, fluxfit, outmask, sset, tell = mutils.extract_and_norm(fitsfile, everyn_break)

    # the "norm" and "good" arrays are what MgiiFinder and CF will operate on
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
    # applied on the "good" arrays, i.e. quantities that have been masked
    norm_good_flux = norm_good_flux.reshape((1, len(norm_good_flux)))
    good_ivar = good_ivar.reshape((1, len(good_ivar)))
    norm_good_std = norm_good_std.reshape((1, len(norm_good_std)))

    mgii_tot = MgiiFinder(vel, norm_good_flux, good_ivar, fwhm, signif_thresh, signif_mask_nsigma=signif_mask_nsigma,
                          signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh)

    ### CF from not masking (computed from "good" masked arrays)
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
    n_real = 50
    for i in range(n_real): # generate n_real of the noise vector
        fnoise.append(rand.normal(0, std_corr*norm_good_std[0])) # taking the 0-th index because norm_good_std has been reshaped above
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

    return vel, norm_good_flux, good_ivar, vel_mid, xi_tot, xi_tot_mask, xi_noise, xi_noise_masked, mgii_tot.fit_gpm

def onespec2(iqso, seed=None, plot=False):
    # 1/25/2022: in progress
    rand = np.random.RandomState(seed) if seed != None else np.random.RandomState()

    vel_data, master_mask, std, fluxfit, outmask, norm_good_std, norm_std, norm_good_flux, good_vel_data, good_ivar, norm_flux, ivar = mutils.init_onespec(iqso)

    norm_good_flux = norm_good_flux.reshape((1, len(norm_good_flux)))
    good_ivar = good_ivar.reshape((1, len(good_ivar)))
    norm_good_std = norm_good_std.reshape((1, len(norm_good_std)))

    mgii_tot = MgiiFinder(good_vel_data, norm_good_flux, good_ivar, fwhm, signif_thresh, signif_mask_nsigma=signif_mask_nsigma,
                          signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh)

    """
    norm_flux = norm_flux.reshape((1, len(norm_flux)))
    ivar = ivar.reshape((1, len(ivar)))
    norm_std = norm_std.reshape((1, len(norm_std)))

    mgii_tot = MgiiFinder(vel_data, norm_flux, ivar, fwhm, signif_thresh,
                          signif_mask_nsigma=signif_mask_nsigma,
                          signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh)
    """
    ### CF from not masking
    meanflux_tot = np.mean(norm_good_flux) # computed from "good" masked arrays
    #deltaf_tot = (norm_good_flux - meanflux_tot) / meanflux_tot
    deltaf_tot = (norm_flux - meanflux_tot) / meanflux_tot
    vel_mid, xi_tot, npix_tot, _ = reion_utils.compute_xi(deltaf_tot, good_vel_data, vmin_corr, vmax_corr, dv_corr, gpm=master_mask)
    xi_mean_tot = np.mean(xi_tot, axis=0)  # not really averaging here since it's one spectrum (i.e. xi_mean_tot = xi_tot)

    ### CF from masking
    meanflux_tot_mask = np.mean(norm_good_flux[mgii_tot.fit_gpm])
    #deltaf_tot_mask = (norm_good_flux - meanflux_tot_mask) / meanflux_tot_mask
    deltaf_tot_mask = (norm_flux - meanflux_tot_mask) / meanflux_tot_mask
    vel_mid, xi_tot_mask, npix_tot_chimask, _ = reion_utils.compute_xi(deltaf_tot_mask, good_vel_data, vmin_corr, vmax_corr,
                                                                       dv_corr, gpm=mgii_tot.fit_gpm)
    xi_mean_tot_mask = np.mean(xi_tot_mask, axis=0)  # again not really averaging here since we only have 1 spectrum

    print("MEAN FLUX", meanflux_tot, meanflux_tot_mask)
    print("mean(DELTA FLUX)", np.mean(deltaf_tot), np.mean(deltaf_tot_mask))

    # fractional path used
    fraction_used = np.sum(mgii_tot.fit_gpm) / mgii_tot.fit_gpm.size
    print("::::: fraction pixels used :::::", fraction_used)

    ### now dealing with CF from pure noise alone (not masking)
    # fnoise = rand.normal(0, norm_good_std)
    fnoise = []
    n_real = 500
    for i in range(n_real):  # generate n_real of the noise vector
        fnoise.append(
            rand.normal(0, norm_good_std[0]))  # taking the 0-th index because norm_good_std has been reshaped above
    fnoise = np.array(fnoise)

    # meanflux_tot = np.mean(fnoise)
    # deltaf_tot = (fnoise - meanflux_tot) / meanflux_tot
    deltaf_tot = fnoise / meanflux_tot
    vel_mid, xi_noise, npix_tot, _ = reion_utils.compute_xi(deltaf_tot, good_vel_data, vmin_corr, vmax_corr, dv_corr)

    ### noise CF with masking
    # meanflux_tot_mask = np.mean(fnoise[mgii_tot.fit_gpm])
    # deltaf_tot_mask = (fnoise - meanflux_tot_mask) / meanflux_tot_mask
    deltaf_tot_mask = fnoise / meanflux_tot_mask
    vel_mid, xi_noise_masked, npix_tot, _ = reion_utils.compute_xi(deltaf_tot_mask, good_vel_data, vmin_corr, vmax_corr, dv_corr,
                                                                   gpm=mgii_tot.fit_gpm)

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
        plt.axvline(vel_doublet.value, color='red', linestyle=':', linewidth=1.5,
                    label='Doublet separation (%0.1f km/s)' % vel_doublet.value)

        # plot with masking
        plt.figure()
        for i in range(n_real):
            plt.plot(vel_mid, xi_noise_masked[i], linewidth=1., c='tab:gray', ls='--', alpha=0.1)
        plt.plot(vel_mid, xi_mean_tot_mask, linewidth=1.5, label='data masked')

        plt.legend(fontsize=15)
        plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
        plt.ylabel(r'$\xi(\Delta v)$', fontsize=18)
        vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
        plt.axvline(vel_doublet.value, color='red', linestyle=':', linewidth=1.5,
                    label='Doublet separation (%0.1f km/s)' % vel_doublet.value)

        plt.show()

    return good_vel_data, norm_good_flux, good_ivar, vel_mid, xi_tot, xi_tot_mask, xi_noise, xi_noise_masked, mgii_tot.fit_gpm

from IPython import embed
def allspec(fitsfile_list, qso_zlist, plot=False, shuffle=False, seed_list=[None, None, None, None]):
    # running onespec() for all the 4 QSOs

    xi_unmask_all = []
    xi_mask_all = []
    xi_noise_unmask_all = []
    xi_noise_mask_all = []

    for ifile, fitsfile in enumerate(fitsfile_list):
        std_corr = corr_all[ifile]
        print("std_corr", std_corr)
        v, f, ivar, vel_mid, xi_unmask, xi_mask, xi_noise, xi_noise_masked, _ = onespec(fitsfile, qso_z=qso_zlist[ifile], shuffle=shuffle, seed=seed_list[ifile], std_corr=std_corr)
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
    xi_noise_unmask_all = np.array(xi_noise_unmask_all) # = (nqso, n_real, n_velmid)
    #xi_mean_noise_unmask = np.mean(xi_noise_unmask_all, axis=0)

    ### masked quantities
    # data
    xi_mask_all = np.array(xi_mask_all)
    xi_mean_mask = np.mean(xi_mask_all, axis=0)
    xi_std_mask = np.std(xi_mask_all, axis=0)

    # noise
    xi_noise_mask_all = np.array(xi_noise_mask_all)
    #xi_mean_noise_mask = np.mean(xi_noise_mask_all, axis=0)

    #embed()
    if plot:
        xi_scale = 1e5
        ymin, ymax = -0.0010 * xi_scale, 0.0006 * xi_scale

        plt.figure()
        for i in range(4):
            for xi in xi_noise_unmask_all[i]: # plotting all 500 realizations of the noise 2PCF (not masked)
                plt.plot(vel_mid, xi*xi_scale, c='k', linewidth=0.5, alpha=0.1)

        for xi in xi_unmask_all:
            plt.plot(vel_mid, xi*xi_scale, linewidth=1.0, c='tab:orange', alpha=0.7)

        plt.errorbar(vel_mid, xi_mean_unmask*xi_scale, yerr=(xi_std_unmask/np.sqrt(4.))*xi_scale, lw=2.0, marker='o', c='tab:orange', ecolor='tab:orange', capthick=2.0, capsize=2, \
                     mec='none', label='data, unmasked', zorder=20)
        #plt.plot(vel_mid, xi_mean_noise_unmask, linewidth=1.5, c='tab:gray', label='noise, unmasked')

        plt.legend(fontsize=15, loc=4)
        plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
        plt.ylabel(r'$\xi(\Delta v) \times 10^5$', fontsize=18)
        vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
        print("vel doublet at", vel_doublet.value)
        plt.axvline(vel_doublet.value, color='green', linestyle=':', linewidth=1.5, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
        #plt.title('vmin_corr = %0.1f, vmax_corr = %0.1f, dv_corr = %0.1f' % (vmin_corr, vmax_corr, dv_corr), fontsize=15)
        plt.ylim([ymin, ymax])

        plt.figure()
        for i in range(4):
            for xi in xi_noise_mask_all[i]: # plotting all 500 realizations of the noise 2PCF (masked)
                plt.plot(vel_mid, xi*xi_scale, c='k', linewidth=0.5, alpha=0.1)

        for xi in xi_mask_all:
            plt.plot(vel_mid, xi*xi_scale, linewidth=1.0, c='tab:orange', alpha=0.7)

        plt.errorbar(vel_mid, xi_mean_mask*xi_scale, yerr=(xi_std_mask / np.sqrt(4.))*xi_scale, lw=2.0, marker='o', c='tab:orange', ecolor='tab:orange', capthick=2.0, capsize=2, \
                     mec='none', label='data, masked', zorder=20)
        #plt.plot(vel_mid, xi_mean_noise_mask, linewidth=1.5, c='tab:gray', label='noise, masked')

        plt.legend(fontsize=15, loc=4)
        plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
        plt.ylabel(r'$\xi(\Delta v) \times 10^5$', fontsize=18)
        vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
        print("vel doublet at", vel_doublet.value)
        plt.axvline(vel_doublet.value, color='green', linestyle=':', linewidth=1.5,
                    label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
        #plt.title('vmin_corr = %0.1f, vmax_corr = %0.1f, dv_corr = %0.1f' % (vmin_corr, vmax_corr, dv_corr),
        #          fontsize=15)
        plt.ylim([ymin, ymax])

        plt.show()

    return vel_mid, xi_mean_unmask, xi_mean_mask, xi_noise_unmask_all, xi_noise_mask_all, xi_unmask_all, xi_mask_all

# testing code (not used; 3/29/2022), instead use allspec() above
def allspec_bccptalk(fitsfile_list, qso_zlist, plot=False, shuffle=False, seed_list=[None, None, None, None], plot_noise_realization=False):
    # running onespec() for all the 4 QSOs

    xi_unmask_all = []
    xi_mask_all = []
    xi_noise_unmask_all = []
    xi_noise_mask_all = []

    for ifile, fitsfile in enumerate(fitsfile_list):
        v, f, ivar, vel_mid, xi_unmask, xi_mask, xi_noise, xi_noise_masked, _ = onespec(fitsfile, qso_z=qso_zlist[ifile], shuffle=shuffle, seed=seed_list[ifile])
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
    xi_noise_unmask_all = np.array(xi_noise_unmask_all) # = (nqso, n_real, n_velmid)
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
        xi_scale = 1e5
        ymin, ymax = -0.0004*xi_scale, 0.0003*xi_scale

        #### plotting unmasked ####
        plt.figure()
        if plot_noise_realization:
            print("yes")
            for i in range(4):
                for xi in xi_noise_unmask_all[i]: # plotting all 500 realizations of the noise 2PCF (not masked)
                    plt.plot(vel_mid, xi, linewidth=0.5, alpha=0.1)

        for xi in xi_unmask_all:
            plt.plot(vel_mid, xi*xi_scale, linewidth=0.7, c='tab:orange', alpha=0.5)

        plt.errorbar(vel_mid, xi_mean_unmask*xi_scale, yerr=(xi_std_unmask/np.sqrt(4.))*xi_scale, marker='o', c='tab:orange', ecolor='tab:orange', capthick=1.5, capsize=2, \
                     mec='none', label='data, unmasked', zorder=20)
        #plt.plot(vel_mid, xi_mean_noise_unmask, linewidth=1.5, c='tab:gray', label='noise, unmasked')

        plt.legend(fontsize=15, loc=4)
        plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
        plt.ylabel(r'$\xi(\Delta v) \times 10^5$', fontsize=18)
        vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
        print("vel doublet at", vel_doublet.value)
        plt.axvline(vel_doublet.value, color='green', linestyle=':', linewidth=1.5, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
        plt.ylim([ymin, ymax])

        #### plotting masked ####
        plt.figure()
        if plot_noise_realization:
            for i in range(4):
                for xi in xi_noise_mask_all[i]: # plotting all 500 realizations of the noise 2PCF (masked)
                    plt.plot(vel_mid, xi, linewidth=0.5, alpha=0.1)

        for xi in xi_mask_all:
            plt.plot(vel_mid, xi*xi_scale, linewidth=0.7, c='tab:orange', alpha=0.5)

        plt.errorbar(vel_mid, xi_mean_mask*xi_scale, yerr=(xi_std_mask / np.sqrt(4.))*xi_scale, marker='o', c='tab:orange', ecolor='tab:orange', capthick=1.5, capsize=2, \
                     mec='none', label='data, masked', zorder=20)
        #plt.plot(vel_mid, xi_mean_noise_mask, linewidth=1.5, c='tab:gray', label='noise, masked')

        plt.legend(fontsize=15, loc=4)
        plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
        plt.ylabel(r'$\xi(\Delta v) \times 10^5$', fontsize=18)
        vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
        print("vel doublet at", vel_doublet.value)
        plt.axvline(vel_doublet.value, color='green', linestyle=':', linewidth=1.5,
                    label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
        plt.ylim([ymin, ymax])

        plt.show()

    return vel_mid, xi_mean_unmask, xi_mean_mask, xi_noise_unmask_all, xi_noise_mask_all

