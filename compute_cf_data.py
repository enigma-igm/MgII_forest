'''
Functions here:
    - init_cgm_fit_gpm
    - onespec
    - allspec
    - plot_allspec
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
import mask_cgm_pdf as mask_cgm

"""
everyn_break = 20
exclude_restwave = 1216 - 1185 # excluding proximity zones; see mutils.qso_exclude_proximity_zone

########## 2PCF settings ##########
mosfire_res = 3610 # K-band for 0.7" slit (https://www2.keck.hawaii.edu/inst/mosfire/grating.html)
fwhm = 90 # used in compute_model_grid.py

qso_namelist = ['J0313-1806', 'J1342+0928', 'J0252-0503', 'J0038-1527', 'J0038-0653']
vmin_corr = 10
vmax_corr = 3500
dv_corr = 90  # 5/23/2022; to ensure npix_corr behaving correctly (smoothly)
corr_all = [0.758, 0.753, 0.701, 0.724, 0.763] # (updated corr for J0038-0653); determined from mutils.plot_allspec_pdf for redshift_bin='all'
median_z = 6.554 # value used in mutils.init_onespec
"""

fwhm = 40
sampling = 3

qso_namelist = ['J0411-0907', 'J0319-1008', 'newqso1', 'newqso2', 'J0313-1806', 'J0038-1527', 'J0252-0503', 'J1342+0928']
qso_zlist = [6.826, 6.8275, 7.0, 7.1, 7.642, 7.034, 7.001, 7.541]
everyn_break_list = (np.ones(len(qso_namelist)) * 20).astype('int')
exclude_restwave = 1216 - 1185
median_z = 6.50
corr_all = [0.669, 0.673, 0.692, 0.73 , 0.697, 0.653, 0.667, 0.72]

vmin_corr, vmax_corr, dv_corr = 10, 3500, 40 # dummy values because we're now using custom binning

def init_cgm_fit_gpm(datapath='/Users/suksientie/Research/MgII_forest/rebinned_spectra/'):
    lowz_mgii_tot_all, highz_mgii_tot_all, allz_mgii_tot_all = mask_cgm.do_allqso_allzbin(datapath)

    lowz_fit_gpm = []
    for i in range(len(lowz_mgii_tot_all)):
        lowz_fit_gpm.append(lowz_mgii_tot_all[i].fit_gpm[0])

    highz_fit_gpm = []
    for i in range(len(highz_mgii_tot_all)):
        highz_fit_gpm.append(highz_mgii_tot_all[i].fit_gpm[0])

    allz_fit_gpm = []
    for i in range(len(allz_mgii_tot_all)):
        allz_fit_gpm.append(allz_mgii_tot_all[i].fit_gpm[0])

    return lowz_fit_gpm, highz_fit_gpm, allz_fit_gpm

def onespec(iqso, redshift_bin, cgm_fit_gpm, plot=False, std_corr=1.0, seed=None, given_bins=None):
    # compute the CF for one QSO spectrum
    # updated 4/14/2022
    # options for redshift_bin are 'low', 'high', 'all'
    # cgm_fit_gpm are gpm from MgiiFinder.py

    raw_data_out, _, all_masks_out = mutils.init_onespec(iqso, redshift_bin)
    wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
    strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, telluric_mask, master_mask = all_masks_out

    ###### CF from not masking CGM ######
    all_masks = master_mask

    norm_good_flux = (flux / fluxfit)[all_masks]
    #norm_good_std = (std / fluxfit)[all_masks]
    #good_wave = wave[all_masks]
    #vel = mutils.obswave_to_vel_2(wave)
    #vel = vel[all_masks]
    #meanflux_tot = np.mean(norm_good_flux)
    #deltaf_tot = (norm_good_flux - meanflux_tot) / meanflux_tot
    #vel_mid, xi_tot, npix_tot, _ = reion_utils.compute_xi(deltaf_tot, vel, vmin_corr, vmax_corr, dv_corr)
    #xi_mean_tot = np.mean(xi_tot, axis=0)  # not really averaging here since it's one spectrum (i.e. xi_mean_tot = xi_tot)

    norm_flux = flux/fluxfit
    vel = mutils.obswave_to_vel_2(wave)
    meanflux_tot = np.mean(norm_good_flux)
    deltaf_tot = (norm_flux - meanflux_tot) / meanflux_tot
    #return vel, np.zeros(norm_flux.shape)
    vel_mid, xi_tot, npix_tot, _ = reion_utils.compute_xi(deltaf_tot, vel, vmin_corr, vmax_corr, dv_corr, given_bins=given_bins, gpm=all_masks)
    xi_mean_tot = np.mean(xi_tot, axis=0)  # not really averaging here since it's one spectrum (i.e. xi_mean_tot = xi_tot)

    ###### CF from masking CGM ######
    norm_good_flux_cgm = norm_good_flux[cgm_fit_gpm]
    meanflux_tot_mask = np.mean(norm_good_flux_cgm)
    #deltaf_tot_mask = (norm_good_flux_cgm - meanflux_tot_mask) / meanflux_tot_mask
    #vel_cgm = vel[all_masks][cgm_fit_gpm]
    deltaf_tot_mask = (norm_good_flux - meanflux_tot_mask) / meanflux_tot_mask
    vel_cgm = vel[all_masks]
    vel_mid, xi_tot_mask, npix_tot_chimask, _ = reion_utils.compute_xi(deltaf_tot_mask, vel_cgm, vmin_corr, vmax_corr, dv_corr, given_bins=given_bins, gpm=cgm_fit_gpm)
    xi_mean_tot_mask = np.mean(xi_tot_mask, axis=0) # again not really averaging here since we only have 1 spectrum

    print("MEAN FLUX", meanflux_tot, meanflux_tot_mask)
    print("mean(DELTA FLUX)", np.mean(deltaf_tot), np.mean(deltaf_tot_mask))

    #print("npix_tot", npix_tot_chimask)
    #print("sum(npix_tot)", np.sum(npix_tot_chimask))

    ###### CF from pure noise (no CGM masking) ######
    rand = np.random.RandomState(seed) if seed != None else np.random.RandomState()
    norm_std = std / fluxfit
    fnoise = []
    fnoise_masked = []
    n_real = 50

    xi_noise = None
    xi_noise_masked = None

    if plot:
        # plot with no masking
        plt.figure(figsize=(12, 5))
        plt.suptitle('%s, %s-z bin' % (qso_namelist[iqso], redshift_bin))
        plt.subplot(121)
        plt.plot(vel_mid, xi_mean_tot, linewidth=1.5, label='data unmasked')
        plt.axhline(0, color='k', ls='--')
        plt.legend(fontsize=15)
        plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
        plt.ylabel(r'$\xi(\Delta v)$', fontsize=18)
        vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
        plt.axvline(vel_doublet.value, color='red', linestyle=':', linewidth=1.5, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)

        # plot with masking
        plt.subplot(122)
        plt.plot(vel_mid, xi_mean_tot_mask, linewidth=1.5, label='data masked')
        plt.axhline(0, color='k', ls='--')
        plt.legend(fontsize=15)
        plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
        plt.ylabel(r'$\xi(\Delta v)$', fontsize=18)
        vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
        plt.axvline(vel_doublet.value, color='red', linestyle=':', linewidth=1.5, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)

        plt.tight_layout()
        plt.show()

    #return vel, norm_good_flux, good_ivar, vel_mid, xi_tot, xi_tot_mask, xi_noise, xi_noise_masked, mgii_tot.fit_gpm
    return vel_mid, xi_tot, xi_tot_mask, xi_noise, xi_noise_masked, npix_tot, npix_tot_chimask

def onespec_o(iqso, redshift_bin, cgm_fit_gpm, plot=False, std_corr=1.0, seed=None, given_bins=None):
    # compute the CF for one QSO spectrum
    # updated 4/14/2022
    # options for redshift_bin are 'low', 'high', 'all'
    # cgm_fit_gpm are gpm from MgiiFinder.py

    raw_data_out, _, all_masks_out = mutils.init_onespec(iqso, redshift_bin)
    wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
    strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, master_mask = all_masks_out

    # v_lo_all, v_hi_all = custom_cf_bin()
    # given_bins = (v_lo_all, v_hi_all)

    ###### CF from not masking CGM ######
    all_masks = mask * redshift_mask * pz_mask * zbin_mask

    norm_good_flux = (flux / fluxfit)[all_masks]
    # norm_good_std = (std / fluxfit)[all_masks]
    # good_wave = wave[all_masks]
    # vel = mutils.obswave_to_vel_2(wave)
    # vel = vel[all_masks]
    # meanflux_tot = np.mean(norm_good_flux)
    # deltaf_tot = (norm_good_flux - meanflux_tot) / meanflux_tot
    # vel_mid, xi_tot, npix_tot, _ = reion_utils.compute_xi(deltaf_tot, vel, vmin_corr, vmax_corr, dv_corr)
    # xi_mean_tot = np.mean(xi_tot, axis=0)  # not really averaging here since it's one spectrum (i.e. xi_mean_tot = xi_tot)

    norm_flux = flux / fluxfit
    vel = mutils.obswave_to_vel_2(wave)
    meanflux_tot = np.mean(norm_good_flux)
    deltaf_tot = (norm_flux - meanflux_tot) / meanflux_tot
    # return vel, np.zeros(norm_flux.shape)
    vel_mid, xi_tot, npix_tot, _ = reion_utils.compute_xi(deltaf_tot, vel, vmin_corr, vmax_corr, dv_corr,
                                                          given_bins=given_bins, gpm=all_masks)
    xi_mean_tot = np.mean(xi_tot,
                          axis=0)  # not really averaging here since it's one spectrum (i.e. xi_mean_tot = xi_tot)

    ###### CF from masking CGM ######
    norm_good_flux_cgm = norm_good_flux[cgm_fit_gpm]
    meanflux_tot_mask = np.mean(norm_good_flux_cgm)
    # deltaf_tot_mask = (norm_good_flux_cgm - meanflux_tot_mask) / meanflux_tot_mask
    # vel_cgm = vel[all_masks][cgm_fit_gpm]
    deltaf_tot_mask = (norm_good_flux - meanflux_tot_mask) / meanflux_tot_mask
    vel_cgm = vel[all_masks]
    vel_mid, xi_tot_mask, npix_tot_chimask, _ = reion_utils.compute_xi(deltaf_tot_mask, vel_cgm, vmin_corr, vmax_corr,
                                                                       dv_corr, given_bins=given_bins, gpm=cgm_fit_gpm)
    xi_mean_tot_mask = np.mean(xi_tot_mask, axis=0)  # again not really averaging here since we only have 1 spectrum

    print("MEAN FLUX", meanflux_tot, meanflux_tot_mask)
    print("mean(DELTA FLUX)", np.mean(deltaf_tot), np.mean(deltaf_tot_mask))

    ###### CF from pure noise (no CGM masking) ######
    rand = np.random.RandomState(seed) if seed != None else np.random.RandomState()
    norm_std = std / fluxfit
    fnoise = []
    fnoise_masked = []
    n_real = 50

    for i in range(n_real): # generate n_real of the noise vector
        r = rand.normal(0, std_corr*norm_std)
        r_masked = r[all_masks]
        fnoise.append(r)
        fnoise_masked.append(r_masked)

    fnoise = np.array(fnoise)
    fnoise_masked = np.array(fnoise_masked)

    all_masks_nreal = np.tile(all_masks, (n_real, 1)) # duplicating the mask n_real times
    cgm_fit_gpm_nreal = np.tile(cgm_fit_gpm, (n_real, 1))
    #fnoise_masked = np.where(all_masks_nreal, fnoise, np.nan)

    #meanflux_tot = np.mean(fnoise)
    #deltaf_tot = (fnoise - meanflux_tot) / meanflux_tot
    deltaf_tot = fnoise / meanflux_tot
    vel_mid, xi_noise, _, _ = reion_utils.compute_xi(deltaf_tot, vel, vmin_corr, vmax_corr, dv_corr, given_bins=given_bins, gpm=all_masks_nreal)

    ###### CF from pure noise (CGM masking) ######
    #meanflux_tot_mask = np.mean(fnoise[mgii_tot.fit_gpm])
    #deltaf_tot_mask = (fnoise - meanflux_tot_mask) / meanflux_tot_mask
    deltaf_tot_mask = fnoise_masked / meanflux_tot_mask
    vel_mid, xi_noise_masked, _, _ = reion_utils.compute_xi(deltaf_tot_mask, vel_cgm, vmin_corr, vmax_corr, dv_corr, given_bins=given_bins, gpm=cgm_fit_gpm_nreal)

    print("mean(DELTA FLUX)", np.mean(deltaf_tot), np.mean(deltaf_tot_mask))

    if plot:
        # plot with no masking
        plt.figure(figsize=(12, 5))
        plt.suptitle('%s, %s-z bin' % (qso_namelist[iqso], redshift_bin))
        plt.subplot(121)
        for i in range(n_real):
            plt.plot(vel_mid, xi_noise[i], linewidth=1., c='tab:gray', ls='--', alpha=0.1)

        plt.plot(vel_mid, xi_mean_tot, linewidth=1.5, label='data unmasked')
        plt.legend(fontsize=15)
        plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
        plt.ylabel(r'$\xi(\Delta v)$', fontsize=18)
        vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
        plt.axvline(vel_doublet.value, color='red', linestyle=':', linewidth=1.5, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)

        # plot with masking
        #plt.figure()
        plt.subplot(122)
        for i in range(n_real):
            plt.plot(vel_mid, xi_noise_masked[i], linewidth=1., c='tab:gray', ls='--', alpha=0.1)
        plt.plot(vel_mid, xi_mean_tot_mask, linewidth=1.5, label='data masked')

        plt.legend(fontsize=15)
        plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
        plt.ylabel(r'$\xi(\Delta v)$', fontsize=18)
        vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
        plt.axvline(vel_doublet.value, color='red', linestyle=':', linewidth=1.5, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)

        plt.tight_layout()
        plt.show()

    # return vel, norm_good_flux, good_ivar, vel_mid, xi_tot, xi_tot_mask, xi_noise, xi_noise_masked, mgii_tot.fit_gpm
    return vel_mid, xi_tot, xi_tot_mask, xi_noise, xi_noise_masked  # , npix_tot, npix_tot_chimask

import pdb
def allspec(nqso, redshift_bin, cgm_fit_gpm_all, plot=False, seed_list=[None, None, None, None], given_bins=None, iqso_to_use=None):
    # running onespec() for all the 4 QSOs

    xi_unmask_all = []
    xi_mask_all = []
    xi_noise_unmask_all = []
    xi_noise_mask_all = []

    if iqso_to_use is None:
        iqso_to_use = np.arange(0, nqso)
    print(iqso_to_use)

    weights_unmasked = []
    weights_masked = []
    #for iqso in range(nqso):
    for iqso in iqso_to_use:
        std_corr = corr_all[iqso]
        vel_mid, xi_unmask, xi_mask, xi_noise, xi_noise_masked, npix_tot, npix_tot_chimask = onespec(iqso, redshift_bin, cgm_fit_gpm_all[iqso], \
                                                                         plot=False, std_corr=std_corr, seed=None, given_bins=given_bins)
        xi_unmask_all.append(xi_unmask[0])
        xi_mask_all.append(xi_mask[0])
        #xi_noise_unmask_all.append(xi_noise[0])
        #xi_noise_mask_all.append(xi_noise_masked[0])
        xi_noise_unmask_all.append(xi_noise)
        xi_noise_mask_all.append(xi_noise_masked)
        weights_unmasked.append(npix_tot.squeeze())
        weights_masked.append(npix_tot_chimask.squeeze())

    weights_masked = np.array(weights_masked)
    weights_unmasked = np.array(weights_unmasked)
    weights_masked = weights_masked/np.sum(weights_masked, axis=0) # pixel weights
    weights_unmasked = weights_unmasked/np.sum(weights_unmasked, axis=0)

    ### un-masked quantities
    # data and noise
    xi_unmask_all = np.array(xi_unmask_all)
    xi_mean_unmask = np.average(xi_unmask_all, axis=0, weights=weights_unmasked)
    #xi_mean_unmask = np.mean(xi_unmask_all, axis=0)

    xi_std_unmask = np.std(xi_unmask_all, axis=0)
    xi_noise_unmask_all = np.array(xi_noise_unmask_all) # = (nqso, n_real, n_velmid)
    #xi_mean_noise_unmask = np.mean(xi_noise_unmask_all, axis=0)

    ### masked quantities
    # data and noise
    xi_mask_all = np.array(xi_mask_all)
    #xi_mean_mask = np.mean(xi_mask_all, axis=0)
    xi_mean_mask = np.average(xi_mask_all, axis=0, weights=weights_masked)
    xi_std_mask = np.std(xi_mask_all, axis=0)
    xi_noise_mask_all = np.array(xi_noise_mask_all)
    #xi_mean_noise_mask = np.mean(xi_noise_mask_all, axis=0)

    if plot:
        plt.figure()
        xi_scale = 1
        ymin, ymax = -0.0010 * xi_scale, 0.002 * xi_scale

        plt.figure()
        for i in range(nqso):
            for xi in xi_noise_unmask_all[i]: # plotting all 500 realizations of the noise 2PCF (not masked)
                plt.plot(vel_mid, xi*xi_scale, c='k', linewidth=0.5, alpha=0.1)

        for xi in xi_unmask_all:
            plt.plot(vel_mid, xi*xi_scale, linewidth=1.0, c='tab:orange', alpha=0.7)

        plt.errorbar(vel_mid, xi_mean_unmask*xi_scale, yerr=(xi_std_unmask/np.sqrt(4.))*xi_scale, lw=2.0, marker='o', c='tab:orange', ecolor='tab:orange', capthick=2.0, capsize=2, \
                     mec='none', label='data, unmasked', zorder=20)
        #plt.plot(vel_mid, xi_mean_noise_unmask, linewidth=1.5, c='tab:gray', label='noise, unmasked')

        plt.legend(fontsize=15, loc=4)
        plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
        plt.ylabel(r'$\xi(\Delta v)$', fontsize=18)
        vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
        print("vel doublet at", vel_doublet.value)
        plt.axvline(vel_doublet.value, color='green', linestyle=':', linewidth=1.5, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
        plt.ylim([ymin, ymax])
        plt.xscale('log')
        plt.tight_layout()

        plt.figure()
        xi_scale = 1e5
        ymin, ymax = -0.0010 * xi_scale, 0.0006 * xi_scale

        for i in range(nqso):
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
        plt.axvline(vel_doublet.value, color='green', linestyle=':', linewidth=1.5, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
        plt.ylim([ymin, ymax])
        plt.xscale('log')
        plt.tight_layout()

        plt.show()

    return vel_mid, xi_mean_unmask, xi_mean_mask, xi_noise_unmask_all, xi_noise_mask_all, xi_unmask_all, xi_mask_all

def plot_allspec(nqso, lowz_cgm_fit_gpm, highz_cgm_fit_gpm, allz_cgm_fit_gpm):

    # running allspec() and plotting the CFs for low-z bin, high-z bin, and all-z bin
    vel_mid_low, xi_mean_unmask_low, xi_mean_mask_low, xi_noise_unmask_low, xi_noise_mask_low, xi_unmask_all_low, xi_mask_all_low = allspec(nqso, 'low', lowz_cgm_fit_gpm)
    xi_std_unmask_low = np.std(xi_unmask_all_low, axis=0)
    xi_std_mask_low = np.std(xi_mask_all_low, axis=0)

    vel_mid_high, xi_mean_unmask_high, xi_mean_mask_high, xi_noise_unmask_high, xi_noise_mask_high, xi_unmask_all_high, xi_mask_all_high = allspec(nqso, 'high', highz_cgm_fit_gpm)
    xi_std_unmask_high = np.std(xi_unmask_all_high, axis=0)
    xi_std_mask_high = np.std(xi_mask_all_high, axis=0)

    vel_mid, xi_mean_unmask, xi_mean_mask, xi_noise_unmask_all, xi_noise_mask_all, xi_unmask_all, xi_mask_all = allspec(nqso, 'all', allz_cgm_fit_gpm)
    xi_std_unmask = np.std(xi_unmask_all, axis=0)
    xi_std_mask = np.std(xi_mask_all, axis=0)

    ##### un-masked #####
    plt.figure(figsize=(14, 5))
    xi_scale = 1
    ymin, ymax = -0.0010 * xi_scale, 0.002 * xi_scale

    plt.subplot(131)
    plt.title('z < %0.3f' % median_z, fontsize=15)
    for i in range(nqso):
        for xi in xi_noise_unmask_low[i]:  # plotting all 500 realizations of the noise 2PCF (not masked)
            plt.plot(vel_mid_low, xi * xi_scale, c='k', linewidth=0.3, alpha=0.1)

    for xi in xi_unmask_all_low:
        plt.plot(vel_mid_low, xi * xi_scale, linewidth=1.0, c='tab:orange', alpha=0.7)

    plt.errorbar(vel_mid_low, xi_mean_unmask_low * xi_scale, yerr=(xi_std_unmask_low / np.sqrt(4.)) * xi_scale, lw=2.0, marker='o',
                 c='tab:orange', ecolor='tab:orange', capthick=2.0, capsize=2, \
                 mec='none', label='data, unmasked', zorder=20)

    #plt.legend(fontsize=15, loc=4)
    plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
    plt.ylabel(r'$\xi(\Delta v)$', fontsize=18)
    vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
    print("vel doublet at", vel_doublet.value)
    plt.axvline(vel_doublet.value, color='green', linestyle='--', linewidth=2.0, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
    plt.ylim([ymin, ymax])
    #plt.xscale('log')

    plt.subplot(132)
    plt.title('z >= %0.3f' % median_z, fontsize=15)
    for i in range(nqso):
        for xi in xi_noise_unmask_high[i]:  # plotting all 500 realizations of the noise 2PCF (not masked)
            plt.plot(vel_mid_high, xi * xi_scale, c='k', linewidth=0.3, alpha=0.1)

    for xi in xi_unmask_all_high:
        plt.plot(vel_mid, xi * xi_scale, linewidth=1.0, c='tab:orange', alpha=0.7)

    plt.errorbar(vel_mid, xi_mean_unmask_high * xi_scale, yerr=(xi_std_unmask_high / np.sqrt(4.)) * xi_scale, lw=2.0,
                 marker='o',
                 c='tab:orange', ecolor='tab:orange', capthick=2.0, capsize=2, \
                 mec='none', label='data, unmasked', zorder=20)

    #plt.legend(fontsize=15, loc=4)
    plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
    vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
    print("vel doublet at", vel_doublet.value)
    plt.axvline(vel_doublet.value, color='green', linestyle='--', linewidth=2.0,
                label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
    plt.ylim([ymin, ymax])
    #plt.xscale('log')

    plt.subplot(133)
    plt.title('All z', fontsize=15)
    for i in range(nqso):
        for xi in xi_noise_unmask_all[i]:  # plotting all 500 realizations of the noise 2PCF (not masked)
            plt.plot(vel_mid, xi * xi_scale, c='k', linewidth=0.3, alpha=0.1)

    for xi in xi_unmask_all:
        plt.plot(vel_mid, xi * xi_scale, linewidth=1.0, c='tab:orange', alpha=0.7)

    plt.errorbar(vel_mid, xi_mean_unmask_high * xi_scale, yerr=(xi_std_unmask / np.sqrt(4.)) * xi_scale, lw=2.0,
                 marker='o',
                 c='tab:orange', ecolor='tab:orange', capthick=2.0, capsize=2, \
                 mec='none', label='data, unmasked', zorder=20)

    #plt.legend(fontsize=15, loc=4)
    plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
    vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
    print("vel doublet at", vel_doublet.value)
    plt.axvline(vel_doublet.value, color='green', linestyle='--', linewidth=2.0,
                label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
    plt.ylim([ymin, ymax])
    #plt.xscale('log')
    plt.tight_layout()

    ##### masked #####
    plt.figure(figsize=(14, 5))
    xi_scale = 1e5
    ymin, ymax = -0.0010 * xi_scale, 0.0006 * xi_scale

    plt.subplot(131)
    plt.title('z < %0.3f' % median_z, fontsize=15)
    for i in range(4):
        for xi in xi_noise_mask_low[i]:  # plotting all 500 realizations of the noise 2PCF (not masked)
            plt.plot(vel_mid_low, xi * xi_scale, c='k', linewidth=0.3, alpha=0.1)

    for xi in xi_mask_all_low:
        plt.plot(vel_mid, xi * xi_scale, linewidth=1.0, c='tab:orange', alpha=0.7)

    plt.errorbar(vel_mid, xi_mean_mask_low * xi_scale, yerr=(xi_std_mask_low / np.sqrt(4.)) * xi_scale, lw=2.0,
                 marker='o',
                 c='tab:orange', ecolor='tab:orange', capthick=2.0, capsize=2, \
                 mec='none', label='data, masked', zorder=20)

    #plt.legend(fontsize=15, loc=4)
    plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
    plt.ylabel(r'$\xi(\Delta v) \times 10^5$', fontsize=18)
    vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
    print("vel doublet at", vel_doublet.value)
    plt.axvline(vel_doublet.value, color='green', linestyle='--', linewidth=2.0,
                label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
    plt.ylim([ymin, ymax])
    #plt.xscale('log')

    plt.subplot(132)
    plt.title('z >= %0.3f' % median_z, fontsize=15)
    for i in range(4):
        for xi in xi_noise_mask_high[i]:  # plotting all 500 realizations of the noise 2PCF (not masked)
            plt.plot(vel_mid_high, xi * xi_scale, c='k', linewidth=0.3, alpha=0.1)

    for xi in xi_mask_all_high:
        plt.plot(vel_mid, xi * xi_scale, linewidth=1.0, c='tab:orange', alpha=0.7)

    plt.errorbar(vel_mid, xi_mean_mask_high * xi_scale, yerr=(xi_std_mask_high / np.sqrt(4.)) * xi_scale, lw=2.0,
                 marker='o',
                 c='tab:orange', ecolor='tab:orange', capthick=2.0, capsize=2, \
                 mec='none', label='data, masked', zorder=20)

    #plt.legend(fontsize=15, loc=4)
    plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
    # plt.ylabel(r'$\xi(\Delta v) \times 10^5$', fontsize=18)
    vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
    print("vel doublet at", vel_doublet.value)
    plt.axvline(vel_doublet.value, color='green', linestyle='--', linewidth=2.0,
                label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
    plt.ylim([ymin, ymax])
    #plt.xscale('log')

    plt.subplot(133)
    plt.title('All z', fontsize=15)
    for i in range(4):
        for xi in xi_noise_mask_all[i]:  # plotting all 500 realizations of the noise 2PCF (not masked)
            plt.plot(vel_mid, xi * xi_scale, c='k', linewidth=0.3, alpha=0.1)

    for xi in xi_mask_all:
        plt.plot(vel_mid, xi * xi_scale, linewidth=1.0, c='tab:orange', alpha=0.7)

    plt.errorbar(vel_mid, xi_mean_mask_high * xi_scale, yerr=(xi_std_mask / np.sqrt(4.)) * xi_scale, lw=2.0,
                 marker='o',
                 c='tab:orange', ecolor='tab:orange', capthick=2.0, capsize=2, \
                 mec='none', label='data, masked', zorder=20)

    #plt.legend(fontsize=15, loc=4)
    plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
    # plt.ylabel(r'$\xi(\Delta v) \times 10^5$', fontsize=18)
    vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
    print("vel doublet at", vel_doublet.value)
    plt.axvline(vel_doublet.value, color='green', linestyle='--', linewidth=2.0,
                label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
    plt.ylim([ymin, ymax])
    #plt.xscale('log')

    plt.tight_layout()
    plt.show()

#################### non-linear dv bins ####################
def custom_cf_bin():
    """
    flux_lores = flux_lores[0:100]  # just looking at a subset
    mean_flux_nless = np.mean(flux_lores)
    delta_f_nless = (flux_lores - mean_flux_nless) / mean_flux_nless

    (vel_mid, xi_nless, npix, xi_nless_zero_lag) = reion_utils.compute_xi(delta_f_nless, np.log10(vel_lores), np.log10(
        vmin), np.log10(vmax), np.log10(dv))
    xi_mean = np.mean(xi_nless, axis=0)
    """

    # log on small-scale
    vmin1, vmax1, dv1 = 10, 550, 0.2
    log_vmin = np.log10(vmin1)
    log_vmax = np.log10(vmax1)
    ngrid = int(round((log_vmax - log_vmin) / dv1) + 1)  # number of grid points including vmin and vmax
    log_v_corr = log_vmin + dv1 * np.arange(ngrid)
    log_v_lo = log_v_corr[:-1]  # excluding the last point (=vmax)
    log_v_hi = log_v_corr[1:]  # excluding the first point (=vmin)
    v_lo1 = 10 ** log_v_lo
    v_hi1 = 10 ** log_v_hi
    v_mid = 10. ** ((log_v_hi + log_v_lo) / 2.0)
    #print(v_mid)

    # linear around peak
    v_bins2 = np.arange(550, 1000, 30)
    v_lo2 = v_bins2[:-1]
    v_hi2 = v_bins2[1:]

    # log on large-scale
    vmin3, vmax3, dv3 = 1000, 3600, 0.1
    log_vmin = np.log10(vmin3)
    log_vmax = np.log10(vmax3)
    ngrid = int(round((log_vmax - log_vmin) / dv3) + 1)  # number of grid points including vmin and vmax
    log_v_corr = log_vmin + dv3 * np.arange(ngrid)
    log_v_lo = log_v_corr[:-1]  # excluding the last point (=vmax)
    log_v_hi = log_v_corr[1:]  # excluding the first point (=vmin)
    v_lo3 = 10 ** log_v_lo
    v_hi3 = 10 ** log_v_hi
    v_mid = 10. ** ((log_v_hi + log_v_lo) / 2.0)
    #print(v_mid)

    v_lo_all = np.concatenate((v_lo1, v_lo2, v_lo3))
    v_hi_all = np.concatenate((v_hi1, v_hi2, v_hi3))

    return v_lo_all, v_hi_all

def custom_cf_bin2():
    """
    flux_lores = flux_lores[0:100]  # just looking at a subset
    mean_flux_nless = np.mean(flux_lores)
    delta_f_nless = (flux_lores - mean_flux_nless) / mean_flux_nless

    (vel_mid, xi_nless, npix, xi_nless_zero_lag) = reion_utils.compute_xi(delta_f_nless, np.log10(vel_lores), np.log10(
        vmin), np.log10(vmax), np.log10(dv))
    xi_mean = np.mean(xi_nless, axis=0)
    """

    # linear around peak and small-scales
    v_bins1 = np.arange(10, 1200, 60)
    v_lo1 = v_bins1[:-1]
    v_hi1 = v_bins1[1:]

    # log on large-scale
    vmin2, vmax2, dv2 = 1200, 3550, 0.1
    log_vmin = np.log10(vmin2)
    log_vmax = np.log10(vmax2)
    ngrid = int(round((log_vmax - log_vmin) / dv2) + 1)  # number of grid points including vmin and vmax
    log_v_corr = log_vmin + dv2 * np.arange(ngrid)
    log_v_lo = log_v_corr[:-1]  # excluding the last point (=vmax)
    log_v_hi = log_v_corr[1:]  # excluding the first point (=vmin)
    v_lo2 = 10 ** log_v_lo
    v_hi2 = 10 ** log_v_hi
    v_mid = 10. ** ((log_v_hi + log_v_lo) / 2.0)

    v_lo_all = np.concatenate((v_lo1, v_lo2))
    v_hi_all = np.concatenate((v_hi1, v_hi2))

    return v_lo_all, v_hi_all

def custom_cf_bin3():

    # linear around peak and small-scales
    dv1 = 60
    v_bins1 = np.arange(10, 1200 + dv1, dv1)

    # increasingly larger dv (=90, 120, 150, 180... 300, 300)
    dv = np.concatenate((np.arange(90, 360, 30), np.ones(10)*300))
    v_bins2 = []
    for i, idv in enumerate(dv):
        if i == 0:
            v_bins2.append(v_bins1[-1] + idv)
        else:
            if v_bins2[-1] < 3500:
                v_bins2.append(v_bins2[i-1] + idv)

    v_bins_all = np.concatenate((v_bins1, v_bins2))
    v_lo = v_bins_all[:-1]
    v_hi = v_bins_all[1:]

    return v_lo, v_hi

def custom_cf_bin4(dv1=40, check=False):

    v_end = 1500
    v_end2 = 3500
    dv2 = 200

    v_bins1 = np.arange(10, v_end + dv1, dv1)
    v_bins2 = np.arange(v_bins1[-1] + dv2, v_end2 + dv2, dv2)
    v_bins = np.concatenate((v_bins1, v_bins2))
    v_lo = v_bins[:-1]
    v_hi = v_bins[1:]

    if check:
        v_mid = (v_hi + v_lo)/2
        print(np.diff(v_mid))
        for i in range(len(v_lo)):
            print(i, v_lo[i], v_hi[i], v_hi[i] - v_lo[i], v_mid[i])

    return v_lo, v_hi

    """
    # this is the binning used for the paper
    #dv1 = 60 # need to be multiples of 30 to ensure correct behavior
    #v_end = 1500

    #dv1 = 80  # since spectra rebinned to 40 km/s
    v_end = 1500

    # linear around peak and small-scales
    v_bins1 = np.arange(10, v_end + dv1, dv1)
    v_lo1 = v_bins1[:-1]
    v_hi1 = v_bins1[1:]

    # larger linear bin size
    #dv2 = 210 # need to be multiples of 30 to ensure correct behavior
    #v_bins2 = np.arange(v_end, 3600 + dv2, dv2)
    dv2 = 200#
    v_end2 = 3500
    v_bins2 = np.arange(v_bins1[-1] + dv2, v_end2 + dv2, dv2)
    v_lo2 = v_bins2[:-1]
    v_hi2 = v_bins2[1:]

    v_lo_all = np.concatenate((v_lo1, v_lo2))
    v_hi_all = np.concatenate((v_hi1, v_hi2))

    return v_lo_all, v_hi_all
    """

from scipy import interpolate
def interp_vbin(vel_mid, xi_mean, kind='linear'):

    f = interpolate.interp1d(vel_mid, xi_mean, kind=kind)

    vmin1, vmax1, dv1 = 55, 3500, 0.05
    log_vmin = np.log10(vmin1)
    log_vmax = np.log10(vmax1)
    ngrid = int(round((log_vmax - log_vmin) / dv1) + 1)  # number of grid points including vmin and vmax
    log_v_corr = log_vmin + dv1 * np.arange(ngrid)
    log_v_lo = log_v_corr[:-1]  # excluding the last point (=vmax)
    log_v_hi = log_v_corr[1:]  # excluding the first point (=vmin)
    v_lo1 = 10 ** log_v_lo
    v_hi1 = 10 ** log_v_hi
    v_mid = 10. ** ((log_v_hi + log_v_lo) / 2.0)

    try:
        xi_mean_new = f(v_mid)
    except ValueError:
        print(v_mid)

    return v_mid, xi_mean_new

def compare_lin_log_bins(deltaf, vel_lores, given_bins, dv_corr, loglegend, title):
    vmin_corr, vmax_corr = 10, 3500

    (vel_mid_lin, xi_nless_lin, npix, xi_nless_zero_lag) = reion_utils.compute_xi(deltaf, vel_lores, vmin_corr, vmax_corr, dv_corr)
    xi_mean_lin = np.mean(xi_nless_lin, axis=0)
    print(len(vel_mid_lin))
    print(vel_mid_lin)

    (vel_mid_log, xi_nless_log, npix, xi_nless_zero_lag) = reion_utils.compute_xi(deltaf, vel_lores, 0, 0, 0, given_bins=given_bins)
    xi_mean_log = np.mean(xi_nless_log, axis=0)
    print(len(vel_mid_log))
    print(vel_mid_log)

    plt.figure(figsize=(12,5))
    plt.suptitle(title, fontsize=16)
    plt.subplot(121)
    plt.plot(vel_mid_lin, xi_mean_lin, 'k-', label='linear with dv=%d' % dv_corr)
    plt.plot(vel_mid_log, xi_mean_log, 'ro', label=loglegend)
    plt.xlabel(r'$\Delta v$ [km/s]')
    plt.ylabel(r'$\xi(\Delta v)$')
    plt.legend()

    plt.subplot(122)
    plt.plot(vel_mid_lin, xi_mean_lin, 'k-', label='linear with dv=%d' % dv_corr)
    plt.plot(vel_mid_log, xi_mean_log, 'ro', label=loglegend)
    plt.xlabel(r'$\Delta v$ [km/s]')
    plt.ylabel(r'$\xi(\Delta v)$')
    plt.xscale('log')
    plt.tight_layout()
    plt.show()

    return vel_mid_log, xi_mean_log, vel_mid_lin, xi_mean_lin

####################
import compute_model_grid_new as cmg

def onespec_chunk(iqso, redshift_bin, cgm_fit_gpm, vel_lores, given_bins=None):
    # cgm_fit_gpm = output from cmg.init_cgm_masking (only CGM mask, other masks added below)

    #dv_corr = 100
    raw_data_out, _, all_masks_out = mutils.init_onespec(iqso, redshift_bin)
    wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
    strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, master_mask = all_masks_out

    nskew_to_match_data, npix_sim_skew = 10, 220

    ###### CF from not masking CGM ######
    all_masks = mask * redshift_mask * pz_mask * zbin_mask

    norm_good_flux = (flux / fluxfit)[all_masks]
    norm_flux = flux / fluxfit
    vel = mutils.obswave_to_vel_2(wave)
    meanflux_tot = np.mean(norm_good_flux)
    deltaf_tot = (norm_flux - meanflux_tot) / meanflux_tot

    deltaf_chunk = cmg.reshape_data_array(deltaf_tot, nskew_to_match_data, npix_sim_skew, data_arr_is_mask=False)
    all_masks_chunk = cmg.reshape_data_array(all_masks, nskew_to_match_data, npix_sim_skew, data_arr_is_mask=True)

    vel_mid, xi_tot, npix_tot, _ = reion_utils.compute_xi(deltaf_chunk, vel_lores, vmin_corr, vmax_corr, dv_corr, given_bins=given_bins, gpm=all_masks_chunk)
    xi_mean_tot = np.mean(xi_tot, axis=0)  # average over nskew_to_match_data

    ###### CF from masking CGM ######
    gpm_onespec_chunk = cmg.reshape_data_array(cgm_fit_gpm, nskew_to_match_data, npix_sim_skew, data_arr_is_mask=True)
    all_masks_chunk *= gpm_onespec_chunk

    norm_flux_chunk = cmg.reshape_data_array(norm_flux, nskew_to_match_data, npix_sim_skew, data_arr_is_mask=False)
    meanflux_tot_mask = np.mean(norm_flux_chunk[all_masks_chunk])

    deltaf_mask_chunk = (norm_flux_chunk - meanflux_tot_mask)/meanflux_tot_mask

    vel_mid, xi_tot_mask, npix_tot_mask, _ = reion_utils.compute_xi(deltaf_mask_chunk, vel_lores, vmin_corr, vmax_corr, dv_corr, given_bins=given_bins, gpm=all_masks_chunk)
    xi_mean_tot_mask = np.mean(xi_tot_mask, axis=0)  # average over nskew_to_match_data

    return vel_mid, xi_mean_tot, xi_mean_tot_mask

def allspec_chunk(nqso, gpm_allspec, redshift_bin, vel_lores, given_bins=None):

    #dv_corr = 100
    #vel_lores, _, _, _, _, _ = mutils.init_skewers_compute_model_grid(dv_corr)

    xi_unmask_all = []
    xi_mask_all = []

    for iqso in range(nqso):
        vel_mid, xi_mean_tot, xi_mean_tot_mask = \
            onespec_chunk(iqso, redshift_bin, gpm_allspec[iqso], vel_lores, given_bins=given_bins)
        xi_unmask_all.append(xi_mean_tot)
        xi_mask_all.append(xi_mean_tot_mask)

    xi_unmask_all = np.array(xi_unmask_all)
    xi_mean_unmask = np.mean(xi_unmask_all, axis=0)

    xi_mask_all = np.array(xi_mask_all)
    xi_mean_mask = np.mean(xi_mask_all, axis=0)

    return vel_mid, xi_mean_unmask, xi_mean_mask
