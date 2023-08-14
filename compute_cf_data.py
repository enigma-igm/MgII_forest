'''
Functions here:
    - init_cgm_fit_gpm
    - onespec
    - allspec
    - plot_allspec
    - custom_cf_bin
    - custom_cf_bin2
    - custom_cf_bin3
    - custom_cf_bin4
    - interp_vbin
    - compare_lin_log_bins
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
from scipy import interpolate
import pdb

####### global variables #######
qso_namelist = ['J0411-0907', 'J0319-1008', 'newqso1', 'newqso2', 'J0313-1806', 'J0038-1527', 'J0252-0503', \
                'J1342+0928', 'J1007+2115', 'J1120+0641']
qso_zlist = [6.826, 6.8275, 7.0, 7.1, 7.642, 7.034, 7.001, 7.541, 7.515, 7.085]

vmin_corr, vmax_corr, dv_corr = 10, 3500, 40 # dummy values because we're now using custom binning

#corr_all = [0.669, 0.673, 0.692, 0.73, 0.697, 0.653, 0.667, 0.72, 0.64, 0.64] #xi_mean_mask_10qso_everyn60_corr.fits
#corr_all = [0.669, 0.673, 0.692, 1.07, 1.01, 1.06, 1.07, 1.00, 0.64, 0.64]

#corr_all = [0.964, 0.975, 1.045, 1.078, 1.01, 1.053, 1.084, 1.006, 0.914, 1.128]
corr_all = [0.93, 0.898, 0.88, 1.051, 0.972, 1.055, 1.086, 0.956, 0.908, 1.059] # after masking strong absorbers
#corr_all = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

#################################
def init_cgm_fit_gpm(datapath='/Users/suksientie/Research/MgII_forest/rebinned_spectra2/', do_not_apply_any_mask=False):
    # initialize the GPM masks from the cgm masking for all redshift bins

    lowz_mgii_tot_all, highz_mgii_tot_all, allz_mgii_tot_all = mask_cgm.do_allqso_allzbin(datapath, do_not_apply_any_mask)

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

def onespec(iqso, redshift_bin, cgm_fit_gpm, fmean_unmask, fmean_mask, plot=False, std_corr=1.0, given_bins=None, ivar_weights=False, subtract_mean_deltaf=False):

    # compute the CF for one QSO spectrum
    # options for redshift_bin are 'low', 'high', 'all'
    # cgm_fit_gpm are gpm from MgiiFinder.py

    raw_data_out, _, all_masks_out = mutils.init_onespec(iqso, redshift_bin)
    wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
    strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, telluric_mask, master_mask = all_masks_out

    ivar *= (fluxfit**2) # normalize by cont
    ivar *= (1/std_corr**2) # apply correction

    #ivar = np.random.permutation(ivar)

    ###### CF from not masking CGM ######
    all_masks = master_mask

    norm_good_flux = (flux / fluxfit)[all_masks]
    #ivar_good = ivar[all_masks]

    norm_flux = flux/fluxfit
    vel = mutils.obswave_to_vel_2(wave)
    meanflux_tot = fmean_unmask
    deltaf_tot = (norm_flux - meanflux_tot) / meanflux_tot

    if subtract_mean_deltaf:
        mean_deltaf_tot = np.mean(deltaf_tot[all_masks])
        deltaf_tot -= np.mean(mean_deltaf_tot)

    if ivar_weights:
        print("use ivar as weights in CF")
        #vel_mid, xi_tot, npix_tot, _ = reion_utils.compute_xi_ivar(deltaf_tot, ivar, vel, vmin_corr, vmax_corr, dv_corr, given_bins=given_bins, gpm=all_masks)
        weights_in = ivar
    else:
        #vel_mid, xi_tot, npix_tot, _ = reion_utils.compute_xi(deltaf_tot, vel, vmin_corr, vmax_corr, dv_corr, given_bins=given_bins, gpm=all_masks)
        weights_in = None

    vel_mid, xi_tot, npix_tot, _ = reion_utils.compute_xi_weights(deltaf_tot, vel, vmin_corr, vmax_corr, dv_corr,
                                                          given_bins=given_bins, gpm=all_masks, weights_in=weights_in)

    xi_mean_tot = np.mean(xi_tot, axis=0)  # not really averaging here since it's one spectrum (i.e. xi_mean_tot = xi_tot)

    ###### CF from masking CGM ######
    """
    norm_good_flux_cgm = norm_good_flux[cgm_fit_gpm]
    meanflux_tot_mask = np.mean(norm_good_flux_cgm)
    #deltaf_tot_mask = (norm_good_flux_cgm - meanflux_tot_mask) / meanflux_tot_mask
    #vel_cgm = vel[all_masks][cgm_fit_gpm]
    deltaf_tot_mask = (norm_good_flux - meanflux_tot_mask) / meanflux_tot_mask
    vel_cgm = vel[all_masks]
    """

    meanflux_tot_mask = fmean_mask
    deltaf_tot_mask = (norm_flux - meanflux_tot_mask) / meanflux_tot_mask

    if subtract_mean_deltaf:
        mean_deltaf_tot_mask = np.mean(deltaf_tot_mask[all_masks * cgm_fit_gpm])
        deltaf_tot_mask -= mean_deltaf_tot_mask

    if ivar_weights:
        print("use ivar as weights in CF")
        #vel_mid, xi_tot_mask, npix_tot_chimask, _ = reion_utils.compute_xi_ivar(deltaf_tot_mask, ivar_good, vel_cgm, vmin_corr, vmax_corr, dv_corr, given_bins=given_bins, gpm=cgm_fit_gpm)
        weights_in = ivar
    else:
        #vel_mid, xi_tot_mask, npix_tot_chimask, _ = reion_utils.compute_xi(deltaf_tot_mask, vel_cgm, vmin_corr, vmax_corr, dv_corr, given_bins=given_bins, gpm=cgm_fit_gpm)
        weights_in = None

    #vel_mid, xi_tot_mask, npix_tot_chimask, _ = reion_utils.compute_xi_weights(deltaf_tot_mask, vel_cgm, vmin_corr, vmax_corr,
    #                                                                   dv_corr, given_bins=given_bins, gpm=cgm_fit_gpm, weights_in=weights_in)
    vel_mid, xi_tot_mask, npix_tot_chimask, _ = reion_utils.compute_xi_weights(deltaf_tot_mask, vel, vmin_corr, vmax_corr, dv_corr, given_bins=given_bins, gpm=all_masks * cgm_fit_gpm, weights_in=weights_in)


    xi_mean_tot_mask = np.mean(xi_tot_mask, axis=0) # again not really averaging here since we only have 1 spectrum

    print("==============")
    print("MEAN FLUX", meanflux_tot, meanflux_tot_mask)
    #print("mean(DELTA FLUX)", np.mean(deltaf_tot[all_masks]), np.mean(deltaf_tot_mask[all_masks * cgm_fit_gpm]))
    print("mean(DELTA FLUX)", np.mean(deltaf_tot), np.mean(deltaf_tot_mask))

    """
    plt.figure()
    plt.title(qso_namelist[iqso])
    plt.hist(deltaf_tot[all_masks], bins=np.arange(-0.5, 0.5, 0.02), histtype='step', color='b')
    plt.hist(deltaf_tot_mask[all_masks], bins=np.arange(-0.5, 0.5, 0.02), histtype='step', label='mask CGM', color='r')
    plt.axvline(np.mean(deltaf_tot_mask[all_masks]), color='r')
    plt.legend()
    """
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
    return vel_mid, xi_tot[0], xi_tot_mask[0], npix_tot, npix_tot_chimask, deltaf_tot, deltaf_tot_mask

def allspec(nqso, redshift_bin, cgm_fit_gpm_all, plot=False, given_bins=None, iqso_to_use=None, ivar_weights=False, subtract_mean_deltaf=False):
    # running onespec() for all QSOs

    xi_unmask_all = []
    xi_mask_all = []
    xi_noise_unmask_all = []
    xi_noise_mask_all = []

    if iqso_to_use is None:
        iqso_to_use = np.arange(0, nqso)
    print(iqso_to_use)

    weights_unmasked = []
    weights_masked = []

    # everyn = 60 (all-z, high-z, low-z), nqso=8
    # normalized ivar, weighted fmean of dataset
    #fmean_global_unmask = [0.9945437229984158, 0.9986658355772385, 0.9914894025444619]
    #fmean_global_mask = [1.0013868138771054, 1.0019725509921347, 1.0009292561522118]

    # everyn = 60 (all-z, high-z, low-z), nqso=10
    # normalized ivar, weighted fmean of dataset
    #fmean_global_unmask = [0.9952775044785822, 0.998804106985206, 0.9927064690772747]
    #fmean_global_mask = [1.001426984289517, 1.0018512443910101, 1.0011034891413138]

    # 7/28/2023: new rebinned spectra, joe's contfit, xshooter fwhm=150 in cgm masking
    #fmean_global_unmask = [0.9942310808994878, 0.9977878806975714, 0.9916679773002758]
    #fmean_global_mask = [1.0024918695120622, 1.0010133171937157, 1.003679032855304]

    # 7/28/2023: new rebinned spectra, my contfit, xshooter fwhm=150 in cgm masking
    fmean_global_unmask = [0.9946550102864571, 0.9977495119854285, 0.992422115121291]
    fmean_global_mask = [1.0014764052901743, 1.0009900106471348, 1.0018462960427013]

    if redshift_bin == 'all':
        i_fmean = 0
    elif redshift_bin == 'high':
        i_fmean = 1
    elif redshift_bin == 'low':
        i_fmean = 2
    fmean_unmask = fmean_global_unmask[i_fmean]
    fmean_mask = fmean_global_mask[i_fmean]

    df_tot_all = []
    df_tot_mask_all = []
    for iqso in iqso_to_use:
        std_corr = corr_all[iqso]
        #vel_mid, xi_unmask, xi_mask, w_tot, w_tot_chimask, _, _ = onespec_old(iqso, redshift_bin, cgm_fit_gpm_all[iqso], plot=False, std_corr=std_corr, given_bins=given_bins, ivar_weights=ivar_weights)
        vel_mid, xi_unmask, xi_mask, w_tot, w_tot_chimask, deltaf_tot, deltaf_tot_mask = onespec(iqso, redshift_bin, cgm_fit_gpm_all[iqso], \
                                fmean_unmask, fmean_mask, plot=False, std_corr=std_corr, given_bins=given_bins, \
                                ivar_weights=ivar_weights, subtract_mean_deltaf=subtract_mean_deltaf)

        xi_unmask_all.append(xi_unmask)
        xi_mask_all.append(xi_mask)
        #xi_noise_unmask_all.append(xi_noise[0])
        #xi_noise_mask_all.append(xi_noise_masked[0])
        #xi_noise_unmask_all.append(xi_noise)
        #xi_noise_mask_all.append(xi_noise_masked)
        weights_unmasked.append(w_tot.squeeze())
        weights_masked.append(w_tot_chimask.squeeze())
        df_tot_all.extend(deltaf_tot)
        df_tot_mask_all.extend(deltaf_tot_mask)

    df_tot_all = np.array(df_tot_all)
    df_tot_mask_all = np.array(df_tot_mask_all)
    print("======== entire sample: mean(DELTA FLUX)", np.mean(df_tot_all.flatten()), np.mean(df_tot_mask_all.flatten()))
    print("======== entire sample: median(DELTA FLUX)", np.median(df_tot_all.flatten()), np.median(df_tot_mask_all.flatten()))

    scale = 1#1e6
    weights_masked = np.array(weights_masked)/scale
    weights_unmasked = np.array(weights_unmasked)/scale

    # no need to express as fraction
    #weights_masked = weights_masked/np.sum(weights_masked, axis=0) # pixel weights
    #weights_unmasked = weights_unmasked/np.sum(weights_unmasked, axis=0)

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
    #xi_mean_mask = np.average(xi_mask_all, axis=0, weights=weights_masked)
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

    return vel_mid, xi_mean_unmask, xi_mean_mask, xi_noise_unmask_all, xi_noise_mask_all, xi_unmask_all, xi_mask_all, \
        weights_masked, weights_unmasked

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

def custom_cf_bin4(dv1=40, dv2=200, check=False):

    v_end = 1500
    v_end2 = 3500
    #dv2 = 200

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

########################################
def frac_weight_per_qso(redshift_bin, nqso, plot=False):
    qso_namelist = ['J0411-0907', 'J0319-1008', 'J0410-0139', 'J0038-0653', 'J0313-1806', 'J0038-1527', 'J0252-0503', 'J1342+0928', 'J1007+2115', 'J1120+0641']
    qso_zlist = [6.826, 6.8275, 7.0, 7.1, 7.642, 7.034, 7.001, 7.541, 7.515, 7.085]
    qso_median_snr = [9.27, 5.51, 3.95, 8.60, 11.42, 14.28, 13.07, 8.57, 7.05, 6.54]  # from Table 1 in current draft (12/6/2022)

    #lowz_cgm_fit_gpm, highz_cgm_fit_gpm, allz_cgm_fit_gpm = init_cgm_fit_gpm()
    lowz_cgm_fit_gpm, highz_cgm_fit_gpm, allz_cgm_fit_gpm = init_cgm_fit_gpm(do_not_apply_any_mask=True)

    if redshift_bin == 'low':
        cgm_fit_gpm = lowz_cgm_fit_gpm
    elif redshift_bin == 'high':
        cgm_fit_gpm = highz_cgm_fit_gpm
    elif redshift_bin == 'all':
        cgm_fit_gpm = allz_cgm_fit_gpm

    given_bins = custom_cf_bin4(dv1=80)
    ivar_weights = True
    vel_mid, xi_mean_unmask, xi_mean_mask, xi_noise_unmask, xi_noise_mask, xi_unmask, xi_mask, w_masked, w_unmasked = \
        allspec(nqso, redshift_bin, cgm_fit_gpm, given_bins=given_bins, ivar_weights=ivar_weights)

    allspec_out = vel_mid, xi_mean_unmask, xi_mean_mask, xi_noise_unmask, xi_noise_mask, xi_unmask, xi_mask, w_masked, w_unmasked

    w_masked_sum = np.sum(w_masked, axis=0)
    frac_w_masked = w_masked / w_masked_sum
    mean_weight = []

    for i in range(nqso):
        print(qso_namelist[i], qso_zlist[i], qso_median_snr[i], '%0.5f' % np.mean(frac_w_masked[i]))
        mean_weight.append(np.mean(frac_w_masked[i]))

    if plot:
        for i in range(nqso):
            plt.plot(vel_mid, frac_w_masked[i], label=qso_namelist[i])
        plt.ylabel('Fractional weight')
        plt.xlabel('V (km/s)')
        plt.legend()
        plt.show()

    return allspec_out, frac_w_masked, mean_weight

def compute_neff(weights_allqso):
    # "weights_allqso" can be frac_w_masked or w_masked returned from frac_weight_per_qso

    # https://statisticaloddsandends.wordpress.com/2021/11/11/what-do-we-mean-by-effective-sample-size/

    # axis=0 is sum over nqso to get the n_eff in each corr bin
    neff = (np.sum(weights_allqso, axis=0)) ** 2 / np.sum(weights_allqso ** 2, axis=0)
    return neff

def fmean_dataset3(norm_flux_allqso, master_mask_allqso, ivar_allqso, master_mask_allqso_mask_cgm):

    nqso = len(norm_flux_allqso) #10

    # normalized quantities
    f_all = []
    f_all_mask_cgm = []
    ivar_all = []
    ivar_all_mask_cgm = []
    f_all_onespec = []
    f_all_mask_cgm_onespec = []

    for iqso in range(nqso):
        f_all.extend(norm_flux_allqso[iqso][master_mask_allqso[iqso]])
        f_all_mask_cgm.extend(norm_flux_allqso[iqso][master_mask_allqso_mask_cgm[iqso]])
        ivar_all.extend(ivar_allqso[iqso][master_mask_allqso[iqso]])
        ivar_all_mask_cgm.extend(ivar_allqso[iqso][master_mask_allqso_mask_cgm[iqso]])

        f_all_onespec.append(np.mean(norm_flux_allqso[iqso][master_mask_allqso[iqso]]))
        f_all_mask_cgm_onespec.append(np.mean(norm_flux_allqso[iqso][master_mask_allqso_mask_cgm[iqso]]))

    fmean_unmask = np.average(f_all, weights=ivar_all)
    fmean_mask = np.average(f_all_mask_cgm, weights=ivar_all_mask_cgm)

    return fmean_unmask, fmean_mask

    # minimal change in fmean when changing the continuum breakpoint (everyn=20, 40, 60)
    #fmean_mask20, fmean_mask40, fmean_mask60
    #(1.0009382101500206, 1.0010164874168208, 1.001426984289517)

    #fmean_unmask20, fmean_unmask40, fmean_unmask60
    #(0.9951233615221605, 0.9950089122951656, 0.9952775044785822)

def weighted_var(xi_allqso, weights_allqso):
    # https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
    # http://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf
    xi_average = np.average(xi_allqso, axis=0, weights=weights_allqso)
    variance = np.average((xi_allqso - xi_average)**2, axis=0, weights=weights_allqso)

    v1 = np.sum(weights_allqso, axis=0)
    v2 = np.sum(weights_allqso ** 2, axis=0)
    bias = 1 - (v2/v1**2)

    unbiased_var = variance / bias

    return unbiased_var

def perform_weighted_bootstrap(data, weights, num_bootstraps):

    bootstrap_means = []
    n = len(data)

    for _ in range(num_bootstraps):
        # Generate bootstrap sample indices with replacement based on the weights
        bootstrap_indices = np.random.choice(range(n), size=n, replace=True)#, p=weights)

        # Create bootstrap sample using the indices
        bootstrap_data = [data[i] for i in bootstrap_indices]
        bootstrap_weights = [weights[i] for i in bootstrap_indices]
        # Calculate the weighted mean of the bootstrap sample
        bootstrap_mean = np.sum(np.multiply(bootstrap_data, bootstrap_weights)) / np.sum(bootstrap_weights)
        bootstrap_means.append(bootstrap_mean)

    # Calculate the standard deviation of the bootstrap means
    bootstrap_errors = np.std(bootstrap_means)

    return bootstrap_errors

def perform_weighted_bootstrap2(data, weights, num_bootstraps):

    bootstrap_means = []
    n = len(data)

    total_weights_all = []
    bs_indices_all = []
    for _ in range(num_bootstraps):
        total_weights = 0
        bs_indices = []
        while total_weights < 1.0:
            i = np.random.randint(n)
            bs_indices.append(i)
            total_weights += weights[i]

        if total_weights <= 1.1:
            total_weights_all.append(total_weights)
            bs_indices_all.append(bs_indices)

    return total_weights_all, bs_indices_all

def perform_weighted_bootstrap3(data, weights, num_bootstraps):

    bootstrap_means = []
    n = len(data)

    for _ in range(num_bootstraps):
        # Generate bootstrap sample indices with replacement based on the weights
        bootstrap_indices = np.random.choice(range(n), size=n, replace=True)

        # Create bootstrap sample using the indices
        bootstrap_data = np.array([data[i] for i in bootstrap_indices])
        bootstrap_weights = np.array([weights[i] for i in bootstrap_indices])

        # Calculate the weighted mean of the bootstrap sample

        bootstrap_mean = np.average(bootstrap_data, axis=0, weights=bootstrap_weights)
        bootstrap_means.append(bootstrap_mean)

    # Calculate the standard deviation of the bootstrap means
    bootstrap_means = np.array(bootstrap_means)
    bootstrap_errors = np.std(bootstrap_means, axis=0)

    return bootstrap_errors, bootstrap_means

######################################## old stuffs
def check_onespec(iqso, redshift_bin, given_bins):
    raw_data_out, _, all_masks_out = mutils.init_onespec(iqso, redshift_bin)
    wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
    strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, telluric_mask, master_mask = all_masks_out

    # ivar = ivar * 1/(std_corr**2) # np.ones(ivar.shape)

    ###### CF from not masking CGM ######
    all_masks = master_mask

    norm_good_flux = (flux / fluxfit)[all_masks]
    ivar_good = ivar[all_masks]

    norm_flux = flux / fluxfit
    vel = mutils.obswave_to_vel_2(wave)
    meanflux_tot = np.mean(norm_good_flux)
    deltaf_tot = (norm_flux - meanflux_tot) / meanflux_tot

    # checking between these 2 using ivar weights
    vel_mid, xi_tot1, w_tot1, _ = reion_utils.compute_xi_weights(deltaf_tot, vel, vmin_corr, vmax_corr, dv_corr,
                                                                 given_bins=given_bins, gpm=all_masks, weights_in=ivar)

    vel_mid, xi_tot2, w_tot2, _ = reion_utils.compute_xi_ivar(deltaf_tot, ivar, vel, vmin_corr, vmax_corr, dv_corr,
                                                              given_bins=given_bins, gpm=all_masks)

    print(xi_tot2 / xi_tot1)

    # checking between these 2 using npix weights
    vel_mid, xi_tot1, npix_tot1, _ = reion_utils.compute_xi_weights(deltaf_tot, vel, vmin_corr, vmax_corr, dv_corr,
                                                                    given_bins=given_bins, gpm=all_masks,
                                                                    weights_in=None)

    vel_mid, xi_tot2, npix_tot2, _ = reion_utils.compute_xi(deltaf_tot, vel, vmin_corr, vmax_corr, dv_corr,
                                                            given_bins=given_bins, gpm=all_masks)

    print(xi_tot2 / xi_tot1)
    return vel_mid, npix_tot1, npix_tot2

def old_onespec(iqso, redshift_bin, cgm_fit_gpm, plot=False, std_corr=1.0, given_bins=None, ivar_weights=False):
    # compute the CF for one QSO spectrum
    # options for redshift_bin are 'low', 'high', 'all'
    # cgm_fit_gpm are gpm from MgiiFinder.py

    raw_data_out, _, all_masks_out = mutils.init_onespec(iqso, redshift_bin)
    wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
    strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, telluric_mask, master_mask = all_masks_out

    # ivar *= (fluxfit**2) # normalize by cont
    # ivar *= (1/std_corr**2) # apply correction

    ###### CF from not masking CGM ######
    all_masks = master_mask

    norm_good_flux = (flux / fluxfit)[all_masks]
    ivar_good = ivar[all_masks]
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

    if ivar_weights:
        print("use ivar as weights in CF")
        # vel_mid, xi_tot, npix_tot, _ = reion_utils.compute_xi_ivar(deltaf_tot, ivar, vel, vmin_corr, vmax_corr, dv_corr, given_bins=given_bins, gpm=all_masks)
        weights_in = ivar
    else:
        # vel_mid, xi_tot, npix_tot, _ = reion_utils.compute_xi(deltaf_tot, vel, vmin_corr, vmax_corr, dv_corr, given_bins=given_bins, gpm=all_masks)
        weights_in = None

    vel_mid, xi_tot, npix_tot, _ = reion_utils.compute_xi_weights(deltaf_tot, vel, vmin_corr, vmax_corr, dv_corr,
                                                                  given_bins=given_bins, gpm=all_masks,
                                                                  weights_in=weights_in)

    xi_mean_tot = np.mean(xi_tot,
                          axis=0)  # not really averaging here since it's one spectrum (i.e. xi_mean_tot = xi_tot)

    ###### CF from masking CGM ######
    """
    norm_good_flux_cgm = norm_good_flux[cgm_fit_gpm]
    meanflux_tot_mask = np.mean(norm_good_flux_cgm)
    #deltaf_tot_mask = (norm_good_flux_cgm - meanflux_tot_mask) / meanflux_tot_mask
    #vel_cgm = vel[all_masks][cgm_fit_gpm]
    deltaf_tot_mask = (norm_good_flux - meanflux_tot_mask) / meanflux_tot_mask
    vel_cgm = vel[all_masks]
    """

    norm_good_flux_cgm = norm_flux[all_masks * cgm_fit_gpm]
    meanflux_tot_mask = np.mean(norm_good_flux_cgm)
    deltaf_tot_mask = (norm_flux - meanflux_tot_mask) / meanflux_tot_mask
    vel_cgm = vel

    """
    if ivar_weights:
        print("use ivar as weights in CF")
        #vel_mid, xi_tot_mask, npix_tot_chimask, _ = reion_utils.compute_xi_ivar(deltaf_tot_mask, ivar_good, vel_cgm, vmin_corr, vmax_corr, dv_corr, given_bins=given_bins, gpm=cgm_fit_gpm)
        weights_in = ivar_good
    else:
        #vel_mid, xi_tot_mask, npix_tot_chimask, _ = reion_utils.compute_xi(deltaf_tot_mask, vel_cgm, vmin_corr, vmax_corr, dv_corr, given_bins=given_bins, gpm=cgm_fit_gpm)
        weights_in = None

    vel_mid, xi_tot_mask, npix_tot_chimask, _ = reion_utils.compute_xi_weights(deltaf_tot_mask, vel_cgm, vmin_corr, vmax_corr,
                                                                       dv_corr, given_bins=given_bins, gpm=cgm_fit_gpm, weights_in=weights_in)
    """

    vel_mid, xi_tot_mask, npix_tot_chimask, _ = reion_utils.compute_xi_weights(deltaf_tot_mask, vel_cgm, vmin_corr,
                                                                               vmax_corr,
                                                                               dv_corr, given_bins=given_bins,
                                                                               gpm=all_masks * cgm_fit_gpm,
                                                                               weights_in=weights_in)

    xi_mean_tot_mask = np.mean(xi_tot_mask, axis=0)  # again not really averaging here since we only have 1 spectrum

    print("==============")
    print("MEAN FLUX", meanflux_tot, meanflux_tot_mask)
    print("mean(DELTA FLUX)", np.mean(deltaf_tot[all_masks]), np.mean(deltaf_tot_mask[all_masks * cgm_fit_gpm]))

    ###### CF from pure noise (no CGM masking) ######
    seed = None
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
        plt.axvline(vel_doublet.value, color='red', linestyle=':', linewidth=1.5,
                    label='Doublet separation (%0.1f km/s)' % vel_doublet.value)

        # plot with masking
        plt.subplot(122)
        plt.plot(vel_mid, xi_mean_tot_mask, linewidth=1.5, label='data masked')
        plt.axhline(0, color='k', ls='--')
        plt.legend(fontsize=15)
        plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
        plt.ylabel(r'$\xi(\Delta v)$', fontsize=18)
        vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
        plt.axvline(vel_doublet.value, color='red', linestyle=':', linewidth=1.5,
                    label='Doublet separation (%0.1f km/s)' % vel_doublet.value)

        plt.tight_layout()
        plt.show()

    # return vel, norm_good_flux, good_ivar, vel_mid, xi_tot, xi_tot_mask, xi_noise, xi_noise_masked, mgii_tot.fit_gpm
    return vel_mid, xi_tot[0], xi_tot_mask[0], npix_tot, npix_tot_chimask, meanflux_tot, meanflux_tot_mask

def fmean_dataset(nqso=8):
    # old
    import compute_model_grid_8qso_fast as cmg8

    z_bin = ['all', 'high', 'low']
    datapath = '/Users/suksientie/Research/MgII_forest/rebinned_spectra/'

    fmean_unmask = []
    fmean_mask = []
    fmean_unmask_onespec = []
    fmean_mask_onespec = []

    for redshift_bin in z_bin:
        vel_data_allqso, norm_flux_allqso, norm_std_allqso, ivar_allqso, \
        master_mask_allqso, master_mask_allqso_mask_cgm, instr_allqso = cmg8.init_dataset(nqso, redshift_bin, datapath)

        # normalized quantities
        f_all = []
        f_all_mask_cgm = []
        ivar_all = []
        ivar_all_mask_cgm = []
        f_all_onespec = []
        f_all_mask_cgm_onespec = []

        for iqso in range(nqso):
            f_all.extend(norm_flux_allqso[iqso][master_mask_allqso[iqso]])
            f_all_mask_cgm.extend(norm_flux_allqso[iqso][master_mask_allqso_mask_cgm[iqso]])
            ivar_all.extend(ivar_allqso[iqso][master_mask_allqso[iqso]])
            ivar_all_mask_cgm.extend(ivar_allqso[iqso][master_mask_allqso_mask_cgm[iqso]])

            f_all_onespec.append(np.mean(norm_flux_allqso[iqso][master_mask_allqso[iqso]]))
            f_all_mask_cgm_onespec.append(np.mean(norm_flux_allqso[iqso][master_mask_allqso_mask_cgm[iqso]]))

        # weighted global mean flux
        fmean_unmask.append(np.average(f_all, weights=ivar_all))
        fmean_mask.append(np.average(f_all_mask_cgm, weights=ivar_all_mask_cgm))
        #fmean_zbin.append(np.mean(f_all))
        #fmean_zbin_mask_cgm.append(np.mean(f_all_mask_cgm))

        fmean_unmask_onespec.append(f_all_onespec)
        fmean_mask_onespec.append(f_all_mask_cgm_onespec)

    return fmean_unmask, fmean_mask, fmean_unmask_onespec, fmean_mask_onespec

def fmean_dataset2(norm_flux_allqso, master_mask_allqso, ivar_allqso, master_mask_allqso_mask_cgm, iqso_remove=None):
    # old
    nqso = 10
    if iqso_remove is not None:
        iqso_to_use = np.delete(np.arange(nqso), iqso_remove)
    else:
        iqso_to_use = np.arange(nqso)

    # normalized quantities
    f_all = []
    f_all_mask_cgm = []
    ivar_all = []
    ivar_all_mask_cgm = []
    f_all_onespec = []
    f_all_mask_cgm_onespec = []

    for iqso in iqso_to_use:
        f_all.extend(norm_flux_allqso[iqso][master_mask_allqso[iqso]])
        f_all_mask_cgm.extend(norm_flux_allqso[iqso][master_mask_allqso_mask_cgm[iqso]])
        ivar_all.extend(ivar_allqso[iqso][master_mask_allqso[iqso]])
        ivar_all_mask_cgm.extend(ivar_allqso[iqso][master_mask_allqso_mask_cgm[iqso]])

        f_all_onespec.append(np.mean(norm_flux_allqso[iqso][master_mask_allqso[iqso]]))
        f_all_mask_cgm_onespec.append(np.mean(norm_flux_allqso[iqso][master_mask_allqso_mask_cgm[iqso]]))

    fmean_unmask = np.average(f_all, weights=ivar_all)
    fmean_mask = np.average(f_all_mask_cgm, weights=ivar_all_mask_cgm)

    return fmean_unmask, fmean_mask

    #fractional diff of fmean remains unaffected within ~0.1% when iteratively removing qso
    #iqso_remove_ls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, [4, 5, 6]]