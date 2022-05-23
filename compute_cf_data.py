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


everyn_break = 20
exclude_restwave = 1216 - 1185 # excluding proximity zones; see mutils.qso_exclude_proximity_zone

########## 2PCF settings ##########
mosfire_res = 3610 # K-band for 0.7" slit (https://www2.keck.hawaii.edu/inst/mosfire/grating.html)
fwhm = 90 # used in compute_model_grid.py

vmin_corr = 10
vmax_corr = 3500
dv_corr = 100  # slightly larger than fwhm
#corr_all = [0.689, 0.640, 0.616, 0.583] # values used by compute_model_grid_new.py
corr_all = [0.758, 0.753, 0.701, 0.724] # determined from mutils.plot_allspec_pdf
median_z = 6.57 # value used in mutils.init_onespec

def init_cgm_fit_gpm():
    lowz_mgii_tot_all, highz_mgii_tot_all, allz_mgii_tot_all = mask_cgm.do_allqso_allzbin()

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

def onespec(iqso, redshift_bin, cgm_fit_gpm, plot=False, std_corr=1.0, seed=None):
    # compute the CF for one QSO spectrum
    # updated 4/14/2022
    # options for redshift_bin are 'low', 'high', 'all'
    # cgm_fit_gpm are gpm from MgiiFinder.py

    raw_data_out, _, all_masks_out = mutils.init_onespec(iqso, redshift_bin)
    wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
    strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, master_mask = all_masks_out

    ###### CF from not masking CGM ######
    all_masks = mask * redshift_mask * pz_mask * zbin_mask

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
    vel_mid, xi_tot, npix_tot, _ = reion_utils.compute_xi(deltaf_tot, vel, vmin_corr, vmax_corr, dv_corr, gpm=all_masks)
    xi_mean_tot = np.mean(xi_tot, axis=0)  # not really averaging here since it's one spectrum (i.e. xi_mean_tot = xi_tot)

    ###### CF from masking CGM ######
    norm_good_flux_cgm = norm_good_flux[cgm_fit_gpm]
    meanflux_tot_mask = np.mean(norm_good_flux_cgm)
    #deltaf_tot_mask = (norm_good_flux_cgm - meanflux_tot_mask) / meanflux_tot_mask
    #vel_cgm = vel[all_masks][cgm_fit_gpm]
    deltaf_tot_mask = (norm_good_flux - meanflux_tot_mask) / meanflux_tot_mask
    vel_cgm = vel[all_masks]
    vel_mid, xi_tot_mask, npix_tot_chimask, _ = reion_utils.compute_xi(deltaf_tot_mask, vel_cgm, vmin_corr, vmax_corr, dv_corr, gpm=cgm_fit_gpm)
    xi_mean_tot_mask = np.mean(xi_tot_mask, axis=0) # again not really averaging here since we only have 1 spectrum

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
    vel_mid, xi_noise, _, _ = reion_utils.compute_xi(deltaf_tot, vel, vmin_corr, vmax_corr, dv_corr, gpm=all_masks_nreal)

    ###### CF from pure noise (CGM masking) ######
    #meanflux_tot_mask = np.mean(fnoise[mgii_tot.fit_gpm])
    #deltaf_tot_mask = (fnoise - meanflux_tot_mask) / meanflux_tot_mask
    deltaf_tot_mask = fnoise_masked / meanflux_tot_mask
    vel_mid, xi_noise_masked, _, _ = reion_utils.compute_xi(deltaf_tot_mask, vel_cgm, vmin_corr, vmax_corr, dv_corr, gpm=cgm_fit_gpm_nreal)

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

    #return vel, norm_good_flux, good_ivar, vel_mid, xi_tot, xi_tot_mask, xi_noise, xi_noise_masked, mgii_tot.fit_gpm
    return vel_mid, xi_tot, xi_tot_mask, xi_noise, xi_noise_masked#, npix_tot, npix_tot_chimask

def allspec(nqso, redshift_bin, cgm_fit_gpm_all, plot=False, seed_list=[None, None, None, None]):
    # running onespec() for all the 4 QSOs

    xi_unmask_all = []
    xi_mask_all = []
    xi_noise_unmask_all = []
    xi_noise_mask_all = []

    for iqso in range(nqso):
        std_corr = corr_all[iqso]
        vel_mid, xi_unmask, xi_mask, xi_noise, xi_noise_masked = onespec(iqso, redshift_bin, cgm_fit_gpm_all[iqso], \
                                                                         plot=False, std_corr=std_corr, seed=None)
        xi_unmask_all.append(xi_unmask[0])
        xi_mask_all.append(xi_mask[0])
        #xi_noise_unmask_all.append(xi_noise[0])
        #xi_noise_mask_all.append(xi_noise_masked[0])
        xi_noise_unmask_all.append(xi_noise)
        xi_noise_mask_all.append(xi_noise_masked)

    ### un-masked quantities
    # data and noise
    xi_unmask_all = np.array(xi_unmask_all)
    xi_mean_unmask = np.mean(xi_unmask_all, axis=0)
    xi_std_unmask = np.std(xi_unmask_all, axis=0)
    xi_noise_unmask_all = np.array(xi_noise_unmask_all) # = (nqso, n_real, n_velmid)
    #xi_mean_noise_unmask = np.mean(xi_noise_unmask_all, axis=0)

    ### masked quantities
    # data and noise
    xi_mask_all = np.array(xi_mask_all)
    xi_mean_mask = np.mean(xi_mask_all, axis=0)
    xi_std_mask = np.std(xi_mask_all, axis=0)
    xi_noise_mask_all = np.array(xi_noise_mask_all)
    #xi_mean_noise_mask = np.mean(xi_noise_mask_all, axis=0)

    if plot:
        plt.figure()
        xi_scale = 1
        ymin, ymax = -0.0010 * xi_scale, 0.002 * xi_scale

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
        plt.ylabel(r'$\xi(\Delta v)$', fontsize=18)
        vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
        print("vel doublet at", vel_doublet.value)
        plt.axvline(vel_doublet.value, color='green', linestyle=':', linewidth=1.5, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
        plt.ylim([ymin, ymax])
        plt.tight_layout()

        plt.figure()
        xi_scale = 1e5
        ymin, ymax = -0.0010 * xi_scale, 0.0006 * xi_scale

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
        plt.axvline(vel_doublet.value, color='green', linestyle=':', linewidth=1.5, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
        plt.ylim([ymin, ymax])
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
    for i in range(4):
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

    plt.subplot(132)
    plt.title('z >= %0.3f' % median_z, fontsize=15)
    for i in range(4):
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

    plt.subplot(133)
    plt.title('All z', fontsize=15)
    for i in range(4):
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
    plt.tight_layout()

    plt.show()

#################### debugging ####################
# 5/23/2022
def allspec_npixcorr(nqso, redshift_bin, cgm_fit_gpm_all, plot=False, seed_list=[None, None, None, None]):
    # same as allspec, but returning the npix for each dv bin

    xi_unmask_all = []
    xi_mask_all = []
    xi_noise_unmask_all = []
    xi_noise_mask_all = []
    npix_unmask_all = []
    npix_mask_all = []

    for iqso in range(nqso):
        std_corr = corr_all[iqso]
        vel_mid, xi_unmask, xi_mask, xi_noise, xi_noise_masked, npix_tot, npix_tot_chimask = onespec(iqso, redshift_bin, cgm_fit_gpm_all[iqso], \
                                                                         plot=False, std_corr=std_corr, seed=None)
        xi_unmask_all.append(xi_unmask[0])
        xi_mask_all.append(xi_mask[0])
        #xi_noise_unmask_all.append(xi_noise[0])
        #xi_noise_mask_all.append(xi_noise_masked[0])
        xi_noise_unmask_all.append(xi_noise)
        xi_noise_mask_all.append(xi_noise_masked)
        npix_unmask_all.append(npix_tot)
        npix_mask_all.append(npix_tot_chimask)

    ### un-masked quantities
    # data and noise
    xi_unmask_all = np.array(xi_unmask_all)
    xi_mean_unmask = np.mean(xi_unmask_all, axis=0)
    xi_std_unmask = np.std(xi_unmask_all, axis=0)
    xi_noise_unmask_all = np.array(xi_noise_unmask_all) # = (nqso, n_real, n_velmid)
    #xi_mean_noise_unmask = np.mean(xi_noise_unmask_all, axis=0)

    ### masked quantities
    # data and noise
    xi_mask_all = np.array(xi_mask_all)
    xi_mean_mask = np.mean(xi_mask_all, axis=0)
    xi_std_mask = np.std(xi_mask_all, axis=0)
    xi_noise_mask_all = np.array(xi_noise_mask_all)
    #xi_mean_noise_mask = np.mean(xi_noise_mask_all, axis=0)

    if plot:
        plt.figure()
        xi_scale = 1
        ymin, ymax = -0.0010 * xi_scale, 0.002 * xi_scale

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
        plt.ylabel(r'$\xi(\Delta v)$', fontsize=18)
        vel_doublet = reion_utils.vel_metal_doublet('Mg II', returnVerbose=False)
        print("vel doublet at", vel_doublet.value)
        plt.axvline(vel_doublet.value, color='green', linestyle=':', linewidth=1.5, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
        plt.ylim([ymin, ymax])
        plt.tight_layout()

        plt.figure()
        xi_scale = 1e5
        ymin, ymax = -0.0010 * xi_scale, 0.0006 * xi_scale

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
        plt.axvline(vel_doublet.value, color='green', linestyle=':', linewidth=1.5, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
        plt.ylim([ymin, ymax])
        plt.tight_layout()

        plt.show()

    return vel_mid, xi_mean_unmask, xi_mean_mask, xi_noise_unmask_all, xi_noise_mask_all, xi_unmask_all, xi_mask_all, npix_unmask_all, npix_mask_all

def onespec_zeros(iqso, redshift_bin, dv_corr):

    raw_data_out, _, all_masks_out = mutils.init_onespec(iqso, redshift_bin)
    wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
    strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, master_mask = all_masks_out

    vel = mutils.obswave_to_vel_2(wave)
    #deltaf_tot = np.zeros(flux.shape)
    deltaf_tot = np.ones(flux.shape)

    vel_mid, xi_tot, npix_tot, _ = reion_utils.compute_xi(deltaf_tot, vel, vmin_corr, vmax_corr, dv_corr)
    xi_mean_tot = np.mean(xi_tot, axis=0)

    return vel_mid, xi_mean_tot, npix_tot, deltaf_tot, vel

def onespec_zeros_dvcorr():

    # running onespec_zeros (above) for various values of dv_corr to see trends of npix vs. velocity lags

    # (5/23/22) conclusion is that we get the correct behavior (i.e. npix smoothly decreasing with velocity lag bin)
    # when using dv_corr that is an integer multiple of the fwhm, otherwise one sees a zig-zaggy pattern
    # see plots documented below
    dv_corr_ls = [90, 120, 150, 180] # plots/debug/npix_dvcorr1.png
    #dv_corr_ls = [100, 130, 200, 400] # plots/debug/npix_dvcorr2.png
    vel_mid_all = []
    npix_tot_all = []

    for dv_corr in dv_corr_ls:
        vel_mid, _, npix_tot, _, _ = onespec_zeros(0, 'all', dv_corr)
        vel_mid_all.append(vel_mid)
        npix_tot_all.append(npix_tot)
        plt.plot(vel_mid, npix_tot[0], '.-', label='dv_corr=%d' % dv_corr)

    plt.legend()
    plt.xlabel('dv (km/s)')
    plt.ylabel('npix_corr')
    plt.show()
    return vel_mid_all, npix_tot_all

import compute_model_grid_new as cmg
def onespec_random(iqso, redshift_bin, vel_lores, mask_chunk):

    vmin_corr, vmax_corr, dv_corr = 10, 3500, 100
    raw_data_out, _, all_masks_out = mutils.init_onespec(iqso, redshift_bin)
    wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
    strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, master_mask = all_masks_out

    vel = mutils.obswave_to_vel_2(wave)
    df = np.random.uniform(-1, 1, len(flux))

    df_chunk = cmg.reshape_data_array(df, 10, 220, data_arr_is_mask=False)
    if type(mask_chunk) == type(None):
        mask_chunk = np.invert(np.isnan(df_chunk))

    vel_mid, xi_tot, npix_tot, _ = reion_utils.compute_xi(df_chunk, vel_lores, vmin_corr, vmax_corr, dv_corr, gpm=mask_chunk)
    xi_mean_tot = np.mean(xi_tot, axis=0)

    plt.figure()
    for i in range(len(xi_tot)):
        plt.plot(vel_mid, xi_tot[i], alpha=0.2)
    plt.plot(vel_mid, xi_mean_tot, 'ko-')
    plt.axvline(1160, color='k', ls='--')

    plt.figure()
    for i in range(len(xi_tot)):
        plt.plot(vel_mid, npix_tot[i], alpha=0.2)
    plt.axvline(1160, color='k', ls='--')

    df_long = df_chunk.flatten()
    mask_long = mask_chunk.flatten()
    vel_lores_long = []
    for i in range(10):
        if len(vel_lores_long) == 0:
            vel_lores_long.extend(vel_lores)
        else:
            vel_lores_long.extend(vel_lores_long[-1] + np.arange(1, len(vel_lores) + 1) * np.diff(vel_lores)[0])

    vel_mid, xi_tot_long, npix_tot_long, _ = reion_utils.compute_xi(df_long, vel_lores_long, vmin_corr, vmax_corr, dv_corr, gpm=mask_long)
    plt.figure()
    plt.plot(vel_mid, xi_tot_long[0])
    plt.plot(vel_mid, xi_mean_tot, 'ko-')
    plt.axvline(1160, color='k', ls='--')

    return vel_mid, xi_mean_tot, xi_tot, npix_tot, xi_tot_long, npix_tot_long