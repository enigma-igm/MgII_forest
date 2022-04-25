'''
Functions here are:
    - obswave_to_vel_2
    - extract_data
    - continuum_normalize
    - custom_mask_J0313
    - custom_mask_J1342
    - custom_mask_J0038
    - extract_and_norm
    - init_skewers_compute_model_grid
    - init_onespec
    - pad_fluxfit
'''

import sys
sys.path.append('/Users/suksientie/codes/enigma')
sys.path.append('/Users/suksientie/Research/data_redux')
sys.path.append('/Users/suksientie/Research/CIV_forest')
sys.path.append('/home/sstie/codes/PypeIt') # for running on IGM cluster
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from astropy.io import fits
from pypeit.core.fitting import iterfit, robust_fit
from pypeit import utils as putils
from pypeit import bspline
import scipy.interpolate as interpolate
from astropy import constants as const
from astropy.table import Table
from astropy.stats import sigma_clip, mad_std
from astropy.modeling.models import Gaussian1D
from enigma.reion_forest import utils
import compute_cf_data as ccf
import pdb

def obswave_to_vel_2(wave_arr):
    # converts wavelength array from Angstrom to km/s using the following relation
    # dv (km/s) =  c * d(log_lambda), and log is natural log
    # Input: 'wave_arr' = numpy array
    # Output: velocity array

    # (01/20/2022) use this, not obswave_to_vel()

    c_kms = const.c.to('km/s').value
    log10_wave = np.log10(wave_arr)
    diff_log10_wave = np.diff(log10_wave) # d(log10_lambda)
    diff_log10_wave = np.append(diff_log10_wave, diff_log10_wave[-1]) # appending the last value twice to make array same size as wave_arr
    dv = c_kms * np.log(10) * diff_log10_wave
    #vel = np.zeros(len(wave_arr))
    #vel[1:] = np.cumsum(dv)
    vel = np.cumsum(dv) #  first pixel is dv; vel = [dv, 2*dv, 3*dv, ....]

    return vel

def extract_data(fitsfile):
    # 'fitsfile' = name of fitsfile containing Pypeit 1d spectrum

    data = fits.open(fitsfile)[1].data
    wave_arr = data['wave_grid_mid'].astype('float64') # midpoint values of wavelength bin
    flux_arr = data['flux'].astype('float64')
    ivar_arr = data['ivar'].astype('float64')
    mask_arr = data['mask'].astype('bool')
    std_arr = np.sqrt(putils.inverse(ivar_arr))
    tell_arr = data['telluric'].astype('float64')

    return wave_arr, flux_arr, ivar_arr, mask_arr, std_arr, tell_arr

def continuum_normalize_new(wave_arr, flux_arr, ivar_arr, mask_arr, std_arr, nbkpt, plot=False):

    # continuum normalize using breakpoint spline method in Pypeit

    # nbkpt: NOT the total number of breakpoints, but instead it's placing a breakpoint at every n-th index,
    # since we're using the 'everyn' argument below. I.e. if nbkpt=20, it places a breakpoint at every 20-th element.
    (sset, outmask) = iterfit(wave_arr, flux_arr, invvar=ivar_arr, inmask=mask_arr, upper=3, lower=3, x2=None,
                              maxiter=10, nord=4, bkpt=None, fullbkpt=None, kwargs_bspline = {'everyn': nbkpt})

    cont_fit, cont_fit_mask = sset.value(wave_arr)

    if plot:
        # plotting provided by Joe
        goodbk = sset.mask
        # This is approximate
        yfit_bkpt = np.interp(sset.breakpoints[goodbk], wave_arr, cont_fit)

        plt.figure(figsize=(12, 5))
        ax = plt.gca()
        was_fit_and_masked = mask_arr & np.logical_not(outmask)
        print(np.sum(was_fit_and_masked))
        ax.plot(wave_arr[mask_arr], flux_arr[mask_arr], color='k', marker='o', markersize=0.4, mfc='k', fillstyle='full',
                linestyle='-', label='Pixels that were fit')
        ax.plot(wave_arr[was_fit_and_masked], flux_arr[was_fit_and_masked], color='red', marker='x', markersize=5.0, mfc='red',
                fillstyle='full', linestyle='None', label='Pixels masked by fit')
        ax.plot(wave_arr, cont_fit, color='cornflowerblue', label='B-spline fit')
        ax.plot(sset.breakpoints[goodbk], yfit_bkpt, color='lawngreen', marker='o', markersize=4.0, mfc='lawngreen',
                fillstyle='full', linestyle='None', label='Good B-spline breakpoints')
        #ax.set_ylim((0.99 * cont_fit.min(), 1.01 * cont_fit.max()))
        plt.ylabel('Flux')
        plt.xlabel('Wave (Ang)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return cont_fit, cont_fit_mask, sset, outmask

################## by-eye strong absorbers masks for each QSO ##################
def custom_mask_J0313(fitsfile, plot=False):

    wave, flux, ivar, mask, std, tell = extract_data(fitsfile)

    mask_wave1 = [19815, 19825]
    mask_wave2 = [19865, 19870]
    mask_wave3 = [23303, 23325]
    mask_wave4 = [23370, 23387]

    all_mask_wave = [mask_wave1, mask_wave2, mask_wave3, mask_wave4]
    strong_abs_gpm = np.ones(wave.shape, dtype=bool) # good pixel mask accounting for strong absorbers

    for mask_wave_i in all_mask_wave:
        a = mask_wave_i[0] < wave
        b = wave < mask_wave_i[1]
        gpm = np.invert(a * b)
        strong_abs_gpm *= gpm

    if plot:
        alpha = 0.3
        plt.plot(wave, flux, c='b', drawstyle='steps-mid')
        plt.plot(wave, std, c='k', drawstyle='steps-mid')
        plt.axvspan(mask_wave1[0], mask_wave1[1], facecolor='r', alpha=alpha)
        plt.axvspan(mask_wave2[0], mask_wave2[1], facecolor='r', alpha=alpha)
        plt.axvspan(mask_wave3[0], mask_wave3[1], facecolor='r', alpha=alpha)
        plt.axvspan(mask_wave4[0], mask_wave4[1], facecolor='r', alpha=alpha)
        plt.show()

    return strong_abs_gpm

def custom_mask_J1342(fitsfile, plot=False):

    wave, flux, ivar, mask, std, tell = extract_data(fitsfile)

    # visually-identified strong absorbers
    mask_wave1 = [21920, 21940]
    mask_wave2 = [21972, 22000]
    mask_wave3 = [20320, 20335]
    mask_wave4 = [20375, 20400]

    all_mask_wave = [mask_wave1, mask_wave2, mask_wave3, mask_wave4]
    strong_abs_gpm = np.ones(wave.shape, dtype=bool) # good pixel mask accounting for strong absorbers

    for mask_wave_i in all_mask_wave:
        a = mask_wave_i[0] < wave
        b = wave < mask_wave_i[1]
        gpm = np.invert(a * b)
        strong_abs_gpm *= gpm

    if plot:
        alpha = 0.3
        plt.plot(wave, flux, c='b', drawstyle='steps-mid')
        plt.plot(wave, std, c='k', drawstyle='steps-mid')
        plt.axvspan(mask_wave1[0], mask_wave1[1], facecolor='r', alpha=alpha)
        plt.axvspan(mask_wave2[0], mask_wave2[1], facecolor='r', alpha=alpha)
        plt.axvspan(mask_wave3[0], mask_wave3[1], facecolor='r', alpha=alpha)
        plt.axvspan(mask_wave4[0], mask_wave4[1], facecolor='r', alpha=alpha)
        plt.show()

    return strong_abs_gpm

def custom_mask_J0252(fitsfile, plot=False):

    # no strong absorbers that I can identify by eye
    # placeholder function in case want to add strong absorbers
    wave, flux, ivar, mask, std, tell = extract_data(fitsfile)
    strong_abs_gpm = np.ones(wave.shape, dtype=bool)

    if plot:
        alpha = 0.3
        plt.plot(wave, flux, c='b', drawstyle='steps-mid')
        plt.plot(wave, std, c='k', drawstyle='steps-mid')
        plt.show()

    return strong_abs_gpm

def custom_mask_J0038(fitsfile, plot=False):

    wave, flux, ivar, mask, std, tell = extract_data(fitsfile)
    # visually-identified strong absorbers
    mask_wave1 = [19777, 19796]
    mask_wave2 = [19828, 19855]

    all_mask_wave = [mask_wave1, mask_wave2]
    strong_abs_gpm = np.ones(wave.shape, dtype=bool)
    for mask_wave_i in all_mask_wave:
        a = mask_wave_i[0] < wave
        b = wave < mask_wave_i[1]
        gpm = np.invert(a * b)
        strong_abs_gpm *= gpm

    if plot:
        alpha = 0.3
        plt.plot(wave, flux, c='b', drawstyle='steps-mid')
        plt.plot(wave, std, c='k', drawstyle='steps-mid')
        plt.axvspan(mask_wave1[0], mask_wave1[1], facecolor='r', alpha=alpha)
        plt.axvspan(mask_wave2[0], mask_wave2[1], facecolor='r', alpha=alpha)
        plt.show()

    return strong_abs_gpm

######################################################
def extract_and_norm(fitsfile, everyn_bkpt, qso_name, plot=False):

    # combine extract_data() and continuum_normalize_new() including by-eye masking of strong absorbers
    wave, flux, ivar, mask, std, tell = extract_data(fitsfile)

    if qso_name == 'J0313-1806':
        print('using custom mask for %s' % qso_name)
        strong_abs_gpm = custom_mask_J0313(fitsfile)

    elif qso_name == 'J1342+0928':
        print('using custom mask for %s' % qso_name)
        strong_abs_gpm = custom_mask_J1342(fitsfile)

    elif qso_name == 'J0252-0503':
        print('using custom mask for %s' % qso_name)
        strong_abs_gpm = custom_mask_J0252(fitsfile)

    elif qso_name == 'J0038-1527':
        print('using custom mask for %s' % qso_name)
        strong_abs_gpm = custom_mask_J0038(fitsfile)

    else:
        print('using custom mask for %s' % qso_name)
        strong_abs_gpm = np.ones(wave.shape, dtype=bool) # dummy mask

    inmask = mask * strong_abs_gpm
    fluxfit, fluxfit_mask, sset, bspline_mask = continuum_normalize_new(wave, flux, ivar, inmask, std, everyn_bkpt, plot=plot)

    return wave, flux, ivar, mask, std, tell, fluxfit, strong_abs_gpm

######################################################
def qso_exclude_proximity_zone(fitsfile, qso_z, qso_name, exclude_rest=1216-1185, plot=False):
    # BAO lyaf: 1040 A < lambda_rest < 1200 A
    # Bosman+2021: lambda_rest < 1185 A
    # the default exclude_rest value uses Bosman cut off

    wave, flux, ivar, mask, std, tell, fluxfit, strong_abs_gpm = extract_and_norm(fitsfile, 20, qso_name)

    redshift_mask = wave <= (2800 * (1 + qso_z))  # removing spectral region beyond qso redshift
    master_mask = mask * strong_abs_gpm * redshift_mask

    good_wave = wave[master_mask]
    obs_wave_max = (2800 - exclude_rest)*(1 + qso_z)
    npix_out = len(np.where(good_wave > obs_wave_max)[0])
    print(npix_out/len(good_wave), obs_wave_max, good_wave.max())

    if plot:
        plt.figure()
        plt.plot(wave, flux)
        plt.axvline(obs_wave_max, color='r', label='obs wave max')
        plt.axvline(2800 * (1 + qso_z), color='k', label='qso redshift')
        plt.legend()
        plt.show()

def qso_redshift_and_pz_mask(wave, qso_z, exclude_rest=1216-1185):

    redshift_mask = wave <= (2800 * (1 + qso_z))  # removing spectral region beyond qso redshift
    obs_wave_max = (2800 - exclude_rest) * (1 + qso_z)
    pz_mask = wave < obs_wave_max

    return redshift_mask, pz_mask, obs_wave_max

def final_qso_pathlength(fitsfile, qso_name, qso_z, exclude_rest=1216-1185, cgm_gpm=None):

    wave, flux, ivar, mask, std, tell, fluxfit, strong_abs_gpm = extract_and_norm(fitsfile, 20, qso_name)
    redshift_mask, pz_mask, obs_wave_max = qso_redshift_and_pz_mask(wave, qso_z, exclude_rest)

    if type(cgm_gpm) == type(None):
        # only using data and PZ masks
        master_mask = mask * redshift_mask * pz_mask
        good_wave = wave[master_mask]

        # (4/25/2022)
        # In[96]: np.median(gz_all), np.min(gz_all), np.max(gz_all)
        # Out[96]: (6.572035934151384, 5.983190982278802, 7.545849071317116)

    else:
        #master_mask = mask * redshift_mask * pz_mask
        #good_wave = wave[master_mask][cgm_gpm]
        master_mask = mask * redshift_mask * pz_mask * cgm_gpm
        good_wave = wave[master_mask]

        # (4/25/2022)
        # In[90]: np.median(gz_all), np.min(gz_all), np.max(gz_all)
        # Out[90]: (6.57430945878276, 5.983190982278802, 7.545849071317116)

    good_z = good_wave / 2800 - 1
    dz_pathlength = good_z.max() - good_z.min()
    print(qso_name, np.min(good_z), np.max(good_z), np.median(good_z), dz_pathlength)

    return good_z

def init_onespec(iqso, redshift_bin, datapath='/Users/suksientie/Research/data_redux/'):
    # initialize all needed quantities for one qso, included all masks, for all subsequent analyses
    # important to make sure fits files and global variables are up to dates

    # datapath = '/Users/suksientie/Research/data_redux/'
    # datapath = '/mnt/quasar/sstie/MgII_forest/z75/'

    # not using the padded fits for J0038-1527 (4/25/2022)
    fitsfile_list = [datapath + 'wavegrid_vel/J0313-1806/vel1234_coadd_tellcorr.fits', \
                     datapath + 'wavegrid_vel/J1342+0928/vel123_coadd_tellcorr.fits', \
                     datapath + 'wavegrid_vel/J0252-0503/vel12_coadd_tellcorr.fits', \
                     datapath + 'wavegrid_vel/J0038-1527/vel1_tellcorr.fits', \
                     datapath + 'wavegrid_vel/J0038-0653/vel1_tellcorr_pad.fits']

    qso_namelist = ['J0313-1806', 'J1342+0928', 'J0252-0503', 'J0038-1527', 'J0038-0653']
    qso_zlist = [7.642, 7.541, 7.001, 7.034, 7.0]
    everyn_break_list = [20, 20, 20, 20, 20]
    exclude_restwave = 1216 - 1185
    median_z = 6.573  # median pixel redshift of measurement (excluding proximity zones)

    fitsfile = fitsfile_list[iqso]
    wave, flux, ivar, mask, std, tell, fluxfit, strong_abs_gpm = extract_and_norm(fitsfile, everyn_break_list[iqso], qso_namelist[iqso])
    redshift_mask, pz_mask, obs_wave_max = qso_redshift_and_pz_mask(wave, qso_zlist[iqso], exclude_restwave)

    if redshift_bin == 'low':
        zbin_mask = wave < (2800 * (1 + median_z))

    elif redshift_bin == 'high':
        zbin_mask = wave >= (2800 * (1 + median_z))

    elif redshift_bin == 'all':
        zbin_mask = np.ones_like(wave, dtype=bool)

    # master mask for measuring 2PCF
    master_mask = mask * strong_abs_gpm * redshift_mask * pz_mask * zbin_mask

    # masked arrays
    good_wave = wave[master_mask]
    good_flux = flux[master_mask]
    good_ivar = ivar[master_mask]
    good_std = std[master_mask]
    good_vel_data = obswave_to_vel_2(good_wave)
    vel_data = obswave_to_vel_2(wave)

    norm_good_flux = good_flux / fluxfit[master_mask]
    norm_good_std = good_std / fluxfit[master_mask]

    raw_data_out = wave, flux, ivar, mask, std, tell, fluxfit
    masked_data_out = good_wave, good_flux, good_ivar, good_std, good_vel_data, norm_good_flux, norm_good_std
    all_masks_out = strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, master_mask

    #return vel_data, master_mask, std, fluxfit, outmask, norm_good_std, norm_std, norm_good_flux, good_vel_data, good_ivar, norm_flux, ivar
    return raw_data_out, masked_data_out, all_masks_out

######################################################
def plot_onespec_pdf(iqso, seed=None, title=None):

    raw_out, masked_out, masks_out = init_onespec(iqso, 'all')
    wave, flux, ivar, mask, std, tell, fluxfit = raw_out
    strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, master_mask = masks_out

    norm_flux = flux/fluxfit
    norm_std = std/fluxfit
    norm_flux = norm_flux[mask * redshift_mask * pz_mask * zbin_mask]
    norm_std = norm_std[mask * redshift_mask * pz_mask * zbin_mask]

    chi = (1 - norm_flux) / norm_std
    corr_factor = mad_std(chi)
    #corr_factor = mad_std(norm_flux)/np.median(norm_std)

    rand = np.random.RandomState(seed) if seed != None else np.random.RandomState()
    gaussian_data = rand.normal(np.median(norm_flux), norm_std * corr_factor)

    plt.figure(figsize=(8, 5))
    if title != None:
        plt.suptitle(title, fontsize=18)

    ##### plot (1 - F) PDF #####
    nbins, oneminf_min, oneminf_max = 71, 1e-5, 1.0
    flux_bins, flux_pdf_data = utils.pdf_calc(1.0 - norm_flux, oneminf_min, oneminf_max, nbins)
    flux_bins, flux_pdf_data_symm = utils.pdf_calc(- (1 - norm_flux), oneminf_min, oneminf_max, nbins)
    flux_bins, flux_pdf_gaussian = utils.pdf_calc(1 - gaussian_data, oneminf_min, oneminf_max, nbins)

    plt.plot(flux_bins, flux_pdf_data, drawstyle='steps-mid', alpha=1.0, lw=2, label='1 - F')
    plt.plot(flux_bins, flux_pdf_data_symm, drawstyle='steps-mid', alpha=1.0, lw=2, label='F - 1')
    plt.plot(flux_bins, flux_pdf_gaussian, drawstyle='steps-mid', alpha=1.0, lw=1, \
             label=r'gaussian ($\sigma = \sigma_{\rm{ipix}}$ * corr), corr=%0.2f' % corr_factor)

    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('PDF')

    plt.tight_layout()
    plt.show()

def plot_allspec_pdf():

    qso_namelist = ['J0313-1806', 'J1342+0928', 'J0252-0503', 'J0038-1527']
    plt.figure(figsize=(12, 8))

    for iqso in range(4):
        seed = None
        redshift_bin = 'high'
        raw_out, masked_out, masks_out = init_onespec(iqso, redshift_bin)
        wave, flux, ivar, mask, std, tell, fluxfit = raw_out
        strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, master_mask = masks_out

        norm_flux = flux/fluxfit
        norm_std = std/fluxfit
        norm_flux = norm_flux[mask * redshift_mask * pz_mask * zbin_mask]
        norm_std = norm_std[mask * redshift_mask * pz_mask * zbin_mask]

        chi = (1 - norm_flux) / norm_std
        corr_factor = mad_std(chi)
        print(corr_factor)
        #corr_all = [0.687, 0.635, 0.617, 0.58]
        #corr_factor = corr_all[iqso]
        rand = np.random.RandomState(seed) if seed != None else np.random.RandomState()
        gaussian_data = rand.normal(np.median(norm_flux), norm_std * corr_factor)

        plt.subplot(2,2,iqso+1)
        plt.title(qso_namelist[iqso])
        nbins, oneminf_min, oneminf_max = 71, 1e-5, 1.0
        flux_bins, flux_pdf_data = utils.pdf_calc(1.0 - norm_flux, oneminf_min, oneminf_max, nbins)
        flux_bins, flux_pdf_data_symm = utils.pdf_calc(- (1 - norm_flux), oneminf_min, oneminf_max, nbins)
        flux_bins, flux_pdf_gaussian = utils.pdf_calc(1 - gaussian_data, oneminf_min, oneminf_max, nbins)

        plt.plot(flux_bins, flux_pdf_data, drawstyle='steps-mid', alpha=1.0, lw=2, label='1 - F')
        plt.plot(flux_bins, flux_pdf_data_symm, drawstyle='steps-mid', alpha=1.0, lw=2, label='F - 1')
        plt.plot(flux_bins, flux_pdf_gaussian, drawstyle='steps-mid', alpha=1.0, lw=1, \
                 label=r'gaussian ($\sigma = \sigma_{\rm{ipix}}$ * corr), corr=%0.2f' % corr_factor) # + '\n' +  r'corr = mad_std(1 - $F_{\rm{ipix}}$)/$\sigma_{\rm{ipix}}$')

        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('PDF')

    plt.tight_layout()
    plt.show()

################ need to deal with all the following later ################


###### compute_model_grid.py testing ######
def init_skewers_compute_model_grid():
    # initialize Nyx skewers for testing purposes in compute_model_grid.py
    file = 'ran_skewers_z75_OVT_xHI_0.50_tau.fits'
    params = Table.read(file, hdu=1)
    skewers = Table.read(file, hdu=2)

    fwhm = 90 # 83
    sampling = 3
    logZ = -3.50

    vel_lores, (flux_lores, flux_lores_igm, flux_lores_cgm, _, _), \
    vel_hires, (flux_hires, flux_hires_igm, flux_hires_cgm, _, _), \
    (oden, v_los, T, xHI), cgm_tuple = utils.create_mgii_forest(params, skewers, logZ, fwhm, sampling=sampling)

    vmin_corr, vmax_corr, dv_corr = 10, 3500, 100

    mean_flux_nless = np.mean(flux_lores)
    delta_f_nless = (flux_lores - mean_flux_nless) / mean_flux_nless
    (vel_mid, xi_nless, npix, xi_nless_zero_lag) = utils.compute_xi(delta_f_nless, vel_lores, vmin_corr, vmax_corr, dv_corr)
    xi_mean = np.mean(xi_nless, axis=0)

    return vel_lores, flux_lores, vel_mid, xi_mean

def init_onespec_tmp(iqso, redshift_bin, datapath, vel_lores, flux_lores, vel_mid, xi_mean, ncopy, cgm_gpm_allspec):

    raw_data_out, _, all_masks_out = init_onespec(iqso, redshift_bin, datapath=datapath)
    wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
    strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, master_mask = all_masks_out

    norm_flux = flux / fluxfit
    norm_std = std / fluxfit
    vel_data = obswave_to_vel_2(wave)

    # generate mock data spectrum
    ranindx, rand_flux_lores, rand_noise_ncopy, noisy_flux_lores_ncopy, nskew_to_match_data, npix_sim_skew = cmg.forward_model_onespec_chunk(vel_data, norm_std, vel_lores, flux_lores, ncopy)
    print(nskew_to_match_data, npix_sim_skew)

    usable_data_mask = mask * redshift_mask * pz_mask * zbin_mask
    usable_data_mask_chunk = cmg.reshape_data_array(usable_data_mask, nskew_to_match_data, npix_sim_skew, True)
    print(np.sum(usable_data_mask_chunk))

    # deal with CGM mask if argued before computing the 2PCF
    if type(cgm_gpm_allspec) == type(None):
        all_mask_chunk = usable_data_mask_chunk
    else:
        gpm_onespec_chunk = cmg.reshape_data_array(cgm_gpm_allspec[iqso], nskew_to_match_data, npix_sim_skew,
                                               True)  # reshaping GPM from cgm masking
        all_mask = usable_data_mask * cgm_gpm_allspec[iqso]
        all_mask_chunk = usable_data_mask_chunk * gpm_onespec_chunk

    cmg.plot_forward_model_onespec_new(noisy_flux_lores_ncopy, rand_noise_ncopy, rand_flux_lores, all_mask, vel_data, norm_flux, 10)


def init_onespec_old(iqso, redshift_bin):
    # initialize all needed data from one qso for testing compute_model_grid.py
    datapath = '/Users/suksientie/Research/data_redux/'
    # datapath = '/mnt/quasar/sstie/MgII_forest/z75/'

    fitsfile_list = [datapath + 'wavegrid_vel/J0313-1806/vel1234_coadd_tellcorr.fits', \
                     datapath + 'wavegrid_vel/J1342+0928/vel123_coadd_tellcorr.fits', \
                     datapath + 'wavegrid_vel/J0252-0503/vel12_coadd_tellcorr.fits', \
                     datapath + 'wavegrid_vel/J0038-1527/vel1_tellcorr_pad.fits']

    qso_namelist = ['J0313-1806', 'J1342+0928', 'J0252-0503', 'J0038-1527']
    qso_zlist = [7.642, 7.541, 7.001, 7.034]
    everyn_break_list = [20, 20, 20, 20]
    exclude_restwave = 1216 - 1185
    median_z = 6.574  # median redshift of measurement (excluding proximity zones)

    fitsfile = fitsfile_list[iqso]
    wave, flux, ivar, mask, std, fluxfit, outmask, sset, tell = extract_and_norm(fitsfile, everyn_break_list[iqso], qso_namelist[iqso])
    vel_data = obswave_to_vel_2(wave)

    redshift_mask = wave <= (2800 * (1 + qso_zlist[iqso]))  # removing spectral region beyond qso redshift
    obs_wave_max = (2800 - exclude_restwave) * (1 + qso_zlist[iqso])
    proximity_zone_mask = wave < obs_wave_max

    if redshift_bin == 'low':
        zbin_mask = wave < (2800 * (1 + median_z))
        zbin_mask_fluxfit = wave[outmask] < (2800 * (1 + median_z))

    elif redshift_bin == 'high':
        zbin_mask = wave >= (2800 * (1 + median_z))
        zbin_mask_fluxfit = wave[outmask] >= (2800 * (1 + median_z))

    elif redshift_bin == 'all':
        zbin_mask = np.ones_like(wave, dtype=bool)
        zbin_mask_fluxfit = np.ones_like(wave[outmask], dtype=bool)

    #master_mask = redshift_mask * outmask # final ultimate mask
    master_mask = redshift_mask * outmask * proximity_zone_mask * zbin_mask

    # masked arrays
    good_wave = wave[master_mask]
    good_flux = flux[master_mask]
    good_ivar = ivar[master_mask]
    good_std = std[master_mask]

    # applying all the data masks above to the fitted continuum
    # fluxfit_redshift = fluxfit[wave[outmask] <= (2800 * (1 + qso_zlist[iqso]))]
    fluxfit_custom_mask = (wave[outmask] <= (2800 * (1 + qso_zlist[iqso]))) * (wave[outmask] < obs_wave_max) * zbin_mask_fluxfit
    fluxfit_redshift = fluxfit[fluxfit_custom_mask]
    norm_good_flux = good_flux / fluxfit_redshift
    norm_good_std = good_std / fluxfit_redshift
    good_vel_data = obswave_to_vel_2(good_wave)

    fluxfit_new = pad_fluxfit(outmask, fluxfit)
    norm_flux = flux / fluxfit_new
    norm_std = std / fluxfit_new

    return vel_data, master_mask, std, fluxfit, outmask, norm_good_std, norm_std, norm_good_flux, good_vel_data, good_ivar, norm_flux, ivar

def pad_fluxfit(outmask, fluxfit):
    # pad the fitted-continuum output from continuum_normalize() so that the array size equals the size of the input data
    # this is because the continuum output from Pypeit's iterfit returns the masked continuum

    # the inputs are really outputs from contiuum_normalize() routine

    outmask_true = np.argwhere(outmask == True).squeeze()
    outmask_false = np.argwhere(outmask == False).squeeze()
    fluxfit_new = np.zeros(outmask.shape) # length of fluxfit_new equals length of raw data

    for i in range(len(outmask)):
        if i in outmask_false:
            fluxfit_new[i] = np.nan # fill the masked pixels with NaNs

    iall_notnan = np.argwhere(np.invert(np.isnan(fluxfit_new))).squeeze()

    for i in range(len(iall_notnan)):
        fluxfit_new[iall_notnan[i]] = fluxfit[i] # fill the non-masked pixels with values

    return fluxfit_new

################################
import compute_model_grid_new as cmg
import pdb
def do_all_onespec_cmg(iqso, redshift_bin, ncopy, vel_lores, flux_lores, std_corr=1.0, cgm_gpm_allspec=None):

    vel_data, master_mask, std, fluxfit, outmask, norm_good_std, norm_std, norm_good_flux, good_vel_data, good_ivar, norm_flux, ivar = \
        init_onespec(iqso, redshift_bin)

    ranindx, rand_flux_lores, rand_noise_ncopy, noisy_flux_lores_ncopy, nskew_to_match_data, npix_sim_skew = \
        cmg.forward_model_onespec_chunk(vel_data, norm_std, vel_lores, flux_lores, ncopy, seed=None, std_corr=std_corr)

    master_mask_chunk = cmg.reshape_data_array(master_mask, nskew_to_match_data, npix_sim_skew, True)

    if type(cgm_gpm_allspec) == type(None):
        pass
    else:
        gpm_onespec_chunk = cmg.reshape_data_array(cgm_gpm_allspec[iqso][0], nskew_to_match_data, npix_sim_skew, True)  # reshaping GPM from cgm masking
        master_mask_chunk *= gpm_onespec_chunk

    flux_masked = []
    noise_masked = []
    for icopy in range(ncopy):
        flux_masked.append(noisy_flux_lores_ncopy[icopy][master_mask_chunk])
        noise_masked.append(rand_noise_ncopy[icopy][master_mask_chunk])
    flux_masked = np.array(flux_masked)
    noise_masked = np.array(noise_masked)
    print("ratio", np.std(norm_good_flux) / np.nanstd(flux_masked.flatten()), np.std(norm_good_flux),
          np.nanstd(flux_masked.flatten()))
    """
    
    ncopy_plot= 5
    cmg.plot_forward_model_onespec(noisy_flux_lores_ncopy, rand_noise_ncopy, rand_flux_lores, master_mask_chunk, \
                                   good_vel_data, norm_good_flux, ncopy_plot)
    plt.show()
    """

# START HERE: generate cf_corr for each QSO
def compare_cf_data_fm(qso_fitsfile, qso_z, iqso, std_corr, vel_lores, flux_lores, ncopy, seed=None, plot=False):

    _, _, _, vel_mid, xi_data, xi_data_cgm_mask, _, _, cgm_masking_gpm = ccf.onespec(qso_fitsfile, qso_z=qso_z)

    vel_data, master_mask, std, fluxfit, outmask, norm_good_std, norm_std, norm_good_flux, good_vel_data, good_ivar, _, _ = init_onespec(iqso)

    # mock data
    flux_lores_rand, master_mask_chunk, norm_std_chunk, flux_lores_rand_noise = \
        cmg.forward_model_onespec_chunk(vel_data, master_mask, norm_std, vel_lores, flux_lores, ncopy, std_corr=std_corr, seed=seed)

    vm, xi_mock, _ = cmg.compute_cf_onespec_chunk(vel_lores, flux_lores_rand_noise, 10, 2000, 100, mask=master_mask_chunk)

    ncopy, nskew, npix = np.shape(flux_lores_rand_noise)
    cgm_gpm = cmg.chunk_gpm_onespec(cgm_masking_gpm, nskew, npix)
    print(cgm_gpm.shape, master_mask_chunk.shape)
    _, xi_mock_cgm_mask, _ = cmg.compute_cf_onespec_chunk(vel_lores, flux_lores_rand_noise, 10, 2000, 100, mask=master_mask_chunk*cgm_gpm)

    if plot:
        plt.figure(figsize=(17, 5.5))
        plt.subplot(121)
        for i in range(50):
            # note: plotting the median here, since the mean will be skewed by the last skewer with most pixels being nan
            plt.plot(vel_mid, np.mean(xi_mock[i+2], axis=0), lw=0.5, alpha=0.3) # plotting each ncopy (averaged over nskew)
        plt.plot(vel_mid, xi_data[0], label='data, unmasked')
        plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
        plt.ylabel(r'$\xi(\Delta v)$', fontsize=18)
        vel_doublet = 768.469
        plt.axvline(vel_doublet, color='red', linestyle=':', linewidth=1.5)
        plt.legend()
        plt.tight_layout()

        plt.subplot(122)
        for i in range(50):
            plt.plot(vel_mid, np.mean(xi_mock_cgm_mask[i + 2], axis=0), lw=0.5, alpha=0.3)
        plt.plot(vel_mid, xi_data_cgm_mask[0], label='data, CGM masked')
        plt.xlabel(r'$\Delta v$ [km/s]', fontsize=18)
        plt.ylabel(r'$\xi(\Delta v)$', fontsize=18)
        vel_doublet = 768.469
        plt.axvline(vel_doublet, color='red', linestyle=':', linewidth=1.5)
        plt.legend()
        plt.tight_layout()

        plt.show()

    return vel_mid, xi_data, xi_data_cgm_mask, vm, xi_mock, xi_mock_cgm_mask



