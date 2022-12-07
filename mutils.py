'''
Functions here are:
    - obswave_to_vel_2
    - extract_data
    - continuum_normalize
    - custom_mask_J0313
    - custom_mask_J1342
    - custom_mask_J0038
    - extract_and_norm
    - qso_exclude_proximity_zone
    - qso_redshift_and_pz_mask
    - final_qso_pathlength
    - init_onespec
    - plot_onespec_pdf
    - plot_allspec_pdf
    - init_skewers_compute_model_grids
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
from astropy.cosmology import FlatLambdaCDM
from pypeit.core.arc import detect_peaks

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

    vel = np.cumsum(dv) #  first pixel is dv; vel = [dv, 2*dv, 3*dv, ....]
    """
    dv_all = np.arange(len(wave_arr) + 1) * dv[0]
    vlo = dv_all[0:-1]
    vhi  = dv_all[1:]
    v_mid = (vlo + vhi)/2
    return v_mid
    """
    return vel

def extract_data(fitsfile):
    # 'fitsfile' = name of fitsfile containing Pypeit 1d spectrum

    data = fits.open(fitsfile)[1].data
    #wave_arr = data['wave_grid_mid'].astype('float64') # midpoint values of wavelength bin
    wave_arr = data['wave'].astype('float64')
    flux_arr = data['flux'].astype('float64')
    ivar_arr = data['ivar'].astype('float64')
    mask_arr = data['mask'].astype('bool')
    std_arr = np.sqrt(putils.inverse(ivar_arr))

    try:
        tell_arr = data['telluric'].astype('float64')
    except KeyError:
        tell_arr = None

    wave_arr = wave_arr.squeeze()
    flux_arr = flux_arr.squeeze()
    ivar_arr = ivar_arr.squeeze()
    mask_arr = mask_arr.squeeze()
    std_arr = std_arr.squeeze()
    if tell_arr is not None:
        tell_arr = tell_arr.squeeze()

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
        print("np.sum(was_fit_and_masked)", np.sum(was_fit_and_masked))
        ax.plot(wave_arr[mask_arr], flux_arr[mask_arr], color='k', marker='o', markersize=0.4, mfc='k', fillstyle='full',
                linestyle='-', label='Pixels that were fit')
        ax.plot(wave_arr[was_fit_and_masked], flux_arr[was_fit_and_masked], color='red', marker='x', markersize=5.0, mfc='red',
                fillstyle='full', linestyle='None', label='Pixels masked by fit')
        ax.plot(wave_arr, cont_fit, color='cornflowerblue', label='B-spline fit')
        ax.plot(sset.breakpoints[goodbk], yfit_bkpt, color='lawngreen', marker='o', markersize=4.0, mfc='lawngreen',
                fillstyle='full', linestyle='None', label='Good B-spline breakpoints')
        #ax.set_ylim((0.99 * cont_fit.min(), 1.01 * cont_fit.max()))
        ax.set_ylim((0.8 * cont_fit.min(), 1.2 * cont_fit.max()))
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
    mask_wave2 = [19863, 19875] #[19865, 19870]
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
    mask_wave2 = [21974, 22000] #[21972, 22000] # updated with rebinned spec
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
    mask_wave1 = [19777, 19804] # [19777, 19796]
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

def custom_mask_J0410(fitsfile, plot=False):

    wave, flux, ivar, mask, std, tell = extract_data(fitsfile)
    # visually-identified strong absorbers
    mask_wave1 = [20638, 20653]
    mask_wave2 = [20688, 20705]
    mask_wave3 = [20742, 20755]

    all_mask_wave = [mask_wave1, mask_wave2, mask_wave3]
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

    elif qso_name == 'J0410-0139':
        print('using custom mask for %s' % qso_name)
        strong_abs_gpm = custom_mask_J0410(fitsfile)

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

    mg2forest_wavemin, mg2forest_wavemax = 19500, 24000
    redshift_mask = (wave <= (2800 * (1 + qso_z))) * (wave >= mg2forest_wavemin)
    #redshift_mask = wave <= (2800 * (1 + qso_z))  # removing spectral region beyond qso redshift
    obs_wave_max = (2800 - exclude_rest) * (1 + qso_z)
    pz_mask = wave < obs_wave_max

    return redshift_mask, pz_mask, obs_wave_max

def telluric_mask(wave):
    #wave_bad_start = [21628, 21778, 22153, 22307, 22745, 22799, 23156, 23394, 23553]
    #wave_bad_end = [21640, 21788, 22169, 22326, 22764, 22815, 23185, 23409, 23566]

    #wave_bad_start = [19791, 19864, 20000, 21628, 22153, 22307, 22745, 22799, 23156, 23394, 23553]
    #wave_bad_end = [19808, 19874, 20060, 21640, 22169, 22326, 22764, 22815, 23185, 23409, 23566]

    wave_bad_start = [20000] #, 20556, 21489]
    wave_bad_end = [20060]#, 20571, 21512]

    telluric_gpm = np.ones(wave.shape, dtype=bool)
    for i in range(len(wave_bad_start)):
        bpm_a = wave_bad_start[i] < wave
        bpm_b = wave < wave_bad_end[i]
        bpm = bpm_a * bpm_b
        telluric_gpm *= np.invert(bpm)

    return telluric_gpm

def final_qso_pathlength(fitsfile, qso_name, qso_z, exclude_rest=1216-1185, cgm_gpm=None):

    wave, flux, ivar, mask, std, tell, fluxfit, strong_abs_gpm = extract_and_norm(fitsfile, 20, qso_name)
    redshift_mask, pz_mask, obs_wave_max = qso_redshift_and_pz_mask(wave, qso_z, exclude_rest)

    # only using data and PZ masks
    if type(cgm_gpm) == type(None):
        master_mask = mask * redshift_mask * pz_mask
        good_wave = wave[master_mask]

        # (4/25/2022)
        # In[96]: np.median(gz_all), np.min(gz_all), np.max(gz_all)
        # Out[96]: (6.572035934151384, 5.983190982278802, 7.545849071317116)

    # including CGM masks (negligible effect because not many pixels are masked by CGM maks)
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
    print("percent pixels excluded", 100*np.sum(np.invert(master_mask))/len(wave))

    return good_z

def init_onespec(iqso, redshift_bin, datapath='/Users/suksientie/Research/MgII_forest/rebinned_spectra/'):

    fitsfile_list = [datapath + 'J0411-0907_dv40_coadd_tellcorr.fits', \
                     datapath + 'J0319-1008_dv40_coadd_tellcorr.fits', \
                     datapath + 'J0410-0139_dv40_coadd_tellcorr.fits', \
                     datapath + 'J0038-0653_dv40_coadd_tellcorr.fits', \
                     datapath + 'J0313-1806_dv40_coadd_tellcorr.fits', \
                     datapath + 'J0038-1527_dv40_coadd_tellcorr.fits', \
                     datapath + 'J0252-0503_dv40_coadd_tellcorr.fits', \
                     datapath + 'J1342+0928_dv40_coadd_tellcorr.fits']

    qso_namelist = ['J0411-0907', 'J0319-1008', 'J0410-0139', 'J0038-0653', 'J0313-1806', 'J0038-1527', 'J0252-0503', 'J1342+0928']
    qso_zlist = [6.826, 6.8275, 7.0, 7.1, 7.642, 7.034, 7.001, 7.541]
    qso_median_snr = [9.29, 5.50, 3.95, 8.60, 11.42, 14.28, 13.07, 8.72] # from Table 1 in current draft (12/6/2022)
    everyn_break_list = (np.ones(len(qso_namelist)) * 20).astype('int')
    exclude_restwave = 1216 - 1185
    median_z = 6.500 # see allqso_pathlength_snr.py

    fitsfile = fitsfile_list[iqso]
    wave, flux, ivar, mask, std, tell, fluxfit, strong_abs_gpm = extract_and_norm(fitsfile, everyn_break_list[iqso], qso_namelist[iqso])
    redshift_mask, pz_mask, obs_wave_max = qso_redshift_and_pz_mask(wave, qso_zlist[iqso], exclude_restwave)
    telluric_gpm = telluric_mask(wave)

    if redshift_bin == 'low':
        zbin_mask = wave < (2800 * (1 + median_z))

    elif redshift_bin == 'high':
        zbin_mask = wave >= (2800 * (1 + median_z))

    elif redshift_bin == 'all':
        zbin_mask = np.ones_like(wave, dtype=bool)

    # master mask for measuring 2PCF
    master_mask = mask * redshift_mask * pz_mask * zbin_mask * telluric_gpm

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
    all_masks_out = strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, telluric_gpm, master_mask

    return raw_data_out, masked_data_out, all_masks_out

def init_onespec_old(iqso, redshift_bin, datapath='/Users/suksientie/Research/data_redux/'):
    # initialize all needed quantities for one qso, included all masks, for all subsequent analyses
    # important to make sure fits files and global variables are up to dates

    # datapath = '/Users/suksientie/Research/data_redux/' # path on laptop
    # datapath = '/mnt/quasar/sstie/MgII_forest/z75/'  # path on IGM server

    # not using the padded fits for J0038-1527 (as of 4/25/2022)
    fitsfile_list = [datapath + 'wavegrid_vel/J0313-1806/vel1234_coadd_tellcorr.fits', \
                     datapath + 'wavegrid_vel/J1342+0928/vel123_coadd_tellcorr.fits', \
                     datapath + 'wavegrid_vel/J0252-0503/vel12_coadd_tellcorr.fits', \
                     datapath + 'wavegrid_vel/J0038-1527/vel1_tellcorr.fits', \
                     datapath + 'wavegrid_vel/J0038-0653/vel1_tellcorr.fits']

    qso_namelist = ['J0313-1806', 'J1342+0928', 'J0252-0503', 'J0038-1527', 'J0038-0653']
    qso_zlist = [7.642, 7.541, 7.001, 7.034, 7.1]
    everyn_break_list = [20, 20, 20, 20, 20]
    exclude_restwave = 1216 - 1185
    median_z = 6.554 # 6.573  # median pixel redshift of measurement (excluding proximity zones)

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
    #master_mask = mask * strong_abs_gpm * redshift_mask * pz_mask * zbin_mask
    master_mask = mask * redshift_mask * pz_mask * zbin_mask

    # masked arrays
    good_wave = wave[master_mask]
    good_flux = flux[master_mask]
    good_ivar = ivar[master_mask]
    good_std = std[master_mask]
    good_vel_data = obswave_to_vel_2(good_wave)
    vel_data = obswave_to_vel_2(wave)

    norm_good_flux = good_flux / fluxfit[master_mask]
    norm_good_std = good_std / fluxfit[master_mask]
    #norm_snr = flux/std # numbers used in group meeting slide
    #print(np.median(norm_snr))

    raw_data_out = wave, flux, ivar, mask, std, tell, fluxfit
    masked_data_out = good_wave, good_flux, good_ivar, good_std, good_vel_data, norm_good_flux, norm_good_std
    all_masks_out = strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, master_mask

    #return vel_data, master_mask, std, fluxfit, outmask, norm_good_std, norm_std, norm_good_flux, good_vel_data, good_ivar, norm_flux, ivar
    return raw_data_out, masked_data_out, all_masks_out

######################################################
# determining the correction factors for each QSO
def plot_onespec_pdf(iqso, seed=None, title=None):

    raw_out, masked_out, masks_out = init_onespec(iqso, 'all')
    wave, flux, ivar, mask, std, tell, fluxfit = raw_out
    strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, telluric_gpm, master_mask = masks_out

    norm_flux = flux/fluxfit
    norm_std = std/fluxfit
    norm_flux = norm_flux[mask * redshift_mask * pz_mask * zbin_mask]
    norm_std = norm_std[mask * redshift_mask * pz_mask * zbin_mask]

    chi = (1 - norm_flux) / norm_std
    corr_factor = mad_std(chi)
    print(corr_factor)
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

def plot_allspec_pdf(redshift_bin, seed_list=[None, None, None, None, None]):

    qso_namelist = ['J0313-1806', 'J1342+0928', 'J0252-0503', 'J0038-1527', 'J0038-0653']
    plt.figure(figsize=(12, 8))

    for iqso in range(len(qso_namelist)):
        seed = seed_list[iqso]
        #redshift_bin = 'high'
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

        plt.subplot(3, 2,iqso+1)
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

######################################################
def init_skewers_compute_model_grid(filename, dv_corr, logZ):
    # initialize Nyx skewers for testing compute_model_grid_new.py

    #filename = 'ran_skewers_z75_OVT_xHI_0.50_tau.fits'
    params = Table.read(filename, hdu=1)
    skewers = Table.read(filename, hdu=2)

    fwhm = 120 #90 # 83
    sampling = 3
    #logZ = -3.50

    vel_lores, (flux_lores, flux_lores_igm, flux_lores_cgm, _, _), \
    vel_hires, (flux_hires, flux_hires_igm, flux_hires_cgm, _, _), \
    (oden, v_los, T, xHI), cgm_tuple = utils.create_mgii_forest(params, skewers, logZ, fwhm, sampling=sampling)

    vmin_corr, vmax_corr = 10, 3500
    mean_flux_nless = np.mean(flux_lores)
    delta_f_nless = (flux_lores - mean_flux_nless) / mean_flux_nless

    return delta_f_nless, vel_lores, flux_lores

    (vel_mid, xi_nless, npix, xi_nless_zero_lag) = utils.compute_xi(delta_f_nless, vel_lores, vmin_corr, vmax_corr, dv_corr)
    xi_mean = np.mean(xi_nless, axis=0)

    return vel_lores, flux_lores, vel_mid, xi_mean, npix, xi_nless

def lya_spikes(fitsfile, zlow, zhigh):
    # fitsfile = '/Users/suksientie/Research/highz_absorbers/J0313m1806_fire_mosfire_nires_tellcorr_contfit.fits'
    # lyb @ 1026
    data = fits.open(fitsfile)[1].data
    wave_arr = data['wave_grid_mid'].astype('float64')  # midpoint values of wavelength bin
    flux_arr = data['flux'].astype('float64')
    ivar_arr = data['ivar'].astype('float64')
    mask_arr = data['mask'].astype('bool')
    std_arr = np.sqrt(putils.inverse(ivar_arr))

    low = (1216 * (1 + zlow)) < wave_arr
    high = wave_arr < 1216 * (1 + zhigh)
    lowhigh = low * high

    print(np.mean(flux_arr[lowhigh]))

    plt.plot(wave_arr[lowhigh], flux_arr[lowhigh], 'k', drawstyle='steps-mid')
    plt.plot(wave_arr[lowhigh], std_arr[lowhigh], 'r', drawstyle='steps-mid')
    #plt.plot(wave_arr[lowhigh], 5*std_arr[lowhigh], 'b', alpha=0.5, drawstyle='steps-mid')
    plt.ylabel('Normalized flux')
    plt.xlabel('Observed wavelength')
    plt.grid()
    plt.show()

def abspath(z1, z2, cosmo=None):
    """
    calculate pathlength between z1 and z2.
    """
    if cosmo is None:
        f = 'ran_skewers_z75_OVT_xHI_0.50_tau.fits'
        par = Table.read(f, hdu=1)
        lit_h, m0, b0, l0 = par['lit_h'][0], par['Om0'][0], par['Ob0'][0], par['Ode0'][0]
        cosmo = FlatLambdaCDM(H0=lit_h*100, Om0=m0, Ob0=b0, Tcmb0=2.725)

    return cosmo.absorption_distance(z1)-cosmo.absorption_distance(z2)

def reweight_factors(nqso, redshift_bin):

    dx_all = []

    for iqso in range(nqso):
        raw_data_out, _, all_masks_out = init_onespec(iqso, redshift_bin)
        wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
        strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, telluric_gpm, master_mask = all_masks_out

        good_zpix = wave[master_mask] / 2800 - 1
        zlow, zhigh = good_zpix.min(), good_zpix.max()
        dx = abspath(zhigh, zlow)
        dx_all.append(dx)

    weight = dx_all / np.sum(dx_all)
    return weight

def mosfire_nires_fwhm():
    c_kms = const.c.to('km/s').value

    nires_mean_res = 2700
    nires_sampling = 2.7
    nires_fwhm = c_kms/nires_mean_res
    nires_fwhm = np.round(nires_fwhm, 2)

    mosfire_Kband_res = 3610
    mosfire_sampling = 2.78
    mosfire_fwhm = c_kms/mosfire_Kband_res
    mosfire_fwhm = np.round(mosfire_fwhm, 2)

    return nires_fwhm, mosfire_fwhm

from pypeit.core import coadd
from pypeit.core.wavecal import wvutils
def coarse_vel_grid():

    wave_method = 'velocity'
    dv = 40
    wave_grid_min = 19500
    fitsfile = '/Users/suksientie/Research/data_redux/silvia/J1917+5003_NIRES_coadd_tellcorr.fits'
    data = fits.open(fitsfile)[1].data
    wave_coadd = data['wave'].astype('float64')
    new_wavegrid, new_wavegrid_mid, dsamp = wvutils.get_wave_grid(wave_coadd, masks=None, wave_method=wave_method,
                                                                  dv=dv, wave_grid_min=wave_grid_min)

    vel_coadd = obswave_to_vel_2(wave_coadd)
    new_velgrid = obswave_to_vel_2(new_wavegrid)

    return vel_coadd, new_velgrid