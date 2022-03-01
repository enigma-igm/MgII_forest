'''
Functions here are:
    - obswave_to_vel_2
    - extract_data
    - continuum_normalize
    - custom_mask_J0313
    - custom_mask_J1342
    - custom_mask_J0038
    - extxract_and_norm
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
import scipy.interpolate as interpolate
from astropy import constants as const
from astropy.table import Table
from enigma.reion_forest import utils
import compute_cf_data as ccf

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

    return wave_arr, flux_arr, ivar_arr, mask_arr, std_arr

def continuum_normalize(wave_arr, flux_arr, ivar_arr, mask_arr, std_arr, nbkpt, plot=False):

    # continuum normalize using breakpoint spline method in Pypeit
    # note: not including the custom masks yet
    (sset, outmask) = iterfit(wave_arr, flux_arr, invvar=ivar_arr, inmask=mask_arr, upper=3, lower=3, x2=None,
                              maxiter=10, nord=4, bkpt=None, fullbkpt=None, kwargs_bspline = {'everyn': nbkpt})

    # flux_fit is the returned continuum
    # note: flux_fit is already masked, so flux_fit.shape == outmask.shape
    _, flux_fit = sset.fit(wave_arr[outmask], flux_arr[outmask], ivar_arr[outmask])

    if plot:
        plt.plot(wave_arr, flux_arr, c='b', alpha=0.3, drawstyle='steps-mid')
        plt.plot(wave_arr[outmask], flux_arr[outmask], c='b', drawstyle='steps-mid')
        plt.plot(wave_arr[outmask], flux_fit, c='r', drawstyle='steps-mid')
        plt.plot(wave_arr[outmask], std_arr[outmask], c='k', drawstyle='steps-mid')
        plt.ylim([0, 0.50])
        plt.show()

    return flux_fit, outmask, sset

###### by-eye strong absorbers masks for each QSO, before continuum-normalizing ######
def custom_mask_J0313(plot=False):
    #fitsfile = '/Users/suksientie/Research/data_redux/mgii_stack_fits/J0313-1806_stacked_coadd_tellcorr.fits'
    fitsfile =  '/Users/suksientie/Research/data_redux/wavegrid_vel/J0313-1806/vel123_coadd_tellcorr.fits'
    wave, flux, ivar, mask, std = extract_data(fitsfile)

    mask_wave1 = [19815, 19825]
    mask_wave2 = [19865, 19870]
    mask_wave3 = [23303, 23325]
    mask_wave4 = [23370, 23387]

    all_mask_wave = [mask_wave1, mask_wave2, mask_wave3, mask_wave4]
    out_gpm = []
    for mask_wave_i in all_mask_wave:
        a = mask_wave_i[0] < wave
        b = wave < mask_wave_i[1]
        gpm = np.invert(a * b)
        out_gpm.append(gpm)
        mask *= gpm

    if plot:
        plt.plot(wave, flux, c='b', alpha=0.3, drawstyle='steps-mid')
        plt.plot(wave, std, c='k', alpha=0.3, drawstyle='steps-mid')
        plt.plot(wave[mask], flux[mask], c='b', drawstyle='steps-mid')
        plt.plot(wave[mask], std[mask], c='k', drawstyle='steps-mid')
        plt.show()

    return wave, flux, ivar, mask, std, out_gpm

def custom_mask_J1342(plot=False):
    #fitsfile = '/Users/suksientie/Research/data_redux/mgii_stack_fits/J1342+0928_stacked_coadd_tellcorr.fits'
    fitsfile = '/Users/suksientie/Research/data_redux/wavegrid_vel/J1342+0928/vel123_coadd_tellcorr.fits'
    wave, flux, ivar, mask, std = extract_data(fitsfile)

    # visually-identified strong absorbers
    mask_wave1 = [21920, 21940]
    mask_wave2 = [21972, 22000]
    mask_wave3 = [20320, 20335]
    mask_wave4 = [20375, 20400]

    all_mask_wave = [mask_wave1, mask_wave2, mask_wave3, mask_wave4]
    out_gpm = []
    for mask_wave_i in all_mask_wave:
        a = mask_wave_i[0] < wave
        b = wave < mask_wave_i[1]
        gpm = np.invert(a * b)
        out_gpm.append(gpm)
        mask *= gpm

    if plot:
        plt.plot(wave, flux, c='b', alpha=0.3, drawstyle='steps-mid')
        plt.plot(wave, std, c='k', alpha=0.3, drawstyle='steps-mid')
        plt.plot(wave[mask], flux[mask], c='b', drawstyle='steps-mid')
        plt.plot(wave[mask], std[mask], c='k', drawstyle='steps-mid')
        plt.show()

    return wave, flux, ivar, mask, std, out_gpm

def custom_mask_J0038(plot=False):
    #fitsfile = '/Users/suksientie/Research/data_redux/2010_done/Redux/J0038-1527_201024_done/J0038-1527_coadd_tellcorr.fits'
    fitsfile = '/Users/suksientie/Research/data_redux/wavegrid_vel/J0038-1527/vel1_tellcorr.fits'
    wave, flux, ivar, mask, std = extract_data(fitsfile)

    # visually-identified strong absorbers
    mask_wave1 = [19777, 19796]
    mask_wave2 = [19828, 19855]

    all_mask_wave = [mask_wave1, mask_wave2]
    out_gpm = []
    for mask_wave_i in all_mask_wave:
        a = mask_wave_i[0] < wave
        b = wave < mask_wave_i[1]
        gpm = np.invert(a * b)
        out_gpm.append(gpm)
        mask *= gpm

    if plot:
        plt.plot(wave, flux, c='b', alpha=0.3, drawstyle='steps-mid')
        plt.plot(wave, std, c='k', alpha=0.3, drawstyle='steps-mid')
        plt.plot(wave[mask], flux[mask], c='b', drawstyle='steps-mid')
        plt.plot(wave[mask], std[mask], c='k', drawstyle='steps-mid')
        plt.show()

    return wave, flux, ivar, mask, std, out_gpm

def extract_and_norm(fitsfile, everyn_bkpt):
    # combining extract_data() and continuum_normalize() including custom masking

    wave, flux, ivar, mask, std = extract_data(fitsfile)
    qso_name = fitsfile.split('/')[-1].split('_')[0]

    if qso_name == 'J0313-1806':
        wave, flux, ivar, mask, std, out_gpm = custom_mask_J0313()

    elif qso_name == 'J1342+0928':
        wave, flux, ivar, mask, std, out_gpm = custom_mask_J1342()

    elif qso_name == 'J0252-0503':
        pass

    elif qso_name == 'J0038-1527':
        wave, flux, ivar, mask, std, out_gpm = custom_mask_J0038()

    fluxfit, outmask, sset = continuum_normalize(wave, flux, ivar, mask, std, everyn_bkpt)
    # fluxfit = fitted continuum from the input breakpoint
    # outmask = final mask including the original data mask and mask returned during continuum fitting
    # sset = object returned from continuum fitting

    return wave, flux, ivar, mask, std, fluxfit, outmask, sset

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

    vmin_corr, vmax_corr, dv_corr = 10, 2000, 100

    mean_flux_nless = np.mean(flux_lores)
    delta_f_nless = (flux_lores - mean_flux_nless) / mean_flux_nless
    (vel_mid, xi_nless, npix, xi_nless_zero_lag) = utils.compute_xi(delta_f_nless, vel_lores, vmin_corr, vmax_corr, dv_corr)
    xi_mean = np.mean(xi_nless, axis=0)

    return vel_lores, flux_lores, vel_mid, xi_mean

def init_onespec(iqso):
    # initialize all needed data from one qso for testing compute_model_grid.py
    datapath = '/Users/suksientie/Research/data_redux/'
    # datapath = '/mnt/quasar/sstie/MgII_forest/z75/'

    fitsfile_list = [datapath + 'wavegrid_vel/J0313-1806/vel123_coadd_tellcorr.fits', \
                     datapath + 'wavegrid_vel/J1342+0928/vel123_coadd_tellcorr.fits', \
                     datapath + 'wavegrid_vel/J0252-0503/vel12_coadd_tellcorr.fits', \
                     datapath + 'wavegrid_vel/J0038-1527/vel1_tellcorr_pad.fits']

    qso_namelist = ['J0313-1806', 'J1342+0928', 'J0252-0503', 'J0038-1527']
    qso_zlist = [7.6, 7.54, 7.0, 7.0]
    everyn_break_list = [20, 20, 20, 20]

    fitsfile = fitsfile_list[iqso]
    wave, flux, ivar, mask, std, fluxfit, outmask, sset = extract_and_norm(fitsfile, everyn_break_list[iqso])
    vel_data = obswave_to_vel_2(wave)

    redshift_mask = wave <= (2800 * (1 + qso_zlist[iqso]))  # removing spectral region beyond qso redshift
    master_mask = redshift_mask * outmask # final ultimate mask

    # masked arrays
    good_wave = wave[master_mask]
    good_flux = flux[master_mask]
    good_ivar = ivar[master_mask]
    good_std = std[master_mask]
    fluxfit_redshift = fluxfit[wave[outmask] <= (2800 * (1 + qso_zlist[iqso]))]
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

import compute_model_grid as cmg
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
