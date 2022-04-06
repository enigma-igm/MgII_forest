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
    tell_arr = data['telluric'].astype('float64')

    return wave_arr, flux_arr, ivar_arr, mask_arr, std_arr, tell_arr

def continuum_normalize(wave_arr, flux_arr, ivar_arr, mask_arr, std_arr, nbkpt, plot=False):

    # continuum normalize using breakpoint spline method in Pypeit
    # note: not including the custom masks yet

    # nbkpt: NOT the total number of breakpoints, but instead it's placing a breakpoint at every n-th index,
    # since we're using the 'everyn' argument below. I.e. if nbkpt=20, it places a breakpoint at every 20-th element.
    (sset, outmask) = iterfit(wave_arr, flux_arr, invvar=ivar_arr, inmask=mask_arr, upper=3, lower=3, x2=None,
                              maxiter=10, nord=4, bkpt=None, fullbkpt=None, kwargs_bspline = {'everyn': nbkpt})

    # flux_fit is the returned continuum
    # note: flux_fit is already masked, so flux_fit.shape == outmask.shape
    _, flux_fit = sset.fit(wave_arr[outmask], flux_arr[outmask], ivar_arr[outmask])

    if plot:
        plt.plot(wave_arr, flux_arr, c='b', alpha=0.3, drawstyle='steps-mid', label='flux')
        plt.plot(wave_arr[outmask], flux_arr[outmask], c='b', drawstyle='steps-mid', label='flux (masked)')
        plt.plot(wave_arr[outmask], flux_fit, c='r', drawstyle='steps-mid', label='continuum fit')
        plt.plot(wave_arr[outmask], std_arr[outmask], c='k', drawstyle='steps-mid', label='sigma (masked)')
        plt.legend()
        #plt.ylim([0, 0.50])
        plt.show()

    return flux_fit, outmask, sset

###### by-eye strong absorbers masks for each QSO, before continuum-normalizing ######
def custom_mask_J0313(plot=False):
    #fitsfile = '/Users/suksientie/Research/data_redux/mgii_stack_fits/J0313-1806_stacked_coadd_tellcorr.fits'
    fitsfile =  '/Users/suksientie/Research/data_redux/wavegrid_vel/J0313-1806/vel1234_coadd_tellcorr.fits'
    wave, flux, ivar, mask, std, tell = extract_data(fitsfile)

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
    wave, flux, ivar, mask, std, tell = extract_data(fitsfile)

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
    fitsfile = '/Users/suksientie/Research/data_redux/wavegrid_vel/J0038-1527/vel1_tellcorr_pad.fits'  #vel1_tellcorr.fits'

    wave, flux, ivar, mask, std, tell = extract_data(fitsfile)
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

def extract_and_norm(fitsfile, everyn_bkpt, qso_name):
    # combining extract_data() and continuum_normalize() including custom masking

    wave, flux, ivar, mask, std, tell = extract_data(fitsfile)
    #qso_name = fitsfile.split('/')[-1].split('_')[0]

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

    return wave, flux, ivar, mask, std, fluxfit, outmask, sset, tell

def all_absorber_redshift(fitsfile_list):

    qso_namelist = ['J0313-1806', 'J1342+0928', 'J0252-0503', 'J0038-1527']
    qso_zlist = [7.642, 7.541, 7.001, 7.034]
    everyn_break_list = [20, 20, 20, 20]
    all_z = []

    for iqso, fitsfile in enumerate(fitsfile_list):
        wave, flux, ivar, mask, std, fluxfit, outmask, sset, tell = extract_and_norm(fitsfile, everyn_break_list[iqso], qso_namelist[iqso])
        redshift_mask = wave[outmask] <= (2800 * (1 + qso_zlist[iqso]))  # removing spectral region beyond qso redshift
        good_wave = wave[outmask][redshift_mask]
        z_abs = good_wave/2800 - 1
        all_z.extend(z_abs)

    print(np.mean(all_z), np.median(all_z))
    return all_z

def qso_pathlength(fitsfile, qso_name, qso_z, exclude_rest=1216-1185):
    # 3/29/2022: forgot that I wrote the same code above (all_absorber_redshift)
    wave, flux, ivar, mask, std, fluxfit, outmask, sset, tell = extract_and_norm(fitsfile, 20, qso_name)

    redshift_mask = wave <= (2800 * (1 + qso_z))  # removing spectral region beyond qso redshift
    obs_wave_max = (2800 - exclude_rest) * (1 + qso_z)
    proximity_zone_mask = wave < obs_wave_max

    master_mask = outmask * redshift_mask * proximity_zone_mask

    good_wave = wave[master_mask]
    good_z = good_wave/2800 - 1
    dz_pathlength = good_z.max() - good_z.min()
    median_z = np.median(good_z)
    print(qso_name, "median z=", median_z, "dz=", dz_pathlength)
    #print(good_z.min(), good_z.max())

    return good_z, outmask, redshift_mask, proximity_zone_mask

def qso_exclude_proximity_zone(fitsfile, qso_z, qso_name, exclude_rest=1216-1185, plot=False):
    # BAO lyaf: 1040 A < lambda_rest < 1200 A
    # Bosman+2021: lambda_rest < 1185 A
    # the default exclude_rest value uses Bosman cut off

    wave, flux, ivar, mask, std, fluxfit, outmask, sset, tell = extract_and_norm(fitsfile, 20, qso_name)

    redshift_mask = wave <= (2800 * (1 + qso_z))  # removing spectral region beyond qso redshift
    master_mask = redshift_mask * outmask

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

import pdb
def init_onespec(iqso, redshift_bin):
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
