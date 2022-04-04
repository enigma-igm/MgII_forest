# Use single cores (forcing it for numpy operations)
import os
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import sys
from astropy.table import Table, hstack, vstack
from IPython import embed
sys.path.append('/Users/suksientie/codes/enigma') # comment out this line if running on IGM cluster
from enigma.reion_forest import utils
from enigma.reion_forest.mgii_find import MgiiFinder
from multiprocessing import Pool
from tqdm import tqdm
import mutils
import pdb

###################### global variables ######################
datapath = '/Users/suksientie/Research/data_redux/'
#datapath = '/mnt/quasar/sstie/MgII_forest/z75/' # on IGM

fitsfile_list = [datapath + 'wavegrid_vel/J0313-1806/vel1234_coadd_tellcorr.fits', \
                 datapath + 'wavegrid_vel/J1342+0928/vel123_coadd_tellcorr.fits', \
                 datapath + 'wavegrid_vel/J0252-0503/vel12_coadd_tellcorr.fits', \
                 datapath + 'wavegrid_vel/J0038-1527/vel1_tellcorr_pad.fits']

qso_namelist = ['J0313-1806', 'J1342+0928', 'J0252-0503', 'J0038-1527']
qso_zlist = [7.642, 7.541, 7.001, 7.034] # precise redshifts from Yang+2021
everyn_break_list = [20, 20, 20, 20] # placing a breakpoint at every 20-th array element (more docs in mutils.continuum_normalize)
                                     # this results in dwave_breakpoint ~ 40 A --> dvel_breakpoint = 600 km/s
exclude_restwave = 1216 - 1185 # excluding proximity zones; see mutils.qso_exclude_proximity_zone
median_z = 6.574 # median redshift of measurement (excluding proximity zones)

########################## helper functions #############################
def imap_unordered_bar(func, args, nproc):
    """
    Display progress bar.
    """
    p = Pool(processes=nproc)
    res_list = []
    with tqdm(total = len(args)) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list

def read_model_grid(modelfile):

    hdu = fits.open(modelfile)
    param = Table(hdu[1].data)
    xi_mock_array = hdu[2].data
    xi_model_array = hdu[3].data
    covar_array = hdu[4].data
    icovar_array = hdu[5].data
    lndet_array = hdu[6].data

    return param, xi_mock_array, xi_model_array, covar_array, icovar_array, lndet_array

def rand_skews_to_match_data(vel_lores, vel_data, tot_nyx_skews, seed=None):
    # vel_lores = Nyx velocity grid (numpy array)
    # vel_data = data velocity grid (numpy array)
    # tot_nyx_skews = total number of Nyx skewers (int)
    # seed = random seed used to grab random Nyx skewers

    tot_vel_data = vel_data.max() - vel_data.min()
    tot_vel_sim = vel_lores.max() - vel_lores.min()
    nskew_to_match_data = int(np.ceil(tot_vel_data / tot_vel_sim))  # assuming tot_vel_data > tot_vel_sim (true here)

    if seed != None:
        rand = np.random.RandomState(seed)
    else:
        rand = np.random.RandomState()

    indx_flux_lores = np.arange(tot_nyx_skews)
    ranindx = rand.choice(indx_flux_lores, replace=False, size=nskew_to_match_data)  # grab a set of random skewers
    return ranindx, nskew_to_match_data

def reshape_data_array(data_arr, nskew_to_match_data, npix_sim_skew, data_arr_is_mask): # 10, 220

    pad_width = nskew_to_match_data * npix_sim_skew - len(data_arr)

    if data_arr_is_mask:
        print("input data array is Bool array, so padding with False")
        # this is the primary use: to reshape the data master into a shape consistent with the mock spectra array
        padded_data_arr = np.pad(data_arr.astype(float), (0, pad_width), mode='constant', constant_values=0)
        padded_data_arr = padded_data_arr.astype(bool)
        assert np.sum(padded_data_arr) == np.sum(data_arr)

    else:
        padded_data_arr = np.pad(data_arr, (0, pad_width), mode='constant', constant_values=np.nan)
        # either of this has to be True, where 2nd condition is to deal with Nan elements

        #assert np.array_equal(data_arr, padded_data_arr[0:len(data_arr)]) or np.nansum(padded_data_arr) == np.nansum(data_arr)
        #assert np.array_equal(data_arr, padded_data_arr[0:len(data_arr)], equal_nan=True) # equal_nan only in Numpy version >=1.19.0
        assert np.array_equal(data_arr, padded_data_arr[0:len(data_arr)]) or np.isclose(np.nansum(padded_data_arr),np.nansum(data_arr))

    new_data_arr = padded_data_arr.reshape(nskew_to_match_data, npix_sim_skew)
    return new_data_arr

def init_cgm_masking(fwhm, signif_thresh=4.0, signif_mask_dv=300.0, signif_mask_nsigma=8, one_minF_thresh = 0.3):
    # returns good pixel mask from cgm masking for all 4 qsos
    # 3/29/22: added proximity zone mask
    # redshift bin mask applied to the outputs later on
    gpm_allspec = []
    for iqso, fitsfile in enumerate(fitsfile_list):
        wave, flux, ivar, mask, std, fluxfit, outmask, sset, tell = mutils.extract_and_norm(fitsfile, everyn_break_list[iqso])

        redshift_mask = wave[outmask] <= (2800 * (1 + qso_zlist[iqso])) # removing spectral region beyond qso redshift
        obs_wave_max = (2800 - exclude_restwave) * (1 + qso_zlist[iqso])
        proximity_zone_mask = wave[outmask] < obs_wave_max

        #good_wave = wave[outmask][redshift_mask][proximity_zone_mask]
        #good_flux = flux[outmask][redshift_mask][proximity_zone_mask]
        #good_ivar = ivar[outmask][redshift_mask][proximity_zone_mask]
        #norm_good_flux = good_flux / fluxfit[redshift_mask]
        good_wave = wave[outmask][redshift_mask * proximity_zone_mask]
        good_flux = flux[outmask][redshift_mask * proximity_zone_mask]
        good_ivar = ivar[outmask][redshift_mask * proximity_zone_mask]
        norm_good_flux = good_flux / fluxfit[redshift_mask * proximity_zone_mask]

        vel_data = mutils.obswave_to_vel_2(good_wave)

        # reshaping to be compatible with MgiiFinder
        norm_good_flux = norm_good_flux.reshape((1, len(norm_good_flux)))
        good_ivar = good_ivar.reshape((1, len(good_ivar)))

        mgii_tot = MgiiFinder(vel_data, norm_good_flux, good_ivar, fwhm, signif_thresh,
                              signif_mask_nsigma=signif_mask_nsigma,
                              signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh)
        gpm_allspec.append(mgii_tot.fit_gpm)

    # 'gpm_allspec' is a list, where gpm for each spec has different length
    return gpm_allspec

########################## one data spectrum #############################
def sample_noise_onespec_chunk(vel_data, norm_std, vel_lores, flux_lores, ncopy, seed=None, std_corr=1.0):

    npix_sim_skew = len(vel_lores)
    tot_nyx_skews = len(flux_lores)
    ranindx, nskew_to_match_data = rand_skews_to_match_data(vel_lores, vel_data, tot_nyx_skews, seed=seed)

    if seed != None:
        rand = np.random.RandomState(seed)
    else:
        rand = np.random.RandomState()

    norm_std_chunk = reshape_data_array(norm_std, nskew_to_match_data, npix_sim_skew, data_arr_is_mask=False)

    rand_noise_ncopy = np.zeros((ncopy, nskew_to_match_data, npix_sim_skew))
    for icopy in range(ncopy):
        onecopy = []
        for iskew in range(nskew_to_match_data):
            onecopy.append(rand.normal(0, std_corr * np.array(norm_std_chunk[iskew])))
        rand_noise_ncopy[icopy] = np.array(onecopy)

    return ranindx, rand_noise_ncopy, nskew_to_match_data, npix_sim_skew

def forward_model_onespec_chunk(vel_data, norm_std, vel_lores, flux_lores, ncopy, seed=None, std_corr=1.0):
    # wrapper to sample_noise_onespec_chunk

    ranindx, rand_noise_ncopy, nskew_to_match_data, npix_sim_skew = sample_noise_onespec_chunk(vel_data, norm_std, vel_lores, flux_lores, ncopy, seed=seed, std_corr=std_corr)
    rand_flux_lores = flux_lores[ranindx]
    #rand_flux_lores_ncopy = np.tile(rand_flux_lores, (ncopy,1,1))
    noisy_flux_lores_ncopy = rand_flux_lores + rand_noise_ncopy # rand_flux_lores automatically broadcasted across the ncopy-axis

    return ranindx, rand_flux_lores, rand_noise_ncopy, noisy_flux_lores_ncopy, nskew_to_match_data, npix_sim_skew

import pdb
def plot_forward_model_onespec(noisy_flux_lores_ncopy, rand_noise_ncopy, rand_flux_lores, master_mask_chunk, good_vel_data, norm_good_flux, ncopy_plot, title=None):

    ncopy, nskew, npix = np.shape(noisy_flux_lores_ncopy)

    plt.figure(figsize=(12, 8))
    if title != None:
        plt.title(title, fontsize=18)

    ##### plot subset of mock spectra
    for i in range(ncopy_plot):
        flux_lores_comb = (noisy_flux_lores_ncopy[i])[master_mask_chunk] # this also flattens the 2d array
        plt.plot(good_vel_data, flux_lores_comb + (i + 1), alpha=0.5, drawstyle='steps-mid')

    plt.plot(good_vel_data, norm_good_flux, 'k', drawstyle='steps-mid')
    #plt.plot(good_vel_data, rand_flux_lores[master_mask_chunk], 'white', drawstyle='steps-mid')
    plt.ylabel('Flux (+ arbitrary offset)', fontsize=15)
    plt.xlabel('Velocity (km/s)', fontsize=15)
    plt.tight_layout()

    ##### plot flux PDF
    plt.figure(figsize=(10, 8))
    if title != None:
        plt.title(title, fontsize=18)
    plt.hist(norm_good_flux, bins=np.arange(0, 3, 0.02), histtype='step', label='data', density=True)

    # applying raw data mask to flux_lores_rand_noise, since we're comparing to "norm_good_flux"
    flux_masked = []
    noise_masked = []
    for icopy in range(ncopy):
        flux_masked.append(noisy_flux_lores_ncopy[icopy][master_mask_chunk])
        noise_masked.append(rand_noise_ncopy[icopy][master_mask_chunk])
    flux_masked = np.array(flux_masked)
    noise_masked = np.array(noise_masked)

    plt.hist(flux_masked.flatten(), bins=np.arange(0, 3, 0.02), histtype='step', label='sim (ncopy=%d), good pix' % ncopy, density=True)
    plt.xlabel('normalized flux', fontsize=15)
    plt.ylabel('PDF', fontsize=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    print("ratio", np.std(norm_good_flux) / np.nanstd(flux_masked.flatten()), np.std(norm_good_flux),
          np.nanstd(flux_masked.flatten()))

    ##### plot (1 - F) PDF
    plt.figure(figsize=(10, 8))
    if title != None:
        plt.title(title, fontsize=18)

    nbins, oneminf_min, oneminf_max = 101, 1e-5, 1.0  # gives d(oneminf) = 0.01
    flux_bins, flux_pdf_data = utils.pdf_calc(1.0 - norm_good_flux, oneminf_min, oneminf_max, nbins)
    _, flux_pdf_flux_lores = utils.pdf_calc(1 - rand_flux_lores.flatten(), oneminf_min, oneminf_max, nbins)
    _, flux_pdf_mock_all = utils.pdf_calc(1.0 - flux_masked.flatten(), oneminf_min, oneminf_max,
                                          nbins)  # all mock spectra
    _, flux_pdf_noise = utils.pdf_calc(noise_masked.flatten(), oneminf_min, oneminf_max, nbins)

    for i in range(ncopy_plot):
        flux_lores_comb = (noisy_flux_lores_ncopy[i])[master_mask_chunk]  # this also flattens the 2d array
        _, flux_pdf_mock = utils.pdf_calc(1.0 - flux_lores_comb, oneminf_min, oneminf_max, nbins)
        plt.plot(flux_bins, flux_pdf_mock, drawstyle='steps-mid', color='g', lw=0.5,
                 alpha=0.6)  # individual mock spectrum

    plt.plot(flux_bins, flux_pdf_data, drawstyle='steps-mid', alpha=1.0, lw=2, label='data')
    plt.plot(flux_bins, flux_pdf_mock_all, drawstyle='steps-mid', color='g', alpha=1.0, lw=2,
             label='sim (ncopy=%d), good pix' % ncopy)
    plt.plot(flux_bins, flux_pdf_noise, drawstyle='steps-mid', color='k', alpha=1.0, lw=2, label='pure noise', zorder=1)

    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('1$-$F')
    plt.ylabel('PDF')
    plt.tight_layout()

def compute_cf_onespec_chunk(vel_lores, noisy_flux_lores_ncopy, vmin_corr, vmax_corr, dv_corr, mask=None):
    # "mask" is any mask, not necessarily cgm mask; need to be of shape (nskew, npix), use the reshape_data_array() function above

    ncopy, nskew, npix = np.shape(noisy_flux_lores_ncopy)

    # compress ncopy and nskew dimensions together because compute_xi() only takes in 2D arrays
    # (tried keeping the original 3D array and looping over ncopy when compute_xi(), but loop process too slow, sth like 1.4 min per 100 copy)
    reshaped_flux = np.reshape(noisy_flux_lores_ncopy, (ncopy * nskew, npix))

    if type(mask) != type(None):
        mask_ncopy = np.tile(mask, (ncopy, 1, 1))
        mask_ncopy = np.reshape(mask_ncopy, (ncopy*nskew, npix)).astype(int) # reshaping the same way as above and recasting to int type so the next line can run
        reshaped_flux_masked = np.where(mask_ncopy, reshaped_flux, np.nan)
        mean_flux = np.nanmean(reshaped_flux_masked)
        mask_ncopy = mask_ncopy.astype(bool)

    else:
        mask_ncopy = None
        mean_flux = np.nanmean(reshaped_flux)

    delta_f = (reshaped_flux - mean_flux)/mean_flux
    (vel_mid, xi_mock, npix_xi, xi_mock_zero_lag) = utils.compute_xi(delta_f, vel_lores, vmin_corr, vmax_corr, dv_corr, gpm=mask_ncopy)

    # reshape xi_mock into the original input shape
    xi_mock = np.reshape(xi_mock, (ncopy, nskew, len(vel_mid)))

    return vel_mid, xi_mock, npix_xi

########################## mock dataset #############################
def mock_mean_covar(xi_mean, ncopy, vel_lores, flux_lores, vmin_corr, vmax_corr, dv_corr, redshift_bin, master_seed=None, cgm_gpm_allspec=None):

    master_rand = np.random.RandomState() if master_seed is None else np.random.RandomState(master_seed)
    qso_seed_list = master_rand.randint(0, 1000000000, 4)  # 4 for 4 qsos, hardwired for now

    # average correction from 4 realizations of 1000 copies
    if redshift_bin == 'all':
        corr_all = [0.687, 0.635, 0.617, 0.58] # [0.689, 0.640, 0.616, 0.583] # before PZ masking
    elif redshift_bin == 'low':
        corr_all = [0.66, 0.59, 0.62, 0.57]
    elif redshift_bin == 'high':
        corr_all = [0.70, 0.69, 0.57, 0.59]

    xi_mock_qso_all = []

    for iqso, fitsfile in enumerate(fitsfile_list):
        # initialize all qso data
        wave, flux, ivar, mask, std, fluxfit, outmask, sset, tell = mutils.extract_and_norm(fitsfile, everyn_break_list[iqso])
        vel_data = mutils.obswave_to_vel_2(wave)

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

        master_mask = redshift_mask * outmask * proximity_zone_mask * zbin_mask # redshift_mask * outmask

        good_wave = wave[master_mask]
        good_vel_data = mutils.obswave_to_vel_2(good_wave)

        fluxfit_custom_mask = (wave[outmask] <= (2800 * (1 + qso_zlist[iqso]))) * (wave[outmask] < obs_wave_max) * zbin_mask_fluxfit
        # fluxfit_redshift = fluxfit[wave[outmask] <= (2800 * (1 + qso_zlist[iqso]))]
        fluxfit_redshift = fluxfit[fluxfit_custom_mask]
        norm_good_flux = flux[master_mask] / fluxfit_redshift
        norm_good_std = std[master_mask] / fluxfit_redshift

        fluxfit_new = mutils.pad_fluxfit(outmask, fluxfit)  # in order to make 'fluxfit' (masked) same shape as raw data
        norm_std = std / fluxfit_new

        # generate mock data spectrum
        _, _, rand_noise_ncopy, noisy_flux_lores_ncopy, nskew_to_match_data, npix_sim_skew = forward_model_onespec_chunk(vel_data, norm_std, vel_lores, flux_lores, ncopy, seed=qso_seed_list[iqso], std_corr=corr_all[iqso])

        # deal with CGM mask if argued before computing the 2PCF
        master_mask_chunk = reshape_data_array(master_mask, nskew_to_match_data, npix_sim_skew, True)
        if type(cgm_gpm_allspec) == type(None):
            all_mask = master_mask_chunk
        else:
            gpm_onespec_chunk = reshape_data_array(cgm_gpm_allspec[iqso][0], nskew_to_match_data, npix_sim_skew, True) # reshaping GPM from cgm masking
            all_mask = master_mask_chunk * gpm_onespec_chunk

        # compute the 2PCF
        vel_mid, xi_mock, npix_xi = compute_cf_onespec_chunk(vel_lores, noisy_flux_lores_ncopy, vmin_corr, vmax_corr, dv_corr, mask=all_mask)
        xi_mock_qso_all.append(xi_mock)

    vel_mid = np.array(vel_mid)
    xi_mock_qso_all = np.array(xi_mock_qso_all)
    nqso, ncopy, nskew, npix = xi_mock_qso_all.shape

    xi_mock_avg_nskew = np.mean(xi_mock_qso_all, axis=2) # average over nskew_to_match_data
    xi_mock_mean = np.mean(xi_mock_avg_nskew, axis=0) # average over nqso
    delta_xi = xi_mock_mean - xi_mean # delta_xi.shape = (ncopy, ncorr)

    ncorr = xi_mean.shape[0]
    covar = np.zeros((ncorr, ncorr))

    for icopy in range(ncopy):
        covar += np.outer(delta_xi[icopy], delta_xi[icopy])  # off-diagonal elements

    xi_mock_keep = xi_mock_mean # xi_mock_mean[:nmock]

    # Divid by ncovar since we estimated the mean from "independent" data
    covar /= ncopy

    return xi_mock_keep, covar

########################## running the grid #############################
def compute_model(args):
    # compute CF and covariance of mock dataset at each point of model grid
    # args: tuple of arguments from parser

    ihi, iZ, xHI, logZ, master_seed, xhi_path, zstr, fwhm, sampling, vmin_corr, vmax_corr, dv_corr, redshift_bin, ncopy, cgm_masking_gpm = args
    rantaufile = os.path.join(xhi_path, 'ran_skewers_' + zstr + '_OVT_' + 'xHI_{:4.2f}'.format(xHI) + '_tau.fits')
    params = Table.read(rantaufile, hdu=1)
    skewers = Table.read(rantaufile, hdu=2)

    # Generate 10,000 Nyx skewers. This takes 5.36s.
    vel_lores, (flux_lores, flux_lores_igm, flux_lores_cgm, _, _), vel_hires, (flux_hires, flux_hires_igm, flux_hires_cgm, _, _), \
    (oden, v_los, T, xHI), cgm_tuple = utils.create_mgii_forest(params, skewers, logZ, fwhm, sampling=sampling)

    # noiseless quantities
    mean_flux_nless = np.mean(flux_lores)
    delta_f_nless = (flux_lores - mean_flux_nless) / mean_flux_nless
    (vel_mid, xi_nless, npix, xi_nless_zero_lag) = utils.compute_xi(delta_f_nless, vel_lores, vmin_corr, vmax_corr, dv_corr)
    xi_mean = np.mean(xi_nless, axis=0)

    # noisy quantities for all QSOs
    xi_mock_keep, covar = mock_mean_covar(xi_mean, ncopy, vel_lores, flux_lores, vmin_corr, vmax_corr, dv_corr, redshift_bin, master_seed=master_seed, cgm_gpm_allspec=cgm_masking_gpm)
    icovar = np.linalg.inv(covar)  # invert the covariance matrix
    sign, logdet = np.linalg.slogdet(covar)  # compute the sign and natural log of the determinant of the covariance matrix

    return ihi, iZ, vel_mid, xi_mock_keep, xi_mean, covar, icovar, logdet

def test_compute_model():

    ihi, iZ = 0, 0 # dummy variables since they are simply returned
    xhi_path = '/Users/suksientie/Research/MgII_forest'
    #xhi_path = '/mnt/quasar/joe/reion_forest/Nyx_output/z75/xHI/' # on IGM cluster
    zstr = 'z75'
    xHI = 0.50
    fwhm = 90
    sampling = 3
    vmin_corr = 10
    vmax_corr = 3500
    dv_corr = fwhm
    ncopy = 5
    #ncovar = 10
    #nmock = 10
    master_seed = 9999 # results in seed list [315203670 242427133 938891646 135124015]
    logZ = -3.5
    cgm_masking = True
    if cgm_masking:
        cgm_masking_gpm = init_cgm_masking(fwhm)
    else:
        cgm_masking_gpm = None

    redshift_bin = 'all' # 'high', 'all'
    args = ihi, iZ, xHI, logZ, master_seed, xhi_path, zstr, fwhm, sampling, vmin_corr, vmax_corr, dv_corr, redshift_bin, ncopy, cgm_masking_gpm

    output = compute_model(args)
    ihi, iZ, vel_mid, xi_mock, xi_mean, covar, icovar, logdet = output

    """
    nhi, nlogZ = 5, 5 # dummy values
    xhi_val = xHI
    ncorr = vel_mid.shape[0]
    xi_mock_array = np.zeros((nhi, nlogZ,) + xi_mock.shape)
    xi_mean_array = np.zeros((nhi, nlogZ,) + xi_mean.shape)
    covar_array = np.zeros((nhi, nlogZ,) + covar.shape)
    icovar_array = np.zeros((nhi, nlogZ,) + icovar.shape)
    lndet_array = np.zeros((nhi, nlogZ))

    # Unpack the output
    for out in output:
        ihi, iZ, vel_mid, xi_mock, xi_mean, covar, icovar, logdet = out
        # Right out a random subset of these so that we don't use so much disk.
        xi_mock_array[ihi, iZ, :, :] = xi_mock
        xi_mean_array[ihi, iZ, :] = xi_mean
        covar_array[ihi, iZ, :, :] = covar
        icovar_array[ihi, iZ, :, :] = icovar
        lndet_array[ihi, iZ] = logdet

    ncovar = 0  # hardwired
    nmock = ncopy  # hardwired
    
    param_model = Table(
        [[ncopy], [ncovar], [nmock], [fwhm], [sampling], [master_seed], [nhi], [xhi_val], [nlogZ], [logZ_vec], [ncorr],
         [vmin_corr], [vmax_corr], [vel_mid]],
        names=(
        'ncopy', 'ncovar', 'nmock', 'fwhm', 'sampling', 'seed', 'nhi', 'xhi', 'nlogZ', 'logZ', 'ncorr', 'vmin_corr',
        'vmax_corr', 'vel_mid'))
    param_out = hstack((params, param_model))

    # Write out to multi-extension fits
    print('Writing out to disk')
    # Write to outfile
    hdu_param = fits.table_to_hdu(param_out)
    hdu_param.name = 'METADATA'
    hdulist = fits.HDUList()
    hdulist.append(hdu_param)
    hdulist.append(fits.ImageHDU(data=xi_mock_array, name='XI_MOCK'))
    hdulist.append(fits.ImageHDU(data=xi_mean_array, name='XI_MEAN'))
    hdulist.append(fits.ImageHDU(data=covar_array, name='COVAR'))
    hdulist.append(fits.ImageHDU(data=icovar_array, name='ICOVAR'))
    hdulist.append(fits.ImageHDU(data=lndet_array, name='LNDET'))
    hdulist.writeto(outfile, overwrite=True)
    """
    return output

###################### main() ######################
def parser():
    import argparse

    parser = argparse.ArgumentParser(description='Create random skewers for MgII forest', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nproc', type=int, help="Number of processors to run on")
    parser.add_argument('--fwhm', type=float, default=90.0, help="spectral resolution in km/s")
    parser.add_argument('--samp', type=float, default=3.0, help="Spectral sampling: pixels per fwhm resolution element")
    parser.add_argument('--vmin', type=float, default=10.0, help="Minimum of velocity bins for correlation function")
    parser.add_argument('--vmax', type=float, default=3500, help="Maximum of velocity bins for correlation function")
    parser.add_argument('--dv', type=float, default=None, help="Width of velocity bins for correlation function. "
                                                               "If not set fwhm will be used")
    parser.add_argument('--ncopy', type=int, default=1000, help="number of forward-modeled spectra for each qso")
    parser.add_argument('--seed', type=int, default=12349876, help="master seed for random number generator")
    parser.add_argument('--nlogZ', type=int, default=201, help="number of bins for logZ models")
    parser.add_argument('--logZmin', type=float, default=-6.0, help="minimum logZ value")
    parser.add_argument('--logZmax', type=float, default=-2.0, help="maximum logZ value")
    parser.add_argument('--cgm_masking', action='store_true', help='whether to mask cgm or not')
    parser.add_argument('--lowz_bin', action='store_true', help='use the low redshift bin, defined to be z < 6.754 (median)')
    parser.add_argument('--highz_bin', action='store_true', help='use the high redshift bin, defined to be z >= 6.754 (median)')
    parser.add_argument('--allz_bin', action='store_true', help='use all redshift bin')
    return parser.parse_args()

def main():

    args = parser()
    nproc = args.nproc
    fwhm = args.fwhm
    sampling = args.samp
    ncopy = args.ncopy
    master_seed = args.seed
    vmin_corr = args.vmin
    vmax_corr = args.vmax
    dv_corr = args.dv if args.dv is not None else fwhm
    cgm_masking = args.cgm_masking
    if cgm_masking:
        cgm_masking_gpm = init_cgm_masking(fwhm)
    else:
        cgm_masking_gpm = None

    if args.lowz_bin:
        redshift_bin = 'low'
    elif args.highz_bin:
        redshift_bin = 'high'
    elif args.allz_bin:
        redshift_bin = 'all'
    else:
        raise ValueError('must set one of these arguments: "--lowz_bin", "--highz_bin", or "--allz_bin"')

    # Grid of metallicities
    nlogZ = args.nlogZ
    logZ_min = args.logZmin
    logZ_max = args.logZmax
    logZ_vec = np.linspace(logZ_min, logZ_max, nlogZ)

    # Read grid of neutral fractions from the 21cm fast xHI fields
    xhi_val, xhi_boxes = utils.read_xhi_boxes() # len(xhi_val) = 51, with d_xhi = 0.02
    #xhi_val = xhi_val[0:3] # testing for production run
    nhi = xhi_val.shape[0]

    # Some file paths and then read in the params table to get the redshift
    zstr = 'z75'
    outpath = '/mnt/quasar/sstie/MgII_forest/' + zstr + '/'
    outfilename = 'corr_func_models_fwhm_{:5.3f}_samp_{:5.3f}'.format(fwhm, sampling) + '.fits'
    outfile = os.path.join(outpath, outfilename)

    xhi_path = '/mnt/quasar/joe/reion_forest/Nyx_output/z75/xHI/'
    files = glob.glob(os.path.join(xhi_path, '*_tau.fits'))
    params = Table.read(files[0], hdu=1)

    args = xhi_path, zstr, fwhm, sampling, vmin_corr, vmax_corr, dv_corr, redshift_bin, ncopy, cgm_masking_gpm
    all_args = []
    seed_vec = np.full(nhi*nlogZ, master_seed) # same seed for each CPU process

    for ihi, xHI in enumerate(xhi_val):
        for iZ, logZ in enumerate(logZ_vec):
            indx = nlogZ * ihi + iZ
            itup = (ihi, iZ, xHI, logZ, seed_vec[indx]) + args
            all_args.append(itup)

    print('Computing nmodel={:d} models on nproc={:d} processors'.format(nhi*nlogZ,nproc))

    output = imap_unordered_bar(compute_model, all_args, nproc)
    #pool = Pool(processes=nproc)
    #output = pool.starmap(compute_model, all_args)

    # Allocate the arrays to hold everything
    ihi, iZ, vel_mid, xi_mock, xi_mean, covar, icovar, logdet = output[0] # ihi and iZ not actually used and simply returned as outputs
    ncorr = vel_mid.shape[0]
    xi_mock_array = np.zeros((nhi, nlogZ,) + xi_mock.shape)
    xi_mean_array = np.zeros((nhi, nlogZ,) + xi_mean.shape)
    covar_array = np.zeros((nhi, nlogZ,) + covar.shape)
    icovar_array = np.zeros((nhi, nlogZ,) + icovar.shape)
    lndet_array = np.zeros((nhi, nlogZ))

    # Unpack the output
    for out in output:
        ihi, iZ, vel_mid, xi_mock, xi_mean, covar, icovar, logdet = out
        # Right out a random subset of these so that we don't use so much disk.
        xi_mock_array[ihi,iZ,:,:] = xi_mock
        xi_mean_array[ihi,iZ,:] = xi_mean
        covar_array[ihi,iZ,:,:] = covar
        icovar_array[ihi,iZ,:,:] = icovar
        lndet_array[ihi,iZ] = logdet

    ncovar = 0 # hardwired
    nmock = ncopy # hardwired
    param_model=Table([[ncopy], [ncovar], [nmock], [fwhm],[sampling],[master_seed], [nhi], [xhi_val], [nlogZ], [logZ_vec], [ncorr], [vmin_corr],[vmax_corr], [vel_mid], [redshift_bin]],
                      names=('ncopy', 'ncovar', 'nmock', 'fwhm', 'sampling', 'seed', 'nhi', 'xhi', 'nlogZ', 'logZ', 'ncorr', 'vmin_corr', 'vmax_corr', 'vel_mid', 'redshift_bin'))
    param_out = hstack((params, param_model))

    # Write out to multi-extension fits
    print('Writing out to disk')
    # Write to outfile
    hdu_param = fits.table_to_hdu(param_out)
    hdu_param.name = 'METADATA'
    hdulist = fits.HDUList()
    hdulist.append(hdu_param)
    hdulist.append(fits.ImageHDU(data=xi_mock_array, name='XI_MOCK'))
    hdulist.append(fits.ImageHDU(data=xi_mean_array, name='XI_MEAN'))
    hdulist.append(fits.ImageHDU(data=covar_array, name='COVAR'))
    hdulist.append(fits.ImageHDU(data=icovar_array, name='ICOVAR'))
    hdulist.append(fits.ImageHDU(data=lndet_array, name='LNDET'))
    hdulist.writeto(outfile, overwrite=True)

###################### run main() ######################
if __name__ == '__main__':
    main()