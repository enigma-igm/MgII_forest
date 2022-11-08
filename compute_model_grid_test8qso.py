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
sys.path.append('/Users/suksientie/codes/enigma') # comment out this line if running on IGM cluster
from enigma.reion_forest.mgii_find import MgiiFinder
from enigma.reion_forest import utils
from astropy.table import Table, hstack, vstack
from IPython import embed
from multiprocessing import Pool
from tqdm import tqdm
import mutils
import mask_cgm_pdf as mask_cgm
import pdb
import compute_cf_data as ccf
import scipy

###################### global variables ######################
datapath='/Users/suksientie/Research/MgII_forest/rebinned_spectra/'

qso_namelist = ['J0411-0907', 'J0319-1008', 'J0410-0139', 'J0038-0653', 'J0313-1806', 'J0038-1527', 'J0252-0503', 'J1342+0928']
qso_zlist = [6.826, 6.8275, 7.0, 7.1, 7.642, 7.034, 7.001, 7.541]
everyn_break_list = (np.ones(len(qso_namelist)) * 20).astype('int')
exclude_restwave = 1216 - 1185
median_z = 6.50
corr_all = [0.669, 0.673, 0.692, 0.73 , 0.697, 0.653, 0.667, 0.72]
nqso_to_use = len(qso_namelist)

nires_fwhm = 111.03
mosfire_fwhm = 83.05
nires_sampling = 2.7
mosfire_sampling = 2.78

qso_fwhm = [nires_fwhm, nires_fwhm, nires_fwhm, mosfire_fwhm, mosfire_fwhm, mosfire_fwhm, mosfire_fwhm, mosfire_fwhm]
qso_sampling = [nires_sampling, nires_sampling, nires_sampling, mosfire_sampling, mosfire_sampling, mosfire_sampling, mosfire_sampling, mosfire_sampling]
given_bins = ccf.custom_cf_bin4(dv1=80)

signif_thresh = 2.0
signif_mask_dv = 300.0 # value used in Hennawi+2021
signif_mask_nsigma = 3
one_minF_thresh = 0.3 # flux threshold

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
    # draw n number of random Nyx skewers that match the total data pathlength

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
    # reshape the data spectrum to match the dimension of the simulated skewers array,
    # which is of shape (nskew_to_match_data, npix_sim_skew)
    # note: set data_arr_is_mask = True if the data array is a mask

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

def forward_model_onespec_chunk(vel_data, norm_std, master_mask, vel_lores, flux_lores, ncovar, seed=None, std_corr=1.0):

    rand = np.random.RandomState(seed) if seed != None else np.random.RandomState()

    npix_sim_skew = len(vel_lores)
    tot_nyx_skews = len(flux_lores)

    ranindx, nskew_to_match_data = rand_skews_to_match_data(vel_lores, vel_data, tot_nyx_skews, seed=seed)

    flux_noise_ncopy = np.zeros((ncovar, nskew_to_match_data, npix_sim_skew))
    flux_noiseless_ncopy = np.zeros((ncovar, nskew_to_match_data, npix_sim_skew))
    master_mask_chunk = reshape_data_array(master_mask, nskew_to_match_data, npix_sim_skew, data_arr_is_mask=True)

    for icopy in range(ncovar):
        ranindx, nskew_to_match_data = rand_skews_to_match_data(vel_lores, vel_data, tot_nyx_skews, seed=seed)

        # adding noise to flux lores
        norm_std[norm_std < 0] = 100  # get rid of negative errors
        noise = rand.normal(0, std_corr * norm_std)
        noise_chunk = reshape_data_array(noise, nskew_to_match_data, npix_sim_skew, data_arr_is_mask=False)

        flux_noise_ncopy[icopy] = flux_lores[ranindx] + noise_chunk
        flux_noiseless_ncopy[icopy] = flux_lores[ranindx]

    return vel_lores, flux_noise_ncopy, flux_noiseless_ncopy, master_mask_chunk, nskew_to_match_data, npix_sim_skew

"""
def sample_noise_onespec_chunk(vel_data, norm_std, vel_lores, flux_lores, ncopy, seed=None, std_corr=1.0):

    # return "ncopy" of simulated noise from data

    npix_sim_skew = len(vel_lores)
    tot_nyx_skews = len(flux_lores)
    ranindx, nskew_to_match_data = rand_skews_to_match_data(vel_lores, vel_data, tot_nyx_skews, seed=seed)

    if seed != None:
        rand = np.random.RandomState(seed)
    else:
        rand = np.random.RandomState()

    # get rid of negative errors
    norm_std[norm_std < 0] = 100

    # reshape data sigma array
    norm_std_chunk = reshape_data_array(norm_std, nskew_to_match_data, npix_sim_skew, data_arr_is_mask=False)

    rand_noise_ncopy = np.zeros((ncopy, nskew_to_match_data, npix_sim_skew))
    for icopy in range(ncopy):
        onecopy = []
        for iskew in range(nskew_to_match_data):
            onecopy.append(rand.normal(0, std_corr * np.array(norm_std_chunk[iskew])))
        rand_noise_ncopy[icopy] = np.array(onecopy)

    #rand_noise_ncopy = np.broadcast_to(std_corr * norm_std_chunk, (ncopy, nskew_to_match_data, npix_sim_skew))

    return ranindx, rand_noise_ncopy, nskew_to_match_data, npix_sim_skew

def forward_model_onespec_chunk(vel_data, norm_std, vel_lores, flux_lores, ncopy, seed=None, std_corr=1.0):
    # wrapper to sample_noise_onespec_chunk

    ranindx, rand_noise_ncopy, nskew_to_match_data, npix_sim_skew = sample_noise_onespec_chunk(vel_data, norm_std, vel_lores, flux_lores, ncopy, seed=seed, std_corr=std_corr)
    rand_flux_lores = flux_lores[ranindx]
    #rand_flux_lores_ncopy = np.tile(rand_flux_lores, (ncopy,1,1))
    noisy_flux_lores_ncopy = rand_flux_lores + rand_noise_ncopy # rand_flux_lores automatically broadcasted across the ncopy-axis

    return ranindx, rand_flux_lores, rand_noise_ncopy, noisy_flux_lores_ncopy, nskew_to_match_data, npix_sim_skew

def plot_forward_model_onespec_new(noisy_flux_lores_ncopy, master_mask, vel_data, norm_flux, ncopy_plot, title=None):

    ncopy, nskew, npix = np.shape(noisy_flux_lores_ncopy)

    plt.figure(figsize=(12, 8))
    if title != None:
        plt.title(title, fontsize=18)

    nan_pad_mask = ~np.isnan(noisy_flux_lores_ncopy[0])
    ##### plot subset of mock spectra
    for i in range(ncopy_plot):
        flux_lores_comb = noisy_flux_lores_ncopy[i][nan_pad_mask]
        plt.plot(vel_data, flux_lores_comb + (i + 1), alpha=0.5, drawstyle='steps-mid', zorder=20)

    plt.plot(vel_data, norm_flux, 'k', drawstyle='steps-mid')

    ind_masked = np.where(master_mask == False)[0]
    for j in range(len(ind_masked)):
        if ind_masked[j]+1 != len(vel_data):
            plt.axvspan(vel_data[ind_masked[j]], vel_data[ind_masked[j]+1], facecolor = 'black', alpha = 0.4)

    plt.ylabel('Flux (+ arbitrary offset)', fontsize=15)
    plt.xlabel('Velocity (km/s)', fontsize=15)
    plt.tight_layout()
    plt.show()
"""

def compute_cf_onespec_chunk(vel_lores, noisy_flux_lores_ncopy, given_bins, mask_chunk=None):
    # "mask" is any mask, not necessarily cgm mask; need to be of shape (nskew, npix), use the reshape_data_array() function above

    vmin_corr, vmax_corr, dv_corr = 0, 0, 0  # dummy values since using custom binning, but still required arguments for compute_xi

    ncopy, nskew, npix = np.shape(noisy_flux_lores_ncopy)
    # compress ncopy and nskew dimensions together because compute_xi() only takes in 2D arrays
    # (tried keeping the original 3D array and looping over ncopy when compute_xi(), but loop process too slow, sth like 1.4 min per 100 copy)
    reshaped_flux = np.reshape(noisy_flux_lores_ncopy, (ncopy * nskew, npix))

    if mask_chunk is not None:
        #magsk_ncopy = np.tile(mask, (ncopy, 1, 1)) # making ncopy of the mask
        #mask_ncopy = np.reshape(mask_ncopy, (ncopy*nskew, npix)).astype(int) # reshaping the same way as above and recasting to int type so the next line can run
        #reshaped_flux_masked = np.where(mask_ncopy, reshaped_flux, np.nan)
        #mean_flux = np.nanmean(reshaped_flux_masked)
        # mask_ncopy = mask_ncopy.astype(bool)

        mask_ncopy = np.tile(mask_chunk, (ncopy, 1, 1))
        mask_ncopy = np.reshape(mask_ncopy, (ncopy * nskew, npix))
        mean_flux = np.nanmean(reshaped_flux[mask_ncopy])
    else:
        mask_ncopy = None
        mean_flux = np.nanmean(reshaped_flux)

    delta_f = (reshaped_flux - mean_flux)/mean_flux
    (vel_mid, xi_mock, npix_xi, xi_mock_zero_lag) = utils.compute_xi(delta_f, vel_lores, vmin_corr, vmax_corr, dv_corr, \
                                                                     given_bins=given_bins, gpm=mask_ncopy)

    # reshape xi_mock into the original input shape
    xi_mock = np.reshape(xi_mock, (ncopy, nskew, len(vel_mid)))
    npix_xi = np.reshape(npix_xi, (ncopy, nskew, len(vel_mid)))
    return vel_mid, xi_mock, npix_xi

def init_dataset(nqso, redshift_bin, datapath):

    norm_std_allqso = []
    vel_data_allqso = []
    master_mask_allqso = []

    instr_allqso = ['nires', 'nires', 'nires', 'mosfire', 'mosfire', 'mosfire', 'mosfire', 'mosfire']
    for iqso in range(nqso):
        raw_data_out, _, all_masks_out = mutils.init_onespec(iqso, redshift_bin, datapath=datapath)
        wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
        strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, telluric_gpm, master_mask = all_masks_out

        norm_flux = flux / fluxfit
        norm_std = std / fluxfit
        vel_data = mutils.obswave_to_vel_2(wave)
        norm_std_allqso.append(norm_std)
        vel_data_allqso.append(vel_data)

        do_not_apply_any_mask = True

        if do_not_apply_any_mask:
            masks_for_cgm_masking = np.ones_like(wave, dtype=bool)
        else:
            masks_for_cgm_masking = mask * redshift_mask * pz_mask * zbin_mask * telluric_gpm

        # masks quantities
        norm_good_flux = (flux / fluxfit)[masks_for_cgm_masking]
        norm_good_ivar = (ivar * (fluxfit ** 2))[masks_for_cgm_masking]
        good_vel_data = vel_data[masks_for_cgm_masking]

        # reshaping to be compatible with MgiiFinder
        norm_good_flux = norm_good_flux.reshape((1, len(norm_good_flux)))
        norm_good_ivar = norm_good_ivar.reshape((1, len(norm_good_ivar)))
        fwhm = qso_fwhm[iqso]

        mgii_tot = MgiiFinder(good_vel_data, norm_good_flux, norm_good_ivar, fwhm, signif_thresh,
                              signif_mask_nsigma=signif_mask_nsigma,
                              signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh)
        gpm_allspec = mgii_tot.fit_gpm[0]
        master_mask_allqso.append(master_mask * gpm_allspec)

    return vel_data_allqso, norm_std_allqso, master_mask_allqso, instr_allqso

def mock_mean_covar_old(xi_mean, ncopy, vel_lores, flux_lores, vmin_corr, vmax_corr, dv_corr, redshift_bin, master_seed=None, cgm_gpm_allspec=None):

    master_rand = np.random.RandomState() if master_seed is None else np.random.RandomState(master_seed)
    qso_seed_list = master_rand.randint(0, 1000000000, nqso_to_use)  # 4 for 4 qsos, hardwired for now

    xi_mock_qso_all = []
    xi_mock_avg_nskew = []
    npix_xi_all = []

    for iqso in range(nqso_to_use):
        vel_data = vel_data_allqso[iqso]
        norm_std = norm_std_allqso[iqso]
        master_mask = master_mask_allqso[iqso]
        std_corr = corr_all[iqso]
        instr = instr_allqso[iqso]

        if instr == 'nires':
            vel_lores = vel_lores_nires_interp
            flux_lores = flux_lores_nires_interp
        elif instr == 'mosfire':
            vel_lores = vel_lores_mosfire_interp
            flux_lores = flux_lores_mosfire_interp

        # generate mock data spectrum
        vel_lores, flux_noise_ncopy, flux_noiseless_ncopy, master_mask_chunk, nskew_to_match_data, npix_sim_skew = \
            forward_model_onespec_chunk(vel_data, norm_std, master_mask, vel_lores, flux_lores, ncopy, seed=None,
                                        std_corr=std_corr)

        # compute the 2PCF
        vel_mid, xi_mock_ncopy, npix_xi = compute_cf_onespec_chunk(vel_lores, flux_noise_ncopy, vmin_corr, vmax_corr,
                                                             dv_corr, mask=master_mask_chunk, given_bins=given_bins)
        # xi_mock_qso_all.append(xi_mock)
        xi_mock_avg_nskew.append(np.mean(xi_mock_ncopy, axis=1))  # average over nskew for each qso


        """
        _, _, rand_noise_ncopy, noisy_flux_lores_ncopy, nskew_to_match_data, npix_sim_skew = forward_model_onespec_chunk(vel_data, norm_std, vel_lores, flux_lores, ncopy, seed=qso_seed_list[iqso], std_corr=corr_all[iqso])

        #usable_data_mask = mask * redshift_mask * pz_mask * zbin_mask
        usable_data_mask = master_mask
        usable_data_mask_chunk = reshape_data_array(usable_data_mask, nskew_to_match_data, npix_sim_skew, True)

        # deal with CGM mask if argued before computing the 2PCF
        if type(cgm_gpm_allspec) == type(None):
            all_mask_chunk = usable_data_mask_chunk
        else:
            gpm_onespec_chunk = reshape_data_array(cgm_gpm_allspec[iqso], nskew_to_match_data, npix_sim_skew, True) # reshaping GPM from cgm masking
            all_mask_chunk = usable_data_mask_chunk * gpm_onespec_chunk

        # compute the 2PCF
        vel_mid, xi_mock, npix_xi = compute_cf_onespec_chunk(vel_lores, noisy_flux_lores_ncopy, vmin_corr, vmax_corr, dv_corr, mask=all_mask_chunk, given_bins=given_bins)
        #xi_mock_qso_all.append(xi_mock)
        xi_mock_avg_nskew.append(np.mean(xi_mock, axis=1)) # average over nskew for each qso
        
        # npix_xi = ncopy, nskew, ncorr
        # npix_xi_all = nqso, ncopy, ncorr
        npix_xi_all.append(np.sum(npix_xi, axis=1))
        """

    vel_mid = np.array(vel_mid)
    npix_xi_all = np.array(npix_xi_all)

    """
    xi_mock_qso_all = np.array(xi_mock_qso_all)
    nqso, ncopy, nskew, npix = xi_mock_qso_all.shape
    xi_mock_avg_nskew = np.mean(xi_mock_qso_all, axis=2) # average over nskew_to_match_data
    xi_mock_mean = np.mean(xi_mock_avg_nskew, axis=0) # average over nqso
    """

    weights = npix_xi_all/np.sum(npix_xi_all, axis=0) # weights = nqso, ncopy, ncorr

    xi_mock_avg_nskew = np.array(xi_mock_avg_nskew)
    nqso, ncopy, npix = xi_mock_avg_nskew.shape
    xi_mock_mean = np.average(xi_mock_avg_nskew, axis=0, weights=weights)

    delta_xi = xi_mock_mean - xi_mean # delta_xi.shape = (ncopy, ncorr)
    ncorr = xi_mean.shape[0]
    covar = np.zeros((ncorr, ncorr))
    for icopy in range(ncopy):
        covar += np.outer(delta_xi[icopy], delta_xi[icopy])  # off-diagonal elements

    xi_mock_keep = xi_mock_mean # xi_mock_mean[:nmock]

    # Divid by ncovar since we estimated the mean from "independent" data
    covar /= ncopy

    return xi_mock_keep, covar

def mock_mean_covar(ncovar, nmock_to_save, vel_data_allqso, norm_std_allqso, master_mask_allqso, instr_allqso, \
                    vel_lores_nires_interp, flux_lores_nires_interp, vel_lores_mosfire_interp, flux_lores_mosfire_interp, given_bins, seed=None):

    master_rand = np.random.RandomState() if seed is None else np.random.RandomState(seed)

    xi_mock_ncopy = []
    xi_mock_ncopy_noiseless = []

    for iqso in range(nqso_to_use):
        vel_data = vel_data_allqso[iqso]
        norm_std = norm_std_allqso[iqso]
        master_mask = master_mask_allqso[iqso]
        std_corr = corr_all[iqso]
        instr = instr_allqso[iqso]

        if instr == 'nires':
            vel_lores = vel_lores_nires_interp
            flux_lores = flux_lores_nires_interp
        elif instr == 'mosfire':
            vel_lores = vel_lores_mosfire_interp
            flux_lores = flux_lores_mosfire_interp

        # generate mock data spectrum
        vel_lores, flux_noise_ncopy, flux_noiseless_ncopy, master_mask_chunk, nskew_to_match_data, npix_sim_skew = \
            forward_model_onespec_chunk(vel_data, norm_std, master_mask, vel_lores, flux_lores, ncovar, seed=None,
                                        std_corr=std_corr)

        # compute the 2PCF (with noise)
        # xi_mock_ncopy = ncopy, nskew, ncorr
        # xi_mock_avg_nskew = nqso, ncopy, ncorr
        # npix_xi = ncopy, nskew, ncorr
        # npix_xi_all = nqso, ncopy, ncorr

        vel_mid, xi_onespec_ncopy, npix_xi = compute_cf_onespec_chunk(vel_lores, flux_noise_ncopy, given_bins, mask_chunk=master_mask_chunk)
        vel_mid, xi_onespec_ncopy_noiseless, npix_xi = compute_cf_onespec_chunk(vel_lores, flux_noiseless_ncopy, given_bins, mask_chunk=master_mask_chunk)

        xi_mock_onespec_ncopy = np.mean(xi_onespec_ncopy, axis=1)
        xi_mock_onespec_noiseless_ncopy = np.mean(xi_onespec_ncopy_noiseless, axis=1)

        xi_mock_ncopy.append(xi_mock_onespec_ncopy)
        xi_mock_ncopy_noiseless.append(xi_mock_onespec_noiseless_ncopy)

    # cast all into np arrays
    vel_mid = np.array(vel_mid)
    xi_mock_ncopy = np.array(xi_mock_ncopy)
    xi_mock_ncopy_noiseless = np.array(xi_mock_ncopy_noiseless)

    xi_mean_ncopy = np.mean(xi_mock_ncopy_noiseless, axis=0) # average over nqso
    xi_mean = np.mean(xi_mean_ncopy, axis=0) # average over ncopy
    xi_mock_mean = np.mean(xi_mock_ncopy, axis=0)

    #delta_xi = xi_mock_mean - xi_mean  # delta_xi.shape = (ncopy, ncorr)
    ncorr = xi_mean.shape[0]
    covar = np.zeros((ncorr, ncorr))
    xi_mock_keep = np.zeros((nmock_to_save, ncorr))

    for imock in range(ncovar):
        delta_xi = xi_mock_mean[imock] - xi_mean
        covar += np.outer(delta_xi, delta_xi)  # off-diagonal elements

        if imock < nmock_to_save:
            xi_mock_keep[imock, :] = xi_mock_mean[imock]

    # Divid by ncovar since we estimated the mean from "independent" data
    covar /= ncovar

    #return xi_mock_keep, covar
    return xi_mock_keep, covar, vel_mid, xi_mean

########################## running the grid #############################
import time
def compute_model(args):
    # compute CF and covariance of mock dataset at each point of model grid
    # args: tuple of arguments from parser

    #ihi, iZ, xHI, logZ, master_seed, xhi_path, zstr, fwhm, sampling, vmin_corr, vmax_corr, dv_corr, redshift_bin, ncopy, cgm_masking_gpm = args
    ihi, iZ, xHI, logZ, master_seed, xhi_path, zstr, redshift_bin, ncovar, nmock, vel_data_allqso, norm_std_allqso, master_mask_allqso, instr_allqso = args

    rantaufile = os.path.join(xhi_path, 'ran_skewers_' + zstr + '_OVT_' + 'xHI_{:4.2f}'.format(xHI) + '_tau.fits')
    params = Table.read(rantaufile, hdu=1)
    skewers = Table.read(rantaufile, hdu=2)

    # Generate 10,000 Nyx skewers. This takes 5.36s.
    # NIRES fwhm and sampling
    #vel_lores_nires, (flux_lores_nires, flux_lores_igm_nires, flux_lores_cgm_nires, _, _), \
    #vel_hires_nires, (flux_hires_nires, flux_hires_igm_nires, flux_hires_cgm_nires, _, _), \
    #(oden, v_los, T, xHI), cgm_tuple = utils.create_mgii_forest(params, skewers, logZ, nires_fwhm, sampling=nires_sampling)
    vel_lores_nires, flux_lores_nires = utils.create_mgii_forest(params, skewers, logZ, nires_fwhm, sampling=nires_sampling, mockcalc=True)

    # MOSFIRE fwhm and sampling
    #vel_lores_mosfire, (flux_lores_mosfire, flux_lores_igm_mosfire, flux_lores_cgm_mosfire, _, _), \
    #vel_hires_mosfire, (flux_hires_mosfire, flux_hires_igm_mosfire, flux_hires_cgm_mosfire, _, _), \
    #(oden, v_los, T, xHI), cgm_tuple = utils.create_mgii_forest(params, skewers, logZ, mosfire_fwhm, sampling=mosfire_sampling)
    vel_lores_mosfire, flux_lores_mosfire = utils.create_mgii_forest(params, skewers, logZ, mosfire_fwhm, sampling=mosfire_sampling, mockcalc=True)

    # interpolate flux lores to dv=40 (nyx)
    dv_coarse = 40
    coarse_grid_all = np.arange(len(vel_lores_nires)+1) * dv_coarse
    vel_lores_nires_interp = (coarse_grid_all[:-1] + coarse_grid_all[1:]) / 2
    flux_lores_nires_interp = scipy.interpolate.interp1d(vel_lores_nires, flux_lores_nires, kind = 'cubic', \
                                                        bounds_error = False, fill_value = np.nan)(vel_lores_nires_interp)

    coarse_grid_all = np.arange(len(vel_lores_mosfire) + 1) * dv_coarse
    vel_lores_mosfire_interp = (coarse_grid_all[:-1] + coarse_grid_all[1:]) / 2
    flux_lores_mosfire_interp = scipy.interpolate.interp1d(vel_lores_mosfire, flux_lores_mosfire, kind='cubic',
                                                       bounds_error=False, fill_value=np.nan)(vel_lores_mosfire_interp)

    del skewers
    del vel_lores_nires
    del vel_lores_mosfire
    del flux_lores_nires
    del flux_lores_mosfire

    start = time.process_time()
    xi_mock_keep, covar, vel_mid, xi_mean = mock_mean_covar(ncovar, nmock, vel_data_allqso, norm_std_allqso, master_mask_allqso, instr_allqso, \
                    vel_lores_nires_interp, flux_lores_nires_interp, vel_lores_mosfire_interp, flux_lores_mosfire_interp, given_bins, seed=None)

    icovar = np.linalg.inv(covar)  # invert the covariance matrix
    sign, logdet = np.linalg.slogdet(covar)  # compute the sign and natural log of the determinant of the covariance matrix
    end = time.process_time()
    print("      mock_mean_covar done in .... ", (end - start) / 60, " min")

    return ihi, iZ, vel_mid, xi_mock_keep, xi_mean, covar, icovar, logdet

"""
def test_compute_model():

    ihi, iZ = 0, 0 # dummy variables since they are simply returned
    xhi_path = '/Users/suksientie/Research/MgII_forest'
    #xhi_path = '/mnt/quasar/joe/reion_forest/Nyx_output/z75/xHI/' # on IGM cluster
    zstr = 'z75'
    xHI = 0.50
    fwhm = 120
    sampling = 3
    vmin_corr = 10
    vmax_corr = 3500
    dv_corr = 100
    ncopy = 5
    #ncovar = 10
    #nmock = 10
    master_seed = 9999 # results in seed list [315203670 242427133 938891646 135124015]
    logZ = -3.5
    cgm_masking = True
    redshift_bin = 'all' # 'low', 'high', 'all'

    if cgm_masking:
        cgm_masking_gpm, _ = init_cgm_masking(redshift_bin, datapath)
    else:
        cgm_masking_gpm = None

    args = ihi, iZ, xHI, logZ, master_seed, xhi_path, zstr, fwhm, sampling, vmin_corr, vmax_corr, dv_corr, redshift_bin, ncopy, cgm_masking_gpm

    output = compute_model(args)
    ihi, iZ, vel_mid, xi_mock, xi_mean, covar, icovar, logdet = output

    return output
"""
###################### main() ######################
def parser():
    import argparse

    parser = argparse.ArgumentParser(description='Create random skewers for MgII forest', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nproc', type=int, help="Number of processors to run on")
    parser.add_argument('--ncovar', type=int, default=1000, help="number of forward-modeled spectra for each qso")
    parser.add_argument('--nmock', type=int, default=500, help="number of mock dataset to save")
    parser.add_argument('--seed', type=int, default=12349876, help="master seed for random number generator")
    parser.add_argument('--nlogZ', type=int, default=201, help="number of bins for logZ models")
    parser.add_argument('--logZmin', type=float, default=-6.0, help="minimum logZ value")
    parser.add_argument('--logZmax', type=float, default=-2.0, help="maximum logZ value")
    parser.add_argument('--lowz_bin', action='store_true', help='use the low redshift bin, defined to be z < 6.754 (median)')
    parser.add_argument('--highz_bin', action='store_true', help='use the high redshift bin, defined to be z >= 6.754 (median)')
    parser.add_argument('--allz_bin', action='store_true', help='use all redshift bin')
    return parser.parse_args()

def main():

    args = parser()
    nproc = args.nproc
    ncovar = args.ncopy
    nmock = args.nmock
    seed = args.seed

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
    outpath = '/mnt/quasar/sstie/MgII_forest/' + zstr + '/8qso/'
    outfilename = 'corr_func_models_{:s}'.format(redshift_bin) + '.fits'
    outfile = os.path.join(outpath, outfilename)

    xhi_path = '/mnt/quasar/joe/reion_forest/Nyx_output/z75/xHI/'
    files = glob.glob(os.path.join(xhi_path, '*_tau.fits'))
    params = Table.read(files[0], hdu=1)

    vel_data_allqso, norm_std_allqso, master_mask_allqso, instr_allqso = init_dataset(nqso_to_use, redshift_bin, datapath)

    #args = xhi_path, zstr, fwhm, sampling, vmin_corr, vmax_corr, dv_corr, redshift_bin, ncopy, cgm_masking_gpm
    args = xhi_path, zstr, redshift_bin, ncovar, nmock, vel_data_allqso, norm_std_allqso, master_mask_allqso, instr_allqso
    all_args = []
    seed_vec = np.full(nhi*nlogZ, seed) # same seed for each CPU process

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

    param_model = Table([[ncovar], [nmock], [seed], [nhi], [xhi_val], [nlogZ], [logZ_vec], [ncorr], [vel_mid], [redshift_bin]],
        names=('ncovar', 'nmock', 'seed', 'nhi', 'xhi', 'nlogZ', 'logZ', 'ncorr', 'vel_mid', 'redshift_bin'))
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