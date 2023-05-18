# Use single cores (forcing it for numpy operations)
import os
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
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
import pdb
from IPython import embed
import compute_cf_data as ccf
import time
import scipy
import warnings
warnings.filterwarnings(action='ignore')#, message='divide by zero encountered in true_divide (compute_model_grid_8qso_fast.py:311)')
#import xi_cython
import mask_cgm_pdf

###################### global variables ######################
datapath='/Users/suksientie/Research/MgII_forest/rebinned_spectra/'
#datapath = '/mnt/quasar/sstie/MgII_forest/z75/rebinned_spectra/'

qso_namelist = ['J0411-0907', 'J0319-1008', 'J0410-0139', 'J0038-0653', 'J0313-1806', 'J0038-1527', 'J0252-0503', \
                'J1342+0928', 'J1007+2115', 'J1120+0641']
qso_zlist = [6.826, 6.8275, 7.0, 7.1, 7.642, 7.034, 7.001, 7.541, 7.515, 7.085]
exclude_restwave = 1216 - 1185
corr_all = [0.669, 0.673, 0.692, 0.73, 0.697, 0.653, 0.667, 0.72, 0.64, 0.64]
nqso_to_use = len(qso_namelist)

nires_fwhm = 111.03
mosfire_fwhm = 83.05
nires_sampling = 2.7
mosfire_sampling = 2.78
xshooter_fwhm = 42.8 # R=7000 quoted in Bosman+2017
xshooter_sampling = 3.7 #https://www.eso.org/sci/facilities/paranal/instruments/xshooter/inst.html

qso_fwhm = [nires_fwhm, nires_fwhm, nires_fwhm, mosfire_fwhm, mosfire_fwhm, mosfire_fwhm, mosfire_fwhm, mosfire_fwhm, nires_fwhm, xshooter_fwhm]
qso_sampling = [nires_sampling, nires_sampling, nires_sampling, mosfire_sampling, mosfire_sampling, mosfire_sampling, mosfire_sampling, mosfire_sampling, nires_sampling, xshooter_sampling]
given_bins = np.array(ccf.custom_cf_bin4(dv1=80))

signif_thresh = 2.0
signif_mask_dv = 300.0 # value used in Hennawi+2021
signif_mask_nsigma = 3
one_minF_thresh = 0.3 # flux threshold

scale_weight = 1#1e6
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
    # seed = input random state

    tot_vel_data = vel_data.max() - vel_data.min()
    tot_vel_sim = vel_lores.max() - vel_lores.min()
    nskew_to_match_data = int(np.ceil(tot_vel_data / tot_vel_sim))  # assuming tot_vel_data > tot_vel_sim (true here)

    rand = np.random.RandomState(seed) if seed is None else seed

    indx_flux_lores = np.arange(tot_nyx_skews)
    ranindx = rand.choice(indx_flux_lores, replace=False, size=nskew_to_match_data)  # grab a set of random skewers
    return ranindx, nskew_to_match_data

def reshape_data_array(data_arr, nskew_to_match_data, npix_sim_skew, data_arr_is_mask):
    # reshape the data spectrum to match the dimension of the simulated skewers array,
    # which is of shape (nskew_to_match_data, npix_sim_skew)
    # note: set data_arr_is_mask = True if the data array is a mask

    pad_width = nskew_to_match_data * npix_sim_skew - len(data_arr)

    # fill with False values if data is a mask
    if data_arr_is_mask:
        print("input data array is Bool array, so padding with False")
        padded_data_arr = np.pad(data_arr.astype(float), (0, pad_width), mode='constant', constant_values=0)
        padded_data_arr = padded_data_arr.astype(bool)
        assert np.sum(padded_data_arr) == np.sum(data_arr)

    # fill with NaN if data is not a mask
    else:
        padded_data_arr = np.pad(data_arr, (0, pad_width), mode='constant', constant_values=np.nan)
        assert np.array_equal(data_arr, padded_data_arr[0:len(data_arr)]) or np.isclose(np.nansum(padded_data_arr),np.nansum(data_arr))

    new_data_arr = padded_data_arr.reshape(nskew_to_match_data, npix_sim_skew)
    return new_data_arr

def forward_model_onespec_chunk(vel_data, norm_std, master_mask, vel_lores, flux_lores, nmock, seed=None, std_corr=1.0):

    rand = np.random.RandomState(seed) if seed is None else seed

    npix_sim_skew = len(vel_lores)
    tot_nyx_skews = len(flux_lores)

    # first run to get nskew in order to initialize arrays
    ranindx, nskew_to_match_data = rand_skews_to_match_data(vel_lores, vel_data, tot_nyx_skews, seed=None)

    flux_noise_ncopy = np.zeros((nmock, nskew_to_match_data, npix_sim_skew))
    flux_noiseless_ncopy = np.zeros((nmock, nskew_to_match_data, npix_sim_skew))
    master_mask_chunk = reshape_data_array(master_mask, nskew_to_match_data, npix_sim_skew, data_arr_is_mask=True)

    norm_std[norm_std < 0] = 100  # get rid of negative errors

    for icopy in range(nmock):
        ranindx, nskew_to_match_data = rand_skews_to_match_data(vel_lores, vel_data, tot_nyx_skews, seed=seed)
        #print("random skewers", ranindx)
        noise = rand.normal(0, std_corr * norm_std) # sample noise vector assuming noise is Gaussian # (npix_data)
        noise_chunk = reshape_data_array(noise, nskew_to_match_data, npix_sim_skew, data_arr_is_mask=False) # (nskew, npix)
        #print("noise", noise[100:120])

        # adding noise to flux lores
        flux_noise_ncopy[icopy] = flux_lores[ranindx] + noise_chunk
        flux_noiseless_ncopy[icopy] = flux_lores[ranindx]

    return vel_lores, flux_noise_ncopy, flux_noiseless_ncopy, master_mask_chunk, nskew_to_match_data, npix_sim_skew

def compute_cf_onespec_chunk_ivarweights(vel_lores, noisy_flux_lores_ncopy, given_bins, weights_in=None, mask_chunk=None):
    # "mask" is any mask, not necessarily cgm mask; need to be of shape (nskew, npix), use the reshape_data_array() function above

    vmin_corr, vmax_corr, dv_corr = 0, 0, 0  # dummy values since using custom binning, but still required arguments for compute_xi

    ncopy, nskew, npix = np.shape(noisy_flux_lores_ncopy)
    # compress ncopy and nskew dimensions together because compute_xi() only takes in 2D arrays
    # (tried keeping the original 3D array and looping over ncopy when compute_xi(), but loop process too slow, sth like 1.4 min per 100 copy)
    reshaped_flux = np.reshape(noisy_flux_lores_ncopy, (ncopy * nskew, npix))

    if mask_chunk is not None:
        if mask_chunk.shape == (ncopy, nskew, npix):
            mask_ncopy = mask_chunk
        else:
            print('tiling mask_chunk')
            mask_ncopy = np.tile(mask_chunk, (ncopy, 1, 1)) # copy mask "ncopy" times

        mask_ncopy = np.reshape(mask_ncopy, (ncopy * nskew, npix))
        mean_flux = np.nanmean(reshaped_flux[mask_ncopy])
    else:
        mask_ncopy = None
        mean_flux = np.nanmean(reshaped_flux)

    if weights_in is not None:
        weights_in_ncopy = np.tile(weights_in, (ncopy, 1, 1))
        weights_in_ncopy = np.reshape(weights_in_ncopy, (ncopy * nskew, npix))
        weights_in = weights_in_ncopy

    delta_f = (reshaped_flux - mean_flux)/mean_flux
    start = time.process_time()
    (vel_mid, xi_mock, w_xi, xi_mock_zero_lag) = utils.compute_xi_weights(delta_f, vel_lores, vmin_corr, vmax_corr, dv_corr, \
                                                                     given_bins=given_bins, gpm=mask_ncopy, weights_in=weights_in)
    end = time.process_time()
    print("compute_xi_weights", end-start)

    # reshape xi_mock into the original input shape
    xi_mock = np.reshape(xi_mock, (ncopy, nskew, len(vel_mid)))
    w_xi = np.reshape(w_xi, (ncopy, nskew, len(vel_mid))) #npix_xi = np.reshape(npix_xi, (ncopy, nskew, len(vel_mid)))

    return vel_mid, xi_mock, w_xi

def init_dataset(nqso, redshift_bin, datapath):

    vel_data_allqso = []
    norm_flux_allqso = []
    norm_ivar_allqso = []
    norm_std_allqso = []
    master_mask_allqso = []
    master_mask_allqso_mask_cgm = []

    instr_allqso = ['nires', 'nires', 'nires', 'mosfire', 'mosfire', 'mosfire', 'mosfire', 'mosfire', 'nires', 'xshooter']

    for iqso in range(nqso):
        raw_data_out, _, all_masks_out = mutils.init_onespec(iqso, redshift_bin, datapath=datapath)
        wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
        strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, telluric_gpm, master_mask = all_masks_out

        norm_flux = flux / fluxfit
        norm_std = std / fluxfit
        norm_ivar = ivar * (fluxfit ** 2)
        vel_data = mutils.obswave_to_vel_2(wave)

        # do not apply any mask before CGM masking to keep the data dimension the same
        # for the purpose of forward modeling; all masks will be applied during the CF computation
        do_not_apply_any_mask = True

        if do_not_apply_any_mask:
            masks_for_cgm_masking = np.ones_like(wave, dtype=bool)
        else:
            masks_for_cgm_masking = mask * redshift_mask * pz_mask * zbin_mask * telluric_gpm

        # masked quantities
        norm_good_flux = norm_flux[masks_for_cgm_masking]
        norm_good_ivar = norm_ivar[masks_for_cgm_masking]
        good_vel_data = vel_data[masks_for_cgm_masking]

        # reshaping to be compatible with MgiiFinder
        norm_good_flux = norm_good_flux.reshape((1, len(norm_good_flux)))
        norm_good_ivar = norm_good_ivar.reshape((1, len(norm_good_ivar)))
        fwhm = qso_fwhm[iqso]

        # J1120+0641
        #if iqso == 9:
        #    signif_mask_nsigma = 2.05

        mgii_tot = MgiiFinder(good_vel_data, norm_good_flux, norm_good_ivar, fwhm, signif_thresh,
                              signif_mask_nsigma=signif_mask_nsigma,
                              signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh)

        # J1120+0641
        if iqso == 9:
            print("masking absorbers from Bosman et al. 2017")
            _, abs_mask_gpm = mask_cgm_pdf.bosman_J1120([4, 4, 3.5], datapath=datapath)
            gpm_allspec = mgii_tot.fit_gpm[0] * abs_mask_gpm
        else:
            gpm_allspec = mgii_tot.fit_gpm[0]

        #######
        tmp_mask = pz_mask * zbin_mask

        norm_std_allqso.append(norm_std[tmp_mask])
        vel_data_allqso.append(vel_data[tmp_mask])
        norm_flux_allqso.append(norm_flux[tmp_mask])
        norm_ivar_allqso.append(norm_ivar[tmp_mask])
        master_mask_allqso.append(master_mask[tmp_mask])
        master_mask_allqso_mask_cgm.append((master_mask * gpm_allspec)[tmp_mask])
        """
        boost = 1.28
        npix_more = int(len(vel_data[tmp_mask])*boost - len(vel_data[tmp_mask]))

        new_vel = vel_data[tmp_mask].tolist() + np.arange(vel_data[tmp_mask][-1] + 40, (vel_data[tmp_mask][-1] + 40) + 40*npix_more, 40).tolist()
        #new_vel = np.arange(vel_data[tmp_mask][0], vel_data[0] + 40*int(len(vel_data[tmp_mask])*1.3), 40)
        new_norm_std = norm_std[tmp_mask].tolist() + norm_std[tmp_mask][0:npix_more].tolist()
        new_norm_flux = norm_flux[tmp_mask].tolist() + norm_flux[tmp_mask][0:npix_more].tolist()
        new_norm_ivar = norm_ivar[tmp_mask].tolist() + norm_ivar[tmp_mask][0:npix_more].tolist()
        new_master_mask = master_mask[tmp_mask].tolist() + master_mask[tmp_mask][0:npix_more].tolist()
        new_master_mask_cgm = (master_mask * gpm_allspec)[tmp_mask].tolist() + (master_mask * gpm_allspec)[tmp_mask][0:npix_more].tolist()

        norm_std_allqso.append(np.array(new_norm_std))
        vel_data_allqso.append(np.array(new_vel))
        norm_flux_allqso.append(np.array(new_norm_flux))
        norm_ivar_allqso.append(np.array(new_norm_ivar))
        master_mask_allqso.append(np.array(new_master_mask))
        master_mask_allqso_mask_cgm.append(np.array(new_master_mask_cgm))
        """

    return vel_data_allqso, norm_flux_allqso, norm_std_allqso, norm_ivar_allqso, \
           master_mask_allqso, master_mask_allqso_mask_cgm, instr_allqso

def mock_mean_covar(ncovar, nmock, vel_data_allqso, norm_std_allqso, master_mask_allqso, instr_allqso, \
                    vel_lores_nires_interp, flux_lores_nires_interp, vel_lores_mosfire_interp, flux_lores_mosfire_interp, \
                    vel_lores_xshooter_interp, flux_lores_xshooter_interp, given_bins, seed=None):

    rand = np.random.RandomState(seed) if seed is None else seed

    xi_mock_ncopy = []
    xi_mock_ncopy_noiseless = []
    w_mock_ncopy = []
    w_mock_ncopy_noiseless = []
    w_mock_nskew_ncopy_allqso = []

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
        elif instr == 'xshooter':
            vel_lores = vel_lores_xshooter_interp
            flux_lores = flux_lores_xshooter_interp

        # generate mock data spectrum (0.3 sec per qso for ncovar x nskew = 1000 x 10 = 10,000 on my Mac)
        start = time.process_time()
        vel_lores, flux_noise_ncopy, flux_noiseless_ncopy, master_mask_chunk, nskew_to_match_data, npix_sim_skew = \
            forward_model_onespec_chunk(vel_data, norm_std, master_mask, vel_lores, flux_lores, nmock, seed=rand,
                                        std_corr=std_corr)
        end = time.process_time()
        print("      iqso = ", iqso)
        print("      nskew = ", nskew_to_match_data)
        print("         forward models done in .... ", (end - start))# / 60, " min")

        norm_std_chunk = reshape_data_array(std_corr * norm_std, nskew_to_match_data, npix_sim_skew, data_arr_is_mask=False)
        norm_ivar_chunk = 1 / (norm_std_chunk ** 2)
        weights_in = norm_ivar_chunk

        # new (2/2023): applying CGM flux mask to simulated data to ensure simulated data are masked similarly
        # applying the flux mask ensures the flux PDF of simulated matches real data
        m = (1 - flux_noise_ncopy) < 0.3 # 0.3 is the 1-F cutoff for masking CGM
        master_mask_chunk_tile = np.tile(master_mask_chunk, (nmock, 1, 1)) * m

        # compute the 2PCF (5 sec per qso for ncovar x nskew = 1000 x 10 = 10,000 skewers on my Mac)
        start = time.process_time()
        vel_mid, xi_onespec_ncopy, w_xi = \
            compute_cf_onespec_chunk_ivarweights(vel_lores, flux_noise_ncopy, given_bins, weights_in=weights_in, mask_chunk=master_mask_chunk_tile) #mask_chunk=master_mask_chunk)
        end = time.process_time()
        print("         compute_cf_onespec_chunk_ivarweights done in .... ", (end - start))# / 60, " min")

        start = time.process_time()
        vel_mid, xi_onespec_ncopy_noiseless, w_xi_noiseless = \
            compute_cf_onespec_chunk_ivarweights(vel_lores, flux_noiseless_ncopy, given_bins, weights_in=weights_in, mask_chunk=master_mask_chunk_tile) #mask_chunk=master_mask_chunk)
        end = time.process_time()
        print("         compute_cf_onespec_chunk_ivarweights done in .... ", (end - start))# / 60, " min")

        w_xi /= 1e5
        w_xi_noiseless /= 1e5
        xi_mock_onespec_ncopy = np.average(xi_onespec_ncopy, axis=1, weights=w_xi)  # averaging over nskew to get CF of onespec
        xi_mock_onespec_noiseless_ncopy = np.average(xi_onespec_ncopy_noiseless, axis=1, weights=w_xi_noiseless)  # same for noiseless onespec

        xi_mock_ncopy.append(xi_mock_onespec_ncopy)
        xi_mock_ncopy_noiseless.append(xi_mock_onespec_noiseless_ncopy)
        w_mock_ncopy.append(np.sum(w_xi, axis=1)) # summing the weights over nskew axis
        w_mock_ncopy_noiseless.append(np.sum(w_xi_noiseless, axis=1)) # summing the weights over nskew axis

        w_mock_nskew_ncopy_allqso.append(w_xi)

        print(np.sum(w_xi, axis=1)[0])

    # cast all into np arrays
    vel_mid = np.array(vel_mid)
    xi_mock_ncopy = np.array(xi_mock_ncopy) # (nqso, ncopy, ncorr)
    xi_mock_ncopy_noiseless = np.array(xi_mock_ncopy_noiseless)

    w_mock_ncopy = np.array(w_mock_ncopy) # (nqso, ncopy, ncorr)
    w_mock_ncopy_noiseless = np.array(w_mock_ncopy_noiseless)  # (nqso, ncopy, ncorr)

    # average over nqso with noiseless CF, then average over ncopy, to get model CF
    xi_mean_ncopy = np.average(xi_mock_ncopy_noiseless, axis=0, weights=w_mock_ncopy_noiseless)  # (ncopy, ncorr)
    xi_mean = np.mean(xi_mean_ncopy, axis=0)  # average over ncopy

    nqso, nmock, ncorr = np.shape(xi_mock_ncopy)
    ncorr = xi_mean.shape[0]
    covar = np.zeros((ncorr, ncorr))
    xi_mock_keep = np.zeros((nmock, ncorr))

    """
    xi_mean_ncopy2 = []
    for icovar in range(ncovar):
        xi_mock_ncopy_noiseless2 = []  # (nqso, ncopy, ncorr)
        w_mock_ncopy_noiseless2 = []

        ran_imock = rand.choice(nmock, replace=True, size=8)
        for i in range(8):
            xi_mock_ncopy_noiseless2.append(xi_mock_ncopy_noiseless[i][ran_imock[i]])
            w_mock_ncopy_noiseless2.append(w_mock_ncopy_noiseless[i][ran_imock[i]])

        xitmp = np.average(xi_mock_ncopy_noiseless2, axis=0, weights=w_mock_ncopy_noiseless2)
        xi_mean_ncopy2.append(xitmp) # ncovar, corr

    print("====== xi_mean_ncopy2", np.shape(xi_mean_ncopy2))
    xi_mean2 = np.mean(xi_mean_ncopy2, axis=0)
    """
    # new (3/2023): including mask to certain CF bins
    lag_mask = mutils.cf_lags_to_mask()
    lag_mask = np.ones_like(vel_mid, dtype=bool)

    covar = np.zeros((np.sum(lag_mask), np.sum(lag_mask)))
    xi_mock_keep = np.zeros((nmock, np.sum(lag_mask)))
    xi_mean = xi_mean[lag_mask]
    vel_mid = vel_mid[lag_mask]

    start = time.process_time()
    for icovar in range(ncovar):
        xi = []
        w = []
        ran_imock = rand.choice(nmock, replace=True, size=nqso)
        for i in range(nqso):
            xi.append(xi_mock_ncopy[i][ran_imock[i]])
            w.append(w_mock_ncopy[i][ran_imock[i]])

        xi_mock_icovar = np.average(xi, axis=0, weights=w)
        xi_mock_icovar = xi_mock_icovar[lag_mask]
        delta_xi = xi_mock_icovar - xi_mean
        covar += np.outer(delta_xi, delta_xi)  # off-diagonal elements

        if icovar < nmock:
            xi_mock_keep[icovar, :] = xi_mock_icovar

    # Divid by ncovar since we estimated the mean from "independent" data
    covar /= ncovar
    end = time.process_time()
    print("         ncovar loop done in .... ", (end - start))  # / 60, " min") # 1 min

    # weights = np.array(w_mock_ncopy / np.sum(w_mock_ncopy, axis=0))
    return xi_mock_keep, covar, vel_mid, xi_mean, w_mock_ncopy, w_mock_ncopy_noiseless, w_mock_nskew_ncopy_allqso

########################## running the grid #############################
def compute_model(args):
    # compute CF and covariance of mock dataset at each point of model grid
    # args: tuple of arguments from parser

    ihi, iZ, xHI, logZ, master_seed, xhi_path, zstr, redshift_bin, ncovar, nmock, vel_data_allqso, norm_std_allqso, master_mask_allqso, instr_allqso = args
    rantaufile = os.path.join(xhi_path, 'ran_skewers_' + zstr + '_OVT_' + 'xHI_{:4.2f}'.format(xHI) + '_tau.fits')
    params = Table.read(rantaufile, hdu=1)
    skewers = Table.read(rantaufile, hdu=2)

    rand = np.random.RandomState(master_seed)

    start = time.process_time()
    # NIRES fwhm and sampling (18 sec for 10,000 skewers on my Mac)
    vel_lores_nires, flux_lores_nires = utils.create_mgii_forest(params, skewers, logZ, nires_fwhm, sampling=nires_sampling, mockcalc=True)
    end = time.process_time()
    print("      NIRES mocks done in .... ", (end - start))# / 60, " min")

    start = time.process_time()
    # MOSFIRE fwhm and sampling
    vel_lores_mosfire, flux_lores_mosfire = utils.create_mgii_forest(params, skewers, logZ, mosfire_fwhm, sampling=mosfire_sampling, mockcalc=True)
    end = time.process_time()
    print("      MOSFIRE mocks done in .... ", (end - start))# / 60, " min")

    # xshooter fwhm and sampling
    vel_lores_xshooter, flux_lores_xshooter = utils.create_mgii_forest(params, skewers, logZ, xshooter_fwhm, sampling=xshooter_sampling, mockcalc=True)
    del skewers

    # interpolate flux lores to dv=40 (nyx); ~0.13 sec for 10,000 skewers on my Mac
    start = time.process_time()
    dv_coarse = 40
    vel_lores_nires_interp = np.arange(vel_lores_nires[0], vel_lores_nires[-1], dv_coarse)
    flux_lores_nires_interp = scipy.interpolate.interp1d(vel_lores_nires, flux_lores_nires, kind = 'cubic', \
                                                        bounds_error = False, fill_value = np.nan)(vel_lores_nires_interp)
    del vel_lores_nires
    del flux_lores_nires

    vel_lores_mosfire_interp = np.arange(vel_lores_mosfire[0], vel_lores_mosfire[-1], dv_coarse)
    flux_lores_mosfire_interp = scipy.interpolate.interp1d(vel_lores_mosfire, flux_lores_mosfire, kind='cubic', \
                                                           bounds_error=False, fill_value=np.nan)(vel_lores_mosfire_interp)

    del vel_lores_mosfire
    del flux_lores_mosfire

    vel_lores_xshooter_interp = np.arange(vel_lores_xshooter[0], vel_lores_xshooter[-1], dv_coarse)
    flux_lores_xshooter_interp = scipy.interpolate.interp1d(vel_lores_xshooter, flux_lores_xshooter, kind='cubic', \
                                                            bounds_error=False, fill_value=np.nan)(vel_lores_xshooter_interp)
    del vel_lores_xshooter
    del flux_lores_xshooter

    end = time.process_time()
    print("      interpolating both mocks done in .... ", (end - start))# / 60, " min")

    start = time.process_time()
    xi_mock_keep, covar, vel_mid, xi_mean, w_mock_ncopy, w_mock_ncopy_noiseless, w_mock_nskew_ncopy_allqso = \
        mock_mean_covar(ncovar, nmock, vel_data_allqso, norm_std_allqso, master_mask_allqso, instr_allqso, \
                    vel_lores_nires_interp, flux_lores_nires_interp, vel_lores_mosfire_interp, flux_lores_mosfire_interp, \
                        vel_lores_xshooter_interp, flux_lores_xshooter_interp, given_bins, seed=rand)
    #icovar = np.linalg.inv(covar)  # invert the covariance matrix
    icovar = np.zeros(covar.shape)
    sign, logdet = np.linalg.slogdet(covar)  # compute the sign and natural log of the determinant of the covariance matrix
    end = time.process_time()
    print("      mock_mean_covar done in .... ", (end - start))# / 60, " min")

    return ihi, iZ, vel_mid, xi_mock_keep, xi_mean, covar, icovar, logdet, w_mock_ncopy, w_mock_ncopy_noiseless, w_mock_nskew_ncopy_allqso


def test_compute_model():

    ihi, iZ = 0, 0 # dummy variables since they are simply returned
    xhi_path = '/Users/suksientie/Research/MgII_forest'
    #xhi_path = '/mnt/quasar/joe/reion_forest/Nyx_output/z75/xHI/' # on IGM cluster
    zstr = 'z75'
    xHI = 0.50 #0.74
    ncovar = 1000000
    nmock = 1000
    master_seed = 99991
    logZ = -4.50
    redshift_bin = 'all'

    vel_data_allqso, norm_flux_allqso, norm_std_allqso, norm_ivar_allqso, master_mask_allqso, master_mask_allqso_mask_cgm, instr_allqso = init_dataset(nqso_to_use, redshift_bin, datapath)
    args = ihi, iZ, xHI, logZ, master_seed, xhi_path, zstr, redshift_bin, ncovar, nmock, vel_data_allqso, norm_std_allqso, master_mask_allqso_mask_cgm, instr_allqso
    output = compute_model(args)
    #ihi, iZ, vel_mid, xi_mock_keep, xi_mean, covar, icovar, logdet, w_mock_ncopy, w_mock_ncopy_noiseless, w_mock_nskew_ncopy_allqso = output

    return output

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
    ncovar = args.ncovar
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
    #xhi_val, xhi_boxes = utils.read_xhi_boxes() # len(xhi_val) = 51, with d_xhi = 0.02
    #xhi_val = xhi_val[0:2] # testing for production run
    xhi_val = np.array([0.2, 0.9])
    nhi = xhi_val.shape[0]

    # Some file paths and then read in the params table to get the redshift
    zstr = 'z75'
    outpath = '/mnt/quasar/sstie/MgII_forest/' + zstr + '/10qso/'
    #outpath = '/Users/suksientie/Research/MgII_forest/'
    outfilename = 'corr_func_models_{:s}'.format(redshift_bin) + '_ivarweights.fits'
    outfile = os.path.join(outpath, outfilename)

    #xhi_path = '/mnt/quasar/joe/reion_forest/Nyx_output/z75/xHI/'
    xhi_path = '/Users/suksientie/Research/MgII_forest'
    files = glob.glob(os.path.join(xhi_path, '*_tau.fits'))
    params = Table.read(files[0], hdu=1)

    vel_data_allqso, norm_flux_allqso, norm_std_allqso, norm_ivar_allqso, master_mask_allqso, master_mask_allqso_mask_cgm, instr_allqso = init_dataset(nqso_to_use, redshift_bin, datapath)

    #args = xhi_path, zstr, redshift_bin, ncovar, nmock, vel_data_allqso, norm_std_allqso, master_mask_allqso, instr_allqso
    args = xhi_path, zstr, redshift_bin, ncovar, nmock, vel_data_allqso, norm_std_allqso, master_mask_allqso_mask_cgm, instr_allqso
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
    ihi, iZ, vel_mid, xi_mock, xi_mean, covar, icovar, logdet, _, _, _ = output[0] # ihi and iZ not actually used and simply returned as outputs
    ncorr = vel_mid.shape[0]
    xi_mock_array = np.zeros((nhi, nlogZ,) + xi_mock.shape)
    xi_mean_array = np.zeros((nhi, nlogZ,) + xi_mean.shape)
    covar_array = np.zeros((nhi, nlogZ,) + covar.shape)
    icovar_array = np.zeros((nhi, nlogZ,) + icovar.shape)
    lndet_array = np.zeros((nhi, nlogZ))

    # Unpack the output
    for out in output:
        ihi, iZ, vel_mid, xi_mock, xi_mean, covar, icovar, logdet, _, _, _ = out
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


###################### old scripts ######################
def old_init_dataset(nqso, redshift_bin, datapath):

    vel_data_allqso = []
    norm_flux_allqso = []
    norm_ivar_allqso = []
    norm_std_allqso = []
    master_mask_allqso = []
    master_mask_allqso_mask_cgm = []

    instr_allqso = ['nires', 'nires', 'nires', 'mosfire', 'mosfire', 'mosfire', 'mosfire', 'mosfire']

    for iqso in range(nqso):
        raw_data_out, _, all_masks_out = mutils.init_onespec(iqso, redshift_bin, datapath=datapath)
        wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
        strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, telluric_gpm, master_mask = all_masks_out

        norm_flux = flux / fluxfit
        norm_std = std / fluxfit
        norm_ivar = ivar * (fluxfit ** 2)
        vel_data = mutils.obswave_to_vel_2(wave)

        norm_std_allqso.append(norm_std)
        vel_data_allqso.append(vel_data)
        norm_flux_allqso.append(norm_flux)
        norm_ivar_allqso.append(norm_ivar)

        # do not apply any mask before CGM masking to keep the data dimension the same
        # for the purpose of forward modeling; all masks will be applied during the CF computation
        do_not_apply_any_mask = True

        if do_not_apply_any_mask:
            masks_for_cgm_masking = np.ones_like(wave, dtype=bool)
        else:
            masks_for_cgm_masking = mask * redshift_mask * pz_mask * zbin_mask * telluric_gpm

        # masked quantities
        norm_good_flux = norm_flux[masks_for_cgm_masking]
        norm_good_ivar = norm_ivar[masks_for_cgm_masking]
        good_vel_data = vel_data[masks_for_cgm_masking]

        # reshaping to be compatible with MgiiFinder
        norm_good_flux = norm_good_flux.reshape((1, len(norm_good_flux)))
        norm_good_ivar = norm_good_ivar.reshape((1, len(norm_good_ivar)))
        fwhm = qso_fwhm[iqso]

        mgii_tot = MgiiFinder(good_vel_data, norm_good_flux, norm_good_ivar, fwhm, signif_thresh,
                              signif_mask_nsigma=signif_mask_nsigma,
                              signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh)
        gpm_allspec = mgii_tot.fit_gpm[0]

        master_mask_allqso.append(master_mask)
        master_mask_allqso_mask_cgm.append(master_mask * gpm_allspec)

    #return vel_data_allqso, norm_std_allqso, master_mask_allqso, instr_allqso, norm_flux_allqso
    return vel_data_allqso, norm_flux_allqso, norm_std_allqso, norm_ivar_allqso, \
           master_mask_allqso, master_mask_allqso_mask_cgm, instr_allqso

def old_mock_mean_covar(ncovar, nmock_to_save, vel_data_allqso, norm_std_allqso, master_mask_allqso, instr_allqso, \
                    vel_lores_nires_interp, flux_lores_nires_interp, vel_lores_mosfire_interp, flux_lores_mosfire_interp, \
                    given_bins, seed=None):

    rand = np.random.RandomState(seed) if seed is None else seed

    xi_mock_ncopy = []
    xi_mock_ncopy_noiseless = []
    w_mock_ncopy = []
    w_mock_ncopy_noiseless = []
    w_mock_nskew_ncopy_allqso = []

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

        # generate mock data spectrum (0.3 sec per qso for ncovar x nskew = 1000 x 10 = 10,000 on my Mac)
        start = time.process_time()
        vel_lores, flux_noise_ncopy, flux_noiseless_ncopy, master_mask_chunk, nskew_to_match_data, npix_sim_skew = \
            forward_model_onespec_chunk(vel_data, norm_std, master_mask, vel_lores, flux_lores, ncovar, seed=rand,
                                        std_corr=std_corr)
        end = time.process_time()
        print("      iqso = ", iqso)
        print("         forward models done in .... ", (end - start))# / 60, " min")

        norm_std_chunk = reshape_data_array(std_corr * norm_std, nskew_to_match_data, npix_sim_skew, data_arr_is_mask=False)
        norm_ivar_chunk = 1 / (norm_std_chunk ** 2)
        weights_in = None #norm_ivar_chunk

        # compute the 2PCF (5 sec per qso for ncovar x nskew = 1000 x 10 = 10,000 skewers on my Mac)
        start = time.process_time()
        vel_mid, xi_onespec_ncopy, w_xi = \
            compute_cf_onespec_chunk_ivarweights(vel_lores, flux_noise_ncopy, given_bins, weights_in=weights_in, mask_chunk=master_mask_chunk)
        end = time.process_time()
        print("         compute_cf_onespec_chunk_ivarweights done in .... ", (end - start))# / 60, " min")

        start = time.process_time()
        vel_mid, xi_onespec_ncopy_noiseless, w_xi_noiseless = \
            compute_cf_onespec_chunk_ivarweights(vel_lores, flux_noiseless_ncopy, given_bins, weights_in=weights_in, mask_chunk=master_mask_chunk)
        end = time.process_time()
        print("         compute_cf_onespec_chunk_ivarweights done in .... ", (end - start))# / 60, " min")

        xi_mock_onespec_ncopy = np.average(xi_onespec_ncopy, axis=1, weights=w_xi)  # averaging over nskew to get CF of onespec
        xi_mock_onespec_noiseless_ncopy = np.average(xi_onespec_ncopy_noiseless, axis=1, weights=w_xi_noiseless)  # same for noiseless onespec

        xi_mock_ncopy.append(xi_mock_onespec_ncopy)
        xi_mock_ncopy_noiseless.append(xi_mock_onespec_noiseless_ncopy)
        w_mock_ncopy.append(np.sum(w_xi, axis=1)) # summing the weights over nskew axis
        w_mock_ncopy_noiseless.append(np.sum(w_xi_noiseless, axis=1)) # summing the weights over nskew axis

        w_mock_nskew_ncopy_allqso.append(w_xi)

    # cast all into np arrays
    vel_mid = np.array(vel_mid)
    xi_mock_ncopy = np.array(xi_mock_ncopy) # (nqso, ncopy, ncorr)
    xi_mock_ncopy_noiseless = np.array(xi_mock_ncopy_noiseless)
    scale = 1#1e6
    w_mock_ncopy = np.array(w_mock_ncopy)/scale # (nqso, ncopy, ncorr)
    w_mock_ncopy_noiseless = np.array(w_mock_ncopy_noiseless)/scale  # (nqso, ncopy, ncorr)

    # average over nqso to get mean CF for a dataset
    xi_mock_mean_ncopy = np.average(xi_mock_ncopy, axis=0, weights=w_mock_ncopy) # (ncopy, ncorr)

    # average over nqso with noiseless CF, then average over ncopy, to get model CF
    xi_mean_ncopy = np.average(xi_mock_ncopy_noiseless, axis=0, weights=w_mock_ncopy_noiseless) # (ncopy, ncorr)
    xi_mean = np.mean(xi_mean_ncopy, axis=0) # average over ncopy

    ncorr = xi_mean.shape[0]
    covar = np.zeros((ncorr, ncorr))
    xi_mock_keep = np.zeros((nmock_to_save, ncorr))

    for imock in range(ncovar):
        delta_xi = xi_mock_mean_ncopy[imock] - xi_mean
        covar += np.outer(delta_xi, delta_xi)  # off-diagonal elements
        if imock < nmock_to_save:
            xi_mock_keep[imock, :] = xi_mock_mean_ncopy[imock]

    # Divid by ncovar since we estimated the mean from "independent" data
    covar /= ncovar

    # weights = np.array(w_mock_ncopy / np.sum(w_mock_ncopy, axis=0))
    return xi_mock_keep, covar, vel_mid, xi_mean, w_mock_ncopy, w_mock_ncopy_noiseless, w_mock_nskew_ncopy_allqso

def compute_cf_onespec_chunk(vel_lores, noisy_flux_lores_ncopy, given_bins, mask_chunk=None):
    # "mask" is any mask, not necessarily cgm mask; need to be of shape (nskew, npix), use the reshape_data_array() function above

    vmin_corr, vmax_corr, dv_corr = 0, 0, 0  # dummy values since using custom binning, but still required arguments for compute_xi

    ncopy, nskew, npix = np.shape(noisy_flux_lores_ncopy)
    # compress ncopy and nskew dimensions together because compute_xi() only takes in 2D arrays
    # (tried keeping the original 3D array and looping over ncopy when compute_xi(), but loop process too slow, sth like 1.4 min per 100 copy)
    reshaped_flux = np.reshape(noisy_flux_lores_ncopy, (ncopy * nskew, npix))

    if mask_chunk is not None:
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