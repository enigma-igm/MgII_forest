"""
Functions here:
    - imap_unordered_bar
    - read_model_grid
    - forward_model_onespec
    - forward_model_allspec
    - init_cgm_masking
    - compute_cf_onespec
    - compute_cf_allspec
    - mock_mean_covar
    - compute_model
    - test_compute_model
    - parser
    - main
"""

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

fitsfile_list = [datapath + 'wavegrid_vel/J0313-1806/vel123_coadd_tellcorr.fits', \
                 datapath + 'wavegrid_vel/J1342+0928/vel123_coadd_tellcorr.fits', \
                 datapath + 'wavegrid_vel/J0252-0503/vel12_coadd_tellcorr.fits', \
                 datapath + 'wavegrid_vel/J0038-1527/vel1_tellcorr_pad.fits']

qso_namelist = ['J0313-1806', 'J1342+0928', 'J0252-0503', 'J0038-1527']
qso_zlist = [7.6, 7.54, 7.0, 7.0]
everyn_break_list = [20, 20, 20, 20]

###################### helper functions ######################
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

#### functions for forward modelling ####
# the default mode is to "chunk", i.e. to assemble the required number of mock skewers WITHOUT combining them into one long skewer
# the reason for not combining into a long skewer is because of edge effects contributing to the 2PCF

import pdb
def forward_model_onespec_chunk(vel_data, master_mask_data, norm_std, vel_lores, flux_lores, ncopy, seed=None, std_corr=1.0, fmean_corr=1.0):
    """
    Mock up an observed QSO spectrum using the required number of Nyx skewers.
    Important: code assumes dv of 'vel_data' matches exactly the dv of the Nyx skewers.

    Inputs:
    'vel_data' = raw vel_data, unmasked (numpy array)
    'master_mask_data' = redshift_mask * outmask (numpy array)
    'norm_std' = std / fluxfit_new (where fluxfit_new is padded fluxfit; see mutils.py), in order to force len(norm_std) = len(vel_data)
        ====> vel_data, master_mask_data, norm_std all have same shape

    'vel_lores' = velocity grid of the Nyx skewer
    'flux_lores' = 10,000 Nyx skewers of the mgii forest
    'ncopy' = number of forward model copies per qso (int)

    'seed' = (optional) random seed used to choose the subset of Nyx skewers and for sampling noise from data for all ncopy
    'std_corr' = (optional) correction to apply to the forward-modeled noise such that the flux PDF of the mock spectra resembles the data flux PDF
    'fmean_corr' = (optional) correction to apply to the mean flux of the forward models

    Outputs: outputs are not masked
    'flux_lores_rand' = shape is (nskew, npix_nyx), where all pixels beyond the data pathlength are set to Nan.
    'master_mask_chunk' = shape is (nskew, npix_nyx); reshaped version of input argument 'master_mask_data', where all pixels beyond data pathlength are set to False
    'norm_std_chunk' = shape is (nskew, npix_nyx); reshape version of input 'norm_std', where
    'flux_lores_rand_noise' = shape is (ncopy, nskew, npix_nyx)
        ====> where nskew is the number of Nyx skewers to match data pathlength and npix_nyx is the number of pixels for each nyx skewer
    """

    tot_vel_data = vel_data.max() - vel_data.min()
    tot_vel_sim = vel_lores.max() - vel_lores.min()
    nskew_to_match_data = int(np.ceil(tot_vel_data / tot_vel_sim))  # assuming tot_vel_data > tot_vel_sim (true here)
    npix_sim_skew = len(flux_lores[0]) # number of pixels in each Nyx skewer
    #print("nskew to match data", nskew_to_match_data)

    if seed != None:
        rand = np.random.RandomState(seed)
    else:
        rand = np.random.RandomState()

    indx_flux_lores = np.arange(flux_lores.shape[0])
    ranindx = rand.choice(indx_flux_lores, replace=False, size=nskew_to_match_data) # grab a set of random skewers
    flux_lores_rand = flux_lores[ranindx]

    # setting some pixels in the last skewer of "flux_lores_rand" to be nan, in order to match with len(vel_data)
    zeropoint_len_skewer = (nskew_to_match_data - 1)*npix_sim_skew # this gives the total pixels up to the 2nd-to-last skewer
    istart = len(vel_data) - zeropoint_len_skewer # all pixels after 'istart' of the last skewer will be set to NaN
    npix_to_set_as_nan = npix_sim_skew - istart # number of NaN pixels in the last skewer
    #print(len(flux_lores_rand[-1][istart:]), npix_to_set_as_nan)
    flux_lores_rand[-1][istart:] = np.nan*np.ones(npix_to_set_as_nan) # all pixels after 'istart' of the last skewer will be set to NaN

    assert np.sum(np.invert(np.isnan(flux_lores_rand.flatten()))) == len(vel_data) # assert that the total number of pixels on simulated spectrum = number of pixels from data

    # chunking master_mask to be the same shape/dimension as flux_lores_rand
    # also want to chunk "norm_std" in order to sample the noise
    master_mask_chunk = []
    norm_std_chunk = []

    for iskew in range(nskew_to_match_data):
        if iskew != nskew_to_match_data - 1:
            istart = npix_sim_skew*iskew
            iend = npix_sim_skew*(iskew+1)
            master_mask_chunk.append(master_mask_data[istart:iend])
            norm_std_chunk.append(norm_std[istart:iend])
        else:
            # the last skewer treated differently because need to set some pixels as false
            last_skew = list(master_mask_data[npix_sim_skew*iskew: len(master_mask_data)]) + list(np.zeros(npix_to_set_as_nan, dtype=bool))
            master_mask_chunk.append(last_skew)

            # the last skewer treated differently because need to set some pixels as Nan
            #last_skew = list(norm_std[npix_sim_skew * iskew: len(norm_std)]) + list(np.zeros(npix_to_set_as_nan, dtype=bool))
            last_skew = list(norm_std[npix_sim_skew * iskew: len(norm_std)]) + list(np.nan*np.zeros(npix_to_set_as_nan))
            norm_std_chunk.append(last_skew)

    # sampling the nosie
    flux_lores_rand_noise = []
    for icopy in range(ncopy):
        onecopy = []
        for iskew in range(nskew_to_match_data):
            # fmean_corr (optional) = correction to apply to the flux
            # std_corr (optional) = correction to apply to the noise
            onecopy.append(rand.normal(fmean_corr*flux_lores_rand[iskew], std_corr*np.array(norm_std_chunk[iskew])))
        flux_lores_rand_noise.append(onecopy)

    """
    for iskew in range(nskew_to_match_data):
        temp = []
        for ipix in range(len(norm_std_chunk[iskew])):
            if np.isnan(norm_std_chunk[iskew][ipix]):
                temp.append(np.nan*np.ones(ncopy))
            else:
                temp.append(rand.normal(flux_lores_rand[iskew][ipix], norm_std_chunk[iskew][ipix], ncopy))
        flux_lores_rand_noise.append(temp)
    """

    flux_lores_rand = np.array(flux_lores_rand)
    master_mask_chunk = np.array(master_mask_chunk)
    norm_std_chunk = np.array(norm_std_chunk)
    flux_lores_rand_noise = np.array(flux_lores_rand_noise)

    # note: outputs are NOT masked
    return flux_lores_rand, master_mask_chunk, norm_std_chunk, flux_lores_rand_noise

def plot_forward_model_onespec(flux_lores_rand_noise, master_mask_chunk, good_vel_data, norm_good_flux, norm_good_std, ncopy_plot, seed=None, title=None):

    ncopy, nskew, npix = np.shape(flux_lores_rand_noise)

    #master_mask_comb = []
    #for iskew in range(nskew):
    #    master_mask_comb.extend(master_mask_chunk[iskew])
    #master_mask_comb = np.array(master_mask_comb)

    plt.figure(figsize=(12, 8))
    if title != None:
        plt.title(title, fontsize=18)

    ##### plot subset of mock spectra
    for i in range(ncopy_plot):
        flux_lores_comb = (flux_lores_rand_noise[i])[master_mask_chunk] # this also flattens the 2d array
        plt.plot(good_vel_data, flux_lores_comb + (i + 1), alpha=0.5, drawstyle='steps-mid')

    plt.plot(good_vel_data, norm_good_flux, 'k', drawstyle='steps-mid')
    plt.ylabel('Flux (+ arbitrary offset)', fontsize=15)
    plt.xlabel('Velocity (km/s)', fontsize=15)
    plt.tight_layout()
    plt.close()

    ##### plot flux PDF
    plt.figure(figsize=(10, 8))
    if title != None:
        plt.title(title, fontsize=18)
    plt.hist(norm_good_flux, bins=np.arange(0, 3, 0.02), histtype='step', label='data', density=True)

    #gauss_noise = np.random.normal(np.mean(norm_good_flux), 1.0, len(norm_good_flux))
    #plt.hist(gauss_noise, bins=np.arange(0, 3, 0.02), histtype='step', label='Gaussian', density=True)

    # applying raw data mask to flux_lores_rand_noise, since we're comparing to "norm_good_flux"
    flux_masked = []
    for icopy in range(ncopy):
        flux_masked.append(flux_lores_rand_noise[icopy][master_mask_chunk])
    flux_masked = np.array(flux_masked)
    plt.hist(flux_masked.flatten(), bins=np.arange(0, 3, 0.02), histtype='step', label='sim (ncopy=%d), good pix' % ncopy, density=True)
    plt.xlabel('normalized flux', fontsize=15)
    plt.ylabel('PDF', fontsize=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    print("ratio", np.std(norm_good_flux) / np.nanstd(flux_masked.flatten()), np.std(norm_good_flux), np.nanstd(flux_masked.flatten()))
    plt.close()

    ##### plot (1 - F) PDF
    nbins, oneminf_min, oneminf_max = 101, 1e-5, 1.0  # gives d(oneminf) = 0.01
    if seed != None:
        rand = np.random.RandomState(seed)
    else:
        rand = np.random.RandomState()
    noise = []
    for i_std in norm_good_std:
        noise.append(rand.normal(0, i_std))
    noise = np.array(noise)

    flux_bins, flux_pdf_data = utils.pdf_calc(1.0 - norm_good_flux, oneminf_min, oneminf_max, nbins)
    _, flux_pdf_mock_all = utils.pdf_calc(1.0 - flux_masked.flatten(), oneminf_min, oneminf_max, nbins) # all mock spectra
    _, flux_pdf_noise = utils.pdf_calc(noise, oneminf_min, oneminf_max, nbins)

    plt.figure()
    if title != None:
        plt.title(title, fontsize=18)

    for i in range(ncopy_plot):
        flux_lores_comb = (flux_lores_rand_noise[i])[master_mask_chunk]  # this also flattens the 2d array
        _, flux_pdf_mock = utils.pdf_calc(1.0 - flux_lores_comb, oneminf_min, oneminf_max, nbins)
        plt.plot(flux_bins, flux_pdf_mock, drawstyle='steps-mid', color='g', lw=0.5, alpha=0.6) # individual mock spectrum

    plt.plot(flux_bins, flux_pdf_data, drawstyle='steps-mid', alpha=1.0, lw=2, label='data')
    plt.plot(flux_bins, flux_pdf_mock_all, drawstyle='steps-mid', color='g', alpha=1.0, lw=2, label='sim (ncopy=%d), good pix' % ncopy)
    plt.plot(flux_bins, flux_pdf_noise, drawstyle='steps-mid', color='k', alpha=1.0, lw=2, label='pure noise', zorder=1)

    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('1$-$F')
    plt.ylabel('PDF')
    plt.tight_layout()

def forward_model_allspec_chunk(vel_lores, flux_lores, ncopy, seed_list=[None, None, None, None], plot=False):

    flores_rand_all = []
    master_mask_chunk_all = []
    good_vel_data_all = []
    flores_rand_noise_all = []
    corr_all = [0.689, 0.640, 0.616, 0.583] # averages from 4 realizations of 1000 copies
    #corr_all = [1.0, 1.0, 1.0, 1.0]
    #fmean_corr_all = [1.0121, 1.009, 1.011, 1.011]

    # corr_fmean
    # np.mean(norm_good_flux), np.nanmean(frand_noise) = (1.0029245693433284, 0.9908431443268844)
    # np.mean(norm_good_flux), np.nanmean(frand_noise) = (1.0001004785756826, 0.9907494214391939)
    # (1.0023226315206812, 0.9914268534769404)
    # (1.002254910667389, 0.9910875060522094)
    for iqso, fitsfile in enumerate(fitsfile_list):
        wave, flux, ivar, mask, std, fluxfit, outmask, sset = mutils.extract_and_norm(fitsfile, everyn_break_list[iqso])
        vel_data = mutils.obswave_to_vel_2(wave)

        redshift_mask = wave <= (2800 * (1 + qso_zlist[iqso]))  # removing spectral region beyond qso redshift
        master_mask = redshift_mask * outmask

        good_wave = wave[master_mask]
        good_vel_data = mutils.obswave_to_vel_2(good_wave)

        fluxfit_redshift = fluxfit[wave[outmask] <= (2800 * (1 + qso_zlist[iqso]))]
        norm_good_flux = flux[master_mask]/ fluxfit_redshift
        norm_good_std = std[master_mask]/ fluxfit_redshift

        fluxfit_new = mutils.pad_fluxfit(outmask, fluxfit) # in order to make 'fluxfit' (masked) same shape as raw data
        norm_std = std / fluxfit_new

        flux_lores_rand, master_mask_chunk, norm_std_chunk, flux_lores_rand_noise = \
            forward_model_onespec_chunk(vel_data, master_mask, norm_std, vel_lores, flux_lores, ncopy, \
                                        seed=seed_list[iqso], std_corr=corr_all[iqso]) #, fmean_corr=fmean_corr_all[iqso])
        flores_rand_all.append(flux_lores_rand)
        master_mask_chunk_all.append(master_mask_chunk)
        good_vel_data_all.append(good_vel_data)
        flores_rand_noise_all.append(flux_lores_rand_noise)

        if plot:
            plot_forward_model_onespec(flux_lores_rand_noise, master_mask_chunk, good_vel_data, norm_good_flux, norm_good_std, 10, seed=seed_list[iqso], title=qso_namelist[iqso] + ' (z=%0.2f)' % qso_zlist[iqso])

    flores_rand_all = np.array(flores_rand_all)
    master_mask_chunk_all = np.array(master_mask_chunk_all)
    good_vel_data_all = np.array(good_vel_data)
    flores_rand_noise_all = np.array(flores_rand_noise_all)

    # note: outputs are NOT masked.
    # can now apply any type of masking onto these outputs
    #   -- to preserve ndarray when masking: masked_arr = np.where(master_mask_chunk, flores_rand_noise[0], np.nan)
    return flores_rand_all, master_mask_chunk_all, good_vel_data_all, flores_rand_noise_all

#### functions for initializing CGM masking ####
def init_cgm_masking(fwhm, signif_thresh=4.0, signif_mask_dv=300.0, signif_mask_nsigma=8, one_minF_thresh = 0.3):
    # returns good pixel mask from cgm masking for all 4 qsos
    gpm_allspec = []
    for iqso, fitsfile in enumerate(fitsfile_list):
        wave, flux, ivar, mask, std, fluxfit, outmask, sset = mutils.extract_and_norm(fitsfile, everyn_break_list[iqso])

        redshift_mask = wave[outmask] <= (2800 * (1 + qso_zlist[iqso])) # removing spectral region beyond qso redshift

        good_wave = wave[outmask][redshift_mask]
        good_flux = flux[outmask][redshift_mask]
        good_ivar = ivar[outmask][redshift_mask]
        norm_good_flux = good_flux / fluxfit[redshift_mask]

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

def chunk_gpm_onespec(gpm_onespec, nskew_to_match_data, npix_sim_skew):

    # setting some pixels in the last (or last n) skewer(s) to be nan
    gpm_onespec = gpm_onespec.squeeze()

    #zeropoint_len_skewer = (nskew_to_match_data - 1) * npix_sim_skew
    zeroindex_simskew = np.arange(nskew_to_match_data)*npix_sim_skew
    i_simskew = np.where(len(gpm_onespec) > zeroindex_simskew)[0][-1]
    zeropoint_len_skewer = zeroindex_simskew[i_simskew]
    #print(i_simskew, zeropoint_len_skewer, len(gpm_onespec))

    #istart = len(gpm_onespec) - zeropoint_len_skewer  # starting index to set nan on the "i_simskew"-th skewer
    #npix_to_set_as_nan = npix_sim_skew - istart

    gpm_chunk = []
    for iskew in range(nskew_to_match_data):
        #if iskew != nskew_to_match_data - 1:
        if iskew != i_simskew:
            if iskew > i_simskew:
                gpm_chunk.append(np.zeros(npix_sim_skew, dtype=bool))
            else:
                istart = npix_sim_skew * iskew
                iend = npix_sim_skew * (iskew + 1)
                gpm_chunk.append(gpm_onespec[istart:iend])

        elif iskew == i_simskew:
            istart = len(gpm_onespec) - zeropoint_len_skewer  # starting index to set nan on the "i_simskew"-th skewer
            npix_to_set_as_nan = npix_sim_skew - istart
            last_skew = list(gpm_onespec[npix_sim_skew * iskew: len(gpm_onespec)]) + list(np.zeros(npix_to_set_as_nan, dtype=bool))
            gpm_chunk.append(last_skew)

        #else:
        #    last_skew = list(gpm_onespec[npix_sim_skew * iskew: len(gpm_onespec)]) + list(np.zeros(npix_to_set_as_nan, dtype=bool))
        #    gpm_chunk.append(last_skew)

    gpm_chunk = np.array(gpm_chunk) # now same shape as master_mask_chunk
    assert np.sum(gpm_onespec) == np.sum(gpm_chunk)

    # final_master_mask = master_mask_chunk * gpm_chunk to combine data mask and cgm mask
    return gpm_chunk

#### functions for computing 2PCF ####
def compute_cf_onespec_chunk(vel_lores, flores_rand_noise, vmin_corr, vmax_corr, dv_corr, mask=None):

    # "mask" is any mask, not necessarily cgm mask

    ncopy, nskew, npix = np.shape(flores_rand_noise)
    # compress ncopy and nskew dimensions together because compute_xi() only takes in 2D arrays
    # (tried keeping the original 3D array and looping over ncopy when compute_xi(), but loop process too slow, sth like 1.4 min per 100 copy)
    reshaped_flux = np.reshape(flores_rand_noise, (ncopy * nskew, npix))

    # 'mask' needs to be of shape (nskew, npix)
    if type(mask) != type(None):
        mask2 = np.zeros((ncopy, nskew, npix)) # duplicate the mask for all ncopy
        for icopy in range(ncopy):
            mask2[icopy] = mask

        mask2 = np.reshape(mask2, (ncopy*nskew, npix)).astype(int) # reshaping the same way as above and recasting to int type
        reshaped_flux_masked = np.where(mask2, reshaped_flux, np.nan)
        mean_flux = np.nanmean(reshaped_flux_masked)

    else:
        mask2 = None
        mean_flux = np.nanmean(reshaped_flux)

    # due to the construction of xi_sum() in enigma.reion_forest.utils.py, which computes npix_sum using the input gpm,
    # this is technically the correct way to do this
    delta_f = (reshaped_flux - mean_flux) / mean_flux
    (vel_mid, xi_mock, npix_xi, xi_mock_zero_lag) = utils.compute_xi(delta_f, vel_lores, vmin_corr, vmax_corr, dv_corr, gpm=mask2)

    # this gives incorrect xi values because the number of masked pixels is not correctly accounted for,
    # if one does not set the 'gpm' in compute_xi.
    #delta_f = (reshaped_flux_masked - mean_flux) / mean_flux
    #(vel_mid, xi_mock, npix_xi, xi_mock_zero_lag) = utils.compute_xi(delta_f, vel_lores, vmin_corr, vmax_corr, dv_corr)

    # this gives the same result  as no. 1 (by virtue of the bad pixels being set to NaN)
    #delta_f = (reshaped_flux_masked - mean_flux) / mean_flux
    #(vel_mid, xi_mock, npix_xi, xi_mock_zero_lag) = utils.compute_xi(delta_f, vel_lores, vmin_corr, vmax_corr, dv_corr, gpm=mask2)

    # reshape xi_mock into the original input shape
    xi_mock = np.reshape(xi_mock, (ncopy, nskew, len(vel_mid)))

    return vel_mid, xi_mock, npix_xi

# too slow
import time
def compute_cf_onespec_loop(vel_lores, flores_rand_noise, vmin_corr, vmax_corr, dv_corr, cgm_masking_gpm=None):
    # ~0.45 min for 100 copy; so ~4.5 min for 1000 copy
    ncopy, nskew, npix = np.shape(flores_rand_noise)
    mean_flux = np.nanmean(flores_rand_noise)
    delta_f = (flores_rand_noise - mean_flux) / mean_flux

    start = time.time()
    xi_mock_ncopy = []
    for icopy in range(ncopy):
        (vel_mid, xi_mock, npix, xi_mock_zero_lag) = utils.compute_xi(delta_f[icopy], vel_lores, vmin_corr, vmax_corr, dv_corr, gpm=cgm_masking_gpm)
        xi_mock_ncopy.append(xi_mock)

    xi_mock_ncopy = np.array(xi_mock_ncopy)
    end = time.time()
    print("finish CF in ", (end-start)/60.)
    return vel_mid, xi_mock_ncopy

def compute_cf_allspec_chunk(forward_model_out, vel_lores, vmin_corr, vmax_corr, dv_corr, cgm_gpm_allspec=None):

    flores_rand_all, master_mask_chunk_all, good_vel_data_all, flores_rand_noise_all = forward_model_out
    nqso, ncopy, nskew, npix = np.shape(flores_rand_noise_all)

    xi_mock_all = []
    for iqso in range(len(flores_rand_noise_all)):
        if type(cgm_gpm_allspec) == type(None):
            mask = master_mask_chunk_all[iqso]
        else:
            gpm_chunk = chunk_gpm_onespec(cgm_gpm_allspec[iqso], nskew, npix)
            mask = master_mask_chunk_all[iqso]*gpm_chunk

        vel_mid, xi_mock, npix_xi = compute_cf_onespec_chunk(vel_lores, flores_rand_noise_all[iqso], vmin_corr, vmax_corr, dv_corr, mask=mask)
        xi_mock_all.append(xi_mock)

    vel_mid = np.array(vel_mid)
    xi_mock_all = np.array(xi_mock_all) # shape = (nqso, ncopy, nskew_to_match_data, npix)

    return vel_mid, xi_mock_all

def mock_mean_covar(xi_mean, ncopy, nmock, vel_lores, flux_lores, vmin_corr, vmax_corr, dv_corr, seed=None, cgm_gpm_allspec=None):
    # TODO: in progress, incorporate new changes

    # calls forward_model_allspec_chunk and compute_cf_allspec_chunk
    rand = np.random.RandomState() if seed is None else np.random.RandomState(seed)
    ncorr = xi_mean.shape[0]
    covar = np.zeros((ncorr, ncorr))

    seed_list = rand.randint(0, 1000000000, 4)  # 4 for 4 qsos, hardwired for now
    fm_out = forward_model_allspec_chunk(vel_lores, flux_lores, ncopy, seed_list=seed_list)
    vel_mid, xi_mock = compute_cf_allspec_chunk(fm_out, vel_lores, vmin_corr, vmax_corr, dv_corr, cgm_gpm_allspec=cgm_gpm_allspec)

    nqso, ncopy, nskew, npix = xi_mock.shape # note the shape

    xi_mock_avg_nskew = np.mean(xi_mock, axis=2) # average over nskew_to_match_data
    xi_mock_mean = np.mean(xi_mock_avg_nskew, axis=0) # average over nqso
    delta_xi = xi_mock_mean - xi_mean

    for icopy in range(ncopy):
        covar += np.outer(delta_xi[icopy], delta_xi[icopy])  # off-diagonal elements

    # xi_mock_keep = np.zeros((nmock, ncorr))
    #for icopy in range(ncopy):
    #    if icopy < nmock:
    #        xi_mock_keep[icopy, :] = xi_mock_mean
    xi_mock_keep = xi_mock_mean[:nmock]

    # Divid by ncovar since we estimated the mean from "independent" data
    covar /= ncopy

    return xi_mock_keep, covar

def compute_model(args):
    # TODO: in progress, incorporate new changes
    # compute CF and covariance of mock dataset at each point of model grid
    # args: tuple of arguments from parser

    ihi, iZ, xHI, logZ, seed, xhi_path, zstr, fwhm, sampling, vmin_corr, vmax_corr, dv_corr, ncopy, ncovar, nmock, cgm_masking_gpm = args
    rantaufile = os.path.join(xhi_path, 'ran_skewers_' + zstr + '_OVT_' + 'xHI_{:4.2f}'.format(xHI) + '_tau.fits')
    #rand = np.random.RandomState(seed)
    params = Table.read(rantaufile, hdu=1)
    skewers = Table.read(rantaufile, hdu=2)

    # Compute skewers. This takes 5.36s for 10,000 skewers
    vel_lores, (flux_lores, flux_lores_igm, flux_lores_cgm, _, _), vel_hires, (flux_hires, flux_hires_igm, flux_hires_cgm, _, _), \
    (oden, v_los, T, xHI), cgm_tuple = utils.create_mgii_forest(params, skewers, logZ, fwhm, sampling=sampling)

    # noiseless mean model
    mean_flux_nless = np.mean(flux_lores)
    delta_f_nless = (flux_lores - mean_flux_nless) / mean_flux_nless
    (vel_mid, xi_nless, npix, xi_nless_zero_lag) = utils.compute_xi(delta_f_nless, vel_lores, vmin_corr, vmax_corr,
                                                                    dv_corr)
    xi_mean = np.mean(xi_nless, axis=0)

    # forward modeled dataset, currently hardwired to take in the 4 QSOs in our current dataset
    #fm_out = forward_model_allspec(vel_lores, flux_lores, ncopy)
    #vel_mid, xi_mock = compute_cf_allspec(fm_out)
    #if cgm_masking:
    #    cgm_masking_gpm = init_cgm_masking(fwhm)
    #else:
    #    cgm_masking_gpm = None

    xi_mock_keep, covar = mock_mean_covar(xi_mean, ncopy, ncovar, nmock, vel_lores, flux_lores, vmin_corr, vmax_corr, dv_corr, seed=seed, cgm_masking_gpm=cgm_masking_gpm)
    icovar = np.linalg.inv(covar)  # invert the covariance matrix
    sign, logdet = np.linalg.slogdet(covar)  # compute the sign and natural log of the determinant of the covariance matrix

    return ihi, iZ, vel_mid, xi_mock_keep, xi_mean, covar, icovar, logdet

def test_compute_model():

    ihi, iZ = 0, 0 # dummy variables since they are simply returned
    xhi_path = '/Users/suksientie/Research/MgII_forest'
    zstr = 'z75'
    xHI = 0.50
    fwhm = 90
    sampling = 3
    vmin_corr = 10
    vmax_corr = 2000
    dv_corr = fwhm
    ncopy = 5
    ncovar = 10
    nmock = 10
    seed = 9999 # results in seed list [315203670 242427133 938891646 135124015]
    logZ = -3.5
    cgm_masking = True
    if cgm_masking:
        cgm_masking_gpm = init_cgm_masking(fwhm)
    else:
        cgm_masking_gpm = None

    args = ihi, iZ, xHI, logZ, seed, xhi_path, zstr, fwhm, sampling, vmin_corr, vmax_corr, dv_corr, ncopy, ncovar, nmock, cgm_masking_gpm

    out = compute_model(args)
    return out

###################### main functions ######################
def parser():
    # TODO: no need ncovar?
    import argparse

    parser = argparse.ArgumentParser(description='Create random skewers for MgII forest', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nproc', type=int, help="Number of processors to run on")
    parser.add_argument('--fwhm', type=float, default=90.0, help="spectral resolution in km/s")
    parser.add_argument('--samp', type=float, default=3.0, help="Spectral sampling: pixels per fwhm resolution element")
    #parser.add_argument('--SNR', type=float, default=100.0, help="signal-to-noise ratio")
    #parser.add_argument('--nqsos', type=int, default=10, help="number of qsos")
    #parser.add_argument('--delta_z', type=float, default=0.6, help="redshift pathlength per qso")
    parser.add_argument('--vmin', type=float, default=30.0, help="Minimum of velocity bins for correlation function")
    parser.add_argument('--vmax', type=float, default=3100, help="Maximum of velocity bins for correlation function")
    parser.add_argument('--dv', type=float, default=None, help="Width of velocity bins for correlation function. "
                                                               "If not set fwhm will be used")
    parser.add_argument('--ncopy', type=int, default=1000, help="number of forward-modeled spectra for each qso")
    parser.add_argument('--ncovar', type=int, default=1000000, help="number of mock datasets for computing covariance")
    parser.add_argument('--nmock', type=int, default=300, help="number of mock datasets to store")
    parser.add_argument('--seed', type=int, default=12349876, help="seed for random number generator")
    parser.add_argument('--nlogZ', type=int, default=201, help="number of bins for logZ models")
    parser.add_argument('--logZmin', type=float, default=-6.0, help="minimum logZ value")
    parser.add_argument('--logZmax', type=float, default=-2.0, help="maximum logZ value")
    parser.add_argument('--cgm_masking', action='store_true', help='whether to mask cgm or not')
    return parser.parse_args()

def main():

    # TODO: incorporate new changes
    args = parser()
    nproc = args.nproc
    fwhm = args.fwhm
    sampling = args.samp
    #SNR = args.SNR
    #nqsos = args.nqsos
    #delta_z = args.delta_z
    ncovar = args.ncovar
    nmock = args.nmock
    ncopy = args.ncopy
    seed = args.seed
    vmin_corr = args.vmin
    vmax_corr = args.vmax
    dv_corr = args.dv if args.dv is not None else fwhm
    cgm_masking = args.cgm_masking
    if cgm_masking:
        cgm_masking_gpm = init_cgm_masking(fwhm)
    else:
        cgm_masking_gpm = None

    #rand = np.random.RandomState(seed)
    # Grid of metallicities
    nlogZ = args.nlogZ
    logZ_min = args.logZmin
    logZ_max = args.logZmax
    logZ_vec = np.linspace(logZ_min, logZ_max, nlogZ)

    # Read grid of neutral fractions from the 21cm fast xHI fields
    xhi_val, xhi_boxes = utils.read_xhi_boxes()
    nhi = xhi_val.shape[0]
    xhi_val = xhi_val[0:3] # testing for production run

    # Some file paths and then read in the params table to get the redshift
    zstr = 'z75'
    outpath = '/mnt/quasar/sstie/MgII_forest/' + zstr + '/'
    outfilename = 'corr_func_models_fwhm_{:5.3f}_samp_{:5.3f}'.format(fwhm, sampling) + '.fits'
    outfile = os.path.join(outpath, outfilename)

    xhi_path = '/mnt/quasar/joe/reion_forest/Nyx_output/z75/xHI/'
    files = glob.glob(os.path.join(xhi_path, '*_tau.fits'))
    params = Table.read(files[0], hdu=1)

    args = xhi_path, zstr, fwhm, sampling, vmin_corr, vmax_corr, dv_corr, ncopy, ncovar, nmock, cgm_masking_gpm
    all_args = []
    seed_vec = np.full(nhi*nlogZ, seed) # same seed for each process

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

    param_model=Table([[ncopy], [ncovar], [nmock], [fwhm],[sampling],[seed], [nhi], [xhi_val], [nlogZ], [logZ_vec], [ncorr], [vmin_corr],[vmax_corr], [vel_mid]],
                      names=('ncopy', 'ncovar', 'nmock', 'fwhm', 'sampling', 'seed', 'nhi', 'xhi', 'nlogZ', 'logZ', 'ncorr', 'vmin_corr', 'vmax_corr', 'vel_mid'))
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


################## old/obsolete ##################
def mock_mean_covar(xi, xi_mean, npath, ncopy, nmock, seed=None):

    """
    Computes the covariance of the 2PCF from realizations of mock dataset.

    Args:
        xi (ndarray): 2PCF of *each* skewer; output of reion_forest.utils.compute_xi
        xi_mean (1D array): mean 2PCF averaging over the 2PCF of all skewers
        npath (int): number of skewers to use for mock data set; determined from nqsos and delta_z
        ncovar (int): number of mock realizations to generate
        nmock (int): number of mock realizations to save/output
        seed: if None, then randomly generate a seed.

    Returns:
        xi_mock_keep (2D array)
        covar (2D array)
    """

    rand = np.random.RandomState(seed) if seed is None else seed
    nskew, ncorr = xi.shape # noisy 2PCF from Nyx skewers, where nskew = 10,000

    # Compute the mean from the "perfect" models
    xi_mock_keep = np.zeros((nmock, ncorr))
    covar = np.zeros((ncorr, ncorr))
    indx = np.arange(nskew)

    for imock in range(ncopy):
        npath = nqso*nskew_to_match_data
        ranindx = rand.choice(indx, replace=False, size=npath)  # return random sampling of 'indx' of size 'npath'
        xi_mock = np.mean(xi[ranindx, :], axis=0)  # mean 2PCF averaging over npath skewers;  for the mock dataset
        delta_xi = xi_mock - xi_mean
        covar += np.outer(delta_xi, delta_xi)
        if imock < nmock:  # saving the first nmock results
            xi_mock_keep[imock, :] = xi_mock

    # Divide by ncovar since we estimated the mean from "independent" data; Eqn (13)
    covar /= ncopy

    return xi_mock_keep, covar
