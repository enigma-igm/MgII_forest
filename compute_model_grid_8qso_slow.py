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
from enigma.reion_forest import utils
from enigma.reion_forest.mgii_find import MgiiFinder
from astropy.table import Table, hstack, vstack
from IPython import embed
from multiprocessing import Pool
from tqdm import tqdm
import mutils
import mask_cgm_pdf as mask_cgm
import pdb
import compute_cf_data as ccf
import scipy
import time

###################### global variables ######################
datapath = '/Users/suksientie/Research/MgII_forest/rebinned_spectra/'
datapath = '/mnt/quasar/sstie/MgII_forest/z75/rebinned_spectra/'

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

# 3 random seeds (cpu seeds, seeds for drawing mock spectra, seeds for drawing noise)

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

def draw_random_skews_to_match_data(vel_lores, vel_data, tot_nyx_skews=10000, seed=None):
    # draw n number of random Nyx skewers that match the total data pathlength

    # vel_lores = Nyx velocity grid (numpy array)
    # vel_data = data velocity grid (numpy array)
    # tot_nyx_skews = total number of Nyx skewers (int)
    # seed = random seed used to grab random Nyx skewers

    tot_vel_data = vel_data.max() - vel_data.min()
    tot_vel_sim = vel_lores.max() - vel_lores.min()
    nskew_to_match_data = int(np.ceil(tot_vel_data / tot_vel_sim))  # assuming tot_vel_data > tot_vel_sim (true here)

    """
    if seed != None:
        rand = np.random.RandomState(seed)
    else:
        rand = np.random.RandomState()
    """
    rand = np.random.RandomState(seed) if seed is None else seed

    indx_flux_lores = np.arange(tot_nyx_skews)
    ranindx = rand.choice(indx_flux_lores, replace=False, size=nskew_to_match_data)  # grab a set of random skewers
    return ranindx, nskew_to_match_data

def reshape_data_array(data_arr, nskew_to_match_data, npix_sim_skew, data_arr_is_mask): # 10, 220
    # reshape the data spectrum to match the dimension of the simulated skewers array,
    # which is of shape (nskew_to_match_data, npix_sim_skew)
    # note: set data_arr_is_mask = True if the data array is a mask

    pad_width = nskew_to_match_data * npix_sim_skew - len(data_arr)

    if data_arr_is_mask:
        #print("input data array is Bool array, so padding with False")
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

def forward_model_onespec_chunk(vel_data, norm_std, master_mask, vel_lores, flux_lores, seed=None, std_corr=1.0):

    #rand = np.random.RandomState(seed) if seed != None else np.random.RandomState()
    rand = np.random.RandomState(seed) if seed is None else seed

    npix_sim_skew = len(vel_lores)
    ranindx, nskew_to_match_data = draw_random_skews_to_match_data(vel_lores, vel_data, seed=seed)

    # adding noise to flux lores
    norm_std[norm_std < 0] = 100 # get rid of negative errors
    noise = rand.normal(0, std_corr * norm_std)

    noise_chunk = reshape_data_array(noise, nskew_to_match_data, npix_sim_skew, data_arr_is_mask=False)
    flux_lores_noise = flux_lores[ranindx] + noise_chunk

    master_mask_chunk = reshape_data_array(master_mask, nskew_to_match_data, npix_sim_skew, data_arr_is_mask=True)

    return vel_lores, flux_lores_noise, flux_lores[ranindx], master_mask_chunk, nskew_to_match_data, npix_sim_skew

def compute_cf_onespec_chunk(vel_lores, flux_lores_noise, given_bins, mask_chunk=None):

    if mask_chunk is None:
        mean_flux = np.nanmean(flux_lores_noise)
    else:
        mean_flux = np.nanmean(flux_lores_noise[mask_chunk])

    vmin_corr, vmax_corr, dv_corr = 0, 0, 0  # dummy values since using custom binning, but still required arguments for compute_xi
    delta_f = (flux_lores_noise - mean_flux) / mean_flux
    (vel_mid, xi_onespec, npix_xi, xi_mock_zero_lag) = utils.compute_xi(delta_f, vel_lores, vmin_corr, vmax_corr, dv_corr, given_bins=given_bins, gpm=mask_chunk)

    xi_onespec = np.mean(xi_onespec, axis=0) # averages over the chunks
    npix_xi = np.sum(npix_xi, axis=0) # summing npix over the chunk

    return vel_mid, xi_onespec, npix_xi

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

def mock_mean_covar(ncovar, nmock_to_save, vel_data_allqso, norm_std_allqso, master_mask_allqso, instr_allqso, \
                    vel_lores_nires_interp, flux_lores_nires_interp, vel_lores_mosfire_interp, flux_lores_mosfire_interp, \
                    given_bins, seed=None):

    ncorr = len(given_bins[0])
    rand = np.random.RandomState(seed) if seed is None else seed # np.random.RandomState() if seed is None else np.random.RandomState(seed)
    #nskew, ncorr = xi_mean.shape

    # Compute the mean from the "perfect" models
    xi_mock_keep = np.zeros((nmock_to_save, ncorr))
    covar = np.zeros((ncorr, ncorr))
    nqso = len(vel_data_allqso)

    xi_mock_ncovar = []
    xi_mock_ncovar_noiseless = []

    for imock in range(ncovar):

        xi_allspec = []
        xi_allspec_noiseless = []
        npix_xi_allspec = []

        for iqso in range(nqso):
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
            vel_lores, flux_lores_noise_onespec, flux_lores_noiseless_onespec, master_mask_chunk, nskew_to_match_data, npix_sim_skew = \
                forward_model_onespec_chunk(vel_data, norm_std, master_mask, vel_lores, flux_lores, seed=rand, std_corr=std_corr)

            # compute CF for this one qso
            vel_mid, xi_onespec, npix_xi = compute_cf_onespec_chunk(vel_lores, flux_lores_noise_onespec, given_bins, mask_chunk=master_mask_chunk)
            xi_allspec.append(xi_onespec)
            npix_xi_allspec.append(npix_xi) # npix_xi = nskew, ncorr

            vel_mid, xi_onespec_noiseless, npix_xi = compute_cf_onespec_chunk(vel_lores, flux_lores_noiseless_onespec, given_bins, mask_chunk=master_mask_chunk)
            xi_allspec_noiseless.append(xi_onespec_noiseless)

        weights = np.array(npix_xi_allspec / np.sum(npix_xi_allspec, axis=0))  # weights = ncorr
        #print("shape of weights", weights.shape)
        #print(weights)

        xi_mock_ncovar.append(np.average(xi_allspec, axis=0, weights=weights)) # averages over nqso
        xi_mock_ncovar_noiseless.append(np.average(xi_allspec_noiseless, axis=0, weights=weights))

    xi_mean = np.mean(xi_mock_ncovar_noiseless, axis=0) # effective mean model is the mean of mean

    for imock in range(ncovar):
        xi_mock = xi_mock_ncovar[imock]
        delta_xi = xi_mock - xi_mean
        covar += np.outer(delta_xi, delta_xi)  # off-diagonal elements

        if imock < nmock_to_save:
            xi_mock_keep[imock, :] = xi_mock

    covar /= ncovar

    return xi_mock_keep, covar, vel_mid, xi_mean

def compute_model(args):

    #ihi, iZ, xHI, logZ, master_seed, xhi_path, zstr, fwhm, sampling, vmin_corr, vmax_corr, dv_corr, redshift_bin, ncopy, cgm_masking_gpm = args
    ihi, iZ, xHI, logZ, master_seed, xhi_path, zstr, redshift_bin, ncovar, nmock, vel_data_allqso, norm_std_allqso, master_mask_allqso, instr_allqso = args
    rantaufile = os.path.join(xhi_path, 'ran_skewers_' + zstr + '_OVT_' + 'xHI_{:4.2f}'.format(xHI) + '_tau.fits')
    params = Table.read(rantaufile, hdu=1)
    skewers = Table.read(rantaufile, hdu=2)

    rand = np.random.RandomState(master_seed)

    # NIRES fwhm and sampling
    start = time.process_time()
    vel_lores_nires, (flux_lores_nires, flux_lores_igm_nires, flux_lores_cgm_nires, _, _), \
    vel_hires_nires, (flux_hires_nires, flux_hires_igm_nires, flux_hires_cgm_nires, _, _), \
    (oden, v_los, T, xHI), cgm_tuple = utils.create_mgii_forest(params, skewers, logZ, nires_fwhm, sampling=nires_sampling)
    end = time.process_time()


    print("      NIRES mocks done in .... ", (end - start) / 60, " min")

    # MOSFIRE fwhm and sampling
    start = time.time()
    vel_lores_mosfire, (flux_lores_mosfire, flux_lores_igm_mosfire, flux_lores_cgm_mosfire, _, _), \
    vel_hires_mosfire, (flux_hires_mosfire, flux_hires_igm_mosfire, flux_hires_cgm_mosfire, _, _), \
    (oden, v_los, T, xHI), cgm_tuple = utils.create_mgii_forest(params, skewers, logZ, mosfire_fwhm, sampling=mosfire_sampling)
    end = time.time()
    print("      MOSFIRE mocks done in .... ", (end - start) / 60, " min")

    # interpolate flux lores to dv=40 (nyx)
    start = time.process_time()
    dv_coarse = 40
    coarse_grid_all = np.arange(len(vel_lores_nires)+1) * dv_coarse
    vel_lores_nires_interp = (coarse_grid_all[:-1] + coarse_grid_all[1:]) / 2
    flux_lores_nires_interp = scipy.interpolate.interp1d(vel_lores_nires, flux_lores_nires, kind = 'cubic', \
                                                        bounds_error = False, fill_value = np.nan)(vel_lores_nires_interp)

    coarse_grid_all = np.arange(len(vel_lores_mosfire) + 1) * dv_coarse
    vel_lores_mosfire_interp = (coarse_grid_all[:-1] + coarse_grid_all[1:]) / 2
    flux_lores_mosfire_interp = scipy.interpolate.interp1d(vel_lores_mosfire, flux_lores_mosfire, kind='cubic',
                                                       bounds_error=False, fill_value=np.nan)(vel_lores_mosfire_interp)
    end = time.process_time()
    print("      interpolating both mocks done in .... ", (end - start) / 60, " min")

    # calls mock_mean_covar
    start = time.process_time()
    xi_mock_keep, covar, vel_mid, xi_mean = mock_mean_covar(ncovar, nmock, vel_data_allqso, norm_std_allqso, master_mask_allqso, instr_allqso, \
                    vel_lores_nires_interp, flux_lores_nires_interp, vel_lores_mosfire_interp,
                    flux_lores_mosfire_interp, given_bins, seed=rand)
    icovar = np.linalg.inv(covar)  # invert the covariance matrix
    sign, logdet = np.linalg.slogdet(covar)  # compute the sign and natural log of the determinant of the covariance matrix
    end = time.process_time()
    print("      mock_mean_covar done in .... ", (end - start) / 60, " min")

    return ihi, iZ, vel_mid, xi_mock_keep, xi_mean, covar, icovar, logdet

###################### main() ######################
def parser():
    import argparse

    parser = argparse.ArgumentParser(description='Create random skewers for MgII forest', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nproc', type=int, help="Number of processors to run on")
    parser.add_argument('--ncovar', type=int, default=1000000, help="number of mock datasets for computing covariance")
    parser.add_argument('--nmock', type=int, default=300, help="number of mock datasets to store")
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

    param_model=Table([[ncovar], [nmock], [seed], [nhi], [xhi_val], [nlogZ], [logZ_vec], [ncorr], [vel_mid], [redshift_bin]],
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