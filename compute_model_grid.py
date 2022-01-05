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

###################### global variables ######################
datapath = '/Users/suksientie/Research/data_redux/'
#datapath = '/mnt/quasar/sstie/MgII_forest/z75/'

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
def forward_model_onespec(vel_data, norm_good_std, master_mask_data, vel_lores, flux_lores, ncopy, seed=None):
    # vel_data is the raw vel_data, unmasked
    # master_mask_data = redshift_mask * outmask
    # norm_good_std = std[redshift_mask][outmask] / fluxfit[redshift_mask]
    # seed is used to randomly sample noise from data for ncopy

    tot_vel_data = vel_data.max() - vel_data.min()
    tot_vel_sim = vel_lores.max() - vel_lores.min()
    nskew_to_match_data = int(np.ceil(tot_vel_data / tot_vel_sim))  # assuming tot_vel_data > tot_vel_sim (true here)
    nskew_to_match_data2 = int(np.ceil(len(vel_data) / len(vel_lores)))
    #print("nskew to match data", nskew_to_match_data, nskew_to_match_data2)

    if seed != None:
        rand = np.random.RandomState(seed)
    else:
        rand = np.random.RandomState()

    indx_flux_lores = np.arange(flux_lores.shape[0])
    ranindx = rand.choice(indx_flux_lores, replace=False, size=nskew_to_match_data) # grab a set of random skewers

    # appending pixel by pixel until simulated skewer equals length of data spectrum
    # this works because dv_data = dv_sim AND dv_data is now uniform
    flux_lores_long = []
    for i in ranindx:
        for j in flux_lores[i]:
            if len(flux_lores_long) < len(vel_data):
                flux_lores_long.append(j)

    flux_lores_long = np.array(flux_lores_long)
    flux_lores_long_masked = flux_lores_long[master_mask_data]
    flux_lores_long_masked_noise = np.zeros((ncopy, len(flux_lores_long_masked)))

    for icopy in range(ncopy):
        temp = []
        for i, f in enumerate(flux_lores_long_masked):
            temp.append(rand.normal(f, norm_good_std[i]))
        flux_lores_long_masked_noise[icopy] = np.array(temp)

    return flux_lores_long, flux_lores_long_masked, flux_lores_long_masked_noise

def forward_model_allspec(vel_lores, flux_lores, ncopy, seed_list=[None, None, None, None], cgm_masking=False, fwhm=None, plot=False):

    flores_long_allspec = []
    flores_long_masked_allspec = []
    flores_long_masked_noise_allspec = []
    master_mask_allspec = []
    good_vel_data_allspec = []

    for iqso, fitsfile in enumerate(fitsfile_list):
        wave, flux, ivar, mask, std, fluxfit, outmask, sset = mutils.extract_and_norm(fitsfile, everyn_break_list[iqso])
        vel_data = mutils.obswave_to_vel_2(wave)

        redshift_mask = wave <= (2800 * (1 + qso_zlist[iqso]))  # removing spectral region beyond qso redshift
        master_mask = redshift_mask * outmask

        # masked arrays
        good_wave = wave[master_mask]
        good_flux = flux[master_mask]
        good_ivar = ivar[master_mask]
        good_std = std[master_mask]
        fluxfit_redshift = fluxfit[wave[outmask] <= (2800 * (1 + qso_zlist[iqso]))]
        norm_good_flux = good_flux / fluxfit_redshift
        norm_good_std = good_std / fluxfit_redshift
        good_vel_data = mutils.obswave_to_vel_2(good_wave)

        flores_long, flores_long_masked, flores_long_masked_noise = forward_model_onespec(vel_data, norm_good_std, master_mask, vel_lores, flux_lores, ncopy, seed=seed_list[iqso])
        flores_long_allspec.append(flores_long)
        flores_long_masked_allspec.append(flores_long_masked)
        flores_long_masked_noise_allspec.append(flores_long_masked_noise)
        master_mask_allspec.append(master_mask)
        good_vel_data_allspec.append(good_vel_data)

        if plot:
            plt.figure(figsize=(12,8))
            plt.title(qso_namelist[iqso], fontsize=18)
            for i in range(10): # plot every 10-th spectrum
                plt.plot(good_vel_data, flores_long_masked_noise[i+10]+i, alpha=0.5, drawstyle='steps-mid')
                if i == 0:
                    plt.plot(good_vel_data, flores_long_masked + i, c='k', alpha=0.8, drawstyle='steps-mid', label='sim spectrum without noise')
            plt.plot(good_vel_data, np.array(norm_good_flux) - 1, c='r', drawstyle='steps-mid', label='data spectrum')
            plt.xlabel('vel (km/s)', fontsize=15)
            plt.legend(fontsize=15)
            plt.tight_layout()

            plt.figure(figsize=(10, 8))
            plt.title(qso_namelist[iqso], fontsize=18)
            plt.hist(norm_good_flux, bins=np.arange(0, 3, 0.02), histtype='step', label = 'data', density = True)
            plt.hist(flores_long_masked_noise.flatten(), bins = np.arange(0, 3, 0.02), histtype = 'step', label = 'sim (ncopy)', density = True)
            plt.xlabel('normalized flux', fontsize=15)
            plt.ylabel('PDF', fontsize=15)
            plt.legend(fontsize=15)

    # can now apply cgm masking onto these outputs
    return flores_long_allspec, flores_long_masked_allspec, flores_long_masked_noise_allspec, master_mask_allspec, good_vel_data_allspec

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
    # gpm for each spec has different length, so cannot be recast into np.ndarray, using list for now
    return gpm_allspec

def compute_cf_onespec(vel_lores_long, flong_noise, vmin_corr, vmax_corr, dv_corr, cgm_masking_gpm=None):
    # vel_lores_long = good_vel_data
    # cgm_masking_gpm = gpm from cgm masking
    mean_flux = np.nanmean(flong_noise)
    delta_f = (flong_noise - mean_flux) / mean_flux

    # using np.nansum in utils.xi_sum
    (vel_mid, xi_mock, npix, xi_mock_zero_lag) = utils.compute_xi(delta_f, vel_lores_long, vmin_corr, vmax_corr, dv_corr, gpm=cgm_masking_gpm)

    return vel_mid, xi_mock

def compute_cf_allspec_old(forward_model_out, vmin_corr, vmax_corr, dv_corr, cgm_masking=False, fwhm=None):
    # "forward_model_out" being the output of forward_model_allspec()

    flores_long_allspec, flores_long_masked_allspec, flores_long_masked_noise_allspec, master_mask_allspec, good_vel_data_allspec = forward_model_out

    if cgm_masking:
        gpm_cgm_allspec = init_cgm_masking(fwhm)

    xi_mock_all = []
    for iqso in range(len(flores_long_allspec)):
        if cgm_masking:
            gpm = gpm_cgm_allspec[iqso]
        else:
            gpm = None

        vel_mid, xi_mock = compute_cf_onespec(good_vel_data_allspec[iqso], flores_long_masked_noise_allspec[iqso], vmin_corr, vmax_corr, dv_corr, gpm)
        xi_mock_all.append(xi_mock)

    vel_mid = np.array(vel_mid)
    xi_mock_all = np.array(xi_mock_all)

    return vel_mid, xi_mock_all

def compute_cf_allspec(forward_model_out, vmin_corr, vmax_corr, dv_corr, cgm_masking_gpm=None):
    # "forward_model_out" being the output of forward_model_allspec()
    # cgm_masking_gpm = gpm from cgm masking for all qsos; basically output of init_cgm_masking()
    flores_long_allspec, flores_long_masked_allspec, flores_long_masked_noise_allspec, master_mask_allspec, good_vel_data_allspec = forward_model_out

    xi_mock_all = []
    for iqso in range(len(flores_long_allspec)):
        if type(cgm_masking_gpm) == type(None):
            gpm = None
        else:
            gpm = cgm_masking_gpm[iqso]

        vel_mid, xi_mock = compute_cf_onespec(good_vel_data_allspec[iqso], flores_long_masked_noise_allspec[iqso], vmin_corr, vmax_corr, dv_corr, cgm_masking_gpm=gpm)
        xi_mock_all.append(xi_mock)

    vel_mid = np.array(vel_mid)
    xi_mock_all = np.array(xi_mock_all)

    return vel_mid, xi_mock_all

def mock_mean_covar(xi_mean, ncopy, ncovar, nmock, vel_lores, flux_lores, vmin_corr, vmax_corr, dv_corr, seed=None, cgm_masking_gpm=None):

    # calls forward_model_allspec and compute_cf_allspec
    rand = np.random.RandomState() if seed is None else np.random.RandomState(seed)

    ncorr = xi_mean.shape[0]
    xi_mock_keep = np.zeros((nmock, ncorr)) # add extra dim?
    covar = np.zeros((ncorr, ncorr)) # add extra dim?

    #if cgm_masking:
    #    cgm_masking_gpm = init_cgm_masking(fwhm)
    #else:
    #    cgm_masking_gpm = None

    for imock in range(ncovar):
        seed_list = rand.randint(0, 1000000000, 4)  # 4 for 4 qsos, hardwired for now
        fm_out = forward_model_allspec(vel_lores, flux_lores, ncopy, seed_list=seed_list)
        vel_mid, xi_mock = compute_cf_allspec(fm_out, vmin_corr, vmax_corr, dv_corr, cgm_masking_gpm=cgm_masking_gpm)
        xi_mock_mean = np.mean(np.mean(xi_mock, axis=1), axis=0) # averaging over ncopy and then over nqso
        delta_xi = xi_mock_mean - xi_mean

        covar += np.outer(delta_xi, delta_xi)  # off-diagonal elements
        if imock < nmock:
            xi_mock_keep[imock, :] = xi_mock_mean
        # Divid by ncovar since we estimated the mean from "independent" data
    covar /= ncovar

    return xi_mock_keep, covar

def compute_model(args):
    # compute CF and covariance of mock dataset at each point of model grid
    # args: tuple of arguments from parser

    ihi, iZ, xHI, logZ, seed, xhi_path, zstr, fwhm, sampling, vmin_corr, vmax_corr, dv_corr, ncopy, ncovar, nmock, cgm_masking = args
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
    if cgm_masking:
        cgm_masking_gpm = init_cgm_masking(fwhm)
    else:
        cgm_masking_gpm = None

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
    cgm_masking = False
    args = ihi, iZ, xHI, logZ, seed, xhi_path, zstr, fwhm, sampling, vmin_corr, vmax_corr, dv_corr, ncopy, ncovar, nmock, cgm_masking

    out = compute_model(args)
    return out

###################### main functions ######################
def parser():
    import argparse

    parser = argparse.ArgumentParser(description='Create random skewers for MgII forest', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nproc', type=int, help="Number of processors to run on")
    parser.add_argument('--fwhm', type=float, default=100.0, help="spectral resolution in km/s")
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

    args = xhi_path, zstr, fwhm, sampling, vmin_corr, vmax_corr, dv_corr, ncopy, ncovar, nmock, cgm_masking
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