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
from matplotlib import rcParams
import time
import sys
import matplotlib.cm as cm
from tqdm.auto import tqdm

from scipy.interpolate import interp1d, interp2d, RectBivariateSpline
from astropy import constants as const
import scipy.ndimage
from astropy.table import Table, hstack, vstack
from IPython import embed
sys.path.append('/Users/suksientie/codes/enigma')
from enigma.reion_forest import utils
#import tqdm
#import enigma.reion_forest.istarmap  # import to apply patch
from multiprocessing import Pool
from tqdm import tqdm
import mutils

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
def forward_model_onespec(vel_data, norm_good_std, vel_lores, flux_lores, ncopy, seed=None):
    # vel_lores.shape = 239
    # flux_lores.shape = (10000, 239)

    tot_vel_data = vel_data.max() - vel_data.min()
    tot_vel_sim = vel_lores.max() - vel_lores.min()
    nskew_to_match_data = int(np.ceil(tot_vel_data/tot_vel_sim)) # assuming tot_vel_data > tot_vel_sim (true here)
    npix_sim = len(flux_lores)
    #print("nskew to match data", nskew_to_match_data, tot_vel_data/tot_vel_sim)
    flux_lores_noise = np.ones((nskew_to_match_data, npix_sim))

    if seed != None:
        rand = np.random.RandomState(seed)
    else:
        rand = np.random.RandomState()

    indx_flux_lores = np.arange(flux_lores.shape[0])
    ranindx = rand.choice(indx_flux_lores, replace=False, size=nskew_to_match_data)

    # stitching simulated spectrum to match length of data
    flux_lores_long = []
    vel_lores_long = []
    for i in ranindx:
        flux_lores_long.extend(flux_lores[i])
        if len(vel_lores_long) == 0:
            vel_lores_long.extend(vel_lores)
        else:
            vel_lores_long.extend(vel_lores_long[-1] + np.arange(1, len(vel_lores) + 1) * np.diff(vel_lores)[0])
    vel_lores_long = np.array(vel_lores_long)
    flux_lores_long = np.array(flux_lores_long)

    # adding noise
    all_flong_noise = np.zeros((ncopy, len(flux_lores_long)))
    for icopy in range(ncopy):
        flux_lores_long_noise = []
        for i, f in enumerate(flux_lores_long):
            if vel_lores_long[i] < vel_data[-1]: # if vpix of simulated spectrum is less than vpix of data
                dist = np.abs(vel_lores_long[i] - vel_data)
                iwant = np.argmin(dist) # assign noise of the nearest neighboring data pixel
                flux_lores_long_noise.append(rand.normal(f, norm_good_std[iwant]))
            else:
                flux_lores_long_noise.append(np.nan) # assign nan if sim skewer is beyond the data
                flux_lores_long[i] = np.nan #...? need this line?

        all_flong_noise[icopy] = flux_lores_long_noise

    # dealing with large masked regions
    ind_large_dv = np.where(np.diff(vel_data) > 100)[0]
    for ind in ind_large_dv:
        assert (vel_data[ind+1] - vel_data[ind]) > 100
        i = np.where(np.logical_and(vel_lores_long >= vel_data[ind], vel_lores_long <= vel_data[ind+1]))[0]
        for icopy in range(ncopy):
            all_flong_noise[icopy][i] = np.ones(len(i))*np.nan

    all_gpm = np.zeros((ncopy, len(flux_lores_long))).astype(bool)
    for icopy in range(ncopy):
        bpm = np.ma.masked_invalid(all_flong_noise[icopy]).mask # bad pixel mask
        all_gpm[icopy] = np.invert(bpm)

    return vel_lores_long, flux_lores_long, all_flong_noise, all_gpm

def forward_model_allspec(vel_lores, flux_lores, ncopy, seed_list=[None, None, None, None], plot=False):

    fitsfile_list = ['/Users/suksientie/Research/data_redux/wavegrid_vel/J0313-1806/vel123_coadd_tellcorr.fits', \
                     '/Users/suksientie/Research/data_redux/wavegrid_vel/J1342+0928/vel123_coadd_tellcorr.fits', \
                     '/Users/suksientie/Research/data_redux/wavegrid_vel/J0252-0503/vel12_coadd_tellcorr.fits', \
                     '/Users/suksientie/Research/data_redux/wavegrid_vel/J0038-1527/vel1_tellcorr_pad.fits']
    qso_namelist = ['J0313-1806', 'J1342+0928', 'J0252-0503', 'J0038-1527']
    qso_zlist = [7.6, 7.54, 7.0, 7.0]
    everyn_break_list = [20, 20, 20, 20]

    out_all = []
    for iqso, fitsfile in enumerate(fitsfile_list):
        wave, flux, ivar, mask, std, fluxfit, outmask, sset = mutils.extract_and_norm(fitsfile, everyn_break_list[iqso])

        x_mask = wave[outmask] <= (2800 * (1 + qso_zlist[iqso])) # removing spectral region beyond qso redshift

        good_wave = wave[outmask][x_mask]
        good_flux = flux[outmask][x_mask]
        good_std = std[outmask][x_mask]
        norm_good_flux = good_flux / fluxfit[x_mask]
        norm_good_std = good_std / fluxfit[x_mask]
        vel_data = mutils.obswave_to_vel_2(good_wave)

        vlores_long, flores_long, flores_long_noise, gpm = forward_model_onespec(vel_data, norm_good_std, vel_lores, flux_lores, ncopy, seed=seed_list[iqso])
        out_all.append([vlores_long, flores_long, flores_long_noise, gpm])

        if plot:
            plt.figure(figsize=(12,8))
            plt.title(qso_namelist[iqso], fontsize=18)
            for i in range(10): # plot every 10-th spectrum
                plt.plot(vlores_long, flores_long_noise[i+10]+i, alpha=0.5)
                if i == 0:
                    plt.plot(vlores_long, flores_long+i, c='k', alpha=0.8, label='sim spectrum without noise')
            plt.plot(vel_data, np.array(norm_good_flux) - 1, c='r', label='data spectrum')
            plt.xlabel('vel (km/s)', fontsize=15)
            plt.legend(fontsize=15)
            plt.tight_layout()

    # out_all = vlores_long, flores_long, flores_long_noise, gpm
    return out_all

def compute_cf_onespec(vel_lores_long, flong_noise, gpm, vmin_corr=10, vmax_corr=2000, dv_corr=60):

    mean_flux = np.nanmean(flong_noise)
    delta_f = (flong_noise - mean_flux) / mean_flux

    # using np.nansum in utils.xi_sum
    (vel_mid, xi_mock, npix, xi_mock_zero_lag) = utils.compute_xi(delta_f, vel_lores_long, vmin_corr, vmax_corr, dv_corr, gpm=gpm)

    return vel_mid, xi_mock

def compute_cf_allspec(forward_model_out, vmin_corr=10, vmax_corr=2000, dv_corr=60):
    # "forward_model_out" being the output of forward_model_allspec()

    xi_mock_all = []

    for iqso, output in enumerate(forward_model_out):
        vlores_long, flores_long, flores_long_noise, gpm = output # unpacking
        vel_mid, xi_mock = compute_cf_onespec(vlores_long, flores_long_noise, gpm, vmin_corr, vmax_corr, dv_corr)
        xi_mock_all.append(xi_mock)

    vel_mid = np.array(vel_mid)
    xi_mock_all = np.array(xi_mock_all)

    return vel_mid, xi_mock_all

# loop mock_mean_covar for each qso?
def mock_mean_covar(xi_mean, ncopy, ncovar, nmock, vel_lores, flux_lores, vmin_corr, vmax_corr, dv_corr, seed=None):

    rand = np.random.RandomState(seed) if seed is None else seed
    seed_list = rand.randint(0, 999999999, 4) # 4 for 4 qsos

    ncorr = xi_mean.shape[0]
    xi_mock_keep = np.zeros((nmock, ncorr)) # add extra dim?
    covar = np.zeros((ncorr, ncorr)) # add extra dim?

    for imock in range(ncovar):
        fm_out = forward_model_allspec(vel_lores, flux_lores, ncopy, seed_list=seed_list)
        vel_mid, xi_mock = compute_cf_allspec(fm_out, vmin_corr, vmax_corr, dv_corr)
        xi_mock_mean = np.mean(np.mean(xi_mock, axis=1), axis=0) # averaging over ncopy and then over nqso
        delta_xi = xi_mock_mean - xi_mean

        covar += np.outer(delta_xi, delta_xi)  # off-diagonal elements
        if imock < nmock:
            xi_mock_keep[imock, :] = xi_mock_mean
        # Divid by ncovar since we estimated the mean from "independent" data
    covar /= ncovar

    return xi_mock_keep, covar

import pdb
def compute_model(args):
    # compute CF and covariance of mock dataset at each point of model grid
    # args: tuple of arguments from parser

    ihi, iZ, xHI, logZ, seed, xhi_path, zstr, fwhm, sampling, vmin_corr, vmax_corr, dv_corr, ncopy, ncovar, nmock = args
    rantaufile = os.path.join(xhi_path, 'ran_skewers_' + zstr + '_OVT_' + 'xHI_{:4.2f}'.format(xHI) + '_tau.fits')
    rand = np.random.RandomState(seed)
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

    # forward modeled dataset, currently hardwired to take in the 4 QSOs in our current dataset (in progress)
    #fm_out = forward_model_allspec(vel_lores, flux_lores, ncopy)
    #vel_mid, xi_mock = compute_cf_allspec(fm_out)
    xi_mock_keep, covar = mock_mean_covar(xi_mean, ncopy, ncovar, nmock, vel_lores, flux_lores, vmin_corr, vmax_corr, dv_corr, seed=None)
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
    ncovar = 100
    nmock = 10
    seed = 9999
    logZ = -3.5
    args = ihi, iZ, xHI, logZ, seed, xhi_path, zstr, fwhm, sampling, vmin_corr, vmax_corr, dv_corr, ncopy, ncovar, nmock

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

    args = xhi_path, zstr, fwhm, sampling, vmin_corr, vmax_corr, dv_corr, ncopy, ncovar, nmock
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