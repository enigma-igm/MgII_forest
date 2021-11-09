
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
from enigma.reion_forest import utils
#import tqdm
#import enigma.reion_forest.istarmap  # import to apply patch
from multiprocessing import Pool
from tqdm import tqdm


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

def mock_mean_covar(xi, xi_mean, npath, ncovar, nmock, seed=None):

    rand = np.random.RandomState(seed) if seed is None else seed
    nskew, ncorr = xi.shape
    # Compute the mean from the "perfect" models
    xi_mock_keep = np.zeros((nmock, ncorr))
    covar = np.zeros((ncorr, ncorr))
    indx = np.arange(nskew)
    for imock in range(ncovar):
        ranindx = rand.choice(indx, replace=False, size=npath) # return random sampling of 'indx' of size 'npath'
        xi_mock = np.mean(xi[ranindx, :], axis=0)
        delta_xi = xi_mock - xi_mean
        covar += np.outer(delta_xi, delta_xi) # off-diagonal elements
        if imock < nmock:
            xi_mock_keep[imock, :] = xi_mock
    # Divid by ncovar since we estimated the mean from "independent" data
    covar /= ncovar

    return xi_mock_keep, covar

def read_model_grid(modelfile):

    hdu = fits.open(modelfile)
    param = Table(hdu[1].data)
    xi_mock_array = hdu[2].data
    xi_model_array = hdu[3].data
    covar_array = hdu[4].data
    icovar_array = hdu[5].data
    lndet_array = hdu[6].data

    return param, xi_mock_array, xi_model_array, covar_array, icovar_array, lndet_array

def resample_data_noise(data_sigma_arr, outshape, rand=None):

    if rand == None:
        rand = np.random.RandomState()

    r = rand.choice(data_sigma_arr, size=outshape, replace=True)
    noise = []
    for ir in r:
        noise.append(rand.normal(0, ir))
    return noise

def parser():
    import argparse

    parser = argparse.ArgumentParser(description='Create random skewers for MgII forest',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('nproc', type=int, help="Number of processors to run on")
    parser.add_argument('--fwhm', type=float, default=100.0, help="spectral resolution in km/s")
    parser.add_argument('--samp', type=float, default=3.0, help="Spectral sampling: pixels per fwhm resolution element")
    parser.add_argument('--SNR', type=float, default=100.0, help="signal-to-noise ratio")
    parser.add_argument('--nqsos', type=int, default=10, help="number of qsos")
    parser.add_argument('--delta_z', type=float, default=0.6, help="redshift pathlength per qso")
    parser.add_argument('--vmin', type=float, default=30.0, help="Minimum of velocity bins for correlation function")
    parser.add_argument('--vmax', type=float, default=3100, help="Maximum of velocity bins for correlation function")
    parser.add_argument('--dv', type=float, default=None, help="Width of velocity bins for correlation function. "
                                                               "If not set fwhm will be used")
    parser.add_argument('--ncovar', type=int, default=1000000, help="number of mock datasets for computing covariance")
    parser.add_argument('--nmock', type=int, default=300, help="number of mock datasets to store")
    parser.add_argument('--seed', type=int, default=1234, help="seed for random number generator")
    parser.add_argument('--nlogZ', type=int, default=201, help="number of bins for logZ models")
    parser.add_argument('--logZmin', type=float, default=-6.0, help="minimum logZ value")
    parser.add_argument('--logZmax', type=float, default=-2.0, help="maximum logZ value")

    return parser.parse_args()

# TODO it seems F_bar should be computed from the realizations
def compute_model(args):

    #ihi, iZ, xHI, logZ, seed, xhi_path, zstr, fwhm, sampling, SNR, vmin_corr, vmax_corr, dv_corr, npath, ncovar, nmock = args
    ihi, iZ, xHI, logZ, seed, xhi_path, zstr, fwhm, sampling, SNR, vmin_corr, vmax_corr, dv_corr, npath, ncovar, nmock, data_sigma_fitsfile = args
    rantaufile = os.path.join(xhi_path, 'ran_skewers_' + zstr + '_OVT_' + 'xHI_{:4.2f}'.format(xHI) + '_tau.fits')
    rand = np.random.RandomState(seed)
    params = Table.read(rantaufile, hdu=1)
    skewers = Table.read(rantaufile, hdu=2)
    # Compute the skewers. This takes 5.36s for 10,000 skewers
    vel_lores, (flux_lores, flux_lores_igm, flux_lores_cgm, _, _), vel_hires, (flux_hires, flux_hires_igm, flux_hires_cgm, _, _), \
    (oden, v_los, T, xHI), cgm_tuple = utils.create_mgii_forest(params, skewers, logZ, fwhm, sampling=sampling)
    # Add noise
    sigma_arr = fits.open(data_sigma_fitsfile)[4].data # last array is the flattened array from all QSO
    noise = resample_data_noise(sigma_arr, flux_lores.shape, rand=rand)
    #noise = rand.normal(0.0, 1.0 / SNR, flux_lores.shape) # TODO: noise might have to be an input argument, e.g. fitsfile
                                                          # randomly sample actual noise from data (all positive though)?
    flux_noise = flux_lores + noise
    # Compute delta_f
    mean_flux = np.mean(flux_noise)
    delta_f = (flux_noise - mean_flux)/mean_flux
    mean_flux_nless = np.mean(flux_lores)
    delta_f_nless = (flux_lores - mean_flux_nless)/mean_flux_nless
    # Compute xi. This takes 46.9s for 10,000 skewers
    (vel_mid, xi, npix, xi_zero_lag) = utils.compute_xi(delta_f, vel_lores, vmin_corr, vmax_corr, dv_corr)
    # Compute xi_noiseless, and use this to compute the mean.
    (vel_mid, xi_nless, npix, xi_nless_zero_lag) = utils.compute_xi(delta_f_nless, vel_lores, vmin_corr, vmax_corr, dv_corr)
    xi_mean = np.mean(xi_nless, axis=0)
    # Compute the covariance 569 ms for 10,000 skewers
    xi_mock, covar = mock_mean_covar(xi, xi_mean, npath, ncovar, nmock, seed=rand)
    icovar = np.linalg.inv(covar) # invert the covariance matrix
    sign, logdet = np.linalg.slogdet(covar) # compute the sign and natural log of the determinant of the covariance matrix
    return ihi, iZ, vel_mid, xi_mock, xi_mean, covar, icovar, logdet

def main():

    args = parser()
    nproc = args.nproc
    fwhm = args.fwhm
    sampling = args.samp
    SNR = args.SNR
    nqsos = args.nqsos
    delta_z = args.delta_z
    ncovar = args.ncovar
    nmock = args.nmock
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

    #nhi = 19
    #xhi_val = 0.05 + np.arange(nhi) * 0.05

    # Some file paths and then read in the params table to get the redshift
    zstr = 'z75'
    outpath = '/mnt/quasar/sstie/MgII_forest/' + zstr + '/'
    #outfilename = 'corr_func_models_fwhm_{:5.3f}_samp_{:5.3f}_SNR_{:5.3f}_nqsos_{:d}'.format(fwhm, sampling, SNR,nqsos) + '.fits'
    outfilename = 'corr_func_models_fwhm_{:5.3f}_samp_{:5.3f}_nqsos_{:d}'.format(fwhm, sampling, nqsos) + '.fits' # omitting SNR in filename

    outfile = os.path.join(outpath, outfilename)
    #xhi_path = os.path.join(outpath, 'xHI')
    xhi_path = '/mnt/quasar/joe/reion_forest/Nyx_output/z75/xHI/'
    files = glob.glob(os.path.join(xhi_path,'*_tau.fits'))
    params = Table.read(files[0],hdu=1)
    skewers = Table.read(files[0], hdu=2)
    z = params['z'][0]
    # Determine our path length by computing one model, since this depends on the padding skewer size, etc.
    c_light = (const.c.to('km/s')).value
    z_min = z - delta_z
    z_eff = (z + z_min) / 2.0
    dv_path = (z - z_min) / (1.0 + z_eff) * c_light
    vel_lores, flux_lores, vel_hires, flux_hires, (oden, v_los, T, xHI), cgm_tuple = utils.create_mgii_forest(params, skewers[0:2], logZ_vec[0],
                                                                                                              fwhm, sampling=sampling, z=z)
    vside_lores = vel_lores.max() - vel_lores.min()
    vside_hires = vel_hires.max() - vel_hires.min()
    dz_side = (vside_lores / c_light) * (1.0 + z_eff)
    npath_float = nqsos * dv_path / vside_lores
    npath = int(np.round(npath_float))
    dz_tot = dz_side*npath
    print('Requested path length for nqsos={:d}'.format(nqsos) + ' covering delta_z={:5.3f}'.format(delta_z) +
    ' corresponds to requested total dz_req = {:5.3f}'.format(delta_z * nqsos) + ',  or {:5.3f}'.format(npath_float) +
    ' total skewers. Rounding to {:d}'.format(npath) + ' or dz_tot={:5.3f}'.format(dz_tot) + '.')
    # Number of mock observations to create
    #pbar = tqdm(total=nlogZ*nhi, desc="Computing models")
    # Create the iterable argument list for map
    args = (xhi_path, zstr, fwhm, sampling, SNR, vmin_corr, vmax_corr, dv_corr, npath, ncovar, nmock)
    all_args = []
    # Use below if you want to give each process a different seed. We actually want the same seed so the models deform
    # continuously with parameters
    seed_vec = np.full(nhi*nlogZ, seed)
    #seed_vec = rand.choice(range(100000), size=nhi*nlogZ, replace=False)
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
    ihi, iZ, vel_mid, xi_mock, xi_mean, covar, icovar, logdet = output[0]
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

    nskew = len(skewers)
    param_model=Table([[nqsos], [delta_z], [dz_tot], [npath], [ncovar], [nmock], [fwhm],[sampling], [SNR], [nskew], [seed], [nhi], [xhi_val],[nlogZ], [logZ_vec],
                      [ncorr], [vmin_corr],[vmax_corr], [vel_mid]],
                      names=('nqsos', 'delta_z', 'dz_tot', 'npath', 'ncovar', 'nmock', 'fwhm', 'sampling', 'SNR', 'nskew', 'seed', 'nhi',
                      'xhi', 'nlogZ', 'logZ', 'ncorr', 'vmin_corr', 'vmax_corr', 'vel_mid'))
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


if __name__ == '__main__':
    main()




