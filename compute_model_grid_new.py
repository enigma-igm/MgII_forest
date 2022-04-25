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
import mask_cgm_pdf as mask_cgm
import pdb

###################### global variables ######################
datapath = '/Users/suksientie/Research/data_redux/'
#datapath = '/mnt/quasar/sstie/MgII_forest/z75/' # on IGM

fitsfile_list = [datapath + 'wavegrid_vel/J0313-1806/vel1234_coadd_tellcorr.fits', \
                 datapath + 'wavegrid_vel/J1342+0928/vel123_coadd_tellcorr.fits', \
                 datapath + 'wavegrid_vel/J0252-0503/vel12_coadd_tellcorr.fits', \
                 datapath + 'wavegrid_vel/J0038-1527/vel1_tellcorr.fits']

qso_namelist = ['J0313-1806', 'J1342+0928', 'J0252-0503', 'J0038-1527']
qso_zlist = [7.642, 7.541, 7.001, 7.034] # precise redshifts from Yang+2021
everyn_break_list = [20, 20, 20, 20] # placing a breakpoint at every 20-th array element (more docs in mutils.continuum_normalize)
                                     # this results in dwave_breakpoint ~ 40 A --> dvel_breakpoint = 600 km/s
exclude_restwave = 1216 - 1185 # excluding proximity zones; see mutils.qso_exclude_proximity_zone
median_z = 6.57 # median redshift of measurement after excluding proximity zones; 4/20/2022
corr_all = [0.758, 0.753, 0.701, 0.724] # 4/20/2022 (determined from mutils.plot_allspec_pdf)

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

def init_cgm_masking(redshift_bin, datapath):

    # retaining the shape of the cgm masks to be the same as data shape, so not applying mask before running CGM masking
    good_vel_data_all, good_wave_data_all, norm_good_flux_all, norm_good_std_all, good_ivar_all, noise_all = \
        mask_cgm.init(redshift_bin, datapath, do_not_apply_any_mask=True)

    mgii_tot_all = mask_cgm.chi_pdf(good_vel_data_all, norm_good_flux_all, good_ivar_all, noise_all, plot=False)
    gpm_allspec = []

    for i in range(len(mgii_tot_all)):
        gpm_allspec.append(mgii_tot_all[i].fit_gpm[0])

    return gpm_allspec, mgii_tot_all

########################## simulating one data spectrum #############################
def sample_noise_onespec_chunk(vel_data, norm_std, vel_lores, flux_lores, ncopy, seed=None, std_corr=1.0):

    # return "ncopy" of simulated noise from data

    npix_sim_skew = len(vel_lores)
    tot_nyx_skews = len(flux_lores)
    ranindx, nskew_to_match_data = rand_skews_to_match_data(vel_lores, vel_data, tot_nyx_skews, seed=seed)

    if seed != None:
        rand = np.random.RandomState(seed)
    else:
        rand = np.random.RandomState()

    # reshape data sigma array
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

def plot_forward_model_onespec_new(noisy_flux_lores_ncopy, rand_noise_ncopy, rand_flux_lores, master_mask, vel_data, norm_flux, ncopy_plot, title=None):

    ncopy, nskew, npix = np.shape(noisy_flux_lores_ncopy)

    plt.figure(figsize=(12, 8))
    if title != None:
        plt.title(title, fontsize=18)

    nan_pad_mask = ~np.isnan(noisy_flux_lores_ncopy[0])
    ##### plot subset of mock spectra
    for i in range(ncopy_plot):
        flux_lores_comb = noisy_flux_lores_ncopy[i][nan_pad_mask]
        plt.plot(vel_data, flux_lores_comb + (i + 1), alpha=0.5, drawstyle='steps-mid')

    plt.plot(vel_data, norm_flux, 'k', drawstyle='steps-mid')

    ind_masked = np.where(master_mask == False)[0]
    for j in range(len(ind_masked)):  # bad way to plot masked pixels
        plt.axvline(vel_data[ind_masked[j]], color='k', alpha=0.2, lw=2)

    plt.ylabel('Flux (+ arbitrary offset)', fontsize=15)
    plt.xlabel('Velocity (km/s)', fontsize=15)
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

    xi_mock_qso_all = []

    for iqso, fitsfile in enumerate(fitsfile_list):
        # initialize all qso data
        raw_data_out, _, all_masks_out = mutils.init_onespec(iqso, redshift_bin, datapath=datapath)
        wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
        strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, master_mask = all_masks_out

        norm_flux = flux / fluxfit
        norm_std = std / fluxfit
        vel_data = mutils.obswave_to_vel_2(wave)

        # generate mock data spectrum
        _, _, rand_noise_ncopy, noisy_flux_lores_ncopy, nskew_to_match_data, npix_sim_skew = forward_model_onespec_chunk(vel_data, norm_std, vel_lores, flux_lores, ncopy, seed=qso_seed_list[iqso], std_corr=corr_all[iqso])

        usable_data_mask = mask * redshift_mask * pz_mask * zbin_mask
        usable_data_mask_chunk = reshape_data_array(usable_data_mask, nskew_to_match_data, npix_sim_skew, True)

        # deal with CGM mask if argued before computing the 2PCF
        if type(cgm_gpm_allspec) == type(None):
            all_mask_chunk = usable_data_mask_chunk
        else:
            gpm_onespec_chunk = reshape_data_array(cgm_gpm_allspec[iqso], nskew_to_match_data, npix_sim_skew, True) # reshaping GPM from cgm masking
            all_mask_chunk = usable_data_mask_chunk * gpm_onespec_chunk

        # compute the 2PCF
        vel_mid, xi_mock, npix_xi = compute_cf_onespec_chunk(vel_lores, noisy_flux_lores_ncopy, vmin_corr, vmax_corr, dv_corr, mask=all_mask_chunk)
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

    if args.lowz_bin:
        redshift_bin = 'low'
    elif args.highz_bin:
        redshift_bin = 'high'
    elif args.allz_bin:
        redshift_bin = 'all'
    else:
        raise ValueError('must set one of these arguments: "--lowz_bin", "--highz_bin", or "--allz_bin"')

    cgm_masking = args.cgm_masking
    if cgm_masking:
        cgm_masking_gpm, _ = init_cgm_masking(redshift_bin, datapath)
    else:
        cgm_masking_gpm = None

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
    outfilename = 'corr_func_models_fwhm_{:5.3f}_samp_{:5.3f}_{:s}'.format(fwhm, sampling, redshift_bin) + '.fits'
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