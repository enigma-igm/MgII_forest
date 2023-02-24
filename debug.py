import compute_cf_data as ccf
from matplotlib import pyplot as plt
from astropy.io import fits
from sklearn.neighbors import KDTree
import compute_cf_data as ccf
import mutils
import numpy as np
import compute_model_grid_8qso_fast as cmg8
from astropy.table import Table
import sys
sys.path.append('/Users/suksientie/codes/enigma')
from enigma.reion_forest import utils
import scipy

def npix_vs_ivar():
    given_bins = ccf.custom_cf_bin4(dv1=80)
    lowz_cgm_fit_gpm, highz_cgm_fit_gpm, allz_cgm_fit_gpm = ccf.init_cgm_fit_gpm(do_not_apply_any_mask=True)
    redshift_bin = 'all'
    nqso = 8

    vel_mid, xi_mean_unmask, xi_mean_mask, xi_noise_unmask, xi_noise_mask, xi_unmask, xi_mask, w_masked, w_unmasked = \
        ccf.allspec(nqso, redshift_bin, allz_cgm_fit_gpm, given_bins=given_bins, ivar_weights=True)

    vel_mid2, xi_mean_unmask2, xi_mean_mask2, xi_noise_unmask2, xi_noise_mask2, xi_unmask2, xi_mask2, w_masked2, w_unmasked2 = \
        ccf.allspec(nqso, redshift_bin, allz_cgm_fit_gpm, given_bins=given_bins, ivar_weights=False)

    qso_namelist = ['J0411-0907', 'J0319-1008', 'J0410-0139', 'J0038-0653', 'J0313-1806', 'J0038-1527', 'J0252-0503', 'J1342+0928']

    plt.figure(figsize=(14,7))
    for i in range(nqso):
        plt.subplot(2, 4, i + 1)
        plt.title(qso_namelist[i])
        plt.plot(vel_mid, xi_mask[i], 'k', label='ivar')
        plt.plot(vel_mid2, xi_mask2[i], 'k', alpha=0.5, label='npix')
        plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 7))
    for i in range(nqso):
        plt.subplot(2, 4, i + 1)
        plt.title(qso_namelist[i])
        plt.plot(vel_mid, xi_unmask[i], 'k', label='ivar')
        plt.plot(vel_mid2, xi_unmask2[i], 'k', alpha=0.5, label='npix')
        plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.title('before cgm masking')
    plt.plot(vel_mid, xi_mean_unmask, 'k', label='ivar')
    plt.plot(vel_mid2, xi_mean_unmask2, 'k', alpha=0.5, label='npix')
    plt.legend()

    plt.subplot(122)
    plt.title('after cgm masking')
    plt.plot(vel_mid, xi_mean_mask, 'k', label='ivar')
    plt.plot(vel_mid2, xi_mean_mask2, 'k', alpha=0.5, label='npix')
    plt.legend()
    plt.show()

def ccf_subtract_mean_deltaf():

    given_bins = ccf.custom_cf_bin4(dv1=80)
    lowz_cgm_fit_gpm, highz_cgm_fit_gpm, allz_cgm_fit_gpm = ccf.init_cgm_fit_gpm(do_not_apply_any_mask=True)
    redshift_bin = 'all'
    nqso = 8

    vel_mid, xi_mean_unmask, xi_mean_mask, xi_noise_unmask, xi_noise_mask, xi_unmask, xi_mask, w_masked, w_unmasked = \
        ccf.allspec(nqso, redshift_bin, allz_cgm_fit_gpm, given_bins=given_bins, ivar_weights=True, subtract_mean_deltaf=True)

    """
    hdulist = fits.HDUList()
    hdulist.append(fits.ImageHDU(vel_mid, name='vel_mid'))
    hdulist.append(fits.ImageHDU(xi_mean_mask, name='xi_mean_mask'))
    hdulist.append(fits.ImageHDU(xi_mean_unmask, name='xi_mean_unmask'))
    hdulist.append(fits.ImageHDU(xi_mask, name='xi_mask'))
    hdulist.append(fits.ImageHDU(xi_unmask, name='xi_unmask'))
    hdulist.writeto('plots/8qso-debug/cf_8qso_allz_ivarweights_everyn60_subtract-df.fits')
    """

    vel_mid2, xi_mean_unmask2, xi_mean_mask2, xi_noise_unmask2, xi_noise_mask2, xi_unmask2, xi_mask2, w_masked2, w_unmasked2 = \
        ccf.allspec(nqso, redshift_bin, allz_cgm_fit_gpm, given_bins=given_bins, ivar_weights=True, subtract_mean_deltaf=False)

    qso_namelist = ['J0411-0907', 'J0319-1008', 'J0410-0139', 'J0038-0653', 'J0313-1806', 'J0038-1527', 'J0252-0503',
                    'J1342+0928']

    plt.figure(figsize=(14, 7))
    for i in range(nqso):
        plt.subplot(2, 4, i + 1)
        plt.title(qso_namelist[i])
        plt.plot(vel_mid, xi_mask[i], 'k', label='subtract')
        plt.plot(vel_mid2, xi_mask2[i], 'k', alpha=0.5, label='no-subtract (old)')
        plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 7))
    for i in range(nqso):
        plt.subplot(2, 4, i + 1)
        plt.title(qso_namelist[i])
        plt.plot(vel_mid, xi_unmask[i], 'k', label='subtract')
        plt.plot(vel_mid2, xi_unmask2[i], 'k', alpha=0.5, label='no-subtract (old)')
        plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.title('before cgm masking')
    plt.plot(vel_mid, xi_mean_unmask, 'k', label='subtract')
    plt.plot(vel_mid2, xi_mean_unmask2, 'k', alpha=0.5, label='no-subtract (old)')
    plt.legend()

    plt.subplot(122)
    plt.title('after cgm masking')
    plt.plot(vel_mid, xi_mean_mask, 'k', label='subtract')
    plt.plot(vel_mid2, xi_mean_mask2, 'k', alpha=0.5, label='no-subtract (old)')
    plt.legend()
    plt.show()

##### negative CF bins #####
def debug_neg_cfbins(delta_f, vel_spec, gpm, neg_lags):

    (v_lo, v_hi) = ccf.custom_cf_bin4(dv1=80)
    v_mid = (v_hi + v_lo) / 2.0
    ncorr = len(v_mid)

    data = np.array([vel_spec])
    data = data.transpose()
    tree = KDTree(data)

    npix_forest = len(vel_spec)
    df_neglag = []
    df_poslag = []

    for iv in range(ncorr):
        ind, dist = tree.query_radius(data, v_hi[iv], return_distance=True)
        for idx in range(npix_forest):
            ibin = (dist[idx] > v_lo[iv]) & (dist[idx] <= v_hi[iv])
            ind_neigh = (ind[idx])[ibin]
            n_neigh = np.sum(ibin)
            #tmp = np.tile(delta_f[:, idx] * gpm[:, idx], (n_neigh, 1)).T * (delta_f[:, ind_neigh] * gpm[:, ind_neigh])

            if gpm[idx]:
                if v_mid[iv] == neg_lags:
                    df_neglag.append(idx)

    return df_neglag, df_poslag

def get_neg_lags():

    given_bins = ccf.custom_cf_bin4(dv1=80)
    lowz_cgm_fit_gpm, highz_cgm_fit_gpm, allz_cgm_fit_gpm = ccf.init_cgm_fit_gpm(do_not_apply_any_mask=True)
    vel_mid, xi_mean_unmask, xi_mean_mask, xi_noise_unmask, xi_noise_mask, xi_unmask, xi_mask, w_masked, w_unmasked = \
        ccf.allspec(10, 'all', allz_cgm_fit_gpm, given_bins = given_bins, ivar_weights=True)

    neg_lags = vel_mid[np.argwhere(xi_mean_mask < 0)].flatten()

    return neg_lags, allz_cgm_fit_gpm

def get_df_velspec_onespec(iqso, redshift_bin):

    corr_all = [1, 1, 1, 0.73, 0.697, 0.653, 0.667, 0.72, 1, 1]

    # everyn = 60 (all-z, high-z, low-z), nqso=10
    # normalized ivar, weighted fmean of dataset
    fmean_global_unmask = [0.9952775044785822, 0.998804106985206, 0.9927064690772747]
    fmean_global_mask = [1.001426984289517, 1.0018512443910101, 1.0011034891413138]
    if redshift_bin == 'all':
        i_fmean = 0
    elif redshift_bin == 'high':
        i_fmean = 1
    elif redshift_bin == 'low':
        i_fmean = 2

    fmean_unmask = fmean_global_unmask[i_fmean]
    fmean_mask = fmean_global_mask[i_fmean]

    raw_data_out, _, all_masks_out = mutils.init_onespec(iqso, redshift_bin)
    wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
    strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, telluric_mask, master_mask = all_masks_out

    ivar *= (fluxfit**2) # normalize by conts
    ivar *= (1/corr_all[iqso]**2) # apply correction

    norm_flux = flux / fluxfit
    vel = mutils.obswave_to_vel_2(wave)
    meanflux_tot_mask = fmean_mask
    deltaf_tot_mask = (norm_flux - meanflux_tot_mask) / meanflux_tot_mask

    return deltaf_tot_mask, vel, master_mask, flux

def remove_qso(cgm_mask):
    nqso = 10
    redshift_bin = 'all'
    given_bins = ccf.custom_cf_bin4(dv1=80)
    ivar_weights = True
    #iqso_remove_ls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, [4, 5, 6]]

    #iqso_to_use_ls = [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6]]
    iqso_to_use_ls = [[4], [5], [6], [4, 5], [4, 6], [5, 6], [4, 5, 6]]
    #iqso_to_use_ls = [[0, 1, 2, 3], [0, 1, 2, 3, 5, 6], [0, 1, 2, 3, 5, 6, 7, 8, 9]]

    xi_mean_mask_remove = []
    xi_mean_unmask_remove = []
    for elem in iqso_to_use_ls:
        #iqso_to_use = elem #np.delete(np.arange(nqso), elem)
        iqso_to_use = np.delete(np.arange(nqso), elem)
        vel_mid, xi_mean_unmask, xi_mean_mask, xi_noise_unmask, xi_noise_mask, xi_unmask, xi_mask, w_masked, w_unmasked = \
            ccf.allspec(nqso, redshift_bin, cgm_mask, given_bins=given_bins, iqso_to_use=iqso_to_use,
                    ivar_weights=ivar_weights, \
                    subtract_mean_deltaf=False)

        xi_mean_mask_remove.append(xi_mean_mask)
        xi_mean_unmask_remove.append(xi_mean_unmask)
    return xi_mean_mask_remove, xi_mean_unmask_remove, iqso_to_use_ls

def contfit_effect():
    # paper_plots/10qso/cf_compare_everyn_v2.png
    xi_20 = np.load('save_cf/xi_mean_mask_10qso_everyn20.npy')
    xi_40 = np.load('save_cf/xi_mean_mask_10qso_everyn40.npy')
    xi_60 = np.load('save_cf/xi_mean_mask_10qso_everyn60.npy')
    xi_80 = np.load('save_cf/xi_mean_mask_10qso_everyn80.npy')
    xi_mean_fluxfit = np.load('save_cf/xi_mean_mask_10qso_mean_fluxfit2.npy')
    xi_ls = [xi_20, xi_40, xi_60, xi_80]
    label = np.arange(20, 100, 20).astype(int)

    fitsfile = fits.open('save_cf/xi_mean_mask_10qso_everyn60.fits')
    vel_mid = fitsfile['VEL_MID'].data

    for i, xi in enumerate(xi_ls):
        plt.plot(vel_mid, xi, label='everyn=%d (%d km/s)' % (label[i], label[i]*40), alpha=0.75)

    plt.plot(vel_mid, xi_mean_fluxfit, 'k', label='mean continuum')
    plt.ylabel('xi')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()





##### MCMC #####
import mcmc_inference as mcmc
from enigma.reion_forest.utils import find_closest
import time

def mockdata_lnlike_max(xhi_guess, logZ_guess):

    # return the max log like and the CF computed from a number of mock data
    # ~1 sec per mock computing time
    start = time.process_time()
    modelfile = '/Users/suksientie/Research/MgII_forest/igm_cluster/10qso/corr_func_models_all_ivarweights.fits'
    nmock = 500 #1000

    lnl_fine_max = []
    xi_data_allmocks = []

    for imock in range(nmock):
        fine_out, coarse_out, data_out, imock, ixhi, iZ = mcmc.init_mockdata(modelfile, xhi_guess, logZ_guess, imock=imock)
        xhi_fine, logZ_fine, lnlike_fine, xi_model_fine = fine_out
        xhi_coarse, logZ_coarse, lnlike_coarse = coarse_out
        xhi_data, logZ_data, xi_data, covar_array, params = data_out

        lnl_fine_max.append(np.max(lnlike_fine))
        xi_data_allmocks.append(xi_data)

    end = time.process_time()
    print((end-start))

    return lnl_fine_max, xi_data_allmocks

def plot_cf_corr(xi_real_data, xi_data_allmocks, ibin, v_mid, neg_lags, saveplot=False, xi_mask_allqso=None):

    #neg_lags, _ = get_neg_lags()
    ibin_neg = []
    for neg in neg_lags:
        ibin_neg.append(np.argwhere(v_mid == neg)[0][0])

    plt.figure(figsize=(13, 8)) # 6x5
    for j in range(len(v_mid)):
        plt.subplot(5, 6, j + 1)
        if j != ibin:
            plt.plot(xi_data_allmocks[:, ibin], xi_data_allmocks[:, j], 'kx')
            if j in ibin_neg:
                plt.plot(xi_real_data[ibin], xi_real_data[j], 'r+', ms=10, markeredgewidth = 2)
            else:
                plt.plot(xi_real_data[ibin], xi_real_data[j], 'g+', ms=10, markeredgewidth = 2)

            if xi_mask_allqso is not None:
                for iqso in range(len(xi_mask_allqso)):
                    plt.scatter(xi_mask_allqso[iqso][ibin], xi_mask_allqso[iqso][j], label=iqso, zorder=10)

            plt.xlabel('xi(dv=%d)' % v_mid[ibin])
            plt.ylabel('xi(dv=%d)' % v_mid[j])

    plt.tight_layout()
    if saveplot:
        plt.savefig('paper_plots/10qso/debug/cf_corr_dv%d.png' % v_mid[ibin])
    plt.show()

def plot_cf_corr_qso(xi_mask_allqso, xi_mean_data, ibin, v_mid):

    plt.figure(figsize=(13, 8))  # 6x5
    for j in range(len(v_mid)):
        plt.subplot(5, 6, j + 1)
        if j != ibin:
            plt.plot(xi_mean_data[ibin], xi_mean_data[j], 'kx', markeredgewidth = 2, zorder=20)
            for iqso in range(len(xi_mask_allqso)):
                plt.scatter(xi_mask_allqso[iqso][ibin], xi_mask_allqso[iqso][j], label=iqso)
    plt.legend()
    plt.tight_layout()
    plt.show()

##### checking forward models #####
def check_fm1(logZ):

    nires_fwhm = 111.03
    mosfire_fwhm = 83.05
    nires_sampling = 2.7
    mosfire_sampling = 2.78
    xshooter_fwhm = 42.8  # R=7000 quoted in Bosman+2017
    xshooter_sampling = 3.7  # https://www.eso.org/sci/facilities/paranal/instruments/xshooter/inst.html

    qso_fwhm = [nires_fwhm, nires_fwhm, nires_fwhm, mosfire_fwhm, mosfire_fwhm, mosfire_fwhm, mosfire_fwhm,
                mosfire_fwhm, nires_fwhm, xshooter_fwhm]
    qso_sampling = [nires_sampling, nires_sampling, nires_sampling, mosfire_sampling, mosfire_sampling,
                    mosfire_sampling, mosfire_sampling, mosfire_sampling, nires_sampling, xshooter_sampling]

    rantaufile = 'ran_skewers_z75_OVT_xHI_0.74_tau.fits'
    params = Table.read(rantaufile, hdu=1)
    skewers = Table.read(rantaufile, hdu=2)

    # NIRES fwhm and sampling (18 sec for 10,000 skewers on my Mac)
    vel_lores_nires, flux_lores_nires = utils.create_mgii_forest(params, skewers, logZ, nires_fwhm,
                                                                 sampling=nires_sampling, mockcalc=True)

    # MOSFIRE fwhm and sampling
    vel_lores_mosfire, flux_lores_mosfire = utils.create_mgii_forest(params, skewers, logZ, mosfire_fwhm,
                                                                     sampling=mosfire_sampling, mockcalc=True)

    # xshooter fwhm and sampling
    vel_lores_xshooter, flux_lores_xshooter = utils.create_mgii_forest(params, skewers, logZ, xshooter_fwhm,
                                                                 sampling=xshooter_sampling, mockcalc=True)
    dv_coarse = 40
    vel_lores_nires_interp = np.arange(vel_lores_nires[0], vel_lores_nires[-1], dv_coarse)
    flux_lores_nires_interp = scipy.interpolate.interp1d(vel_lores_nires, flux_lores_nires, kind='cubic', \
                                                         bounds_error=False, fill_value=np.nan)(vel_lores_nires_interp)

    vel_lores_mosfire_interp = np.arange(vel_lores_mosfire[0], vel_lores_mosfire[-1], dv_coarse)
    flux_lores_mosfire_interp = scipy.interpolate.interp1d(vel_lores_mosfire, flux_lores_mosfire, kind='cubic',
                                                           bounds_error=False, fill_value=np.nan)(vel_lores_mosfire_interp)

    vel_lores_xshooter_interp = np.arange(vel_lores_xshooter[0], vel_lores_xshooter[-1], dv_coarse)
    flux_lores_xshooter_interp = scipy.interpolate.interp1d(vel_lores_xshooter, flux_lores_xshooter, kind='cubic',
                                                           bounds_error=False, fill_value=np.nan)(vel_lores_xshooter_interp)

    out1 = vel_lores_nires_interp, flux_lores_nires_interp, vel_lores_mosfire_interp, flux_lores_mosfire_interp, vel_lores_xshooter_interp, flux_lores_xshooter_interp
    return out1

def check_fm2(vel_data, norm_std, master_mask, nmock, rand, std_corr, instr, out1):

    vel_lores_nires_interp, flux_lores_nires_interp, vel_lores_mosfire_interp, flux_lores_mosfire_interp, vel_lores_xshooter_interp, flux_lores_xshooter_interp = out1

    if instr == 'nires':
        vel_lores = vel_lores_nires_interp
        flux_lores = flux_lores_nires_interp
    elif instr == 'mosfire':
        vel_lores = vel_lores_mosfire_interp
        flux_lores = flux_lores_mosfire_interp
    elif instr == 'xshooter':
        vel_lores = vel_lores_xshooter_interp
        flux_lores = flux_lores_xshooter_interp

    vel_lores, flux_noise_ncopy, flux_noiseless_ncopy, master_mask_chunk, nskew_to_match_data, npix_sim_skew = \
        cmg8.forward_model_onespec_chunk(vel_data, norm_std, master_mask, vel_lores, flux_lores, nmock, seed=rand,
                                    std_corr=std_corr)

    return vel_lores, flux_noise_ncopy, flux_noiseless_ncopy, master_mask_chunk

def check_fm3(out1, plot=True):

    qso_namelist = ['J0411-0907', 'J0319-1008', 'J0410-0139', 'J0038-0653', 'J0313-1806', 'J0038-1527', 'J0252-0503', \
                    'J1342+0928', 'J1007+2115', 'J1120+0641']
    corr_all = [0.669, 0.673, 0.692, 0.73, 0.697, 0.653, 0.667, 0.72, 0.64, 0.64]
    #corr_all = np.ones(10)
    rand = np.random.RandomState(2001)
    nmock = 1000

    vel_data_allqso, norm_flux_allqso, norm_std_allqso, norm_ivar_allqso, master_mask_allqso, master_mask_allqso_mask_cgm, instr_allqso = \
        cmg8.init_dataset(10, 'all', '/Users/suksientie/Research/MgII_forest/rebinned_spectra/')

    if plot:
        plt.figure(figsize=(14,8))

    for i in range(len(vel_data_allqso)):
        vel_lores, flux_noise_ncopy, flux_noiseless_ncopy, master_mask_chunk = check_fm2(vel_data_allqso[i], norm_std_allqso[i], \
master_mask_allqso_mask_cgm[i], nmock, rand, corr_all[i], instr_allqso[i], out1)

        m = (1 - flux_noise_ncopy) < 0.3
        master_mask_chunk_tile = np.tile(master_mask_chunk, (nmock, 1, 1)) * m

        npix_before = np.sum(master_mask_chunk)
        npix_after = np.sum(master_mask_chunk_tile)/nmock
        print(npix_after/npix_before)

        if plot:
            plt.subplot(2, 5, i+1)
            plt.title(qso_namelist[i])
            plt.hist(flux_noise_ncopy[master_mask_chunk_tile].flatten(), bins=np.arange(0, 1.5, 0.02), color='k', lw=1.5, histtype = 'step', density = True, label='FM')
            plt.hist(norm_flux_allqso[i][master_mask_allqso_mask_cgm[i]], bins = np.arange(0, 1.5, 0.02), color='r', lw=1.5, histtype = 'step', density = True, label='data')
            #plt.hist(flux_noise_ncopy.flatten(), bins=np.arange(0, 1.2, 0.02), color='k', alpha=0.5, histtype='step', density=True)
            #plt.hist(norm_flux_allqso[i], bins=np.arange(0, 1.2, 0.01), color='r', alpha=0.5, histtype='step', density=True)
            #print(np.nanmean(flux_noise_ncopy[master_mask_chunk_tile].flatten()), np.nanmean(norm_flux_allqso[i][master_mask_allqso_mask_cgm[i]]))
            plt.legend(loc=2)
            plt.xlabel(r'$F_{\mathrm{norm}}$')
    if plot:
        plt.tight_layout()
        plt.show()
