'''
Functions here:
    - init
    - init_mockdata
    - run_mcmc
    - corrfunc_plot

    (misc functions)
    - plot_corrmatrix
    - corr_matrix
    - plot_single_corr_elem
    - lnlike_plot_slice
    - lnlike_plot_logZ_many
    - lnlike_plot_xhi_many
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import emcee
import corner
import time
from scipy import optimize, interpolate
from IPython import embed
import pdb
#import sys
#sys.path.append('/Users/suksientie/codes/enigma')
from enigma.reion_forest.compute_model_grid import read_model_grid
from enigma.reion_forest.utils import find_closest, vel_mgii
from enigma.reion_forest import inference
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.ticker import AutoMinorLocator
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.io import fits
import compute_cf_data
import compute_model_grid_new as cmg
import argparse
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
import mutils

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--modelfile', type=str)
args = parser.parse_args()

seed = 711019 #args.seed
seed = None
if seed == None:
    seed = np.random.randint(0, 1000000)
    print(seed)

rand = np.random.RandomState(seed)
nqso = 10

def init(modelfile, redshift_bin, given_bins, lag_mask=None, figpath=None, xi_mean_data=None, covar_array_fine=None, lnlike_fine_in=None):
    """
    redshift_bin: 'all', 'low', 'high'
    xi_mean_data: data measurement (if not provided, then calculate on the fly)
    covar_array_fine: finely-interpolated covariance array; if provided, lnlike_fine will be calculated from this
    """
    params, xi_mock_array, xi_model_array, covar_array, icovar_array, lndet_array = read_model_grid(modelfile)
    logZ_coarse = params['logZ'].flatten()
    xhi_coarse = params['xhi'].flatten()
    vel_corr = params['vel_mid'].flatten()
    #vel_min = params['vmin_corr'][0]
    #vel_max = params['vmax_corr'][0]
    nlogZ = params['nlogZ'][0]
    nhi = params['nhi'][0]

    xhi_data, logZ_data = 0.5, -3.50  # bogus numbers
    seed_list = [None] * nqso

    if xi_mean_data is None:
        # using the chunked data for the MCMC...
        #cgm_fit_gpm_all, _ = cmg.init_cgm_masking(redshift_bin, datapath='/Users/suksientie/Research/data_redux/') # only CGM masks, other masks added later in ccf.onespec_chunk
        #vel_mid, xi_mean_unmask, xi_mean_mask = compute_cf_data.allspec_chunk(nqso, cgm_fit_gpm_all, redshift_bin, vel_lores, given_bins=given_bins)

        # using the data in its original format for MCMC
        #lowz_cgm_fit_gpm, highz_cgm_fit_gpm, allz_cgm_fit_gpm = compute_cf_data.init_cgm_fit_gpm()
        lowz_cgm_fit_gpm, highz_cgm_fit_gpm, allz_cgm_fit_gpm = compute_cf_data.init_cgm_fit_gpm(do_not_apply_any_mask=True)
        if redshift_bin == 'low':
            cgm_fit_gpm_all = lowz_cgm_fit_gpm
        elif redshift_bin == 'high':
            cgm_fit_gpm_all = highz_cgm_fit_gpm
        elif redshift_bin == 'all':
            cgm_fit_gpm_all = allz_cgm_fit_gpm

        #iqso_to_use = range(3, nqso)
        #vel_mid, xi_mean_unmask, xi_mean_mask, _, _, _, _ = compute_cf_data.allspec(nqso, redshift_bin, cgm_fit_gpm_all, plot=False, seed_list=seed_list, given_bins=given_bins, iqso_to_use=iqso_to_use)

        vel_mid, xi_mean_unmask, xi_mean_mask, _, _, _, _, _, _  = compute_cf_data.allspec(nqso, redshift_bin, cgm_fit_gpm_all, \
                                                plot=False, given_bins=given_bins, iqso_to_use=None, ivar_weights=True)
        #import cf_chunk_check as ccc
        #vel_mid, xi_avg, xi_avg_chunk = ccc.check_allspec()
        #xi_mean_mask = xi_avg_chunk

        xi_data = xi_mean_mask
    else:
        xi_data = xi_mean_data

    #lag_mask = mutils.cf_lags_to_mask()
    #xi_data = xi_data[lag_mask]
    #xi_mask = np.ones_like(xi_data, dtype=bool)  # Boolean array

    if lag_mask is not None:
        xi_model_array, xi_mock_array, covar_array, lndet_array = mutils.extract_subarr(lag_mask, xi_model_array, xi_mock_array, covar_array)
        xi_data = xi_data[lag_mask]

    xi_mask = np.ones_like(xi_data, dtype=bool)  # Boolean array

    # Interpolate the likelihood onto a fine grid to speed up the MCMC
    nlogZ = logZ_coarse.size
    nlogZ_fine = 1001
    logZ_fine_min = logZ_coarse.min()
    logZ_fine_max = logZ_coarse.max()
    dlogZ_fine = (logZ_fine_max - logZ_fine_min) / (nlogZ_fine - 1)
    logZ_fine = logZ_fine_min + np.arange(nlogZ_fine) * dlogZ_fine

    nhi = xhi_coarse.size
    nhi_fine = 1001
    xhi_fine_min = 0.0
    xhi_fine_max = 1.0
    dxhi = (xhi_fine_max - xhi_fine_min) / (nhi_fine - 1)
    xhi_fine = np.arange(nhi_fine) * dxhi

    # Loop over the coarse grid and evaluate the likelihood at each location
    lnlike_coarse = np.zeros((nhi, nlogZ,))
    for ixhi, xhi in enumerate(xhi_coarse):
        for iZ, logZ in enumerate(logZ_coarse):
            lnlike_coarse[ixhi, iZ] = inference.lnlike_calc(xi_data, xi_mask, xi_model_array[ixhi, iZ, :], lndet_array[ixhi, iZ],
                                                  covar_array[ixhi, iZ, :, :])

    xi_model_fine = inference.interp_model(xhi_fine, logZ_fine, xhi_coarse, logZ_coarse, xi_model_array)

    if lnlike_fine_in is not None:
        print("using input lnlike_fine")
        lnlike_fine = lnlike_fine_in
    else:
        if covar_array_fine is not None:
            print("using interpolated covariance matrix to compute lnL")
            lndet_array_fine = inference.interp_lnlike(xhi_fine, logZ_fine, xhi_coarse, logZ_coarse, lndet_array)
            lnlike_fine = np.zeros((nhi_fine, nlogZ_fine,))

            for ixhi, xhi in enumerate(xhi_fine):
                for iZ, logZ in enumerate(logZ_fine):
                    lnlike_fine[ixhi, iZ] = inference.lnlike_calc(xi_data, xi_mask, xi_model_fine[ixhi, iZ, :],
                                                                    lndet_array_fine[ixhi, iZ],
                                                                    covar_array_fine[ixhi, iZ, :, :])
        else:
            lnlike_fine = inference.interp_lnlike(xhi_fine, logZ_fine, xhi_coarse, logZ_coarse, lnlike_coarse, kx=3, ky=3)

    # Make a 2d surface plot of the likelhiood
    logZ_fine_2d, xhi_fine_2d = np.meshgrid(logZ_fine, xhi_fine)
    if figpath is not None:
        lnlikefile = figpath + redshift_bin + 'z_lnlike.pdf'
        inference.lnlike_plot(xhi_fine_2d, logZ_fine_2d, lnlike_fine, lnlikefile)

    fine_out = xhi_fine, logZ_fine, lnlike_fine, xi_model_fine
    coarse_out = xhi_coarse, logZ_coarse, lnlike_coarse
    data_out = xhi_data, logZ_data, xi_data, covar_array, params

    return fine_out, coarse_out, data_out

def init_mockdata(modelfile, xhi_guess, logZ_guess, imock=None):

    # modelfile = 'igm_cluster/corr_func_models_fwhm_90.000_samp_3.000.fits'
    params, xi_mock_array, xi_model_array, covar_array, icovar_array, lndet_array = read_model_grid(modelfile)
    logZ_coarse = params['logZ'].flatten()
    xhi_coarse = params['xhi'].flatten()
    vel_corr = params['vel_mid'].flatten()
    #vel_min = params['vmin_corr'][0]
    #vel_max = params['vmax_corr'][0]
    nlogZ = params['nlogZ'][0]
    nhi = params['nhi'][0]

    if imock is None:
        # Pick the mock data that we will run with
        nmock = xi_mock_array.shape[2]
        imock = rand.choice(np.arange(nmock), size=1)

    # find the closest model values to guesses
    ixhi = find_closest(xhi_coarse, xhi_guess)
    iZ = find_closest(logZ_coarse, logZ_guess)
    print("imock, ixhi, iZ", imock, ixhi, iZ)

    xhi_data = xhi_coarse[ixhi]
    logZ_data = logZ_coarse[iZ]
    xi_data = xi_mock_array[ixhi, iZ, imock, :].flatten()
    xi_mask = np.ones_like(xi_data, dtype=bool) # Boolean array

    # Interpolate the likelihood onto a fine grid to speed up the MCMC
    nlogZ = logZ_coarse.size
    nlogZ_fine = 1001
    logZ_fine_min = logZ_coarse.min()
    logZ_fine_max = logZ_coarse.max()
    dlogZ_fine = (logZ_fine_max - logZ_fine_min) / (nlogZ_fine - 1)
    logZ_fine = logZ_fine_min + np.arange(nlogZ_fine) * dlogZ_fine

    nhi = xhi_coarse.size
    nhi_fine = 1001
    xhi_fine_min = 0.0
    xhi_fine_max = 1.0
    dxhi = (xhi_fine_max - xhi_fine_min) / (nhi_fine - 1)
    xhi_fine = np.arange(nhi_fine) * dxhi

    # Loop over the coarse grid and evaluate the likelihood at each location
    lnlike_coarse = np.zeros((nhi, nlogZ,))
    for ixhi, xhi in enumerate(xhi_coarse):
        for iZ, logZ in enumerate(logZ_coarse):
            lnlike_coarse[ixhi, iZ] = inference.lnlike_calc(xi_data, xi_mask, xi_model_array[ixhi, iZ, :], lndet_array[ixhi, iZ],
                                                  covar_array[ixhi, iZ, :, :])

    lnlike_fine = inference.interp_lnlike(xhi_fine, logZ_fine, xhi_coarse, logZ_coarse, lnlike_coarse)
    xi_model_fine = inference.interp_model(xhi_fine, logZ_fine, xhi_coarse, logZ_coarse, xi_model_array)

    # Make a 2d surface plot of the likelhiood
    logZ_fine_2d, xhi_fine_2d = np.meshgrid(logZ_fine, xhi_fine)
    #lnlikefile = figpath + 'lnlike.pdf'
    lnlikefile = None
    #inference.lnlike_plot(xhi_fine_2d, logZ_fine_2d, lnlike_fine, lnlikefile)

    fine_out = xhi_fine, logZ_fine, lnlike_fine, xi_model_fine
    coarse_out = xhi_coarse, logZ_coarse, lnlike_coarse
    data_out = xhi_data, logZ_data, xi_data, covar_array, params

    return fine_out, coarse_out, data_out, imock, ixhi, iZ

def new_init(modelfile, redshift_bin, given_bins, lag_mask=None, figpath=None, xi_mean_data=None, covar_array_fine=None):

    params, xi_mock_array, xi_model_array, covar_array, icovar_array, lndet_array = read_model_grid(modelfile)
    logZ_coarse = params['logZ'].flatten()
    xhi_coarse = params['xhi'].flatten()
    vel_corr = params['vel_mid'].flatten()
    #vel_min = params['vmin_corr'][0]
    #vel_max = params['vmax_corr'][0]
    nlogZ = params['nlogZ'][0]
    nhi = params['nhi'][0]

    xhi_data, logZ_data = 0.5, -3.50  # bogus numbers
    seed_list = [None] * nqso

    if xi_mean_data is None:
        # using the chunked data for the MCMC...
        #cgm_fit_gpm_all, _ = cmg.init_cgm_masking(redshift_bin, datapath='/Users/suksientie/Research/data_redux/') # only CGM masks, other masks added later in ccf.onespec_chunk
        #vel_mid, xi_mean_unmask, xi_mean_mask = compute_cf_data.allspec_chunk(nqso, cgm_fit_gpm_all, redshift_bin, vel_lores, given_bins=given_bins)

        # using the data in its original format for MCMC
        #lowz_cgm_fit_gpm, highz_cgm_fit_gpm, allz_cgm_fit_gpm = compute_cf_data.init_cgm_fit_gpm()
        lowz_cgm_fit_gpm, highz_cgm_fit_gpm, allz_cgm_fit_gpm = compute_cf_data.init_cgm_fit_gpm(do_not_apply_any_mask=True)
        if redshift_bin == 'low':
            cgm_fit_gpm_all = lowz_cgm_fit_gpm
        elif redshift_bin == 'high':
            cgm_fit_gpm_all = highz_cgm_fit_gpm
        elif redshift_bin == 'all':
            cgm_fit_gpm_all = allz_cgm_fit_gpm

        #iqso_to_use = range(3, nqso)
        #vel_mid, xi_mean_unmask, xi_mean_mask, _, _, _, _ = compute_cf_data.allspec(nqso, redshift_bin, cgm_fit_gpm_all, plot=False, seed_list=seed_list, given_bins=given_bins, iqso_to_use=iqso_to_use)

        vel_mid, xi_mean_unmask, xi_mean_mask, _, _, _, _, _, _  = compute_cf_data.allspec(nqso, redshift_bin, cgm_fit_gpm_all, \
                                                plot=False, given_bins=given_bins, iqso_to_use=None, ivar_weights=True)
        #import cf_chunk_check as ccc
        #vel_mid, xi_avg, xi_avg_chunk = ccc.check_allspec()
        #xi_mean_mask = xi_avg_chunk

        xi_data = xi_mean_mask
    else:
        xi_data = xi_mean_data

    if lag_mask is not None:
        xi_model_array, xi_mock_array, covar_array, lndet_array = mutils.extract_subarr(lag_mask, xi_model_array, xi_mock_array, covar_array)
        xi_data = xi_data[lag_mask]

    xi_mask = np.ones_like(xi_data, dtype=bool)  # Boolean array

    # Interpolate the likelihood onto a fine grid to speed up the MCMC
    nlogZ = logZ_coarse.size
    nlogZ_fine = 1001
    logZ_fine_min = logZ_coarse.min()
    logZ_fine_max = logZ_coarse.max()
    dlogZ_fine = (logZ_fine_max - logZ_fine_min) / (nlogZ_fine - 1)
    logZ_fine = logZ_fine_min + np.arange(nlogZ_fine) * dlogZ_fine

    nhi = xhi_coarse.size
    nhi_fine = 1001
    xhi_fine_min = 0.0
    xhi_fine_max = 1.0
    dxhi = (xhi_fine_max - xhi_fine_min) / (nhi_fine - 1)
    xhi_fine = np.arange(nhi_fine) * dxhi

    # Loop over the coarse grid and evaluate the likelihood at each location
    lnlike_coarse = np.zeros((nhi, nlogZ,))
    for ixhi, xhi in enumerate(xhi_coarse):
        for iZ, logZ in enumerate(logZ_coarse):
            lnlike_coarse[ixhi, iZ] = inference.lnlike_calc(xi_data, xi_mask, xi_model_array[ixhi, iZ, :], lndet_array[ixhi, iZ],
                                                  covar_array[ixhi, iZ, :, :])

    xi_model_fine = inference.interp_model(xhi_fine, logZ_fine, xhi_coarse, logZ_coarse, xi_model_array)

    if covar_array_fine is not None:
        print("using interpolated covariance matrix to compute lnL")
        lndet_array_fine = inference.interp_lnlike(xhi_fine, logZ_fine, xhi_coarse, logZ_coarse, lndet_array)
        lnlike_fine = np.zeros((nhi_fine, nlogZ_fine,))

        for ixhi, xhi in enumerate(xhi_fine):
            for iZ, logZ in enumerate(logZ_fine):
                lnlike_fine[ixhi, iZ] = inference.lnlike_calc(xi_data, xi_mask, xi_model_fine[ixhi, iZ, :],
                                                              lndet_array_fine[ixhi, iZ],
                                                              covar_array_fine[ixhi, iZ, :, :])
    else:
        lnlike_fine = inference.interp_lnlike(xhi_fine, logZ_fine, xhi_coarse, logZ_coarse, lnlike_coarse, kx=3, ky=3)

    # Make a 2d surface plot of the likelhiood
    logZ_fine_2d, xhi_fine_2d = np.meshgrid(logZ_fine, xhi_fine)
    if figpath is not None:
        lnlikefile = figpath + redshift_bin + 'z_lnlike.pdf'
        inference.lnlike_plot(xhi_fine_2d, logZ_fine_2d, lnlike_fine, lnlikefile)

    fine_out = xhi_fine, logZ_fine, lnlike_fine, xi_model_fine
    coarse_out = xhi_coarse, logZ_coarse, lnlike_coarse
    data_out = xhi_data, logZ_data, xi_data, covar_array, params

    return fine_out, coarse_out, data_out

def run_mcmc(fine_out, coarse_out, data_out, redshift_bin, figpath, nsteps=100000, burnin=1000, nwalkers=40, \
             linearZprior=False, savefits_chain=None, actual_data=True, input_xi_err=None, inferred_model='mean', \
             lag_mask=None, plotcorrfunc=True):

    xhi_fine, logZ_fine, lnlike_fine, xi_model_fine = fine_out
    xhi_coarse, logZ_coarse, lnlike_coarse = coarse_out
    xhi_data, logZ_data, xi_data, covar_array, params = data_out

    # find optimal starting points for each walker
    chi2_func = lambda *args: -2 * inference.lnprob(*args)
    logZ_fine_min = logZ_fine.min()
    logZ_fine_max = logZ_fine.max()
    #bounds = [(0.8, 1.0), (logZ_fine_min, logZ_fine_max)] if not linearZprior else [(0.8, 1.0), (0.0, np.power(10.0,logZ_fine_max))]
    bounds = [(0.0, 1.0), (logZ_fine_min, logZ_fine_max)] if not linearZprior else [(0.0, 1.0), (0.0, np.power(10.0, logZ_fine_max))]
    print("bounds", bounds)
    args = (lnlike_fine, xhi_fine, logZ_fine, linearZprior)
    result_opt = optimize.differential_evolution(chi2_func,bounds=bounds, popsize=25, recombination=0.7,
                                                 disp=True, polish=True, args=args, seed=rand)
    ndim = 2 # xhi and logZ

    # initialize walkers about the maximum within a ball of 1% of the parameter domain
    pos = [[np.clip(result_opt.x[i] + 1e-2*(bounds[i][1] - bounds[i][0])*rand.randn(1)[0],bounds[i][0],bounds[i][1]) for i in range(ndim)] for i in range(nwalkers)]
    # randomly initialize walkers
    #print("random initialize walkers")
    #pos = [[bounds[i][0] + (bounds[i][1] - bounds[i][0])*rand.rand(1)[0] for i in range(ndim)] for i in range(nwalkers)]

    # I think this seeds the random number generator which will make the emcee results reproducible. Create an issue on this
    np.random.seed(rand.randint(0,seed, size=1)[0])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, inference.lnprob, args=args)
    sampler.run_mcmc(pos, nsteps, progress=True)

    try:
        tau = sampler.get_autocorr_time()
    except:
        print('Autocorr time does not converge')
    else:
        print('Autocorrelation time')
        print('tau_xhi = {:7.2f}, tau_logZ = {:7.2f}'.format(tau[0],tau[1]))
        acceptfrac = sampler.acceptance_fraction
        print("Acceptance fraction per walker")
        print(acceptfrac)

    flat_samples = sampler.get_chain(discard=burnin, thin=250, flat=True)
    if linearZprior:
        param_samples = flat_samples.copy()
        param_samples[:,1] = np.log10(param_samples[:,1])
    else:
        param_samples = flat_samples

    theta_mean = np.mean(param_samples, axis=0)
    # Average the diagonal instead?
    covar_mean = inference.covar_model(theta_mean, xhi_coarse, logZ_coarse, covar_array)
    xi_err = np.sqrt(np.diag(covar_mean))
    vel_corr = params['vel_mid'].flatten()
    if lag_mask is not None:
        vel_corr = vel_corr[lag_mask]

    if savefits_chain is not None:
        hdulist = fits.HDUList()
        hdulist.append(fits.ImageHDU(data=sampler.get_chain(), name='all_chain'))
        hdulist.append(fits.ImageHDU(data=sampler.get_chain(flat=True), name='all_chain_flat'))
        hdulist.append(
            fits.ImageHDU(data=sampler.get_chain(discard=burnin, flat=True), name='all_chain_discard_burnin'))
        hdulist.append(fits.ImageHDU(data=param_samples, name='param_samples'))
        hdulist.append(fits.ImageHDU(data=vel_corr, name='vel_corr'))
        hdulist.append(fits.ImageHDU(data=xi_err, name='xi_err'))
        hdulist.writeto(savefits_chain, overwrite=True)

    ############# make all MCMC plots #############
    # Make the walker plot, use the true values in the chain
    var_label = [r'$\langle x_{\rm HI}\rangle$', '[Mg/H]']
    truths = [xhi_data, np.power(10.0,logZ_data)] if linearZprior else [xhi_data, logZ_data]
    chain = sampler.get_chain()

    if linearZprior:
        inference.walker_plot(chain, truths, var_label, figpath + '%sz_walkers_upperlim.pdf' % redshift_bin)
    else:
        inference.walker_plot(chain, truths, var_label, figpath + '%sz_walkers.pdf' % redshift_bin)

    # Make the corner plot, again use the true values in the chain
    if actual_data:
        fig = corner.corner(param_samples, labels=var_label, levels=(0.68,), color='k',
                            show_titles=True, title_kwargs={"fontsize": 15}, label_kwargs={'fontsize': 20},
                            data_kwargs={'ms': 1.0, 'alpha': 0.1})
    else:
        fig = corner.corner(flat_samples, labels=var_label, truths=truths, levels = (0.68,), color='k', truth_color='darkgreen',
                            show_titles=True, title_kwargs={"fontsize": 15}, label_kwargs={'fontsize':20},
                            data_kwargs={'ms': 1.0, 'alpha': 0.1})

    if linearZprior:
        cornerfile = figpath + '%sz_corner_plot_upperlim.pdf' % redshift_bin
    else:
        cornerfile = figpath + '%sz_corner_plot.pdf' % redshift_bin

    for ax in fig.get_axes():
          #ax.tick_params(axis='both', which='major', labelsize=14)
          #ax.tick_params(axis='both', which='minor', labelsize=12)
          ax.tick_params(labelsize=12)
    plt.close()
    fig.savefig(cornerfile)

    lower = np.array([bounds[0][0], bounds[1][0]])
    upper = np.array([bounds[0][1], bounds[1][1]])
    param_limits = [lower, upper],
    #param_names = ['xHI', 'logZ']
    #labels = param_names
    #ranges = dict(zip(param_names, [[lower[i], upper[i]] for i in range(ndim)]))
    #triangle_plot([samples], param_names, labels, ranges, filename=figpath + 'triangle.pdf', show_plot=True)
    if linearZprior:
        corrfile = figpath + '%sz_corr_func_data_upperlim.pdf' % redshift_bin
    else:
        corrfile = figpath + '%sz_corr_func_data.pdf' % redshift_bin

    # Upper limit on metallicity for pristine case
    if linearZprior:
        ixhi_prior = flat_samples[:,0] > 0.95
        logZ_95 = np.percentile(param_samples[ixhi_prior,1], 95.0)
        print('Obtained 95% upper limit of {:6.4f}'.format(logZ_95))

    if plotcorrfunc:
        if actual_data:
            corrfunc_plot_new(xi_data, param_samples, params, xhi_fine, logZ_fine, xi_model_fine, xhi_coarse, logZ_coarse, covar_array,
                          corrfile, redshift_bin, nrand=200, rand=rand, input_xi_err=input_xi_err, inferred_model=inferred_model, lag_mask=lag_mask)
        else:
            save_xi_err = None
            #inference.corrfunc_plot(xi_data, param_samples, params, xhi_fine, logZ_fine, xi_model_fine, xhi_coarse, logZ_coarse, covar_array, xhi_data, logZ_data, corrfile, rand=rand)
            corrfunc_plot(xi_data, param_samples, params, xhi_fine, logZ_fine, xi_model_fine, xhi_coarse, logZ_coarse, covar_array,
                          corrfile, redshift_bin, nrand=300, rand=rand, save_xi_err=save_xi_err, inferred_model=inferred_model)

    extra_out = xi_data, param_samples, params, xhi_fine, logZ_fine, xi_model_fine, xhi_coarse, logZ_coarse, covar_array, corrfile, redshift_bin, rand
    return sampler, param_samples, flat_samples, extra_out

def run_mcmc_lite(fine_out, coarse_out, data_out, \
                  nsteps=100000, burnin=1000, nwalkers=40, linearZprior=False, savefits_chain=None):

    # for running many mocks for inference test
    xhi_fine, logZ_fine, lnlike_fine, xi_model_fine = fine_out
    xhi_coarse, logZ_coarse, lnlike_coarse = coarse_out
    xhi_data, logZ_data, xi_data, covar_array, params = data_out

    # find optimal starting points for each walker
    chi2_func = lambda *args: -2 * inference.lnprob(*args)
    logZ_fine_min = logZ_fine.min()
    logZ_fine_max = logZ_fine.max()
    #bounds = [(0.8, 1.0), (logZ_fine_min, logZ_fine_max)] if not linearZprior else [(0.8, 1.0), (0.0, np.power(10.0,logZ_fine_max))]
    bounds = [(0.0, 1.0), (logZ_fine_min, logZ_fine_max)] if not linearZprior else [(0.0, 1.0), (0.0, np.power(10.0, logZ_fine_max))]
    print("bounds", bounds)
    args = (lnlike_fine, xhi_fine, logZ_fine, linearZprior)
    result_opt = optimize.differential_evolution(chi2_func,bounds=bounds, popsize=25, recombination=0.7,
                                                 disp=True, polish=True, args=args, seed=rand)
    ndim = 2 # xhi and logZ

    # initialize walkers about the maximum within a ball of 1% of the parameter domain
    pos = [[np.clip(result_opt.x[i] + 1e-2*(bounds[i][1] - bounds[i][0])*rand.randn(1)[0],bounds[i][0],bounds[i][1]) for i in range(ndim)] for i in range(nwalkers)]
    # randomly initialize walkers
    #print("random initialize walkers")
    #pos = [[bounds[i][0] + (bounds[i][1] - bounds[i][0])*rand.rand(1)[0] for i in range(ndim)] for i in range(nwalkers)]

    # I think this seeds the random number generator which will make the emcee results reproducible. Create an issue on this
    np.random.seed(rand.randint(0,seed, size=1)[0])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, inference.lnprob, args=args)
    sampler.run_mcmc(pos, nsteps, progress=True)

    flat_samples = sampler.get_chain(discard=burnin, thin=250, flat=True)
    if linearZprior:
        param_samples = flat_samples.copy()
        param_samples[:, 1] = np.log10(param_samples[:, 1])
    else:
        param_samples = flat_samples

    theta_mean = np.mean(param_samples, axis=0)
    # Average the diagonal instead?
    covar_mean = inference.covar_model(theta_mean, xhi_coarse, logZ_coarse, covar_array)
    xi_err = np.sqrt(np.diag(covar_mean))
    vel_corr = params['vel_mid'].flatten()

    if savefits_chain is not None:
        hdulist = fits.HDUList()
        hdulist.append(fits.ImageHDU(data=sampler.get_chain(), name='all_chain'))
        hdulist.append(fits.ImageHDU(data=sampler.get_chain(flat=True), name='all_chain_flat'))
        hdulist.append(fits.ImageHDU(data=sampler.get_chain(discard=burnin, flat=True), name='all_chain_discard_burnin'))
        hdulist.append(fits.ImageHDU(data=param_samples, name='param_samples'))
        hdulist.append(fits.ImageHDU(data=vel_corr, name='vel_corr'))
        hdulist.append(fits.ImageHDU(data=xi_err, name='xi_err'))
        hdulist.writeto(savefits_chain, overwrite=True)

    return sampler, param_samples, flat_samples

def xi_err_for_masked_bins_hack(mcmc_outfits_unmasked, mcmc_outfits_masked, savefits):
    mcmc_out1 = fits.open(mcmc_outfits_unmasked)
    xi_err1 = mcmc_out1['xi_err'].data
    vel_corr1 = mcmc_out1['vel_corr'].data

    mcmc_out2 = fits.open(mcmc_outfits_masked)
    xi_err2 = mcmc_out2['xi_err'].data
    vel_corr2 = mcmc_out2['vel_corr'].data

    tmp_xi_err = []
    tmp_vel_corr = []
    for i in range(len(vel_corr1)):
        if vel_corr1[i] not in vel_corr2:
            tmp_xi_err.append(xi_err1[i])
            tmp_vel_corr.append(vel_corr1[i])

    a = np.concatenate((tmp_xi_err, xi_err2))
    b = np.concatenate((tmp_vel_corr, vel_corr2))

    i = np.argsort(b)
    final_xi_err = a[i]

    hdulist = fits.HDUList()
    hdulist.append(fits.ImageHDU(data=vel_corr1, name='vel_corr'))
    hdulist.append(fits.ImageHDU(data=final_xi_err, name='xi_err'))
    hdulist.writeto(savefits, overwrite=True)

    return vel_corr1, final_xi_err


################################## plotting ##################################
def old_corrfunc_plot(xi_data, samples, params, xhi_fine, logZ_fine, xi_model_fine, xhi_coarse, logZ_coarse, covar_array, \
                  corrfile, nrand=50, rand=None, save_xi_err=None, vel_mid_compare=None, xi_mean_compare=None, \
                  label_compare=None, plot_draws=True, inferred_model='mean'):

    # adapted from enigma.reion_forest.inference.corrfunc_plot
    if rand is None:
        rand = np.random.RandomState(1234)

    factor = 1e5
    fx = plt.figure(1, figsize=(12, 9))
    # left, bottom, width, height
    rect = [0.12, 0.12, 0.84, 0.75]
    axis = fx.add_axes(rect)
    vel_corr = params['vel_mid'].flatten()
    #vel_min = params['vmin_corr']
    #vel_max = params['vmax_corr']

    vmin, vmax = 0.4*vel_corr.min(), 1.02*vel_corr.max()
    # Compute the mean model from the samples
    xi_model_samp = inference.xi_model(samples, xhi_fine, logZ_fine, xi_model_fine)
    if inferred_model == 'mean':
        xi_model_mean = np.mean(xi_model_samp, axis=0)
    elif inferred_model == 'median':
        xi_model_mean = np.median(xi_model_samp, axis=0)
    # Compute the covariance at the mean model
    theta_mean = np.mean(samples,axis=0)
    # Average the diagonal instead?
    covar_mean = inference.covar_model(theta_mean, xhi_coarse, logZ_coarse, covar_array)
    xi_err = np.sqrt(np.diag(covar_mean))

    if save_xi_err is not None:
        np.save(save_xi_err, xi_err)

    # Grab some realizations
    imock = rand.choice(np.arange(samples.shape[0]), size=nrand)
    xi_model_rand = xi_model_samp[imock, :]
    ymin = factor*np.min(xi_data - 1.3*xi_err)
    ymax = factor*np.max(xi_data + 1.6*xi_err)

    axis.set_xlabel(r'$\Delta v$ (km/s)', fontsize=26)
    axis.set_ylabel(r'$\xi(\Delta v)\times 10^5$', fontsize=26, labelpad=-4)
    axis.tick_params(axis="x", labelsize=16)
    axis.tick_params(axis="y", labelsize=16)

    axis.errorbar(vel_corr, factor*xi_data, yerr=factor*xi_err, marker='o', ms=6, color='black', ecolor='black', capthick=2,
                  capsize=4, mec='none', ls='none', label='data', zorder=20)

    axis.plot(vel_corr, factor*xi_model_mean, linewidth=2.0, color='red', zorder=10, label='inferred model')

    if vel_mid_compare is not None:
        axis.plot(vel_mid_compare, factor * xi_mean_compare, linewidth=2.0, color='blue', zorder=10, label=label_compare)

    #true_xy = (vmin + 0.44*(vmax-vmin), 0.60*ymax)
    #xhi_xy  = (vmin + 0.385*(vmax-vmin), 0.52*ymax)
    #Z_xy    = (vmin + 0.283*(vmax-vmin), 0.44*ymax)
    #xhi_label  = r'$\langle x_{{\rm HI}}\rangle = {:3.2f}$'.format(xhi_data)
    #logZ_label = '      ' + r'$[{{\rm Mg\slash H}}]={:5.2f}$'.format(logZ_data)
    #axis.annotate('True', xy=true_xy, xytext=true_xy, textcoords='data', xycoords='data', color='darkgreen', annotation_clip=False,fontsize=16, zorder=25)
    #axis.annotate(xhi_label, xy=xhi_xy, xytext=xhi_xy, textcoords='data', xycoords='data', color='darkgreen', annotation_clip=False,fontsize=16, zorder=25)
    #axis.annotate(logZ_label, xy=Z_xy, xytext=Z_xy, textcoords='data', xycoords='data', color='darkgreen', annotation_clip=False,fontsize=16, zorder=25)

    # error bar
    percent_lower = (1.0-0.6827)/2.0
    percent_upper = 1.0 - percent_lower
    param = np.median(samples, axis=0)
    param_lower = param - np.percentile(samples, 100*percent_lower, axis=0)
    param_upper = np.percentile(samples, 100*percent_upper, axis=0) - param

    #infr_xy = (vmin + 0.1*(vmax-vmin), 0.90*ymax)
    #xhi_xy  = (vmin + 0.05*(vmax-vmin), 0.80*ymax)
    #Z_xy    = (vmin - 0.05*(vmax-vmin), 0.70*ymax)
    infr_xy = (1800, (0.55 * ymax))
    xhi_xy = (1700, (0.42 * ymax))
    Z_xy = (1355, (0.29 * ymax))

    xhi_label  = r'$\langle x_{{\rm HI}}\rangle = {:3.2f}^{{+{:3.2f}}}_{{-{:3.2f}}}$'.format(param[0], param_upper[0], param_lower[0])
    logZ_label = '          ' + r'$[{{\rm Mg\slash H}}]={:5.2f}^{{+{:3.2f}}}_{{-{:3.2f}}}$'.format(param[1], param_upper[1], param_lower[1])
    axis.annotate('Inferred', xy=infr_xy, xytext=infr_xy, textcoords='data', xycoords='data', color='red', annotation_clip=False,fontsize=16, zorder=25)
    axis.annotate(xhi_label, xy=xhi_xy, xytext=xhi_xy, textcoords='data', xycoords='data', color='red', annotation_clip=False,fontsize=16, zorder=25)
    axis.annotate(logZ_label, xy=Z_xy, xytext=Z_xy, textcoords='data', xycoords='data', color='red', annotation_clip=False,fontsize=16, zorder=25)

    if plot_draws:
        for ind in range(nrand):
            label = 'posterior draws' if ind == 0 else None
            axis.plot(vel_corr, factor*xi_model_rand[ind, :], linewidth=0.5, color='cornflowerblue', alpha=0.7, zorder=0, label=label)

    """
    # (Mar 2023) HACK: plotting masked data points
    xi_mean_data = np.load('save_cf/xi_mean_mask_10qso_everyn60.npy')
    v_lo, v_hi = given_bins
    vel_mid = (v_hi + v_lo) / 2
    ibad = np.array([11, 14, 18])  # lags 930, 1170, 1490
    vel_corr_masked = vel_mid[ibad]
    #axis.errorbar(vel_corr[ibad], factor * xi_data[ibad], yerr=factor * xi_err[ibad], marker='x', ms=6, color='red', ecolor='red',
    #              capthick=2, capsize=4,mec='none', ls='none', label='masked', zorder=25)
    axis.plot(vel_corr_masked, factor * xi_mean_data[ibad], 'kx', ms=8, mew=2, label='masked', zorder=-10)
    """
    axis.tick_params(right=True, which='both')
    axis.minorticks_on()
    axis.set_xlim((vmin, vmax))
    axis.set_ylim((ymin, ymax))

    # Make the new upper x-axes in cMpc
    z = params['z'][0]
    nlogZ = params['nlogZ'][0]
    nhi = params['nhi'][0]
    cosmo = FlatLambdaCDM(H0=100.0 * params['lit_h'][0], Om0=params['Om0'][0], Ob0=params['Ob0'][0])
    Hz = (cosmo.H(z))
    a = 1.0 / (1.0 + z)
    rmin = (vmin * u.km / u.s / a / Hz).to('Mpc').value
    rmax = (vmax * u.km / u.s / a / Hz).to('Mpc').value
    atwin = axis.twiny()
    atwin.set_xlabel('R (cMpc)', fontsize=26, labelpad=8)
    atwin.xaxis.tick_top()
    # atwin.yaxis.tick_right()
    atwin.axis([rmin, rmax, ymin, ymax])
    atwin.tick_params(top=True)
    atwin.xaxis.set_minor_locator(AutoMinorLocator())
    atwin.tick_params(axis="x", labelsize=16)

    axis.annotate('MgII doublet', xy=(1030, 0.90 * ymax), xytext=(1030, 0.90* ymax), fontsize=16, color='black')
    axis.annotate('separation', xy=(1070, 0.82 * ymax), xytext=(1070, 0.82 * ymax), fontsize=16, color='black')
    axis.annotate('', xy=(780, 0.88 * ymax), xytext=(1010, 0.88* ymax),
                fontsize=16, arrowprops={'arrowstyle': '-|>', 'lw': 4, 'color': 'black'}, va='center', color='black')


    # Plot a vertical line at the MgII doublet separation
    vel_mg = vel_mgii()
    axis.vlines(vel_mg.value, ymin, ymax, color='black', linestyle='--', linewidth=1.2)

    #axis.legend(fontsize=16,loc='lower left', bbox_to_anchor=(1800, 0.69*ymax), bbox_transform=axis.transData)
    axis.legend(fontsize=16)

    fx.tight_layout()
    plt.show()
    fx.savefig(corrfile)
    plt.close()

def corrfunc_plot(xi_data, samples, params, xhi_fine, logZ_fine, xi_model_fine, xhi_coarse, logZ_coarse, covar_array, \
                  corrfile, redshift_bin, nrand=50, rand=None, save_xi_err=None, vel_mid_compare=None, xi_mean_compare=None, \
                  label_compare=None, plot_draws=True, inferred_model='mean', lag_mask=None):

    # adapted from enigma.reion_forest.inference.corrfunc_plot
    if rand is None:
        rand = np.random.RandomState(1234)

    factor = 1e5
    fx = plt.figure(1, figsize=(12, 9))
    # left, bottom, width, height
    rect = [0.12, 0.12, 0.84, 0.75]
    axis = fx.add_axes(rect)
    vel_corr = params['vel_mid'].flatten()
    if lag_mask is not None:
        vel_corr = vel_corr[lag_mask]

    # Compute the mean model from the samples
    xi_model_samp = inference.xi_model(samples, xhi_fine, logZ_fine, xi_model_fine)
    if inferred_model == 'mean':
        xi_model_mean = np.mean(xi_model_samp, axis=0)
    elif inferred_model == 'median':
        xi_model_mean = np.median(xi_model_samp, axis=0)
    # Compute the covariance at the mean model
    theta_mean = np.mean(samples,axis=0)
    # Average the diagonal instead?
    covar_mean = inference.covar_model(theta_mean, xhi_coarse, logZ_coarse, covar_array)
    xi_err = np.sqrt(np.diag(covar_mean))

    if save_xi_err is not None:
        np.save(save_xi_err, xi_err)

    # Grab some realizations
    imock = rand.choice(np.arange(samples.shape[0]), size=nrand)
    xi_model_rand = xi_model_samp[imock, :]

    if redshift_bin == 'all':
        vmin, vmax = 0.4 * vel_corr.min(), 1.02 * vel_corr.max()
        ymin = 1.2*factor * np.min(xi_data - 1.5 * xi_err)
        ymax = factor * np.max(xi_data + 1.6 * xi_err)

        # (Mar 2023) HACK for plotting masked data points
        cf = fits.open('save_cf/xi_mean_mask_10qso_everyn60.fits')
        xi_mean_data = cf['XI_MEAN_MASK'].data
        _, ibad = mutils.cf_lags_to_mask()
        vel_mid = params['vel_mid'].flatten()
        vel_corr_masked = vel_mid[ibad]

    elif redshift_bin == 'high':
        vmin, vmax = 0.1 * vel_corr.min(), 1.02 * vel_corr.max()
        ymin = factor * np.min(xi_data - 1.5 * xi_err)
        ymax = 1.5 * factor * np.max(xi_data + 1.6 * xi_err)

        cf = fits.open('save_cf/xi_mean_mask_10qso_everyn60_highz.fits')
        xi_mean_data = cf['XI_MEAN_MASK'].data
        _, ibad = mutils.cf_lags_to_mask_highz()
        vel_mid = params['vel_mid'].flatten()
        vel_corr_masked = vel_mid[ibad]

    elif redshift_bin == 'low':
        vmin, vmax = 0.4 * vel_corr.min(), 1.02 * vel_corr.max()
        ymin = 1.2 * factor * np.min(xi_data - 1.5 * xi_err)
        ymax = factor * np.max(xi_data + 1.6 * xi_err)

        cf = fits.open('save_cf/xi_mean_mask_10qso_everyn60_lowz.fits')
        xi_mean_data = cf['XI_MEAN_MASK'].data
        _, ibad = mutils.cf_lags_to_mask_lowz()
        vel_mid = params['vel_mid'].flatten()
        vel_corr_masked = vel_mid[ibad]

    axis.set_xlabel(r'$\Delta v$ (km/s)', fontsize=26)
    axis.set_ylabel(r'$\xi(\Delta v)\times 10^5$', fontsize=26, labelpad=-4)
    axis.tick_params(axis="x", labelsize=16)
    axis.tick_params(axis="y", labelsize=16)

    #true_xy = (vmin + 0.44*(vmax-vmin), 0.60*ymax)
    #xhi_xy  = (vmin + 0.385*(vmax-vmin), 0.52*ymax)
    #Z_xy    = (vmin + 0.283*(vmax-vmin), 0.44*ymax)
    #xhi_label  = r'$\langle x_{{\rm HI}}\rangle = {:3.2f}$'.format(xhi_data)
    #logZ_label = '      ' + r'$[{{\rm Mg\slash H}}]={:5.2f}$'.format(logZ_data)
    #axis.annotate('True', xy=true_xy, xytext=true_xy, textcoords='data', xycoords='data', color='darkgreen', annotation_clip=False,fontsize=16, zorder=25)
    #axis.annotate(xhi_label, xy=xhi_xy, xytext=xhi_xy, textcoords='data', xycoords='data', color='darkgreen', annotation_clip=False,fontsize=16, zorder=25)
    #axis.annotate(logZ_label, xy=Z_xy, xytext=Z_xy, textcoords='data', xycoords='data', color='darkgreen', annotation_clip=False,fontsize=16, zorder=25)

    # error bar
    percent_lower = (1.0-0.6827)/2.0
    percent_upper = 1.0 - percent_lower
    param = np.median(samples, axis=0)
    param_lower = param - np.percentile(samples, 100*percent_lower, axis=0)
    param_upper = np.percentile(samples, 100*percent_upper, axis=0) - param

    #infr_xy = (vmin + 0.1*(vmax-vmin), 0.90*ymax)
    #xhi_xy  = (vmin + 0.05*(vmax-vmin), 0.80*ymax)
    #Z_xy    = (vmin - 0.05*(vmax-vmin), 0.70*ymax)
    infr_xy = (1800, (0.55 * ymax))
    xhi_xy = (1700, (0.42 * ymax))
    Z_xy = (1355, (0.29 * ymax))

    xhi_label  = r'$\langle x_{{\rm HI}}\rangle = {:3.2f}^{{+{:3.2f}}}_{{-{:3.2f}}}$'.format(param[0], param_upper[0], param_lower[0])
    logZ_label = '          ' + r'$[{{\rm Mg\slash H}}]={:5.2f}^{{+{:3.2f}}}_{{-{:3.2f}}}$'.format(param[1], param_upper[1], param_lower[1])
    #print(xhi_label)
    #print(logZ_label)

    axis.annotate('Inferred', xy=infr_xy, xytext=infr_xy, textcoords='data', xycoords='data', color='red', annotation_clip=False,fontsize=16, zorder=25)
    axis.annotate(xhi_label, xy=xhi_xy, xytext=xhi_xy, textcoords='data', xycoords='data', color='red', annotation_clip=False,fontsize=16, zorder=25)
    axis.annotate(logZ_label, xy=Z_xy, xytext=Z_xy, textcoords='data', xycoords='data', color='red', annotation_clip=False,fontsize=16, zorder=25)

    axis.plot(vel_corr, factor * xi_model_mean, linewidth=2.0, color='red', zorder=10, label='inferred model')

    if plot_draws:
        for ind in range(nrand):
            label = 'posterior draws' if ind == 0 else None
            axis.plot(vel_corr, factor * xi_model_rand[ind, :], linewidth=0.5, color='cornflowerblue', alpha=0.7, zorder=0, label=label)

    axis.errorbar(vel_corr, factor * xi_data, yerr=factor * xi_err, marker='o', ms=6, color='black', ecolor='black',
                  capthick=2, capsize=4, mec='none', ls='none', label='data', zorder=20)

    axis.plot(vel_corr_masked, factor * xi_mean_data[ibad], 'kx', ms=8, mew=2, label='masked', zorder=0)

    if vel_mid_compare is not None:
        axis.plot(vel_mid_compare, factor * xi_mean_compare, linewidth=2.0, color='blue', zorder=10, label=label_compare)

    axis.tick_params(right=True, which='both')
    axis.minorticks_on()
    axis.set_xlim((vmin, vmax))
    axis.set_ylim((ymin, ymax))

    # Make the new upper x-axes in cMpc
    z = params['z'][0]
    nlogZ = params['nlogZ'][0]
    nhi = params['nhi'][0]
    cosmo = FlatLambdaCDM(H0=100.0 * params['lit_h'][0], Om0=params['Om0'][0], Ob0=params['Ob0'][0])
    Hz = (cosmo.H(z))
    a = 1.0 / (1.0 + z)
    rmin = (vmin * u.km / u.s / a / Hz).to('Mpc').value
    rmax = (vmax * u.km / u.s / a / Hz).to('Mpc').value
    atwin = axis.twiny()
    atwin.set_xlabel('R (cMpc)', fontsize=26, labelpad=8)
    atwin.xaxis.tick_top()
    # atwin.yaxis.tick_right()
    atwin.axis([rmin, rmax, ymin, ymax])
    atwin.tick_params(top=True)
    atwin.xaxis.set_minor_locator(AutoMinorLocator())
    atwin.tick_params(axis="x", labelsize=16)

    axis.annotate('MgII doublet', xy=(1030, 0.90 * ymax), xytext=(1030, 0.90* ymax), fontsize=16, color='black')
    axis.annotate('separation', xy=(1070, 0.82 * ymax), xytext=(1070, 0.82 * ymax), fontsize=16, color='black')
    axis.annotate('', xy=(780, 0.88 * ymax), xytext=(1010, 0.88* ymax),
                fontsize=16, arrowprops={'arrowstyle': '-|>', 'lw': 4, 'color': 'black'}, va='center', color='black')


    # Plot a vertical line at the MgII doublet separation
    vel_mg = vel_mgii()
    axis.vlines(vel_mg.value, ymin, ymax, color='black', linestyle='--', linewidth=1.2)

    #axis.legend(fontsize=16,loc='lower left', bbox_to_anchor=(1800, 0.69*ymax), bbox_transform=axis.transData)
    axis.legend(fontsize=16, loc='lower right')

    fx.tight_layout()
    plt.show()
    fx.savefig(corrfile)
    plt.close()

def corrfunc_plot_new(xi_data, samples, params, xhi_fine, logZ_fine, xi_model_fine, xhi_coarse, logZ_coarse, covar_array, \
                  corrfile, redshift_bin, nrand=50, rand=None, input_xi_err=None, plot_draws=True, inferred_model='mean', \
                      lag_mask=None):

    # adapted from enigma.reion_forest.inference.corrfunc_plot
    if rand is None:
        rand = np.random.RandomState(1234)

    factor = 1e5
    fx = plt.figure(1, figsize=(12, 9))
    # left, bottom, width, height
    rect = [0.12, 0.12, 0.84, 0.75]
    axis = fx.add_axes(rect)
    vel_corr_all = params['vel_mid'].flatten()
    if lag_mask is not None:
        vel_corr = vel_corr_all[lag_mask]
    else:
        vel_corr = vel_corr_all

    # Compute the mean model from the samples
    xi_model_samp = inference.xi_model(samples, xhi_fine, logZ_fine, xi_model_fine)
    if inferred_model == 'mean':
        xi_model_mean = np.mean(xi_model_samp, axis=0)
    elif inferred_model == 'median':
        xi_model_mean = np.median(xi_model_samp, axis=0)
    # Compute the covariance at the mean model
    theta_mean = np.mean(samples,axis=0)

    covar_mean = inference.covar_model(theta_mean, xhi_coarse, logZ_coarse, covar_array)
    xi_err = np.sqrt(np.diag(covar_mean))

    # Grab some realizations
    imock = rand.choice(np.arange(samples.shape[0]), size=nrand)
    xi_model_rand = xi_model_samp[imock, :]

    if redshift_bin == 'all':
        vmin, vmax = 0.4 * vel_corr.min(), 1.02 * vel_corr.max()
        ymin = 1.2*factor * np.min(xi_data - 2 * xi_err)
        ymax = factor * np.max(xi_data + 1.6 * xi_err)

        # (Mar 2023) HACK for plotting masked data points
        #cf = fits.open('save_cf/paper/xi_mean_mask_10qso_everyn60_corr.fits')
        #xi_mean_data = cf['XI_MEAN_MASK'].data
        #_, ibad = mutils.cf_lags_to_mask()
        #vel_mid = params['vel_mid'].flatten()
        #vel_corr_bad = vel_mid[ibad]
        ibad, vel_corr_bad = [], []

    elif redshift_bin == 'high':
        vmin, vmax = 0.1 * vel_corr.min(), 1.02 * vel_corr.max()
        ymin = factor * np.min(xi_data - 1.5 * xi_err)
        ymax = 1.5 * factor * np.max(xi_data + 1.6 * xi_err)

        cf = fits.open('save_cf/paper/xi_mean_mask_10qso_everyn60_highz.fits')
        xi_mean_data = cf['XI_MEAN_MASK'].data
        _, ibad = mutils.cf_lags_to_mask_highz()
        vel_mid = params['vel_mid'].flatten()
        vel_corr_bad = vel_mid[ibad]

    elif redshift_bin == 'low':
        vmin, vmax = 0.4 * vel_corr.min(), 1.02 * vel_corr.max()
        ymin = 1.2 * factor * np.min(xi_data - 2 * xi_err)
        ymax = factor * np.max(xi_data + 1.6 * xi_err)

        cf = fits.open('save_cf/paper/xi_mean_mask_10qso_everyn60_lowz.fits')
        xi_mean_data = cf['XI_MEAN_MASK'].data
        _, ibad = mutils.cf_lags_to_mask_lowz()
        vel_mid = params['vel_mid'].flatten()
        vel_corr_bad = vel_mid[ibad]

    axis.set_xlabel(r'$\Delta v$ (km/s)', fontsize=26)
    axis.set_ylabel(r'$\xi(\Delta v)\times 10^5$', fontsize=26, labelpad=-4)
    axis.tick_params(axis="x", labelsize=16)
    axis.tick_params(axis="y", labelsize=16)

    # error bar
    #percent_lower = (1.0-0.6827)/2.0
    percent_lower = (1.0 - 0.68) / 2.0
    percent_upper = 1.0 - percent_lower
    param = np.median(samples, axis=0)
    param_lower = param - np.percentile(samples, 100*percent_lower, axis=0)
    param_upper = np.percentile(samples, 100*percent_upper, axis=0) - param

    infr_xy = (1800, (0.55 * ymax))
    xhi_xy = (1700, (0.42 * ymax))
    Z_xy = (1355, (0.29 * ymax))

    xhi_label  = r'$\langle x_{{\rm HI}}\rangle = {:3.2f}^{{+{:3.2f}}}_{{-{:3.2f}}}$'.format(param[0], param_upper[0], param_lower[0])
    logZ_label = '          ' + r'$[{{\rm Mg\slash H}}]={:5.2f}^{{+{:3.2f}}}_{{-{:3.2f}}}$'.format(param[1], param_upper[1], param_lower[1])

    axis.annotate('Inferred', xy=infr_xy, xytext=infr_xy, textcoords='data', xycoords='data', color='red', annotation_clip=False,fontsize=16, zorder=25)
    axis.annotate(xhi_label, xy=xhi_xy, xytext=xhi_xy, textcoords='data', xycoords='data', color='red', annotation_clip=False,fontsize=16, zorder=25)
    axis.annotate(logZ_label, xy=Z_xy, xytext=Z_xy, textcoords='data', xycoords='data', color='red', annotation_clip=False,fontsize=16, zorder=25)

    axis.plot(vel_corr, factor * xi_model_mean, linewidth=2.0, color='red', zorder=10, label='inferred model')

    if plot_draws:
        for ind in range(nrand):
            label = 'posterior draws' if ind == 0 else None
            axis.plot(vel_corr, factor * xi_model_rand[ind, :], linewidth=0.5, color='cornflowerblue', alpha=0.7, zorder=0, label=label)

    axis.errorbar(vel_corr, factor * xi_data, yerr=factor * xi_err, marker='o', ms=6, color='black', ecolor='black',
                  capthick=2, capsize=4, mec='none', ls='none', label='data', zorder=20)

    # (Mar 2023) HACK for plotting masked data points
    #if input_xi_err is not None:
    #    axis.errorbar(vel_corr_bad, factor * xi_mean_data[ibad], yerr=factor * input_xi_err[ibad], marker='x', mew=2, ms=6, color='black', ecolor='black',
    #                  capthick=2, capsize=4, mec='black', ls='none', label='masked', zorder=20, alpha=0.5)
    #else:
    #    axis.plot(vel_corr_bad, factor * xi_mean_data[ibad], 'kx', ms=8, mew=2, label='masked', zorder=0)

    axis.tick_params(right=True, which='both')
    axis.minorticks_on()
    axis.set_xlim((vmin, vmax))
    axis.set_ylim((ymin, ymax))

    # Make the new upper x-axes in cMpc
    z = params['z'][0]
    nlogZ = params['nlogZ'][0]
    nhi = params['nhi'][0]
    cosmo = FlatLambdaCDM(H0=100.0 * params['lit_h'][0], Om0=params['Om0'][0], Ob0=params['Ob0'][0])
    Hz = (cosmo.H(z))
    a = 1.0 / (1.0 + z)
    rmin = (vmin * u.km / u.s / a / Hz).to('Mpc').value
    rmax = (vmax * u.km / u.s / a / Hz).to('Mpc').value
    atwin = axis.twiny()
    atwin.set_xlabel('R (cMpc)', fontsize=26, labelpad=8)
    atwin.xaxis.tick_top()
    # atwin.yaxis.tick_right()
    atwin.axis([rmin, rmax, ymin, ymax])
    atwin.tick_params(top=True)
    atwin.xaxis.set_minor_locator(AutoMinorLocator())
    atwin.tick_params(axis="x", labelsize=16)

    axis.annotate('MgII doublet', xy=(1030, 0.90 * ymax), xytext=(1030, 0.90* ymax), fontsize=16, color='black')
    axis.annotate('separation', xy=(1070, 0.82 * ymax), xytext=(1070, 0.82 * ymax), fontsize=16, color='black')
    axis.annotate('', xy=(780, 0.88 * ymax), xytext=(1010, 0.88* ymax),
                fontsize=16, arrowprops={'arrowstyle': '-|>', 'lw': 4, 'color': 'black'}, va='center', color='black')


    # Plot a vertical line at the MgII doublet separation
    vel_mg = vel_mgii()
    axis.vlines(vel_mg.value, ymin, ymax, color='black', linestyle='--', linewidth=1.2)

    #axis.legend(fontsize=16,loc='lower left', bbox_to_anchor=(1800, 0.69*ymax), bbox_transform=axis.transData)
    axis.legend(fontsize=16, loc='lower right')

    fx.tight_layout()
    plt.show()
    fx.savefig(corrfile)
    plt.close()

def corrfunc_plot_jwst_prop(xi_data, samples, params, xhi_fine, logZ_fine, xi_model_fine, xhi_coarse, logZ_coarse, covar_array, \
                  corrfile, nrand=50, rand=None, save_xi_err=None, vel_mid_compare=None, xi_mean_compare=None, \
                  label_compare=None, plot_draws=True, inferred_model='mean'):

    # adapted from enigma.reion_forest.inference.corrfunc_plot
    if rand is None:
        rand = np.random.RandomState(1234)

    factor = 1e5
    fx = plt.figure(1, figsize=(10, 7))
    # left, bottom, width, height
    rect = [0.12, 0.12, 0.84, 0.75]
    axis = fx.add_axes(rect)
    vel_corr = params['vel_mid'].flatten()
    #vel_min = params['vmin_corr']
    #vel_max = params['vmax_corr']

    vmin, vmax = 0.4*vel_corr.min(), 1.02*vel_corr.max()
    # Compute the mean model from the samples
    xi_model_samp = inference.xi_model(samples, xhi_fine, logZ_fine, xi_model_fine)
    if inferred_model == 'mean':
        xi_model_mean = np.mean(xi_model_samp, axis=0)
    elif inferred_model == 'median':
        xi_model_mean = np.median(xi_model_samp, axis=0)
    # Compute the covariance at the mean model
    theta_mean = np.mean(samples,axis=0)
    # Average the diagonal instead?
    covar_mean = inference.covar_model(theta_mean, xhi_coarse, logZ_coarse, covar_array)
    xi_err = np.sqrt(np.diag(covar_mean))

    if save_xi_err is not None:
        np.save(save_xi_err, xi_err)

    # Grab some realizations
    imock = rand.choice(np.arange(samples.shape[0]), size=nrand)
    xi_model_rand = xi_model_samp[imock, :]
    #ymin = factor*np.min(xi_data - 1.3*xi_err)
    #ymax = factor*np.max(xi_data + 1.6*xi_err )
    ymin = factor * -0.00055
    ymax = factor * 0.00115

    axis.set_xlabel(r'$\Delta v$ (km/s)', fontsize=26)
    axis.set_ylabel(r'$\xi(\Delta v)\times 10^5$', fontsize=26, labelpad=-4)
    axis.tick_params(axis="x", labelsize=16)
    axis.tick_params(axis="y", labelsize=16)
    axis.tick_params(right=True, which='both')
    axis.minorticks_on()
    axis.set_xlim((vmin, vmax))
    axis.set_ylim((ymin, ymax))

    axis.errorbar(vel_corr, factor*xi_data, yerr=factor*xi_err, marker='o', ms=6, color='black', ecolor='black', capthick=2,
                  capsize=4,
                  mec='none', ls='none', label='ground-based data', zorder=20)
    #axis.plot(vel_corr, factor*xi_model_mean, linewidth=2.0, color='red', zorder=10, label='inferred model')
    if vel_mid_compare is not None:
        axis.plot(vel_mid_compare, factor * xi_mean_compare, linewidth=2.0, color='red', zorder=10, label='Fiducial IGM model')


    #true_xy = (vmin + 0.44*(vmax-vmin), 0.60*ymax)
    #xhi_xy  = (vmin + 0.385*(vmax-vmin), 0.52*ymax)
    #Z_xy    = (vmin + 0.283*(vmax-vmin), 0.44*ymax)
    #xhi_label  = r'$\langle x_{{\rm HI}}\rangle = {:3.2f}$'.format(xhi_data)
    #logZ_label = '      ' + r'$[{{\rm Mg\slash H}}]={:5.2f}$'.format(logZ_data)
    #axis.annotate('True', xy=true_xy, xytext=true_xy, textcoords='data', xycoords='data', color='darkgreen', annotation_clip=False,fontsize=16, zorder=25)
    #axis.annotate(xhi_label, xy=xhi_xy, xytext=xhi_xy, textcoords='data', xycoords='data', color='darkgreen', annotation_clip=False,fontsize=16, zorder=25)
    #axis.annotate(logZ_label, xy=Z_xy, xytext=Z_xy, textcoords='data', xycoords='data', color='darkgreen', annotation_clip=False,fontsize=16, zorder=25)

    # error bar
    percent_lower = (1.0-0.6827)/2.0
    percent_upper = 1.0 - percent_lower
    param = np.median(samples, axis=0)
    param_lower = param - np.percentile(samples, 100*percent_lower, axis=0)
    param_upper = np.percentile(samples, 100*percent_upper, axis=0) - param

    #infr_xy = (vmin + 0.1*(vmax-vmin), 0.90*ymax)
    #xhi_xy  = (vmin + 0.05*(vmax-vmin), 0.80*ymax)
    #Z_xy    = (vmin - 0.05*(vmax-vmin), 0.70*ymax)
    infr_xy = (1800, (0.55 * ymax))
    xhi_xy = (1700, (0.42 * ymax))
    Z_xy = (1355, (0.29 * ymax))

    """
    xhi_label  = r'$\langle x_{{\rm HI}}\rangle = {:3.2f}^{{+{:3.2f}}}_{{-{:3.2f}}}$'.format(param[0], param_upper[0], param_lower[0])
    logZ_label = '          ' + r'$[{{\rm Mg\slash H}}]={:5.2f}^{{+{:3.2f}}}_{{-{:3.2f}}}$'.format(param[1], param_upper[1], param_lower[1])
    axis.annotate('Inferred', xy=infr_xy, xytext=infr_xy, textcoords='data', xycoords='data', color='red', annotation_clip=False,fontsize=16, zorder=25)
    axis.annotate(xhi_label, xy=xhi_xy, xytext=xhi_xy, textcoords='data', xycoords='data', color='red', annotation_clip=False,fontsize=16, zorder=25)
    axis.annotate(logZ_label, xy=Z_xy, xytext=Z_xy, textcoords='data', xycoords='data', color='red', annotation_clip=False,fontsize=16, zorder=25)
    """
    if plot_draws:
        for ind in range(nrand):
            label = 'posterior draws' if ind == 0 else None
            axis.plot(vel_corr, factor*xi_model_rand[ind, :], linewidth=0.5, color='cornflowerblue', alpha=0.7, zorder=0, label=label)

    axis.tick_params(right=True, which='both')
    axis.minorticks_on()
    axis.set_xlim((vmin, vmax))
    axis.set_ylim((ymin, ymax))

    # Make the new upper x-axes in cMpc
    z = params['z'][0]
    nlogZ = params['nlogZ'][0]
    nhi = params['nhi'][0]
    cosmo = FlatLambdaCDM(H0=100.0 * params['lit_h'][0], Om0=params['Om0'][0], Ob0=params['Ob0'][0])
    Hz = (cosmo.H(z))
    a = 1.0 / (1.0 + z)
    rmin = (vmin * u.km / u.s / a / Hz).to('Mpc').value
    rmax = (vmax * u.km / u.s / a / Hz).to('Mpc').value
    atwin = axis.twiny()
    atwin.set_xlabel('R (cMpc)', fontsize=26, labelpad=8)
    atwin.xaxis.tick_top()
    # atwin.yaxis.tick_right()
    atwin.axis([rmin, rmax, ymin, ymax])
    atwin.tick_params(top=True)
    atwin.xaxis.set_minor_locator(AutoMinorLocator())
    atwin.tick_params(axis="x", labelsize=16)


    axis.annotate('MgII doublet', xy=(1030, 0.90 * ymax), xytext=(1030, 0.90* ymax), fontsize=16, color='black')
    axis.annotate('separation', xy=(1070, 0.82 * ymax), xytext=(1070, 0.82 * ymax), fontsize=16, color='black')
    axis.annotate('', xy=(780, 0.88 * ymax), xytext=(1010, 0.88* ymax),
                fontsize=16, arrowprops={'arrowstyle': '-|>', 'lw': 4, 'color': 'black'}, va='center', color='black')


    # Plot a vertical line at the MgII doublet separation
    vel_mg = vel_mgii()
    axis.vlines(vel_mg.value, ymin, ymax, color='black', linestyle='--', linewidth=1.2)

    #axis.legend(fontsize=16,loc='lower left', bbox_to_anchor=(1800, 0.69*ymax), bbox_transform=axis.transData)
    axis.legend(fontsize=16)

    #fx.tight_layout()
    plt.show()
    fx.savefig(corrfile)
    plt.close()

def plot_corrmatrix(coarse_out, data_out, logZ_want, xhi_want, vmin=None, vmax=None, plot_covar=False, interp_covar=True):
    # plotting the correlation matrix (copied from CIV_forest/metal_corrfunc.py)

    xhi_coarse, logZ_coarse, lnlike_coarse = coarse_out
    xhi_data, logZ_data, xi_data, covar_array, params = data_out
    vel_corr = params['vel_mid'].flatten()
    vmin_corr = vel_corr[0]
    vmax_corr = vel_corr[-1]

    ixhi = find_closest(xhi_coarse, xhi_want)
    iZ = find_closest(logZ_coarse, logZ_want)
    xhi_data = xhi_coarse[ixhi]
    logZ_data = logZ_coarse[iZ]
    #covar = covar_array[ixhi, iZ, :, :]
    print(ixhi, iZ)

    if interp_covar:
        vel_corr_want = np.linspace(vmin_corr, vmax_corr, 41)
        covar = interp_cov(np.array(vel_corr), vel_corr_want, covar_array, ixhi, iZ)
    else:
        covar = covar_array[ixhi, iZ, :, :]

    # correlation matrix; easier to visualize compared to covar matrix
    corr = covar / np.sqrt(np.outer(np.diag(covar), np.diag(covar)))
    if plot_covar:
        corr = covar
    print(corr.min(), corr.max())

    if vmin == None:
        vmin = corr.min()
    if vmax == None:
        vmax = corr.max()

    plt.figure(figsize=(8, 8))
    if plot_covar:
        t = 'Covariance matrix'
    else:
        t = 'Correlation matrix'

    plt.imshow(corr, origin='lower', interpolation='nearest', extent=[vmin_corr, vmax_corr, vmin_corr, vmax_corr], \
               vmin=vmin, vmax=vmax, cmap='inferno')
    plt.xlabel(r'$\Delta v$ (km/s)', fontsize=15)
    plt.ylabel(r'$\Delta v$ (km/s)', fontsize=15)
    plt.title('%s: For model (logZ, xHI) = (%0.2f, %0.2f)' % (t, logZ_data, xhi_data))
    plt.colorbar()

    plt.show()

def interp_cov(ori_vel_corr_grid, want_vel_corr_grid, covar_array, ixhi, iZ):
    subcovar = covar_array[ixhi, iZ]
    f = interpolate.interp2d(ori_vel_corr_grid, ori_vel_corr_grid, subcovar, kind='linear')
    subcovar_interp = f(want_vel_corr_grid, want_vel_corr_grid)
    return subcovar_interp

def corr_matrix(covar_array):

    # constructing the correlation matrix
    corr_array = []
    for i in range(len(covar_array)):
        corr_array_dim1 = []
        for j in range(len(covar_array[i])):
            corr = covar_array[i, j] / np.sqrt(np.outer(np.diag(covar_array[i, j]), np.diag(
                covar_array[i, j])))  # correlation matrix; see Eqn 14 of Hennawi+ 2020
            corr_array_dim1.append(corr)
        corr_array.append(corr_array_dim1)

    corr_array = np.array(corr_array)
    return corr_array

def plot_single_corr_elem(coarse_out, data_out, corr_array, rand_ixhi=None, rand_ilogZ=None, rand_i=None, rand_j=None, savefig=None):

    xhi_coarse, logZ_coarse, lnlike_coarse = coarse_out
    _, _, _, covar_array, params = data_out
    nxHI, nlogZ, ncorr, _ = np.shape(covar_array)

    if rand_i == None and rand_j == None:
        rand_i, rand_j = np.random.randint(ncorr), np.random.randint(ncorr)
        print(rand_i, rand_j)

    if rand_ixhi == None:
        rand_ixhi = np.random.randint(nxHI)
        print(rand_ixhi, xhi_coarse[rand_ixhi])

    if rand_ilogZ == None:
        rand_ilogZ = np.random.randint(nlogZ)
        print(rand_ilogZ, logZ_coarse[rand_ilogZ])

    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.title('logZ = %0.2f (ilogZ = %d)' % (logZ_coarse[rand_ilogZ], rand_ilogZ))
    plt.plot(xhi_coarse, corr_array[:,rand_ilogZ, rand_i, rand_j], 'k.-', label='($i,j$)=(%d,%d)' % (rand_i, rand_j))
    plt.xlabel('xHI')
    plt.ylabel(r'Corr$_{ij}$=Cov$_{ij}$/$\sqrt{Cov_{ii}Cov_{jj}}$')
    plt.legend()

    plt.subplot(122)
    plt.title('xHI = %0.2f (ixHI = %d)' % (xhi_coarse[rand_ixhi], rand_ixhi))
    plt.plot(logZ_coarse, corr_array[rand_ixhi, :, rand_i, rand_j], 'k.-')
    plt.xlabel('logZ')
    plt.tight_layout()

    if savefig is not None:
        plt.savefig(savefig)
        plt.close()
    else:
        plt.show()

def lnlike_plot_slice(xhi_arr, logZ_arr, lnlike_arr, xhi_want, logZ_want):
    ixhi = find_closest(xhi_arr, xhi_want)
    iZ = find_closest(logZ_arr, logZ_want)

    lnlike_arr = lnlike_arr - lnlike_arr.max()
    lnlike_arr = np.exp(lnlike_arr)

    plt.subplot(121)
    plt.plot(logZ_arr, lnlike_arr[ixhi])
    plt.xlabel('logZ')
    plt.ylabel('exp(lnL-lnL_max)')

    plt.subplot(122)
    plt.plot(xhi_arr, lnlike_arr[:,iZ])
    plt.xlabel('xHI')

    plt.tight_layout()
    plt.suptitle('model: (xhi, logZ) = (%0.2f, %0.2f)' % (xhi_want, logZ_want), fontsize=15)
    plt.show()

def lnlike_plot_logZ_many(xhi_arr, logZ_arr, lnlike_arr, xhi_want_arr):

    lnlike_arr_new = lnlike_arr - lnlike_arr.max()
    lnlike_arr_new = np.exp(lnlike_arr_new)

    for i in range(len(xhi_want_arr)):
        ixhi = find_closest(xhi_arr, xhi_want_arr[i])
        plt.plot(logZ_arr, lnlike_arr_new[ixhi], label='xHI = %0.2f' % xhi_want_arr[i])

    plt.xlabel('logZ')
    plt.ylabel('likelihood')

    plt.legend()
    plt.show()

def lnlike_plot_xhi_many(xhi_arr, logZ_arr, lnlike_arr, logZ_want_arr):

    lnlike_arr_new = lnlike_arr - lnlike_arr.max()
    lnlike_arr_new = np.exp(lnlike_arr_new)

    for i in range(len(logZ_want_arr)):
        iZ = find_closest(logZ_arr, logZ_want_arr[i])
        plt.plot(xhi_arr, lnlike_arr_new[:, iZ], label='logZ = %0.2f' % logZ_want_arr[i])

    plt.xlabel('xHI')
    plt.ylabel('likelihood')

    plt.legend()
    plt.show()

import seaborn as sns
def lnlike_heatmap(lnlike_grid, xhi_grid, logz_grid, cbar_label, vmin=None, vmax=None):
    ax = sns.heatmap(lnlike_grid, xticklabels=logz_grid, yticklabels=xhi_grid, vmin=vmin, vmax=vmax, cbar_kws={'label': cbar_label})
    if lnlike_grid.shape[0] > 100:
        nstep1, nstep2 = 110, 110
    else:
        nstep1, nstep2 = 12, 6
    ax.set_xticks(ax.get_xticks()[::nstep1])
    ax.set_xticklabels(np.round(logz_grid[::nstep1],2))
    ax.set_yticks(ax.get_yticks()[::nstep2])
    ax.set_yticklabels(xhi_grid[::nstep2])
    ax.set_xlabel('logZ')
    ax.set_ylabel('xHI')
    plt.tight_layout()
    plt.show()

##################### interpolate covariance matrix for lnlike_fine calculation #####################
def interp_covar(modelfile, lag_mask=None):

    params, xi_mock_array, xi_model_array, covar_array, icovar_array, lndet_array = read_model_grid(modelfile)
    logZ_coarse = params['logZ'].flatten()
    xhi_coarse = params['xhi'].flatten()

    # Interpolate the likelihood onto a fine grid to speed up the MCMC
    nlogZ = logZ_coarse.size
    nlogZ_fine = 1001
    logZ_fine_min = logZ_coarse.min()
    logZ_fine_max = logZ_coarse.max()
    dlogZ_fine = (logZ_fine_max - logZ_fine_min) / (nlogZ_fine - 1)
    logZ_fine = logZ_fine_min + np.arange(nlogZ_fine) * dlogZ_fine

    nhi = xhi_coarse.size
    nhi_fine = 1001
    xhi_fine_min = 0.0
    xhi_fine_max = 1.0
    dxhi = (xhi_fine_max - xhi_fine_min) / (nhi_fine - 1)
    xhi_fine = np.arange(nhi_fine) * dxhi

    if lag_mask is not None:
        xi_model_array, xi_mock_array, covar_array, lndet_array = mutils.extract_subarr(lag_mask, xi_model_array,
                                                                                        xi_mock_array, covar_array)

    nhi_fine = xhi_fine.size
    nlogZ_fine = logZ_fine.size
    # Interpolate the model onto the fine grid as well, as we will need this for plotting
    ncorr = covar_array.shape[2]

    # 25 sec for 1001 x 1001
    covar_fine = np.zeros((nhi_fine, nlogZ_fine, ncorr, ncorr))
    for icorr in range(ncorr):
        for jcorr in range(ncorr):
            covar_interp_func = RectBivariateSpline(xhi_coarse, logZ_coarse, covar_array[:, :, icorr, jcorr])
            covar_fine[:, :, icorr, jcorr] = covar_interp_func(xhi_fine, logZ_fine)

    return covar_fine

def old_interp_covar(xhi_fine, logZ_fine, xhi_coarse, logZ_coarse, covar_array):

    nhi_fine = xhi_fine.size
    nlogZ_fine = logZ_fine.size
    # Interpolate the model onto the fine grid as well, as we will need this for plotting
    ncorr = covar_array.shape[2]

    # 25 sec for 1001 x 1001
    covar_fine = np.zeros((nhi_fine, nlogZ_fine, ncorr, ncorr))
    for icorr in range(ncorr):
        for jcorr in range(ncorr):
            covar_interp_func = RectBivariateSpline(xhi_coarse, logZ_coarse, covar_array[:, :, icorr, jcorr])
            covar_fine[:, :, icorr, jcorr] = covar_interp_func(xhi_fine, logZ_fine)

    return covar_fine