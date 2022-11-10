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

from scipy import optimize
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

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--modelfile', type=str)
args = parser.parse_args()


seed = args.seed
if seed == None:
    seed = np.random.randint(0, 1000000)
    print(seed)

rand = np.random.RandomState(seed)
nqso = 8
#figpath = '/Users/suksientie/Research/MgII_forest/paper_plots/8qso/mcmc/'

# for init_mockdata()
logZ_guess = -5.0 #-4.50 # -3.70
xhi_guess  = 0.50 # 0.74
given_bins = compute_cf_data.custom_cf_bin4(dv1=80)

def init(modelfile, redshift_bin, figpath, given_bins, vel_lores=None):
    # vel_lores = np.load('vel_lores_nyx.npy')

    # options for redshift_bin: 'all', 'low', 'high'
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

    # using the chunked data for the MCMC...
    #cgm_fit_gpm_all, _ = cmg.init_cgm_masking(redshift_bin, datapath='/Users/suksientie/Research/data_redux/') # only CGM masks, other masks added later in ccf.onespec_chunk
    #vel_mid, xi_mean_unmask, xi_mean_mask = compute_cf_data.allspec_chunk(nqso, cgm_fit_gpm_all, redshift_bin, vel_lores, given_bins=given_bins)

    # using the data in its original format for MCMC
    lowz_cgm_fit_gpm, highz_cgm_fit_gpm, allz_cgm_fit_gpm = compute_cf_data.init_cgm_fit_gpm()
    if redshift_bin == 'low':
        cgm_fit_gpm_all = lowz_cgm_fit_gpm
    elif redshift_bin == 'high':
        cgm_fit_gpm_all = highz_cgm_fit_gpm
    elif redshift_bin == 'all':
        cgm_fit_gpm_all = allz_cgm_fit_gpm
    vel_mid, xi_mean_unmask, xi_mean_mask, _, _, _, _ = compute_cf_data.allspec(nqso, redshift_bin, cgm_fit_gpm_all, plot=False, seed_list=seed_list, given_bins=given_bins)

    xi_data = xi_mean_mask
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

    lnlike_fine = inference.interp_lnlike(xhi_fine, logZ_fine, xhi_coarse, logZ_coarse, lnlike_coarse, kx=3, ky=3)
    xi_model_fine = inference.interp_model(xhi_fine, logZ_fine, xhi_coarse, logZ_coarse, xi_model_array)

    # Make a 2d surface plot of the likelhiood
    logZ_fine_2d, xhi_fine_2d = np.meshgrid(logZ_fine, xhi_fine)
    lnlikefile = figpath + 'lnlike.pdf'
    inference.lnlike_plot(xhi_fine_2d, logZ_fine_2d, lnlike_fine, lnlikefile)

    fine_out = xhi_fine, logZ_fine, lnlike_fine, xi_model_fine
    coarse_out = xhi_coarse, logZ_coarse, lnlike_coarse
    data_out = xhi_data, logZ_data, xi_data, covar_array, params

    return fine_out, coarse_out, data_out

def init_mockdata(modelfile):

    # modelfile = 'igm_cluster/corr_func_models_fwhm_90.000_samp_3.000.fits'
    params, xi_mock_array, xi_model_array, covar_array, icovar_array, lndet_array = read_model_grid(modelfile)
    logZ_coarse = params['logZ'].flatten()
    xhi_coarse = params['xhi'].flatten()
    vel_corr = params['vel_mid'].flatten()
    vel_min = params['vmin_corr'][0]
    vel_max = params['vmax_corr'][0]
    nlogZ = params['nlogZ'][0]
    nhi = params['nhi'][0]

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
    lnlikefile = figpath + 'lnlike.pdf'
    inference.lnlike_plot(xhi_fine_2d, logZ_fine_2d, lnlike_fine, lnlikefile)

    fine_out = xhi_fine, logZ_fine, lnlike_fine, xi_model_fine
    coarse_out = xhi_coarse, logZ_coarse, lnlike_coarse
    data_out = xhi_data, logZ_data, xi_data, covar_array, params

    return fine_out, coarse_out, data_out

def run_mcmc(fine_out, coarse_out, data_out, redshift_bin, figpath, nsteps=100000, burnin=1000, nwalkers=40, \
             linearZprior=False, savefits_chain=None, actual_data=True, save_xi_err=None):

    xhi_fine, logZ_fine, lnlike_fine, xi_model_fine = fine_out
    xhi_coarse, logZ_coarse, lnlike_coarse = coarse_out
    xhi_data, logZ_data, xi_data, covar_array, params = data_out

    # find optimal starting points for each walker
    chi2_func = lambda *args: -2 * inference.lnprob(*args)
    logZ_fine_min = logZ_fine.min()
    logZ_fine_max = logZ_fine.max()
    #bounds = [(0.0,1.0), (logZ_fine_min, logZ_fine_max)] if not linearZprior else [(0.95, 1.0), (0.0, np.power(10.0,logZ_fine_max))]
    bounds = [(0.0, 1.0), (logZ_fine_min, logZ_fine_max)] if not linearZprior else [(0.0, 1.0), (0.0, np.power(10.0, logZ_fine_max))]
    print("bounds", bounds)
    args = (lnlike_fine, xhi_fine, logZ_fine, linearZprior)
    result_opt = optimize.differential_evolution(chi2_func,bounds=bounds, popsize=25, recombination=0.7,
                                                 disp=True, polish=True, args=args, seed=rand)
    ndim = 2 # xhi and logZ

    # initialize walkers about the maximum within a ball of 1% of the parameter domain
    pos = [[np.clip(result_opt.x[i] + 1e-2*(bounds[i][1] - bounds[i][0])*rand.randn(1)[0],bounds[i][0],bounds[i][1]) for i in range(ndim)] for i in range(nwalkers)]
    # randomly initialize walkers
    #pos = [[bounds[i][0] + (bounds[i][1] - bounds[i][0])*rand.rand(1)[0] for i in range(ndim)] for i in range(nwalkers)]

    # I think this seeds the random number generator which will make the emcee results reproducible. Create an issue on this
    np.random.seed(rand.randint(0,seed, size=1)[0])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, inference.lnprob, args=args)
    sampler.run_mcmc(pos, nsteps, progress=True)

    tau = sampler.get_autocorr_time()
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

    if actual_data:
        corrfunc_plot(xi_data, param_samples, params, xhi_fine, logZ_fine, xi_model_fine, xhi_coarse, logZ_coarse, covar_array,
                      corrfile, nrand=300, rand=rand, save_xi_err=save_xi_err)
    else:
        inference.corrfunc_plot(xi_data, param_samples, params, xhi_fine, logZ_fine, xi_model_fine, xhi_coarse,
                                logZ_coarse, covar_array, xhi_data, logZ_data, corrfile, rand=rand)

    # Upper limit on metallicity for pristine case
    if linearZprior:
        ixhi_prior = flat_samples[:,0] > 0.95
        logZ_95 = np.percentile(param_samples[ixhi_prior,1], 95.0)
        print('Obtained 95% upper limit of {:6.4f}'.format(logZ_95))

    if savefits_chain != None:
        hdulist = fits.HDUList()
        hdulist.append(fits.ImageHDU(data=sampler.get_chain(), name='all_chain'))
        hdulist.append(fits.ImageHDU(data=sampler.get_chain(flat=True), name='all_chain_flat'))
        hdulist.append(fits.ImageHDU(data=sampler.get_chain(discard=burnin, flat=True), name='all_chain_discard_burnin'))
        hdulist.append(fits.ImageHDU(data=param_samples, name='param_samples'))
        hdulist.writeto(savefits_chain, overwrite=True)

    return sampler, param_samples, flat_samples

################################## plotting ##################################
def corrfunc_plot(xi_data, samples, params, xhi_fine, logZ_fine, xi_model_fine, xhi_coarse, logZ_coarse, covar_array, \
                  corrfile, nrand=50, rand=None, save_xi_err=None):

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
    xi_model_mean = np.mean(xi_model_samp, axis=0)
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
                  capsize=4,
                  mec='none', ls='none', label='data', zorder=20)
    axis.plot(vel_corr, factor*xi_model_mean, linewidth=2.0, color='red', zorder=10, label='inferred model')

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

    fx.tight_layout()
    fx.savefig(corrfile)
    plt.close()
    #plt.show()

def plot_corrmatrix(coarse_out, data_out, logZ_want, xhi_want, vmin=None, vmax=None, plot_covar=False):
    # plotting the correlation matrix (copied from CIV_forest/metal_corrfunc.py)

    xhi_coarse, logZ_coarse, lnlike_coarse = coarse_out
    xhi_data, logZ_data, xi_data, covar_array, params = data_out
    vmin_corr = params['vmin_corr'][0]
    vmax_corr = params['vmax_corr'][0]

    ixhi = find_closest(xhi_coarse, xhi_want)
    iZ = find_closest(logZ_coarse, logZ_want)
    xhi_data = xhi_coarse[ixhi]
    logZ_data = logZ_coarse[iZ]
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
    plt.imshow(corr, origin='lower', interpolation='nearest', extent=[vmin_corr, vmax_corr, vmin_corr, vmax_corr], \
               vmin=vmin, vmax=vmax, cmap='inferno')
    plt.xlabel(r'$\Delta v$ (km/s)', fontsize=15)
    plt.ylabel(r'$\Delta v$ (km/s)', fontsize=15)
    plt.title('For model (logZ, xHI) = (%0.2f, %0.2f)' % (logZ_data, xhi_data))
    plt.colorbar()

    plt.show()

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

def plot_single_corr_elem(coarse_out, data_out, corr_array, rand_ixhi=None, rand_ilogZ=None, rand_i=None, rand_j=None):

    xhi_coarse, logZ_coarse, lnlike_coarse = coarse_out
    xhi_data, logZ_data, xi_data, covar_array, params = data_out
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

    plt.subplot(121)
    plt.title('ilogZ = %d, logZ = %0.2f' % (rand_ilogZ, logZ_coarse[rand_ilogZ]))
    plt.plot(xhi_coarse, corr_array[:,rand_ilogZ, rand_i, rand_j], label='($i,j$)=(%d,%d)' % (rand_i, rand_j))
    plt.xlabel('xHI')
    plt.ylabel(r'Corr$_{ij}$=Cov$_{ij}$/$\sqrt{Cov_{ii}Cov_{jj}}$')
    plt.legend()

    plt.subplot(122)
    plt.title('ixHI = %d, xHI = %0.2f' % (rand_ixhi, xhi_coarse[rand_ixhi]))
    plt.plot(logZ_coarse, corr_array[rand_ixhi, :, rand_i, rand_j])
    plt.xlabel('logZ')

    #plt.yscale('log')
    #plt.legend()
    plt.show()

def lnlike_plot_slice(xhi_arr, logZ_arr, lnlike_arr, xhi_want, logZ_want):
    ixhi = find_closest(xhi_arr, xhi_want)
    iZ = find_closest(logZ_arr, logZ_want)

    lnlike_arr = lnlike_arr - lnlike_arr.max()
    lnlike_arr = np.exp(lnlike_arr)

    plt.subplot(121)
    plt.plot(logZ_arr, lnlike_arr[ixhi])
    plt.xlabel('logZ')
    plt.ylabel('likelihood')

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

