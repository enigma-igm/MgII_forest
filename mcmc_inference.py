'''
Functions here:
    - init
    - run_mcmc
    - corrfunc_plot
'''

import numpy as np
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

seed = 988765 #2213142
rand = np.random.RandomState(seed)
figpath = '/Users/suksientie/Research/MgII_forest/plots/mcmc_out/'

logZ_guess = -5.0 #-4.50 # -3.70
xhi_guess  = 0.50 # 0.74
linearZprior = False

datapath = '/Users/suksientie/Research/data_redux/'
fitsfile_list = [datapath + 'wavegrid_vel/J0313-1806/vel1234_coadd_tellcorr.fits', \
                 datapath + 'wavegrid_vel/J1342+0928/vel123_coadd_tellcorr.fits', \
                 datapath + 'wavegrid_vel/J0252-0503/vel12_coadd_tellcorr.fits', \
                 datapath + 'wavegrid_vel/J0038-1527/vel1_tellcorr_pad.fits']

qso_namelist = ['J0313-1806', 'J1342+0928', 'J0252-0503', 'J0038-1527']
qso_zlist = [7.6, 7.54, 7.0, 7.0]

def init(modelfile, actual_data=True):
    # modelfile = 'igm_cluster/corr_func_models_fwhm_90.000_samp_3.000.fits'
    params, xi_mock_array, xi_model_array, covar_array, icovar_array, lndet_array = read_model_grid(modelfile)
    logZ_coarse = params['logZ'].flatten()
    xhi_coarse = params['xhi'].flatten()
    vel_corr = params['vel_mid'].flatten()
    vel_min = params['vmin_corr'][0]
    vel_max = params['vmax_corr'][0]
    nlogZ = params['nlogZ'][0]
    nhi = params['nhi'][0]

    if actual_data:
        print("==== using actual data ==== ")
        xhi_data, logZ_data = 0.5, -3.50  # bogus numbers
        vel_mid, xi_mean_unmask, xi_mean_mask, _, _, _, _ = compute_cf_data.allspec(fitsfile_list, qso_zlist, plot=False)
        xi_data = xi_mean_mask
        xi_mask = np.ones_like(xi_data, dtype=bool)  # Boolean array

    else:
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
                                                  icovar_array[ixhi, iZ, :, :])

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

def run_mcmc(fine_out, coarse_out, data_out, nsteps=100000, burnin=1000, nwalkers=40, savefits_chain=None, actual_data=True):

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

    inference.walker_plot(chain, truths, var_label, figpath + 'walkers.pdf')

    # Make the corner plot, again use the true values in the chain
    if actual_data:
        fig = corner.corner(param_samples, labels=var_label, levels=(0.68,), color='k',
                            show_titles=True, title_kwargs={"fontsize": 15}, label_kwargs={'fontsize': 20},
                            data_kwargs={'ms': 1.0, 'alpha': 0.1})
    else:
        fig = corner.corner(flat_samples, labels=var_label, truths=truths, levels = (0.68,), color='k', truth_color='darkgreen',
                            show_titles=True, title_kwargs={"fontsize": 15}, label_kwargs={'fontsize':20},
                            data_kwargs={'ms': 1.0, 'alpha': 0.1})

    cornerfile = figpath + 'corner_plot.pdf' 
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
    corrfile = figpath + 'corr_func_data.pdf'
    if actual_data:
        corrfunc_plot(xi_data, param_samples, params, xhi_fine, logZ_fine, xi_model_fine, xhi_coarse, logZ_coarse, covar_array,
                      corrfile, nrand=300, rand=rand)
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

def corrfunc_plot(xi_data, samples, params, xhi_fine, logZ_fine, xi_model_fine, xhi_coarse, logZ_coarse, covar_array, \
                  corrfile, nrand=50, rand=None):

    # adapted from enigma.reion_forest.inference.corrfunc_plot
    if rand is None:
        rand = np.random.RandomState(1234)

    factor = 1e5
    fx = plt.figure(1, figsize=(12, 9))
    # left, bottom, width, height
    rect = [0.12, 0.12, 0.84, 0.75]
    axis = fx.add_axes(rect)
    vel_corr = params['vel_mid'].flatten()
    vel_min = params['vmin_corr']
    vel_max = params['vmax_corr']

    vmin, vmax = 0.4*vel_corr.min(), 1.02*vel_corr.max()
    # Compute the mean model from the samples
    xi_model_samp = inference.xi_model(samples, xhi_fine, logZ_fine, xi_model_fine)
    xi_model_mean = np.mean(xi_model_samp, axis=0)
    # Compute the covariance at the mean model
    theta_mean = np.mean(samples,axis=0)
    # Average the diagonal instead?
    covar_mean = inference.covar_model(theta_mean, xhi_coarse, logZ_coarse, covar_array)
    xi_err = np.sqrt(np.diag(covar_mean))
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

    infr_xy = (vmin + 0.1*(vmax-vmin), 0.90*ymax)
    xhi_xy  = (vmin + 0.05*(vmax-vmin), 0.80*ymax)
    Z_xy    = (vmin - 0.05*(vmax-vmin), 0.70*ymax)
    #infr_xy = (vmin + 0.74 * (vmax - vmin), 0.60 * ymax)
    #xhi_xy = (vmin + 0.685 * (vmax - vmin), 0.52 * ymax)
    #Z_xy = (vmin + 0.54 * (vmax - vmin), 0.44 * ymax)
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