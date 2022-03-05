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
from enigma.reion_forest.utils import find_closest
from enigma.reion_forest import inference

seed = 2213142
rand = np.random.RandomState(seed)
figpath = '/Users/suksientie/Research/MgII_forest/plots/mcmc_out/'

# For metal limits
#logZ_guess = -6.0
#xhi_guess  = 1.0
logZ_guess = -3.70
xhi_guess  = 0.74
#logZ_guess = -3.20
#xhi_guess  = 0.1
linearZprior = False

def init(modelfile):
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

def run_mcmc(fine_out, coarse_out, data_out, nsteps=100000, burnin=1000, nwalkers=40):

    xhi_fine, logZ_fine, lnlike_fine, xi_model_fine = fine_out
    xhi_coarse, logZ_coarse, lnlike_coarse = coarse_out
    xhi_data, logZ_data, xi_data, covar_array, params = data_out

    # find optimal starting points for each walker
    chi2_func = lambda *args: -2 * inference.lnprob(*args)
    logZ_fine_min = logZ_fine.min()
    logZ_fine_max = logZ_fine.max()
    #bounds = [(0.0,1.0), (logZ_fine_min, logZ_fine_max)] if not linearZprior else [(0.95, 1.0), (0.0, np.power(10.0,logZ_fine_max))]
    bounds = [(0.2, 0.65), (logZ_fine_min, logZ_fine_max)] if not linearZprior else [(0.20, 0.65), (0.0, np.power(10.0, logZ_fine_max))]
    print("bounds", bounds)
    args = (lnlike_fine, xhi_fine, logZ_fine, linearZprior)
    result_opt = optimize.differential_evolution(chi2_func,bounds=bounds, popsize=25, recombination=0.7,
                                                 disp=True, polish=True, args=args, seed=rand)
    ndim = 2 # xhi and logZ
    #pos = [[result_opt.x[i] for i in range(ndim)] + 1.0e-1*rand.randn(ndim) for i in range(nwalkers)]

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
    #fig = corner.corner(flat_samples, labels=var_label, truths=truths, levels = (0.68,), color='k', truth_color='darkgreen',
    #                    show_titles=True, title_kwargs={"fontsize": 15}, label_kwargs={'fontsize':20},
    #                    data_kwargs={'ms': 1.0, 'alpha': 0.1})
    fig = corner.corner(param_samples, labels=var_label, levels=(0.68,), color='k',
                        show_titles=True, title_kwargs={"fontsize": 15}, label_kwargs={'fontsize': 20},
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
    inference.corrfunc_plot(xi_data, param_samples, params, xhi_fine, logZ_fine, xi_model_fine, xhi_coarse, logZ_coarse, covar_array,
                  xhi_data, logZ_data, corrfile, rand=rand)
    # Lower limit on metallicity for pristine case
    if linearZprior:
        ixhi_prior = flat_samples[:,0] > 0.95
        logZ_95 = np.percentile(param_samples[ixhi_prior,1], 95.0)
        print('Obtained 95% upper limit of {:6.4f}'.format(logZ_95))

    return sampler, param_samples, flat_samples
