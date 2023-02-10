import mcmc_inference as mcmc
import compute_cf_data as ccf
from astropy.io import fits

#modelfile = 'igm_cluster/8qso/corr_func_models_all.fits'
#figpath = '/Users/suksientie/Research/MgII_forest/mcmc/8qso/npixweights/doublecheck/'

modelfile = 'igm_cluster/8qso/corr_func_models_all_ivarweights_everyn60_ncovar1e6.fits'
#figpath = 'plots/8qso-debug/'
figpath = '/Users/suksientie/Research/MgII_forest/plots/8qso-debug/ivarweights_globalfmean_everyn60_subtract-df/median_model/'
redshift_bin = 'all'
given_bins = ccf.custom_cf_bin4(dv1=80)

cf = fits.open('plots/8qso-debug/cf_8qso_allz_ivarweights_globalfmean_everyn60_subtract-df.fits')
xi_mean_data = cf['xi_mean_mask'].data
#xi_mean_data = None

# initialize
fine_out, coarse_out, data_out = mcmc.init(modelfile, redshift_bin, given_bins, figpath, xi_mean_data=xi_mean_data)

# log Z prior
savefits_chain = figpath + redshift_bin + 'z_mcmc_chain.fits'
save_xi_err = figpath + redshift_bin + 'z_xi_err.npy'
sampler, param_samples, flat_samples = mcmc.run_mcmc(fine_out, coarse_out, data_out, redshift_bin, figpath, \
                                                     nsteps=100000, burnin=1000, nwalkers=40, \
             linearZprior=False, savefits_chain=savefits_chain, actual_data=True, save_xi_err=save_xi_err, \
            inferred_model='median')

"""
# linear Z prior
savefits_chain = figpath + redshift_bin + 'z_mcmc_chain_upperlim.fits'
sampler, param_samples, flat_samples = mcmc.run_mcmc(fine_out, coarse_out, data_out, redshift_bin, figpath, \
                                                     nsteps=100000, burnin=1000, nwalkers=40, \
             linearZprior=True, savefits_chain=savefits_chain, actual_data=True, save_xi_err=None)
"""
