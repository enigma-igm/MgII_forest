import mcmc_inference as mcmc
import compute_cf_data as ccf
from astropy.io import fits
import numpy as np
import mutils

#modelfile = 'igm_cluster/8qso/corr_func_models_all.fits'
#figpath = '/Users/suksientie/Research/MgII_forest/mcmc/8qso/npixweights/doublecheck/'

mgii_dir = '/Users/suksientie/Research/MgII_forest/'
modelfile = mgii_dir + 'igm_cluster/10qso/corr_func_models_high_ivarweights.fits'
figpath = mgii_dir + 'mcmc/10qso/xi_mask_extract_subarr/'
redshift_bin = 'high'
given_bins = ccf.custom_cf_bin4(dv1=80)

if redshift_bin == 'all':
    cf = fits.open('save_cf/xi_mean_mask_10qso_everyn60.fits')
    covar_fine_file = mgii_dir + 'igm_cluster/10qso/covar_fine_all_ivarweights_subarr.npy' #covar_fine_all_ivarweights_lagmask.npy
    lag_mask = mutils.cf_lags_to_mask()

elif redshift_bin == 'high':
    cf = fits.open('save_cf/xi_mean_mask_10qso_everyn60_highz.fits')
    covar_fine_file = mgii_dir + 'igm_cluster/10qso/covar_fine_high_ivarweights_subarr.npy'
    lag_mask = mutils.cf_lags_to_mask_highz()

elif redshift_bin == 'low':
    cf = fits.open('save_cf/xi_mean_mask_10qso_everyn60_lowz.fits')
    covar_fine_file = mgii_dir + 'igm_cluster/10qso/covar_fine_low_ivarweights_subarr.npy'
    lag_mask = mutils.cf_lags_to_mask_lowz()

xi_mean_data = cf['XI_MEAN_MASK'].data
covar_array_fine = np.load(covar_fine_file)

# initialize
fine_out, coarse_out, data_out = mcmc.init(modelfile, redshift_bin, given_bins, lag_mask=lag_mask, figpath=figpath, \
                                           xi_mean_data=xi_mean_data, covar_array_fine=covar_array_fine)

# log Z priors
savefits_chain = figpath + redshift_bin + 'z_mcmc_chain.fits'
save_xi_err = figpath + redshift_bin + 'z_xi_err.npy'
sampler, param_samples, flat_samples = mcmc.run_mcmc(fine_out, coarse_out, data_out, redshift_bin, figpath, \
                                                     nsteps=100000, burnin=1000, nwalkers=40, \
             linearZprior=False, savefits_chain=savefits_chain, actual_data=True, save_xi_err=save_xi_err)

# linear Z prior
savefits_chain = figpath + redshift_bin + 'z_mcmc_chain_upperlim.fits'
sampler, param_samples, flat_samples = mcmc.run_mcmc(fine_out, coarse_out, data_out, redshift_bin, figpath, \
                                                     nsteps=100000, burnin=1000, nwalkers=40, \
             linearZprior=True, savefits_chain=savefits_chain, actual_data=True, save_xi_err=None)
