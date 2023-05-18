import mcmc_inference as mcmc
import compute_cf_data as ccf
from astropy.io import fits
import numpy as np
import mutils

#modelfile = 'igm_cluster/8qso/corr_func_models_all.fits'
#figpath = '/Users/suksientie/Research/MgII_forest/mcmc/8qso/npixweights/doublecheck/'

redshift_bin = 'low'
mgii_dir = '/Users/suksientie/Research/MgII_forest/'
given_bins = ccf.custom_cf_bin4(dv1=80)
modelfile = mgii_dir + 'igm_cluster/10qso/corr_func_models_%s_ivarweights.fits' % redshift_bin

subarr_flag = True # flag to extract sub arrays and do inference using velocity lag bins that are not masked

if subarr_flag:
    figpath = mgii_dir + 'mcmc/10qso/paper/xi_mask_extract_subarr2/'
else:
    figpath = mgii_dir + 'mcmc/10qso/paper/xi_fullarr/'

if redshift_bin == 'all':
    cf = fits.open('save_cf/paper/xi_mean_mask_10qso_everyn60_corr.fits')
    if subarr_flag:
        covar_fine_file = mgii_dir + 'igm_cluster/10qso/covar_fine_all_ivarweights_subarr.npy'
        lag_mask, _ = mutils.cf_lags_to_mask()
    else:
        covar_fine_file = mgii_dir + 'igm_cluster/10qso/covar_fine_all_ivarweights.npy'
        lag_mask = None

elif redshift_bin == 'high':
    cf = fits.open('save_cf/paper/xi_mean_mask_10qso_everyn60_highz.fits')
    if subarr_flag:
        covar_fine_file = mgii_dir + 'igm_cluster/10qso/covar_fine_high_ivarweights_subarr.npy'
        lag_mask, _ = mutils.cf_lags_to_mask_highz()
    else:
        covar_fine_file = mgii_dir + 'igm_cluster/10qso/covar_fine_high_ivarweights.npy'
        lag_mask = None

elif redshift_bin == 'low':
    cf = fits.open('save_cf/paper/xi_mean_mask_10qso_everyn60_lowz.fits')
    if subarr_flag:
        covar_fine_file = mgii_dir + 'igm_cluster/10qso/covar_fine_low_ivarweights_subarr.npy'
        lag_mask, _ = mutils.cf_lags_to_mask_lowz()
    else:
        covar_fine_file = mgii_dir + 'igm_cluster/10qso/covar_fine_high_ivarweights.npy'
        lag_mask = None

xi_mean_data = cf['XI_MEAN_MASK'].data
covar_array_fine = np.load(covar_fine_file)

# initialize
fine_out, coarse_out, data_out = mcmc.init(modelfile, redshift_bin, given_bins, lag_mask=lag_mask, figpath=figpath, \
                                           xi_mean_data=xi_mean_data, covar_array_fine=covar_array_fine)

# log Z priors
savefits_chain = figpath + redshift_bin + 'z_mcmc_chain.fits' # xi_err saved here
input_xi_err = None #figpath + redshift_bin + 'z_xi_err.npy'

sampler, param_samples, flat_samples, extra_out = mcmc.run_mcmc(fine_out, coarse_out, data_out, redshift_bin, figpath, \
                                                     nsteps=100000, burnin=1000, nwalkers=40, \
             linearZprior=False, savefits_chain=savefits_chain, actual_data=True, input_xi_err=input_xi_err, lag_mask=lag_mask)

# plot error bars on masked velocity lags
if subarr_flag:
    mcmc_out_full_arr = mgii_dir + 'mcmc/10qso/paper/xi_fullarr/%sz_mcmc_chain.fits' % redshift_bin
    input_xi_errfile = fits.open(mcmc_out_full_arr)
    input_xi_err = input_xi_errfile['xi_err'].data

    xi_data, param_samples, params, xhi_fine, logZ_fine, xi_model_fine, xhi_coarse, logZ_coarse, covar_array, corrfile, redshift_bin, rand = extra_out
    mcmc.corrfunc_plot_new(xi_data, param_samples, params, xhi_fine, logZ_fine, xi_model_fine, xhi_coarse, logZ_coarse,
                      covar_array, corrfile, redshift_bin, nrand=200, rand=rand, input_xi_err=input_xi_err, lag_mask=lag_mask)

# linear Z prior to get upper limits
if subarr_flag:
    savefits_chain = figpath + redshift_bin + 'z_mcmc_chain_upperlim.fits'
    sampler, param_samples, flat_samples, _ = mcmc.run_mcmc(fine_out, coarse_out, data_out, redshift_bin, figpath, \
                                                         nsteps=100000, burnin=1000, nwalkers=40, \
                 linearZprior=True, savefits_chain=savefits_chain, actual_data=True, input_xi_err=None, lag_mask=lag_mask)


