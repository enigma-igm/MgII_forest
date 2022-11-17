import mcmc_inference as mcmc
import compute_cf_data as ccf

modelfile = 'igm_cluster/5qso/corr_func_models_all.fits'
redshift_bin = 'all'
figpath = '/Users/suksientie/Research/MgII_forest/mcmc/5qso/'
given_bins = ccf.custom_cf_bin4(dv1=80)

# initialize
fine_out, coarse_out, data_out = mcmc.init(modelfile, redshift_bin, figpath, given_bins)

# log Z prior
savefits_chain = figpath + 'mcmc_chain_' + redshift_bin + 'z.fits'
save_xi_err = figpath + 'xi_err.npy'
sampler, param_samples, flat_samples = mcmc.run_mcmc(fine_out, coarse_out, data_out, redshift_bin, figpath, \
                                                     nsteps=100000, burnin=1000, nwalkers=40, \
             linearZprior=False, savefits_chain=savefits_chain, actual_data=True, save_xi_err=save_xi_err)

# linear Z prior
savefits_chain = figpath + 'mcmc_chain_' + redshift_bin + 'z_upperlim.fits'
sampler, param_samples, flat_samples = mcmc.run_mcmc(fine_out, coarse_out, data_out, redshift_bin, figpath, \
                                                     nsteps=100000, burnin=1000, nwalkers=40, \
             linearZprior=True, savefits_chain=savefits_chain, actual_data=True, save_xi_err=None)