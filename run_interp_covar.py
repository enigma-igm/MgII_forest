import mcmc_inference as mcmc
import numpy as np
import compute_model_grid_8qso_fast as cmg8
import mutils

"""
modelfile = '/Users/suksientie/Research/MgII_forest/igm_cluster/10qso/corr_func_models_all_ivarweights_lagmask.fits'
save_covar_fine = '/Users/suksientie/Research/MgII_forest/igm_cluster/10qso/covar_fine_all_ivarweights_lagmask.npy'

params, xi_mock_array, xi_model_array, covar_array, icovar_array, lndet_array = cmg8.read_model_grid(modelfile)
logZ_coarse = params['logZ'].flatten()
xhi_coarse = params['xhi'].flatten()

nlogZ_fine = 1001
logZ_fine_min = logZ_coarse.min()
logZ_fine_max = logZ_coarse.max()
nhi_fine = 1001
xhi_fine_min = 0.0
xhi_fine_max = 1.0

xhi_fine = np.linspace(xhi_fine_min, xhi_fine_max, nhi_fine)
logZ_fine = np.linspace(logZ_fine_min, logZ_fine_max, nlogZ_fine)

covar_fine = mcmc.interp_covar(xhi_fine, logZ_fine, xhi_coarse, logZ_coarse, covar_array)
np.save(save_covar_fine, covar_fine)
"""

redshift_bin = 'low'
modelfile = '/Users/suksientie/Research/MgII_forest/igm_cluster/10qso/rebin/corr_func_models_%s_ivarweights.fits' % redshift_bin

############## interpolate covariance for subarr
save_covar_fine = '/Users/suksientie/Research/MgII_forest/igm_cluster/10qso/rebin/covar_fine_%s_ivarweights_subarr.npy' % redshift_bin

if redshift_bin == 'all':
    lag_mask, _ = mutils.cf_lags_to_mask()

elif redshift_bin == 'high':
    lag_mask, _ = mutils.cf_lags_to_mask_highz()

elif redshift_bin == 'low':
    lag_mask, _ = mutils.cf_lags_to_mask_lowz()

covar_fine = mcmc.interp_covar(modelfile, lag_mask=lag_mask)
np.save(save_covar_fine, covar_fine)

############## interpolate covariance for fullarr
save_covar_fine = '/Users/suksientie/Research/MgII_forest/igm_cluster/10qso/rebin/covar_fine_%s_ivarweights.npy' % redshift_bin
covar_fine = mcmc.interp_covar(modelfile, lag_mask=None)
np.save(save_covar_fine, covar_fine)
