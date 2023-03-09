import mcmc_inference as mcmc
import numpy as np
import compute_model_grid_8qso_fast as cmg8

modelfile = '/Users/suksientie/Research/MgII_forest/igm_cluster/10qso/corr_func_models_all_ivarweights.fits'
save_covar_fine = '/Users/suksientie/Research/MgII_forest/igm_cluster/10qso/covar_fine_all_ivarweights.npy'

params, xi_mock_array, xi_model_array, covar_array, icovar_array, lndet_array = cmg8.read_model_grid(modelfile)
logZ_coarse = params['logZ'].flatten()
xhi_coarse = params['xhi'].flatten()

xhi_fine = np.linspace(0, 1, 1001)
logZ_fine = np.linspace(-5.5, -2.5, 1001)

covar_fine = mcmc.interp_covar(xhi_fine, logZ_fine, xhi_coarse, logZ_coarse, covar_array)
np.save(save_covar_fine, covar_fine)