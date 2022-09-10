import numpy as np
from astropy.cosmology import FlatLambdaCDM
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.table import Table
import compute_cf_data as ccf
import sys
sys.path.append('/Users/suksientie/codes/enigma')
from enigma.reion_forest import utils as reion_utils

### Figure settings
font = {'family' : 'serif', 'weight' : 'normal'}#, 'size': 11}
plt.rc('font', **font)
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['xtick.minor.width'] = 1.5
mpl.rcParams['ytick.minor.width'] = 1.5
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.minor.size'] = 4

xytick_size = 16
xylabel_fontsize = 20
legend_fontsize = 12
black_shaded_alpha = 0.15

savefig_unmasked = 'paper_plots/cf_masked_models.pdf'

nqso = 4
median_z = 6.57
seed_list=[None, None, None, None]
given_bins = ccf.custom_cf_bin4() #None #if using given_bins, see ccf.custom_cf_bin2() for vmin_corr, vmax_corr, dv_corr

factor = 1e5
vmin, vmax = 0, 3500
ymin, ymax = -0.0007 * factor, 0.0012 * factor

####### running allspec() and plotting the CFs for low-z bin, high-z bin, and all-z bin
lowz_cgm_fit_gpm, highz_cgm_fit_gpm, allz_cgm_fit_gpm = ccf.init_cgm_fit_gpm()

vel_mid, xi_mean_unmask, xi_mean_mask, xi_noise_unmask_all, xi_noise_mask_all, xi_unmask_all, xi_mask_all = \
    ccf.allspec(nqso, 'all', allz_cgm_fit_gpm, seed_list=seed_list, given_bins=given_bins)
xi_std_unmask = np.std(xi_unmask_all, axis=0)
xi_std_mask = np.std(xi_mask_all, axis=0)

#####
nqso, nreal, nvel = xi_noise_mask_all.shape

xi_noise_mask = np.reshape(xi_noise_mask_all, (nqso*nreal, nvel))
xi_noise_mask_mean = np.mean(xi_noise_mask, axis=0)
xi_noise_mask_std = np.std(xi_noise_mask, axis=0)

"""
# Create upper axis in cMpc
lit_h = 0.67038
Om0 = 0.3192
Ob0 = 0.04964
cosmo = FlatLambdaCDM(H0=100.0 * lit_h, Om0=Om0, Ob0=Ob0)
z = params['z'][0]
Hz = (cosmo.H(z))
a = 1.0 / (1.0 + z)
rmin = (vmin*u.km/u.s/a/Hz).to('Mpc').value
rmax = (vmax*u.km/u.s/a/Hz).to('Mpc').value
"""

####### plot masked CF #######
#### models
fwhm = 90 # 83
sampling = 3

xhi_models = [0.50]
logZ_models = [-3.0, -3.2, -3.5]

#xhi_models = [0.2, 0.50]
#logZ_models = [-3.0, -3.5]

xi_mean_models = []
for xhi in xhi_models:
    filename = 'ran_skewers_z75_OVT_xHI_%0.2f_tau.fits' % xhi
    params = Table.read(filename, hdu=1)
    skewers = Table.read(filename, hdu=2)
    #skewers = skewers[0:100]

    xi_mean_models_logZ = []
    for logZ in logZ_models:
        vel_lores, (flux_lores, flux_lores_igm, flux_lores_cgm, _, _), \
        vel_hires, (flux_hires, flux_hires_igm, flux_hires_cgm, _, _), \
        (oden, v_los, T, xHI), cgm_tuple = reion_utils.create_mgii_forest(params, skewers, logZ, fwhm, sampling=sampling)

        mean_flux_nless = np.mean(flux_lores)
        delta_f_nless = (flux_lores - mean_flux_nless) / mean_flux_nless

        (vel_mid_log, xi_nless_log, npix, xi_nless_zero_lag) = reion_utils.compute_xi(delta_f_nless, vel_lores, 0, 0, 0, \
                                                                                      given_bins=given_bins)
        xi_mean_log = np.mean(xi_nless_log, axis=0)
        xi_mean_models_logZ.append(xi_mean_log)
    xi_mean_models.append(xi_mean_models_logZ)

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
fig.subplots_adjust(left=0.12, bottom=0.15, right=0.98, top=0.93, wspace=0, hspace=0.)

yerr = xi_noise_mask_std * factor
ax1.errorbar(vel_mid, xi_mean_mask * factor, yerr=yerr, lw=2.0, \
             fmt='o', c='black', ecolor='black', capthick=2.0, capsize=2,  mec='none', zorder=20) #, alpha=0.8)

for ixhi, xhi in enumerate(xhi_models):
    for ilogZ, logZ in enumerate(logZ_models):
        if len(xhi_models) == 1:
            label = r'[Mg/H] = $%0.1f$' % logZ
        else:
            label = r'($x_{\mathrm{HI}}$, [Mg/H]) = (%0.2f, $%0.1f$)' % (xhi, logZ)
        ax1.plot(vel_mid_log, xi_mean_models[ixhi][ilogZ] * factor, lw=2.0, label=label)

ax1.annotate('MgII doublet', xy=(1030, 0.85 * ymax), xytext=(1030, 0.85* ymax), fontsize=xytick_size, color='black')
ax1.annotate('separation', xy=(1070, 0.75 * ymax), xytext=(1070, 0.75 * ymax), fontsize=xytick_size, color='black')
ax1.annotate('', xy=(780, 0.88 * ymax), xytext=(1010, 0.88* ymax),
                fontsize=16, arrowprops={'arrowstyle': '-|>', 'lw': 4, 'color': 'black'}, va='center', color='magenta')

ax1.set_xlabel(r'$\Delta v$ (km/s)', fontsize=xylabel_fontsize)
ax1.set_ylabel(r'$\xi(\Delta v)\times 10^5$', fontsize=xylabel_fontsize)
ax1.axvline(768.469, color='black', linestyle='--', linewidth=2.0)
ax1.set_ylim([ymin, ymax])
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.legend(loc=1, fontsize=legend_fontsize)
ax1.tick_params(which='both', labelsize=xytick_size)

plt.savefig(savefig_unmasked)
plt.show()
plt.close()