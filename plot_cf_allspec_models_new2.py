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

factor = 1e5
vmin, vmax = 0, 3500
ymin, ymax = -0.0007 * factor, 0.0012 * factor

fwhm = 120
sampling = 3

xhi_models = [0.50]
logZ_models = [-3.0, -3.2, -3.5]
given_bins1 = ccf.custom_cf_bin4(dv1=40)
given_bins2 = ccf.custom_cf_bin4(dv1=80)

xi_mean_models_bin1 = []
xi_mean_models_bin2 = []
vel_mid_bin1 = []
vel_mid_bin2 = []

for xhi in xhi_models:
    filename = 'ran_skewers_z75_OVT_xHI_%0.2f_tau.fits' % xhi
    params = Table.read(filename, hdu=1)
    skewers = Table.read(filename, hdu=2)

    xi_mean_models_logZ_bin1 = []
    xi_mean_models_logZ_bin2 = []

    for logZ in logZ_models:
        vel_lores, (flux_lores, flux_lores_igm, flux_lores_cgm, _, _), \
        vel_hires, (flux_hires, flux_hires_igm, flux_hires_cgm, _, _), \
        (oden, v_los, T, xHI), cgm_tuple = reion_utils.create_mgii_forest(params, skewers, logZ, fwhm, sampling=sampling)

        mean_flux_nless = np.mean(flux_lores)
        delta_f_nless = (flux_lores - mean_flux_nless) / mean_flux_nless

        (vel_mid_log, xi_nless_log, npix, xi_nless_zero_lag) = reion_utils.compute_xi(delta_f_nless, vel_lores, 0, 0, 0, given_bins=given_bins1)
        xi_mean_log = np.mean(xi_nless_log, axis=0)
        xi_mean_models_logZ_bin1.append(xi_mean_log)
        vel_mid_bin1 = vel_mid_log

        (vel_mid_log, xi_nless_log, npix, xi_nless_zero_lag) = reion_utils.compute_xi(delta_f_nless, vel_lores, 0, 0, 0, given_bins=given_bins2)
        xi_mean_log = np.mean(xi_nless_log, axis=0)
        xi_mean_models_logZ_bin2.append(xi_mean_log)
        vel_mid_bin2 = vel_mid_log

    xi_mean_models_bin1.append(xi_mean_models_logZ_bin1)
    xi_mean_models_bin2.append(xi_mean_models_logZ_bin2)

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
fig.subplots_adjust(left=0.12, bottom=0.15, right=0.98, top=0.93, wspace=0, hspace=0.)

for ixhi, xhi in enumerate(xhi_models):
    for ilogZ, logZ in enumerate(logZ_models):
        if len(xhi_models) == 1:
            label = r'[Mg/H] = $%0.1f$' % logZ
        else:
            label = r'($x_{\mathrm{HI}}$, [Mg/H]) = (%0.2f, $%0.1f$)' % (xhi, logZ)

        ax1.plot(vel_mid_bin1, xi_mean_models_bin1[ixhi][ilogZ] * factor, lw=2.0, label=label)
        ax1.plot(vel_mid_bin2, xi_mean_models_bin2[ixhi][ilogZ] * factor, 'k:', lw=2.0)

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

plt.show()
plt.close()