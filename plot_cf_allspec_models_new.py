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

xytick_size = 16 + 2
xylabel_fontsize = 20 + 5
legend_fontsize = 16 + 5
black_shaded_alpha = 0.15

savefig_unmasked = 'paper_plots/10qso/cf_masked_models.pdf'

given_bins = np.array(ccf.custom_cf_bin4(dv1=80))
factor = 1e5
vmin, vmax = 0, 3500
ymin, ymax = -0.00017 * factor, 0.00021 * factor

####### plot masked CF #######
#### models
# using mosfire specs since measurement dominated by mosfire quasars
fwhm = 120 #83.05
sampling = 3 #2.78

xhi_models = [0.50]
logZ_models = [-3.2, -3.5, -4.0] #[-3.0, -3.2, -3.5]

cf = fits.open('save_cf/paper_new/xi_10qso_everyn60_corr_allz.fits') # 'save_cf/paper/xi_mean_mask_10qso_everyn60_corr.fits'
vel_mid = cf['vel_mid'].data
xi_mean_mask = cf['XI_MEAN_MASK'].data
xi_err = np.load('save_cf/paper_new/xi_err_allz.npy') #'allz_xi_err_paper.npy'

xi_mean_models = []
for xhi in xhi_models:
    filename = 'ran_skewers_z75_OVT_xHI_%0.2f_tau.fits' % xhi
    params = Table.read(filename, hdu=1)
    skewers = Table.read(filename, hdu=2)

    xi_mean_models_logZ = []
    for logZ in logZ_models:
        vel_lores, (flux_lores, flux_lores_igm, flux_lores_cgm, _, _), \
        vel_hires, (flux_hires, flux_hires_igm, flux_hires_cgm, _, _), \
        (oden, v_los, T, xHI), cgm_tuple = reion_utils.create_mgii_forest(params, skewers, logZ, fwhm, sampling=sampling)

        mean_flux_nless = np.mean(flux_lores)
        delta_f_nless = (flux_lores - mean_flux_nless) / mean_flux_nless

        (vel_mid_log, xi_nless_log, npix, xi_nless_zero_lag) = reion_utils.compute_xi_weights(delta_f_nless, vel_lores, 0, 0, 0, given_bins=given_bins)

        xi_mean_log = np.mean(xi_nless_log, axis=0)
        xi_mean_models_logZ.append(xi_mean_log)
    xi_mean_models.append(xi_mean_models_logZ)

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(11, 8))
fig.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.88, wspace=0, hspace=0.)

#yerr = xi_noise_mask_std * factor
yerr = xi_err #(xi_std_unmask / np.sqrt(nqso)) * factor
ax1.errorbar(vel_mid, xi_mean_mask * factor, yerr=yerr * factor, lw=2.0, \
             fmt='o', c='black', ecolor='black', capthick=2.0, capsize=2,  mec='none', zorder=20) #, alpha=0.8)

for ixhi, xhi in enumerate(xhi_models):
    for ilogZ, logZ in enumerate(logZ_models):
        if len(xhi_models) == 1:
            label = r'[Mg/H] = $%0.1f$' % logZ
        else:
            label = r'($x_{\mathrm{HI}}$, [Mg/H]) = (%0.2f, $%0.1f$)' % (xhi, logZ)
        ax1.plot(vel_mid_log, xi_mean_models[ixhi][ilogZ] * factor, lw=2.0, label=label)

ax1.annotate('MgII doublet', xy=(1030, 0.85 * ymax), xytext=(1030, 0.85* ymax), fontsize=legend_fontsize, color='black')
ax1.annotate('separation', xy=(1070, 0.75 * ymax), xytext=(1070, 0.75 * ymax), fontsize=legend_fontsize, color='black')
ax1.annotate('', xy=(780, 0.88 * ymax), xytext=(1010, 0.88* ymax),
                fontsize=16, arrowprops={'arrowstyle': '-|>', 'lw': 4, 'color': 'black'}, va='center', color='magenta')

ax1.set_xlabel(r'$\Delta v$ (km/s)', fontsize=xylabel_fontsize)
ax1.set_ylabel(r'$\xi(\Delta v)\times 10^5$', fontsize=xylabel_fontsize)
ax1.axvline(768.469, color='black', linestyle='--', linewidth=2.0)

ax1.set_xlim([vmin, vmax])
ax1.set_ylim([ymin, ymax])
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.legend(loc=1, fontsize=legend_fontsize)
ax1.tick_params(which='both', labelsize=xytick_size)

# Create upper axis in cMpc
cosmo = FlatLambdaCDM(H0=100.0 * 0.6704, Om0=0.3192, Ob0=0.04964) # Nyx values
z = 6.469# all-z
Hz = (cosmo.H(z))
a = 1.0 / (1.0 + z)
rmin = (vmin*u.km/u.s/a/Hz).to('Mpc').value
rmax = (vmax*u.km/u.s/a/Hz).to('Mpc').value

# Make the new upper x-axes
atwin = ax1.twiny()
atwin.set_xlabel('R (cMpc)', fontsize=xylabel_fontsize)#, labelpad=8)
atwin.xaxis.tick_top()

atwin.axis([rmin, rmax, ymin, ymax])
atwin.tick_params(top=True)
atwin.xaxis.set_minor_locator(AutoMinorLocator())
atwin.tick_params(axis="x", labelsize=xytick_size)

if savefig_unmasked != None:
    plt.savefig(savefig_unmasked)
plt.show()
plt.close()