import numpy as np
from astropy.cosmology import FlatLambdaCDM
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
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
legend_fontsize = 14

savefig_masked = 'paper_plots/cf_masked.pdf'
savefig_unmasked = 'paper_plots/cf_unmasked.pdf'

nqso = 4
median_z = 6.57
seed_list=[None, None, None, None]
given_bins=None

####### running allspec() and plotting the CFs for low-z bin, high-z bin, and all-z bin
lowz_cgm_fit_gpm, highz_cgm_fit_gpm, allz_cgm_fit_gpm = ccf.init_cgm_fit_gpm()

vel_mid_low, xi_mean_unmask_low, xi_mean_mask_low, xi_noise_unmask_low, xi_noise_mask_low, xi_unmask_all_low, xi_mask_all_low = \
    ccf.allspec(nqso, 'low', lowz_cgm_fit_gpm, seed_list=seed_list, given_bins=given_bins)

xi_std_unmask_low = np.std(xi_unmask_all_low, axis=0)
xi_std_mask_low = np.std(xi_mask_all_low, axis=0)

vel_mid_high, xi_mean_unmask_high, xi_mean_mask_high, xi_noise_unmask_high, xi_noise_mask_high, xi_unmask_all_high, xi_mask_all_high = \
    ccf.allspec(nqso, 'high', highz_cgm_fit_gpm)
xi_std_unmask_high = np.std(xi_unmask_all_high, axis=0)
xi_std_mask_high = np.std(xi_mask_all_high, axis=0)

vel_mid, xi_mean_unmask, xi_mean_mask, xi_noise_unmask_all, xi_noise_mask_all, xi_unmask_all, xi_mask_all = \
    ccf.allspec(nqso, 'all', allz_cgm_fit_gpm)
xi_std_unmask = np.std(xi_unmask_all, axis=0)
xi_std_mask = np.std(xi_mask_all, axis=0)

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

xi_scale = 1
vmin, vmax = 0, 3500
ymin, ymax = -0.0010 * xi_scale, 0.002 * xi_scale

####### plot un-masked CF #######
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(14, 5), sharey=True)
fig.subplots_adjust(left=0.12, bottom=0.15, right=0.98, top=0.93, wspace=0, hspace=0.)

for i in range(nqso):
    for xi in xi_noise_unmask_low[i]:  # plotting all 500 realizations of the noise 2PCF (not masked)
        ax1.plot(vel_mid_low, xi * xi_scale, c='k', linewidth=0.3, alpha=0.1)

for xi in xi_unmask_all_low:
    ax1.plot(vel_mid_low, xi * xi_scale, linewidth=1.0, c='tab:orange', alpha=0.7)

ax1.errorbar(vel_mid_low, xi_mean_unmask_low * xi_scale, yerr=(xi_std_unmask_low / np.sqrt(nqso)) * xi_scale, lw=2.0, \
             marker='o', c='tab:orange', ecolor='tab:orange', capthick=2.0, capsize=2, mec='none', zorder=20)

ax1.text(2500, 0.85*ymax, r'$z < %0.2f$' % median_z, fontsize=xytick_size) #, linespacing=1.8)
ax1.set_xlabel(r'$\Delta v$ [km/s]', fontsize=xylabel_fontsize)
ax1.set_ylabel(r'$\xi(\Delta v)$', fontsize=xylabel_fontsize)
vel_doublet = 768.469
ax1.axvline(vel_doublet, color='green', linestyle='--', linewidth=2.0)
ax1.set_ylim([ymin, ymax])
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.tick_params(top=True, which='both', labelsize=xytick_size)

for i in range(nqso):
    for xi in xi_noise_unmask_high[i]:  # plotting all 500 realizations of the noise 2PCF (not masked)
        ax2.plot(vel_mid_high, xi * xi_scale, c='k', linewidth=0.3, alpha=0.1)

for xi in xi_unmask_all_high:
    ax2.plot(vel_mid, xi * xi_scale, linewidth=1.0, c='tab:orange', alpha=0.7)

ax2.errorbar(vel_mid, xi_mean_unmask_high * xi_scale, yerr=(xi_std_unmask_high / np.sqrt(nqso)) * xi_scale, lw=2.0, \
             marker='o', c='tab:orange', ecolor='tab:orange', capthick=2.0, capsize=2, mec='none', zorder=20)

ax2.text(2500, 0.85*ymax, r'$z \geq %0.2f$' % median_z, fontsize=xytick_size) #, linespacing=1.8)
ax2.set_xlabel(r'$\Delta v$ [km/s]', fontsize=xylabel_fontsize)
ax2.axvline(vel_doublet, color='green', linestyle='--', linewidth=2.0)
ax2.set_ylim([ymin, ymax])
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax2.tick_params(top=True, which='both', labelsize=xytick_size)

for i in range(nqso):
    for xi in xi_noise_unmask_all[i]:  # plotting all 500 realizations of the noise 2PCF (not masked)
        ax3.plot(vel_mid, xi * xi_scale, c='k', linewidth=0.3, alpha=0.1)

for xi in xi_unmask_all:
    ax3.plot(vel_mid, xi * xi_scale, linewidth=1.0, c='tab:orange', alpha=0.7)

ax3.errorbar(vel_mid, xi_mean_unmask_high * xi_scale, yerr=(xi_std_unmask / np.sqrt(nqso)) * xi_scale, lw=2.0, \
             marker='o', c='tab:orange', ecolor='tab:orange', capthick=2.0, capsize=2,  mec='none', zorder=20)

ax3.text(2500, 0.85*ymax, r'All $z$', fontsize=xytick_size) #, linespacing=1.8)
ax3.set_xlabel(r'$\Delta v$ [km/s]', fontsize=xylabel_fontsize)
ax3.axvline(vel_doublet, color='green', linestyle='--', linewidth=2.0)
ax3.set_ylim([ymin, ymax])
ax3.xaxis.set_minor_locator(AutoMinorLocator())
ax3.yaxis.set_minor_locator(AutoMinorLocator())
ax3.tick_params(top=True, which='both', labelsize=xytick_size)

plt.savefig(savefig_unmasked)
plt.show()
plt.close()

####### plot masked CF #######
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(14, 5), sharey=True)
fig.subplots_adjust(left=0.12, bottom=0.15, right=0.98, top=0.93, wspace=0, hspace=0.)

for i in range(nqso):
    for xi in xi_noise_mask_low[i]:  # plotting all 500 realizations of the noise 2PCF (not masked)
        ax1.plot(vel_mid_low, xi * xi_scale, c='k', linewidth=0.3, alpha=0.1)

for xi in xi_mask_all_low:
    ax1.plot(vel_mid_low, xi * xi_scale, linewidth=1.0, c='tab:orange', alpha=0.7)

ax1.errorbar(vel_mid_low, xi_mean_mask_low * xi_scale, yerr=(xi_std_mask_low / np.sqrt(nqso)) * xi_scale, lw=2.0, \
             marker='o', c='tab:orange', ecolor='tab:orange', capthick=2.0, capsize=2, mec='none', zorder=20)

ax1.text(2500, 0.85*ymax, r'$z < %0.2f$' % median_z, fontsize=xytick_size) #, linespacing=1.8)
ax1.set_xlabel(r'$\Delta v$ [km/s]', fontsize=xylabel_fontsize)
ax1.set_ylabel(r'$\xi(\Delta v)$', fontsize=xylabel_fontsize)
vel_doublet = 768.469
ax1.axvline(vel_doublet, color='green', linestyle='--', linewidth=2.0)
ax1.set_ylim([ymin, ymax])
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.tick_params(top=True, which='both', labelsize=xytick_size)

for i in range(nqso):
    for xi in xi_noise_mask_high[i]:  # plotting all 500 realizations of the noise 2PCF (not masked)
        ax2.plot(vel_mid_high, xi * xi_scale, c='k', linewidth=0.3, alpha=0.1)

for xi in xi_mask_all_high:
    ax2.plot(vel_mid, xi * xi_scale, linewidth=1.0, c='tab:orange', alpha=0.7)

ax2.errorbar(vel_mid, xi_mean_mask_high * xi_scale, yerr=(xi_std_mask_high / np.sqrt(nqso)) * xi_scale, lw=2.0, \
             marker='o', c='tab:orange', ecolor='tab:orange', capthick=2.0, capsize=2, mec='none', zorder=20)

ax2.text(2500, 0.85*ymax, r'$z \geq %0.2f$' % median_z, fontsize=xytick_size) #, linespacing=1.8)
ax2.set_xlabel(r'$\Delta v$ [km/s]', fontsize=xylabel_fontsize)
ax2.axvline(vel_doublet, color='green', linestyle='--', linewidth=2.0)
ax2.set_ylim([ymin, ymax])
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax2.tick_params(top=True, which='both', labelsize=xytick_size)

for i in range(nqso):
    for xi in xi_noise_mask_all[i]:  # plotting all 500 realizations of the noise 2PCF (not masked)
        ax3.plot(vel_mid, xi * xi_scale, c='k', linewidth=0.3, alpha=0.1)

for xi in xi_mask_all:
    ax3.plot(vel_mid, xi * xi_scale, linewidth=1.0, c='tab:orange', alpha=0.7)

ax3.errorbar(vel_mid, xi_mean_mask_high * xi_scale, yerr=(xi_std_mask / np.sqrt(nqso)) * xi_scale, lw=2.0, \
             marker='o', c='tab:orange', ecolor='tab:orange', capthick=2.0, capsize=2,  mec='none', zorder=20)

ax3.text(2500, 0.85*ymax, r'All $z$', fontsize=xytick_size) #, linespacing=1.8)
ax3.set_xlabel(r'$\Delta v$ [km/s]', fontsize=xylabel_fontsize)
ax3.axvline(vel_doublet, color='green', linestyle='--', linewidth=2.0)
ax3.set_ylim([ymin, ymax])
ax3.xaxis.set_minor_locator(AutoMinorLocator())
ax3.yaxis.set_minor_locator(AutoMinorLocator())
ax3.tick_params(top=True, which='both', labelsize=xytick_size)

plt.savefig(savefig_masked)
plt.show()
plt.close()