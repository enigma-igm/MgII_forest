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
from enigma.reion_forest import utils as reion_utilsf
from enigma.reion_forest.compute_model_grid import read_model_grid
import argparse
import mutils

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
qso_alpha = 0.6

parser = argparse.ArgumentParser()
parser.add_argument('--zbin', type=str, help="options: all, high, low")
parser.add_argument('--ivarweights', action='store_true', default=False)
args = parser.parse_args()
redshift_bin = args.zbin

"""
qso_namelist = ['J0313-1806', 'J1342+0928', 'J0252-0503', 'J0038-1527', 'unpublished \n new QSO']
median_z = 6.554
seed_list=[None, None, None, None, None]
given_bins = ccf.custom_cf_bin4()
"""

qso_namelist = ['J0411-0907', 'J0319-1008', 'newqso1', 'newqso2', 'J0313-1806', 'J0038-1527', 'J0252-0503', 'J1342+0928']
nqso = len(qso_namelist)
median_z = 6.50
#seed_list = [None] * nqso
given_bins = ccf.custom_cf_bin4(dv1=80)
savefig = 'paper_plots/8qso/cf_%sz_%dqso_ivar.pdf' % (redshift_bin, nqso)
#savefig = None
iqso_to_use = None #np.array([0])
ivar_weights = args.ivarweights

if iqso_to_use is None:
    iqso_to_use = np.arange(0, nqso)

#######
lowz_cgm_fit_gpm, highz_cgm_fit_gpm, allz_cgm_fit_gpm = ccf.init_cgm_fit_gpm()

if redshift_bin == 'low':
    cgm_fit_gpm = lowz_cgm_fit_gpm
elif redshift_bin == 'high':
    cgm_fit_gpm = highz_cgm_fit_gpm
elif redshift_bin == 'all':
    cgm_fit_gpm = allz_cgm_fit_gpm

vel_mid, xi_mean_unmask, xi_mean_mask, xi_noise_unmask, xi_noise_mask, xi_unmask, xi_mask = \
    ccf.allspec(nqso, redshift_bin, cgm_fit_gpm, given_bins=given_bins, iqso_to_use=iqso_to_use, ivar_weights=ivar_weights)

xi_std_unmask = np.std(xi_unmask, axis=0, ddof=1) # ddof=1 means std normalized to N-1
xi_std_mask = np.std(xi_mask, axis=0, ddof=1)

#######
vel_corr = vel_mid

vmin, vmax = 0.4*vel_corr.min(), 1.02*vel_corr.max()
factor = 1e5
ymin = factor*(-0.001)
ymax = factor*(0.002)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharex=True, sharey=True)
fig.subplots_adjust(left=0.12, bottom=0.15, right=0.98, top=0.93, wspace=0, hspace=0.)

#######
"""
nqso, nreal, nvel = xi_noise_mask.shape

xi_noise_unmask = np.reshape(xi_noise_unmask, (nqso*nreal, nvel))
xi_noise_unmask_mean = np.mean(xi_noise_unmask, axis=0)
xi_noise_unmask_std = np.std(xi_noise_unmask, axis=0)
#p16_unmask = np.percentile(xi_noise_unmask*factor, 16, axis=0) # 1-sigma
#p84_unmask = np.percentile(xi_noise_unmask*factor, 84, axis=0)

xi_noise_mask = np.reshape(xi_noise_mask, (nqso*nreal, nvel))
xi_noise_mask_mean = np.mean(xi_noise_mask, axis=0)
xi_noise_mask_std = np.std(xi_noise_mask, axis=0)
#p16_mask = np.percentile(xi_noise_mask*factor, 16, axis=0) # 1-sigma
#p84_mask = np.percentile(xi_noise_mask*factor, 84, axis=0)
"""
#######
#ax1.fill_between(vel_mid, p16_unmask, p84_unmask, color='k', alpha=0.2, ec=None)
#ax1.fill_between(vel_mid, (xi_noise_unmask_mean-xi_noise_unmask_std)*factor, (xi_noise_unmask_mean+xi_noise_unmask_std)*factor, color='k', alpha=0.2, ec=None)

#for iqso, xi in enumerate(xi_unmask):
#    ax1.plot(vel_mid, xi * factor, alpha=qso_alpha, linewidth=1.0, label=qso_namelist[iqso])
for i in range(len(xi_unmask)):
    xi = xi_unmask[i]
    ax1.plot(vel_mid, xi * factor, alpha=qso_alpha, linewidth=1.0, label=qso_namelist[iqso_to_use[i]])

yerr = (xi_std_unmask / np.sqrt(nqso)) * factor
#yerr = xi_noise_unmask_std * factor
ax1.errorbar(vel_mid, xi_mean_unmask * factor, yerr=yerr, lw=2.0, \
             fmt='o-', c='black', ecolor='black', capthick=2.0, capsize=2,  mec='none', zorder=20, label='all QSOs')

#ax1.set_xticks(range(500, 4000, 500))
ax1.set_yticks(range(int(ymin), int(ymax) + 50, 50))
ax1.set_xlabel(r'$\Delta v$ (km/s)', fontsize=xylabel_fontsize)
ax1.set_ylabel(r'$\xi(\Delta v)\times 10^5$', fontsize=xylabel_fontsize, labelpad=-4)
ax1.tick_params(which="both", labelsize=xytick_size)

vel_doublet = 768.469
ax1.axvline(vel_doublet, color='black', linestyle='--', linewidth=2.0)
ax1.minorticks_on()
ax1.set_xlim((vmin, vmax))
ax1.set_ylim((ymin, ymax))

ax1.annotate('MgII doublet', xy=(1030, 0.85 * ymax), xytext=(1030, 0.85* ymax), fontsize=xytick_size, color='black')
ax1.annotate('separation', xy=(1070, 0.75 * ymax), xytext=(1070, 0.75 * ymax), fontsize=xytick_size, color='black')
ax1.annotate('', xy=(780, 0.88 * ymax), xytext=(1010, 0.88* ymax),
                fontsize=16, arrowprops={'arrowstyle': '-|>', 'lw': 4, 'color': 'black'}, va='center', color='magenta')

bbox = dict(boxstyle="round", fc="0.9") # fc is shading of the box, sth like alpha
ax1.annotate('before masking CGM', xy=(1900, -75), xytext=(2000, -75), textcoords='data', xycoords='data', annotation_clip=False, fontsize=legend_fontsize, bbox=bbox)

#######
#ax2.fill_between(vel_mid, p16_mask, p84_mask, color='k', alpha=0.2, ec=None)
#ax2.fill_between(vel_mid, (xi_noise_mask_mean-xi_noise_mask_std)*factor, (xi_noise_mask_mean+xi_noise_mask_std)*factor, color='k', alpha=0.2, ec=None)

#for iqso, xi in enumerate(xi_mask):
#    ax2.plot(vel_mid, xi * factor, alpha=qso_alpha, linewidth=1.0, label=qso_namelist[iqso])
for i in range(len(xi_mask)):
    xi = xi_mask[i]
    ax2.plot(vel_mid, xi * factor, alpha=qso_alpha, linewidth=1.0, label=qso_namelist[iqso_to_use[i]])

yerr = (xi_std_mask / np.sqrt(nqso)) * factor
#yerr_cov = np.load('mcmc/8qso/xi_err.npy') * factor
ax2.errorbar(vel_mid, xi_mean_mask * factor, yerr=yerr, lw=2.0, \
             fmt='o-', c='black', ecolor='black', capthick=2.0, capsize=2,  mec='none', zorder=20, label='all QSOs')
#ax2.errorbar(vel_mid, xi_mean_mask * factor, yerr=yerr_cov, lw=2.0, \
#             fmt='o-', c='black', ecolor='red', capthick=2.0, capsize=2,  mec='none', zorder=20)
#np.save('plot_cf_allspec_new_xi_std_mask.npy', xi_std_mask)

ax2.set_xlabel(r'$\Delta v$ (km/s)', fontsize=xylabel_fontsize)
ax2.tick_params(which="both", right=True, labelsize=xytick_size)

vel_doublet = 768.469
ax2.axvline(vel_doublet, color='black', linestyle='--', linewidth=2.0)
ax2.minorticks_on()
ax2.set_xlim((vmin, vmax))
ax2.set_ylim((ymin, ymax))
ax2.legend(loc=1, fontsize=legend_fontsize)

ax2.annotate('MgII doublet', xy=(1030, 0.85 * ymax), xytext=(1030, 0.85* ymax), fontsize=xytick_size, color='black')
ax2.annotate('separation', xy=(1070, 0.75 * ymax), xytext=(1070, 0.75 * ymax), fontsize=xytick_size, color='black')
ax2.annotate('', xy=(780, 0.88 * ymax), xytext=(1010, 0.88* ymax),
                fontsize=16, arrowprops={'arrowstyle': '-|>', 'lw': 4, 'color': 'black'}, va='center', color='magenta')

bbox = dict(boxstyle="round", fc="0.9") # fc is shading of the box, sth like alpha
ax2.annotate('after masking CGM', xy=(1900, -75), xytext=(2000, -75), textcoords='data', xycoords='data', annotation_clip=False, fontsize=legend_fontsize, bbox=bbox)

if savefig != None:
    plt.savefig(savefig)
plt.show()
plt.close()



