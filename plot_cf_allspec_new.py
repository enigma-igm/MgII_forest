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
parser.add_argument('--zbin', required=True, type=str, help="options: all, high, low")
parser.add_argument('--ivarweights', action='store_true', default=False, help="whether to use inverse variance weighting to compute the CF")
parser.add_argument('--subtractdf', action='store_true', default=False)
parser.add_argument('--xi_err_file', default=None, help="use the error bars from this file, if provided")
parser.add_argument('--savefig', default=None)
parser.add_argument('--save_cf_out', default=None)

args = parser.parse_args()
redshift_bin = args.zbin
subtract_mean_deltaf = args.subtractdf
ivar_weights = args.ivarweights
xi_err_file = args.xi_err_file
savefig = args.savefig
save_cf_out = args.save_cf_out

datapath = '/Users/suksientie/Research/MgII_forest/rebinned_spectra2/'

qso_namelist = ['J0411-0907', 'J0319-1008', 'J0410-0139', 'J0038-0653', 'J0313-1806', 'J0038-1527', 'J0252-0503', \
                'J1342+0928', 'J1007+2115', 'J1120+0641']

nqso = len(qso_namelist)
given_bins = ccf.custom_cf_bin4(dv1=80)

iqso_to_use = None #np.array([4,8]) #None #np.array([0,3,4,5,6,7,8,9])
if iqso_to_use is None:
    iqso_to_use = np.arange(0, nqso)

#######
#lowz_cgm_fit_gpm, highz_cgm_fit_gpm, allz_cgm_fit_gpm = ccf.init_cgm_fit_gpm()
lowz_cgm_fit_gpm, highz_cgm_fit_gpm, allz_cgm_fit_gpm = ccf.init_cgm_fit_gpm(datapath=datapath, do_not_apply_any_mask=True)

if redshift_bin == 'low':
    cgm_fit_gpm = lowz_cgm_fit_gpm
    median_z = 6.235

elif redshift_bin == 'high':
    cgm_fit_gpm = highz_cgm_fit_gpm
    median_z = 6.715

elif redshift_bin == 'all':
    cgm_fit_gpm = allz_cgm_fit_gpm
    median_z = 6.469

vel_mid, xi_mean_unmask, xi_mean_mask, xi_noise_unmask, xi_noise_mask, xi_unmask, xi_mask, w_masked, w_unmasked = \
    ccf.allspec(nqso, redshift_bin, cgm_fit_gpm, given_bins=given_bins, iqso_to_use=iqso_to_use, ivar_weights=ivar_weights, \
                subtract_mean_deltaf=subtract_mean_deltaf)

xi_std_unmask = np.std(xi_unmask, axis=0, ddof=1) # ddof=1 means std normalized to N-1
xi_std_mask = np.std(xi_mask, axis=0, ddof=1)

if save_cf_out is not None:
    hdulist = fits.HDUList()
    hdulist.append(fits.ImageHDU(data=vel_mid, name='vel_mid'))
    hdulist.append(fits.ImageHDU(data=xi_mean_unmask, name='xi_mean_unmask'))
    hdulist.append(fits.ImageHDU(data=xi_mean_mask, name='xi_mean_mask'))
    hdulist.append(fits.ImageHDU(data=xi_unmask, name='xi_unmask'))
    hdulist.append(fits.ImageHDU(data=xi_mask, name='xi_mask'))
    hdulist.append(fits.ImageHDU(data=w_masked, name='w_masked'))
    hdulist.append(fits.ImageHDU(data=w_unmasked, name='w_unmasked'))
    hdulist.writeto(save_cf_out, overwrite=True)

#######
vel_corr = vel_mid
vmin, vmax = 0.4*vel_corr.min(), 1.02*vel_corr.max()

factor = 1e5
ymin = factor*(-0.001)
ymax = factor*(0.002)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharex=True, sharey=True)
fig.subplots_adjust(left=0.09, bottom=0.15, right=0.98, top=0.88, wspace=0, hspace=0.)

#######
for i in range(len(xi_unmask)):
    xi = xi_unmask[i]
    ax1.plot(vel_mid, xi * factor, 'o', mec='none', ms=8, alpha=qso_alpha, linewidth=1.0, label=qso_namelist[iqso_to_use[i]])

yerr = (xi_std_unmask / np.sqrt(nqso)) * factor
#neff = ccf.compute_neff(w_unmasked)
#std_weighted = np.sqrt(ccf.weighted_var(xi_unmask, w_unmasked))
#yerr = std_weighted/np.sqrt(neff) * factor

ax1.errorbar(vel_mid, xi_mean_unmask * factor, yerr=yerr, lw=2.5, \
             fmt='o', c='black', ecolor='black', ms=8, capthick=2.5, capsize=2,  mec='none', zorder=20, label='all QSOs')
ax1.axhline(0, c='k', ls=":", lw=2.0)

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
ax1.annotate('before masking CGM', xy=(1800, -85), xytext=(1800, -85), textcoords='data', xycoords='data', annotation_clip=False, fontsize=legend_fontsize+5, bbox=bbox)

# Create upper axis in cMpc
cosmo = FlatLambdaCDM(H0=100.0 * 0.6704, Om0=0.3192, Ob0=0.04964) # Nyx values
z = median_z
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

#######
#ax2.fill_between(vel_mid, p16_mask, p84_mask, color='k', alpha=0.2, ec=None)
#ax2.fill_between(vel_mid, (xi_noise_mask_mean-xi_noise_mask_std)*factor, (xi_noise_mask_mean+xi_noise_mask_std)*factor, color='k', alpha=0.2, ec=None)

#for iqso, xi in enumerate(xi_mask):
#    ax2.plot(vel_mid, xi * factor, alpha=qso_alpha, linewidth=1.0, label=qso_namelist[iqso])
for i in range(len(xi_mask)):
    xi = xi_mask[i]
    ax2.plot(vel_mid, xi * factor, 'o', mec='none', ms=8, alpha=qso_alpha, linewidth=1.0, label=qso_namelist[iqso_to_use[i]])

if xi_err_file is None:
    yerr = (xi_std_mask / np.sqrt(nqso)) * factor
else:
    print("Using error from covariance matrix")
    yerr = np.load(xi_err_file) * factor

ax2.errorbar(vel_mid, xi_mean_mask * factor, yerr=yerr, lw=2.5, \
             fmt='o', c='black', ecolor='black', ms=8, capthick=2.5, capsize=2, mec='none', zorder=20)#, label='all QSOs')
ax2.axhline(0, c='k', ls=":", lw=2.0)
ax2.set_xlabel(r'$\Delta v$ (km/s)', fontsize=xylabel_fontsize)
ax2.tick_params(which="both", right=True, labelsize=xytick_size)

vel_doublet = 768.469
ax2.axvline(vel_doublet, color='black', linestyle='--', linewidth=2.0)
ax2.minorticks_on()
ax2.set_xlim((vmin, vmax))
ax2.set_ylim((ymin, ymax))
ax2.legend(loc=1, fontsize=legend_fontsize-1)


ax2.annotate('MgII doublet', xy=(1030, 0.85 * ymax), xytext=(1030, 0.85* ymax), fontsize=xytick_size, color='black')
ax2.annotate('separation', xy=(1070, 0.75 * ymax), xytext=(1070, 0.75 * ymax), fontsize=xytick_size, color='black')
ax2.annotate('', xy=(780, 0.88 * ymax), xytext=(1010, 0.88* ymax),
                fontsize=16, arrowprops={'arrowstyle': '-|>', 'lw': 4, 'color': 'black'}, va='center', color='magenta')

bbox = dict(boxstyle="round", fc="0.9") # fc is shading of the box, sth like alpha
ax2.annotate('after masking CGM', xy=(1885, -85), xytext=(1885, -85), textcoords='data', xycoords='data', annotation_clip=False, fontsize=legend_fontsize+5, bbox=bbox)

# Create upper axis in cMpc
cosmo = FlatLambdaCDM(H0=100.0 * 0.6704, Om0=0.3192, Ob0=0.04964) # Nyx values
z = median_z
Hz = (cosmo.H(z))
a = 1.0 / (1.0 + z)
rmin = (vmin*u.km/u.s/a/Hz).to('Mpc').value
rmax = (vmax*u.km/u.s/a/Hz).to('Mpc').value
# Make the new upper x-axes
atwin = ax2.twiny()
atwin.set_xlabel('R (cMpc)', fontsize=xylabel_fontsize)#, labelpad=8)
atwin.xaxis.tick_top()

atwin.axis([rmin, rmax, ymin, ymax])
atwin.tick_params(top=True)
atwin.xaxis.set_minor_locator(AutoMinorLocator())
atwin.tick_params(axis="x", labelsize=xytick_size)

if savefig != None:
    plt.savefig(savefig)
plt.show()
plt.close()



