import numpy as np
from astropy.cosmology import FlatLambdaCDM
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from astropy.io import fits
import sys
sys.path.append('/Users/suksientie/Research/data_redux')
from scripts import rdx_utils
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

fitsfile_list = ['/Users/suksientie/Research/data_redux/mgii_stack_fits/J0313-1806_stacked_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/mgii_stack_fits/J1342+0928_stacked_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/mgii_stack_fits/J0252-0503_stacked_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/2010_done/Redux/J0038-1527_201024_done/J0038-1527_coadd_tellcorr.fits']
qso_zlist = [7.6, 7.54, 7.0, 7.0]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(len(fitsfile_list), figsize=(16, 10), sharex=True)
fig.subplots_adjust(left=0.1, bottom=0.07, right=0.98, top=0.93, wspace=0, hspace=0.)
color_ls = ['r', 'g', 'b', 'orange']
savefig = 'all_bspline_fit.png'

plot_normalized = False
if plot_normalized:
    ymin, ymax = -0.05, 1.9
else:
    ymin, ymax = -0.1, 0.55

wave_min, wave_max = 19500, 24000
zmin, zmax = wave_min / 2800 - 1, wave_max / 2800 - 1
everyn_break_list = [20, 20, 20, 20]

all_wave = []
all_cont_flux = []
all_norm_std = []
all_mask = []
all_xmask = []

for i, fitsfile in enumerate(fitsfile_list):
    wave, flux, ivar, mask, std = mutils.extract_data(fitsfile)
    qso_name = fitsfile.split('/')[-1].split('_')[0]

    if qso_name == 'J0313-1806':
        wave, flux, ivar, mask, std, out_gpm = mutils.custom_mask_J0313()

    elif qso_name == 'J1342+0928':
        wave, flux, ivar, mask, std, out_gpm = mutils.custom_mask_J1342()

    elif qso_name == 'J0252-0503':
        pass

    elif qso_name == 'J0038-1527':
        wave, flux, ivar, mask, std, out_gpm = mutils.custom_mask_J0038()

    fluxfit, outmask = mutils.continuum_normalize(wave, flux, ivar, mask, std, everyn_break_list[i])
    print(len(fluxfit), np.sum(outmask))

    good_wave, good_flux, good_std = wave[outmask], flux[outmask], std[outmask]
    norm_good_flux = good_flux/fluxfit
    norm_good_std = good_std/fluxfit

    if i == 0: plot_ax = ax1
    elif i == 1: plot_ax = ax2
    elif i == 2: plot_ax = ax3
    elif i == 3: plot_ax = ax4

    if plot_normalized:
        plot_ax.plot(good_wave, norm_good_flux, c=color_ls[i], drawstyle='steps-mid', label=qso_name + ' (z=%0.2f)' % qso_zlist[i])
        plot_ax.plot(good_wave, norm_good_std, c='k', alpha=0.7, drawstyle='steps-mid')
        plot_ax.set_ylabel(r'$F_{\mathrm{norm}}$', fontsize=xylabel_fontsize)
        #if i == 0:
        #    plot_ax.set_ylim([0, 2.0])
    else:
        plot_ax.plot(wave, flux, c=color_ls[i], alpha=0.3, drawstyle='steps-mid')
        plot_ax.plot(good_wave, good_flux, c=color_ls[i], drawstyle='steps-mid', label=qso_name + ' (z=%0.2f)' % qso_zlist[i])
        plot_ax.plot(good_wave, fluxfit, c='k', lw=2.5, drawstyle='steps-mid', label='bspline with everyn=%d' % everyn_break_list[i])
        plot_ax.plot(good_wave, good_std, c='k', alpha=0.7, drawstyle='steps-mid')
        plot_ax.set_ylabel('Flux', fontsize=xylabel_fontsize)
        #if i == 0:
        #    plot_ax.set_ylim([-0.1, 0.55])

    plot_ax.axvline((qso_zlist[i] + 1) * 2800, ls='--', lw=2)
    plot_ax.xaxis.set_minor_locator(AutoMinorLocator())
    plot_ax.yaxis.set_minor_locator(AutoMinorLocator())
    plot_ax.tick_params(top=True, which='both', labelsize=xytick_size)
    plot_ax.set_xlim([wave_min, wave_max])
    plot_ax.set_ylim([ymin, ymax])
    #plot_ax.set_ylabel(r'$F_{\mathrm{norm}}$', fontsize=xylabel_fontsize)
    plot_ax.legend(fontsize=legend_fontsize)
    if i == 3:
        plot_ax.set_xlabel('obs wavelength (A)', fontsize=xylabel_fontsize)

atwin = ax1.twiny()
atwin.set_xlabel('absorber redshift', fontsize=xylabel_fontsize)
atwin.axis([zmin, zmax, ymin, ymax])
atwin.tick_params(top=True, axis="x", labelsize=xytick_size)
atwin.xaxis.set_minor_locator(AutoMinorLocator())

plt.tight_layout()
plt.savefig(savefig)
#plt.show()
