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
savefig = 'all_spectra.png'
fig.subplots_adjust(left=0.1, bottom=0.07, right=0.98, top=0.93, wspace=0, hspace=0.)
color_ls = ['r', 'g', 'b', 'orange']

wave_min, wave_max = 19500, 24000
zmin, zmax = wave_min / 2800 - 1, wave_max / 2800 - 1
ymin, ymax = 0., 1.9

all_wave = []
all_cont_flux = []
all_norm_std = []
all_mask = []
all_xmask = []

for i, fitsfile in enumerate(fitsfile_list):
    #wave, flux, ivar, std, mask, cont_flux, norm_std = rdx_utils.continuum_normalize(fitsfile, qso_zlist[i])
    wave, flux, ivar, mask, std = mutils.extract_data(fitsfile)
    cont_flux = flux
    norm_std = std

    x_mask = wave[mask] <= (2800 * (1 + qso_zlist[i])) # redshift cutoff

    if i == 0: plot_ax = ax1
    elif i == 1: plot_ax = ax2
    elif i == 2: plot_ax = ax3
    elif i == 3: plot_ax = ax4

    plot_ax.plot(wave, cont_flux, color=color_ls[i], alpha=0.4, drawstyle='steps-mid')
    label = fitsfile.split('/')[-1].split('_')[0] + ' (z=%0.2f)' % qso_zlist[i]
    plot_ax.plot(wave[mask][x_mask], cont_flux[mask][x_mask], color=color_ls[i], label=label, drawstyle='steps-mid')

    plot_ax.plot(wave, norm_std, color='k', alpha=0.4, drawstyle='steps-mid')
    plot_ax.plot(wave[mask][x_mask], norm_std[mask][x_mask], color='k', alpha=0.8, drawstyle='steps-mid')
    plot_ax.xaxis.set_minor_locator(AutoMinorLocator())
    plot_ax.yaxis.set_minor_locator(AutoMinorLocator())
    plot_ax.tick_params(top=True, which='both', labelsize=xytick_size)
    plot_ax.set_xlim([wave_min, wave_max])
    #plot_ax.set_ylim([ymin, ymax])

    #plot_ax.set_ylabel(r'$F_{\mathrm{norm}}$', fontsize=xylabel_fontsize)
    plot_ax.set_ylabel('Un-normalized flux', fontsize=xylabel_fontsize-5)
    plot_ax.legend(loc=1, fontsize=legend_fontsize)
    if i == 3:
        plot_ax.set_xlabel('obs wavelength (A)', fontsize=xylabel_fontsize)

atwin = ax1.twiny()
atwin.set_xlabel('absorber redshift', fontsize=xylabel_fontsize)
atwin.axis([zmin, zmax, ymin, ymax])
atwin.tick_params(top=True, axis="x", labelsize=xytick_size)
atwin.xaxis.set_minor_locator(AutoMinorLocator())

#plt.savefig(savefig)
plt.show()
