import numpy as np
from astropy.cosmology import FlatLambdaCDM
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from astropy.io import fits
import sys
sys.path.append('/Users/suksientie/Research/data_redux')
import mutils
import pdb

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

### Some controls
plot_normalized = True
redshift_bin = 'all'

fitsfile_list = ['/Users/suksientie/Research/data_redux/wavegrid_vel/J0313-1806/vel1234_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/wavegrid_vel/J1342+0928/vel123_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/wavegrid_vel/J0252-0503/vel12_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/wavegrid_vel/J0038-1527/vel1_tellcorr.fits']

qso_namelist =['J0313-1806', 'J1342+0928', 'J0252-0503', 'J0038-1527']
qso_zlist = [7.642, 7.541, 7.001, 7.034]
exclude_restwave = 1216 - 1185 # excluding proximity zones; see mutils.qso_exclude_proximity_zone

fig, (ax1, ax2, ax3, ax4) = plt.subplots(len(fitsfile_list), figsize=(16, 10), sharex=True)
fig.subplots_adjust(left=0.1, bottom=0.07, right=0.96, top=0.93, wspace=0, hspace=0.)
color_ls = ['k', 'k', 'k', 'k'] # ['r', 'g', 'b', 'orange']

if plot_normalized:
    ymin, ymax = -0.05, 1.9
else:
    ymin, ymax = -0.05, 0.55

wave_min, wave_max = 19500, 24100
zmin, zmax = wave_min / 2800 - 1, wave_max / 2800 - 1
everyn_break_list = [20, 20, 20, 20]

for i, fitsfile in enumerate(fitsfile_list):
    raw_data_out, masked_data_out, all_masks_out = mutils.init_onespec(i, redshift_bin)

    wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
    #good_wave, good_flux, good_ivar, good_std, good_vel_data, norm_good_flux, norm_good_std = masked_data_out
    strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, master_mask = all_masks_out
    all_masks = mask * redshift_mask * pz_mask * zbin_mask
    median_snr = np.nanmedian(flux / std)

    print(1 - np.sum(all_masks)/len(all_masks))
    print(median_snr)

    if i == 0: plot_ax = ax1
    elif i == 1: plot_ax = ax2
    elif i == 2: plot_ax = ax3
    elif i == 3: plot_ax = ax4

    if plot_normalized:
        plot_ax.plot(wave, flux/fluxfit, c=color_ls[i], drawstyle='steps-mid', label=qso_namelist[i] + ' (z=%0.2f)' % qso_zlist[i])
        plot_ax.plot(wave, std/fluxfit, c='k', alpha=0.5, drawstyle='steps-mid')
        plot_ax.set_ylabel(r'$F_{\mathrm{norm}}$', fontsize=xylabel_fontsize)
        ind_masked = np.where(all_masks == False)[0]
        for j in range(len(ind_masked)): # bad way to plot masked pixels
            plot_ax.axvline(wave[ind_masked[j]], color='k', alpha=0.2, lw=2)

    else:
        plot_ax.plot(wave, flux, c=color_ls[i], alpha=1.0, drawstyle='steps-mid', label=qso_namelist[i] + ' (z=%0.2f)' % qso_zlist[i])
        plot_ax.plot(wave, tell*(ymax-0.02), alpha=0.3, drawstyle='steps-mid') # telluric
        plot_ax.plot(wave, fluxfit, c='r', lw=1.0, drawstyle='steps-mid')
        plot_ax.plot(wave, std, c='k', alpha=0.5, drawstyle='steps-mid')
        plot_ax.set_ylabel('Flux', fontsize=xylabel_fontsize)

    plot_ax.xaxis.set_minor_locator(AutoMinorLocator())
    plot_ax.yaxis.set_minor_locator(AutoMinorLocator())
    plot_ax.tick_params(top=True, which='both', labelsize=xytick_size)
    plot_ax.axvline((qso_zlist[i] + 1) * 2800, ls='--', c=color_ls[i], lw=2)
    #plot_ax.axvline(obs_wave_max, ls='--', c='k', lw=2)
    plot_ax.set_xlim([wave_min, wave_max])

    plot_ax.set_ylim([ymin, ymax])
    plot_ax.legend(fontsize=legend_fontsize, loc='upper center')
    if i == 3:
        plot_ax.set_xlabel(r'obs wavelength ($\mathrm{{\AA}}$)', fontsize=xylabel_fontsize)

atwin = ax1.twiny()
atwin.set_xlabel('redshift', fontsize=xylabel_fontsize)
atwin.axis([zmin, zmax, ymin, ymax])
atwin.tick_params(top=True, axis="x", labelsize=xytick_size)
atwin.xaxis.set_minor_locator(AutoMinorLocator())

plt.tight_layout()
plt.savefig("combined_coadds.png")
plt.show()
