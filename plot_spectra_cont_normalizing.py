import numpy as np
from astropy.cosmology import FlatLambdaCDM
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from astropy.io import fits
import sys
sys.path.append('/Users/suksientie/Research/data_redux')
import mutils
import argparse
import mask_cgm_pdf

### Some controls
parser = argparse.ArgumentParser()
parser.add_argument('--plotnorm', default=False, action='store_true')
parser.add_argument('--zbin', type=str, help="options: all, high, low")
parser.add_argument('--nqso', type=int, help="if 4, then exclude J0038-0653")
parser.add_argument('--savefig', type=str, default=None)
args = parser.parse_args()

plot_normalized = args.plotnorm
redshift_bin = args.zbin
nqso_to_plot = args.nqso
savefig = args.savefig

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

qso_namelist =['J0313-1806', 'J1342+0928', 'J0252-0503', 'J0038-1527', 'J0038-0653']
qso_zlist = [7.642, 7.541, 7.001, 7.034, 7.1]
exclude_restwave = 1216 - 1185 # excluding proximity zones; see mutils.qso_exclude_proximity_zone
everyn_break_list = [20, 20, 20, 20, 20]

fig, plot_ax_all = plt.subplots(nqso_to_plot, figsize=(16, 11), sharex=True, sharey=True)
fig.subplots_adjust(left=0.1, bottom=0.07, right=0.96, top=0.93, wspace=0, hspace=0.)
color_ls = ['k', 'k', 'k', 'k', 'k']

if plot_normalized:
    ymin, ymax = -0.05, 2.1
else:
    ymin, ymax = -0.05, 0.65

wave_min, wave_max = 19500, 24100
zmin, zmax = wave_min / 2800 - 1, wave_max / 2800 - 1

# CGM masks
good_vel_data_all, good_wave_all, norm_good_flux_all, norm_good_std_all, norm_good_ivar_all, noise_all = \
    mask_cgm_pdf.init(redshift_bin=redshift_bin, do_not_apply_any_mask=True)
mgii_tot_all = mask_cgm_pdf.chi_pdf(good_vel_data_all, norm_good_flux_all, norm_good_ivar_all, noise_all, plot=False, savefig=None)

"""
# plot telluric on upper panel
raw_data_out, masked_data_out, all_masks_out = mutils.init_onespec(0, redshift_bin)
wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
plot_ax = plot_ax_all[0]
plot_ax.plot(wave, tell * 0.2, alpha=0.3, drawstyle='steps-mid')  # telluric
#plot_ax.xaxis.set_minor_locator(AutoMinorLocator())
#plot_ax.yaxis.set_minor_locator(AutoMinorLocator())
plot_ax.tick_params(top=True, right=True, which='both', labelsize=xytick_size) 
plot_ax.set_xlim([wave_min, wave_max])
plot_ax.set_ylim([0, 0.25])
"""

for i in range(nqso_to_plot):
    raw_data_out, masked_data_out, all_masks_out = mutils.init_onespec(i, redshift_bin)
    wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
    #good_wave, good_flux, good_ivar, good_std, good_vel_data, norm_good_flux, norm_good_std = masked_data_out
    strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, master_mask = all_masks_out

    pix, tell_gpm = mutils.mask_telluric_lines(tell)
    #all_masks = mask * redshift_mask * pz_mask * zbin_mask * tell_gpm
    all_masks = mask * redshift_mask * pz_mask * zbin_mask
    print("masked fraction", 1 - np.sum(all_masks) / len(all_masks))

    median_snr = np.nanmedian((flux / std)[all_masks])
    print("median snr", median_snr)

    mgii_tot = mgii_tot_all[i]
    fs_mask = mgii_tot.fit_gpm[0]
    all_masks = all_masks * fs_mask

    plot_ax = plot_ax_all[i]

    if plot_normalized:
        plot_ax.plot(wave, flux/fluxfit, c=color_ls[i], drawstyle='steps-mid', label=qso_namelist[i] + ' (z=%0.2f)' % qso_zlist[i])
        plot_ax.plot(wave, std/fluxfit, c='k', alpha=0.5, drawstyle='steps-mid')
        plot_ax.plot(wave, tell*(ymax-0.05), alpha=0.3, drawstyle='steps-mid') # telluric
        ind_masked = np.where(all_masks == False)[0]
        for j in range(len(ind_masked)): # bad way to plot masked pixels
            plot_ax.axvline(wave[ind_masked[j]], color='k', alpha=0.1, lw=2)

    else:
        plot_ax.plot(wave, flux, c=color_ls[i], alpha=1.0, drawstyle='steps-mid', label=qso_namelist[i] + ' (z=%0.2f)' % qso_zlist[i])
        plot_ax.plot(wave, tell*(ymax-0.02), alpha=0.3, drawstyle='steps-mid') # telluric
        plot_ax.plot(wave, fluxfit, c='r', lw=1.0, drawstyle='steps-mid')
        plot_ax.plot(wave, std, c='k', alpha=0.5, drawstyle='steps-mid')

    plot_ax.xaxis.set_minor_locator(AutoMinorLocator())
    plot_ax.yaxis.set_minor_locator(AutoMinorLocator())
    plot_ax.tick_params(top=True, right=True, which='both', labelsize=xytick_size)
    plot_ax.axvline((qso_zlist[i] + 1) * 2800, ls='--', c=color_ls[i], lw=2)
    plot_ax.set_xlim([wave_min, wave_max])
    plot_ax.set_ylim([ymin, ymax])
    plot_ax.legend(fontsize=legend_fontsize, bbox_to_anchor=(0., 0.8, 1., .102), loc='upper center')
    if i == (nqso_to_plot - 1):
        plot_ax.set_xlabel(r'obs wavelength ($\mathrm{{\AA}}$)', fontsize=xylabel_fontsize)

atwin = plot_ax_all[0].twiny()
atwin.set_xlabel('redshift', fontsize=xylabel_fontsize)
atwin.axis([zmin, zmax, ymin, ymax])
atwin.tick_params(top=True, axis="x", labelsize=xytick_size)
atwin.xaxis.set_minor_locator(AutoMinorLocator())

if plot_normalized:
    fig.text(0.04, 0.5, r'$F_{\mathrm{norm}}$', va='center', rotation='vertical', fontsize=xylabel_fontsize+5)
else:
    fig.text(0.04, 0.5, r'Flux $(10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\mathrm{{\AA}}^{-1})$', va='center', rotation='vertical', fontsize=xylabel_fontsize)

"""
if plot_normalized:
    plt.savefig("paper_plots/combined_coadds_norm.pdf")
else:
    plt.savefig("paper_plots/combined_coadds.pdf")
"""
if savefig is not None:
    plt.savefig(savefig)
plt.show()
