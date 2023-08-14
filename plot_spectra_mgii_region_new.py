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
import mutils
import mutils2 as m2

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
savefig = True

datapath = '/Users/suksientie/Research/MgII_forest/rebinned_spectra2/'
qso_namelist = ['J0411-0907', 'J0319-1008', 'J0410-0139', 'J0038-0653', 'J0313-1806', 'J0038-1527', 'J0252-0503', 'J1342+0928', 'J1007+2115', 'J1120+0641']
qso_zlist = [6.826, 6.8275, 7.0, 7.1, 7.642, 7.034, 7.001, 7.541, 7.515, 7.085]
#everyn_break_list = (np.ones(len(qso_namelist)) * 20).astype('int')
#exclude_restwave = 1216 - 1185
redshift_bin = 'all'

# CGM masks
good_vel_data_all, good_wave_all, norm_good_flux_all, norm_good_std_all, norm_good_ivar_all, noise_all, _, _ = \
    mask_cgm_pdf.init(redshift_bin=redshift_bin, do_not_apply_any_mask=True, datapath=datapath)
mgii_tot_all = mask_cgm_pdf.chi_pdf(good_vel_data_all, norm_good_flux_all, norm_good_ivar_all, noise_all, plot=False, savefig=None)

################################
iqso_to_plot = [4, 7, 8] # z > 7.5
nqso_to_plot = len(iqso_to_plot)

wavemin = 22000
dwave = 700

ymin, ymax = 0.75, 1.32
err_plot_scale = [0.72, 0.72, 0.68]

fig, ax_all = plt.subplots(nqso_to_plot, figsize=(16, 2.25*nqso_to_plot), sharex=True, sharey=True)
fig.subplots_adjust(left=0.065, bottom=0.12, right=0.96, top=0.89, wspace=0, hspace=0)

for ind, iqso in enumerate(iqso_to_plot):
    raw_data_out, masked_data_out, all_masks_out = mutils.init_onespec(iqso, redshift_bin, datapath=datapath)
    wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
    strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, telluric_mask, master_mask = all_masks_out

    #all_masks = mask * redshift_mask * pz_mask * zbin_mask * telluric_mask
    #print("masked fraction", 1 - np.sum(all_masks) / len(all_masks))
    #median_snr = np.nanmedian((flux / std)[all_masks])
    #print("median snr", median_snr)

    #import pdb
    #pdb.set_trace()
    mgii_tot = mgii_tot_all[iqso]
    fs_mask = mgii_tot.fit_gpm[0]
    #all_masks = all_masks * fs_mask
    all_masks = master_mask * fs_mask

    ax = ax_all[ind]
    #ax.annotate(qso_namelist[i], xy=(xmin + 100, ymax * 0.88), fontsize=18, bbox=dict(boxstyle='round', ec="k", fc="white"))
    wave_mask = (wave >= wavemin) * (wave <= (wavemin + dwave))
    ax.plot(wave[wave_mask], (flux / fluxfit)[wave_mask], c='k', drawstyle='steps-mid', label=qso_namelist[iqso])
    ax.plot(wave[wave_mask], (std / fluxfit)[wave_mask] + err_plot_scale[ind], c='r', alpha=0.4, drawstyle='steps-mid')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    if ind > 0:
        ax.tick_params(top=True, right=True, which='both', labelsize=xytick_size)
    else:
        ax.tick_params(right=True, which='both', labelsize=xytick_size)

    ax.set_ylabel(r'$F_{\mathrm{norm}}$', fontsize=xylabel_fontsize)
    ax.set_xlim([wavemin, wavemin + dwave])
    ax.set_ylim([ymin, ymax])
    ax.axhline(1, ls='--', c='k', alpha=0.7)
    ax.legend(fontsize=legend_fontsize, loc=2)
    if ind == (nqso_to_plot - 1):
        ax.set_xlabel(r'obs wavelength ($\mathrm{{\AA}}$)', fontsize=xylabel_fontsize)

    #ind_masked = np.where(master_mask[wave_mask] * strong_abs_gpm[wave_mask] == False)[0]
    #for j in range(len(ind_masked)):  # bad way to plot masked pixels
    #    ax.axvline(wave[wave_mask][ind_masked[j]], color='k', alpha=0.2, lw=5)
    ind_masked = np.where(all_masks[wave_mask] == False)[0]
    for j in range(len(ind_masked)):  # bad way to plot masked pixels
        ax.axvline(wave[wave_mask][ind_masked[j]], color='k', alpha=0.2, lw=5)

atwin = ax_all[0].twiny()
atwin.set_xlabel('redshift', fontsize=xylabel_fontsize)
zmin, zmax = wavemin / 2800 - 1, (wavemin + dwave) / 2800 - 1
atwin.axis([zmin, zmax, ymin, ymax])
atwin.tick_params(top=True, axis="x", labelsize=xytick_size)
atwin.xaxis.set_minor_locator(AutoMinorLocator())

if savefig:
    plt.savefig('paper_plots/10qso/spec_mg2forest-z7p5.pdf')
    plt.close()
if savefig is False:
    plt.show()

################################
iqso_to_plot = [0, 1, 5, 6, 9] # z < 7.5
nqso_to_plot = len(iqso_to_plot)

wavemin = 20600
dwave = 700

ymin, ymax = 0.75, 1.32
err_plot_scale = [0.72, 0.68, 0.75, 0.75, 0.72]

fig, ax_all = plt.subplots(nqso_to_plot, figsize=(16, 2.25*nqso_to_plot), sharex=True, sharey=True)
fig.subplots_adjust(left=0.065, bottom=0.12, right=0.96, top=0.89, wspace=0, hspace=0)

for ind, iqso in enumerate(iqso_to_plot):
    raw_data_out, masked_data_out, all_masks_out = mutils.init_onespec(iqso, redshift_bin, datapath=datapath)
    wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
    strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, telluric_mask, master_mask = all_masks_out

    #all_masks = mask * redshift_mask * pz_mask * zbin_mask * telluric_mask
    #print("masked fraction", 1 - np.sum(all_masks) / len(all_masks))
    #median_snr = np.nanmedian((flux / std)[all_masks])
    #print("median snr", median_snr)

    #import pdb
    #pdb.set_trace()
    mgii_tot = mgii_tot_all[iqso]
    fs_mask = mgii_tot.fit_gpm[0]
    #all_masks = all_masks * fs_mask
    all_masks = master_mask * fs_mask

    ax = ax_all[ind]
    #ax.annotate(qso_namelist[i], xy=(xmin + 100, ymax * 0.88), fontsize=18, bbox=dict(boxstyle='round', ec="k", fc="white"))
    wave_mask = (wave >= wavemin) * (wave <= (wavemin + dwave))
    ax.plot(wave[wave_mask], (flux / fluxfit)[wave_mask], c='k', drawstyle='steps-mid', label=qso_namelist[iqso])
    ax.plot(wave[wave_mask], (std / fluxfit)[wave_mask] + err_plot_scale[ind], c='r', alpha=0.4, drawstyle='steps-mid')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    if ind > 0:
        ax.tick_params(top=True, right=True, which='both', labelsize=xytick_size)
    else:
        ax.tick_params(right=True, which='both', labelsize=xytick_size)

    ax.set_ylabel(r'$F_{\mathrm{norm}}$', fontsize=xylabel_fontsize)
    ax.set_xlim([wavemin, wavemin + dwave])
    ax.set_ylim([ymin, ymax])
    ax.axhline(1, ls='--', c='k', alpha=0.7)
    ax.legend(fontsize=legend_fontsize, loc=2)
    if ind == (nqso_to_plot - 1):
        ax.set_xlabel(r'obs wavelength ($\mathrm{{\AA}}$)', fontsize=xylabel_fontsize)

    #ind_masked = np.where(master_mask[wave_mask] * strong_abs_gpm[wave_mask] == False)[0]
    #for j in range(len(ind_masked)):  # bad way to plot masked pixels
    #    ax.axvline(wave[wave_mask][ind_masked[j]], color='k', alpha=0.2, lw=5)
    ind_masked = np.where(all_masks[wave_mask] == False)[0]
    for j in range(len(ind_masked)):  # bad way to plot masked pixels
        ax.axvline(wave[wave_mask][ind_masked[j]], color='k', alpha=0.2, lw=5)

atwin = ax_all[0].twiny()
atwin.set_xlabel('redshift', fontsize=xylabel_fontsize)
zmin, zmax = wavemin / 2800 - 1, (wavemin + dwave) / 2800 - 1
atwin.axis([zmin, zmax, ymin, ymax])
atwin.tick_params(top=True, axis="x", labelsize=xytick_size)
atwin.xaxis.set_minor_locator(AutoMinorLocator())

if savefig:
    plt.savefig('paper_plots/10qso/spec_mg2forest-z7.pdf')
    plt.close()
if savefig is False:
    plt.show()


