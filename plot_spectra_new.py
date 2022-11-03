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

datapath = '/Users/suksientie/Research/MgII_forest/rebinned_spectra/'
qso_namelist = ['J0411-0907', 'J0319-1008', 'J0410-0139', 'J0038-0653', 'J0313-1806', 'J0038-1527', 'J0252-0503', 'J1342+0928']
qso_zlist = [6.826, 6.8275, 7.0, 7.1, 7.642, 7.034, 7.001, 7.541]
everyn_break_list = (np.ones(len(qso_namelist)) * 20).astype('int')
exclude_restwave = 1216 - 1185
nqso_to_plot = len(qso_namelist)
redshift_bin = 'all'

# CGM masks
good_vel_data_all, good_wave_all, norm_good_flux_all, norm_good_std_all, norm_good_ivar_all, noise_all = \
    mask_cgm_pdf.init(redshift_bin=redshift_bin, do_not_apply_any_mask=True, datapath=datapath)
mgii_tot_all = mask_cgm_pdf.chi_pdf(good_vel_data_all, norm_good_flux_all, norm_good_ivar_all, noise_all, plot=False, savefig=None)

xmin = 19500
ymin = -0.05
ymax_ls = [0.8, 0.48, 0.4, 0.6, 0.35, 0.6, 0.5, 0.5]
ymin_norm, ymax_norm = -0.05, 2.3

savefig = True

for i in range(nqso_to_plot):
#for i in range(1):

    raw_data_out, masked_data_out, all_masks_out = mutils.init_onespec(i, redshift_bin, datapath=datapath)
    wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
    strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, telluric_mask, master_mask = all_masks_out

    all_masks = mask * redshift_mask * pz_mask * zbin_mask * telluric_mask
    print("masked fraction", 1 - np.sum(all_masks) / len(all_masks))

    median_snr = np.nanmedian((flux / std)[all_masks])
    print("median snr", median_snr)

    mgii_tot = mgii_tot_all[i]
    fs_mask = mgii_tot.fit_gpm[0]
    all_masks = all_masks * fs_mask

    ymax = ymax_ls[i]
    xmax = wave.max()

    fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 10), sharex=True)
    fig.subplots_adjust(left=0.1, bottom=0.08, right=0.96, top=0.93, wspace=0, hspace=0.)
    ax1.annotate(qso_namelist[i] + '\n', xy=(xmin + 100, ymax * 0.8), fontsize=18)
    ax1.plot(wave, flux, c='k', drawstyle='steps-mid')
    ax1.plot(wave, fluxfit, c='r', drawstyle='steps-mid') #, label='continuum fit')
    ax1.plot(wave, std, c='k', alpha=0.5, drawstyle='steps-mid')#, label='sigma')
    ind_masked = np.where(mask * strong_abs_gpm == False)[0]
    for j in range(len(ind_masked)):  # bad way to plot masked pixels
        ax1.axvline(wave[ind_masked[j]], color='k', alpha=0.1, lw=2)

    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(top=True, right=True, which='both', labelsize=xytick_size)
    ax1.axvline((qso_zlist[i] + 1) * 2800, ls='--', c='k', lw=3)

    ax1.set_xlim([xmin, xmax])
    ax1.set_ylim([ymin, ymax])
    ax1.set_ylabel(r'Flux' + '\n $(10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\mathrm{{\AA}}^{-1})$', fontsize=xylabel_fontsize)

    ax2.plot(wave, flux / fluxfit, c='k', drawstyle='steps-mid')
    ax2.plot(wave, std / fluxfit, c='k', alpha=0.5, drawstyle='steps-mid')
    ax2.plot(wave, tell * 2, alpha=0.5, drawstyle='steps-mid')  # telluric
    ind_masked = np.where(all_masks == False)[0]
    for j in range(len(ind_masked)):  # bad way to plot masked pixels
        ax2.axvline(wave[ind_masked[j]], color='k', alpha=0.1, lw=2)

    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.tick_params(top=True, right=True, which='both', labelsize=xytick_size)
    ax2.set_xlim([xmin, xmax])
    ax2.set_ylim([ymin_norm, ymax_norm])
    ax2.set_xlabel(r'obs wavelength ($\mathrm{{\AA}}$)', fontsize=xylabel_fontsize)
    ax2.set_ylabel(r'$F_{\mathrm{norm}}$', fontsize=xylabel_fontsize+5)

    atwin = ax1.twiny()
    atwin.set_xlabel('redshift', fontsize=xylabel_fontsize)
    zmin, zmax = xmin / 2800 - 1, xmax / 2800 - 1
    atwin.axis([zmin, zmax, ymin, ymax])
    atwin.tick_params(top=True, axis="x", labelsize=xytick_size)
    atwin.xaxis.set_minor_locator(AutoMinorLocator())

    if savefig:
        plt.savefig('paper_plots/8qso/spec_%s.png' % qso_namelist[i])
        plt.close()
    if savefig is False:
        plt.show()
