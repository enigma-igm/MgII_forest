import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import sys
from astropy.table import Table, hstack, vstack
from IPython import embed
sys.path.append('/Users/suksientie/codes/enigma') # comment out this line if running on IGM cluster
from enigma.reion_forest import utils
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
import mutils
import compute_model_grid_new as cmg

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
legend_fontsize = 12
black_shaded_alpha = 0.25
fm_spec_alpha = 0.6

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), sharex=True, sharey=True)
fig.subplots_adjust(left=0.12, bottom=0.15, right=0.98, top=0.93, wspace=0, hspace=0.)
ax_plot = axes.flatten()
savefig = 'paper_plots/forward_model_specs.pdf'

###################### data variables ######################
datapath = '/Users/suksientie/Research/data_redux/'

fitsfile_list = [datapath + 'wavegrid_vel/J0313-1806/vel1234_coadd_tellcorr.fits', \
                 datapath + 'wavegrid_vel/J1342+0928/vel123_coadd_tellcorr.fits', \
                 datapath + 'wavegrid_vel/J0252-0503/vel12_coadd_tellcorr.fits', \
                 datapath + 'wavegrid_vel/J0038-1527/vel1_tellcorr.fits']

qso_namelist = ['J0313-1806', 'J1342+0928', 'J0252-0503', 'J0038-1527']
qso_zlist = [7.642, 7.541, 7.001, 7.034] # precise redshifts from Yang+2021
everyn_break_list = [20, 20, 20, 20] # placing a breakpoint at every 20-th array element (more docs in mutils.continuum_normalize)
                                     # this results in dwave_breakpoint ~ 40 A --> dvel_breakpoint = 600 km/s
exclude_restwave = 1216 - 1185 # excluding proximity zones; see mutils.qso_exclude_proximity_zone
median_z = 6.57 # median redshift of measurement after excluding proximity zones; 4/20/2022
corr_all = [0.758, 0.753, 0.701, 0.724] # 4/20/2022 (determined from mutils.plot_allspec_pdf)

redshift_bin = 'all'
qso_seed_list = [77221056, 77221057, 77221058, 77221059]
ncopy_plot = 5
ncopy = 1000

###################### nyx skewers #########################
filename = 'ran_skewers_z75_OVT_xHI_0.50_tau.fits'
params = Table.read(filename, hdu=1)
skewers = Table.read(filename, hdu=2)

fwhm = 90 # 83
sampling = 3
logZ = -3.50

vel_lores, (flux_lores, flux_lores_igm, flux_lores_cgm, _, _), \
vel_hires, (flux_hires, flux_hires_igm, flux_hires_cgm, _, _), \
(oden, v_los, T, xHI), cgm_tuple = utils.create_mgii_forest(params, skewers, logZ, fwhm, sampling=sampling)

###################### forward models #########################
for iqso, fitsfile in enumerate(fitsfile_list):
    # initialize all qso data
    raw_data_out, _, all_masks_out = mutils.init_onespec(iqso, redshift_bin, datapath=datapath)
    wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
    strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, master_mask = all_masks_out

    norm_flux = flux / fluxfit
    norm_std = std / fluxfit
    vel_data = mutils.obswave_to_vel_2(wave)

    # generate mock data spectrum
    _, _, rand_noise_ncopy, noisy_flux_lores_ncopy, nskew_to_match_data, npix_sim_skew = cmg.forward_model_onespec_chunk(
        vel_data, norm_std, vel_lores, flux_lores, ncopy, seed=qso_seed_list[iqso], std_corr=corr_all[iqso])

    ncopy, nskew, npix = np.shape(noisy_flux_lores_ncopy)
    nan_pad_mask = ~np.isnan(noisy_flux_lores_ncopy[0])

    # plot subset of mock spectra
    ax = ax_plot[iqso]
    for i in range(ncopy_plot):
        flux_lores_comb = noisy_flux_lores_ncopy[i][nan_pad_mask]
        ax.plot(vel_data, flux_lores_comb + (i + 1), 'tab:blue', alpha=fm_spec_alpha, drawstyle='steps-mid', zorder=20)

    ax.plot(vel_data, norm_flux, 'k', drawstyle='steps-mid', label=qso_namelist[iqso])

    ind_masked = np.where(master_mask == False)[0]
    for j in range(len(ind_masked)):
        if ind_masked[j] + 1 != len(vel_data):
            ax.axvspan(vel_data[ind_masked[j]], vel_data[ind_masked[j] + 1], facecolor='black', alpha=black_shaded_alpha)

    if iqso == 0 or iqso == 2:
        ax.set_ylabel(r'$F_{\mathrm{norm}}$ (+ arbitrary offset)', fontsize=xylabel_fontsize)
    if iqso == 2 or iqso == 3:
        ax.set_xlabel('Velocity (km/s)', fontsize=xylabel_fontsize)

    ax.legend(loc=2, fontsize=legend_fontsize)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', labelsize=xytick_size)

plt.savefig(savefig)
plt.show()
plt.close()

