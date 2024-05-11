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
import mask_cgm_pdf
import compute_model_grid_8qso_fast as cmg8
import scipy
import pdb

###################### setting for figures ######################
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
legend_fontsize = 16
black_shaded_alpha = 0.25
fm_spec_alpha = 0.6

exclude_new_qso = True # excluding Jinyi and Banados new qso
sharex = False
sharey = True
vel_factor = 1# 100

ymin, ymax = -0.5, 8
xmin, xmax = 0, 62000

redshift_bin = 'all'
savefig = True #'paper_plots/10qso/forward_model_specs_%sz.pdf' % redshift_bin

###################### fixed data variables ######################
datapath='/Users/suksientie/Research/MgII_forest/rebinned_spectra2/'

qso_namelist = ['J0411-0907', 'J0319-1008', 'J0410-0139', 'newqso2', 'J0313-1806', 'J0038-1527', 'J0252-0503', \
                    'J1342+0928', 'J1007+2115', 'J1120+0641']
qso_zlist = [6.826, 6.8275, 7.0, 7.1, 7.642, 7.034, 7.001, 7.541, 7.515, 7.085]

corr_all = [0.93, 0.898, 0.88, 1.051, 0.972, 1.055, 1.086, 0.956, 0.908, 1.059] # 7/28/2023

nires_fwhm = 111.03
mosfire_fwhm = 83.05
nires_sampling = 2.7
mosfire_sampling = 2.78
xshooter_fwhm = 42.8
xshooter_sampling = 3.7

###################### flexible data variables ######################
#iqso_to_use = [0, 1, 4, 5, 6, 7, 8, 9] # omitting new qso in the plotting
iqso_to_use = [2, 3]
logZ = -4.50 # choosing model with no signal
ncovar = 5 # just mocking for plots

###################### forward models ######################
nqso = 10
#vel_data_allqso, norm_std_allqso, master_mask_allqso, instr_allqso, norm_flux_allqso = cmg8.init_dataset(nqso, redshift_bin, datapath)
vel_data_allqso, norm_flux_allqso, norm_std_allqso, norm_ivar_allqso, master_mask_allqso, master_mask_allqso_mask_cgm, instr_allqso = cmg8.init_dataset(
    nqso, redshift_bin, datapath)

rantaufile = 'ran_skewers_z75_OVT_xHI_0.50_tau.fits'
params = Table.read(rantaufile, hdu=1)
skewers = Table.read(rantaufile, hdu=2)

vel_lores_nires, flux_lores_nires = utils.create_mgii_forest(params, skewers, logZ, nires_fwhm, sampling=nires_sampling, mockcalc=True)
vel_lores_mosfire, flux_lores_mosfire = utils.create_mgii_forest(params, skewers, logZ, mosfire_fwhm, sampling=mosfire_sampling, mockcalc=True)
vel_lores_xshooter, flux_lores_xshooter = utils.create_mgii_forest(params, skewers, logZ, xshooter_fwhm, sampling=xshooter_sampling, mockcalc=True)

dv_coarse = 40

#vel_lores_nires_interp, flux_lores_nires_interp = cmg8.rebin_nyx_skewers(vel_lores_nires, flux_lores_nires, dv_coarse)
vel_lores_nires_interp = np.arange(vel_lores_nires[0], vel_lores_nires[-1], dv_coarse)
flux_lores_nires_interp = scipy.interpolate.interp1d(vel_lores_nires, flux_lores_nires, kind = 'cubic', bounds_error = False, fill_value = np.nan)(vel_lores_nires_interp)

#vel_lores_mosfire_interp, flux_lores_mosfire_interp = cmg8.rebin_nyx_skewers(vel_lores_mosfire, flux_lores_mosfire, dv_coarse)
vel_lores_mosfire_interp = np.arange(vel_lores_mosfire[0], vel_lores_mosfire[-1], dv_coarse)
flux_lores_mosfire_interp = scipy.interpolate.interp1d(vel_lores_mosfire, flux_lores_mosfire, kind='cubic', bounds_error=False, fill_value=np.nan)(vel_lores_mosfire_interp)

#vel_lores_xshooter_interp, flux_lores_xshooter_interp = cmg8.rebin_nyx_skewers(vel_lores_xshooter, flux_lores_xshooter, dv_coarse)
vel_lores_xshooter_interp = np.arange(vel_lores_xshooter[0], vel_lores_xshooter[-1], dv_coarse)
flux_lores_xshooter_interp = scipy.interpolate.interp1d(vel_lores_xshooter, flux_lores_xshooter, kind='cubic', bounds_error=False, fill_value=np.nan)(vel_lores_xshooter_interp)

assert np.sum(np.isnan(flux_lores_nires_interp)) == 0
assert np.sum(np.isnan(flux_lores_mosfire_interp)) == 0

for iqso in iqso_to_use:
#for iqso in [0]:
    vel_data = vel_data_allqso[iqso]
    norm_std = norm_std_allqso[iqso]
    master_mask = master_mask_allqso_mask_cgm[iqso]
    std_corr = corr_all[iqso]
    instr = instr_allqso[iqso]
    norm_flux = norm_flux_allqso[iqso]

    if instr == 'nires':
        vel_lores = vel_lores_nires_interp
        flux_lores = flux_lores_nires_interp
    elif instr == 'mosfire':
        vel_lores = vel_lores_mosfire_interp
        flux_lores = flux_lores_mosfire_interp
    elif instr == 'xshooter':
        vel_lores = vel_lores_xshooter_interp
        flux_lores = flux_lores_xshooter_interp

    # generate mock data spectrum
    rand = np.random.RandomState()

    vel_lores, flux_noise_ncopy, flux_noiseless_ncopy, master_mask_chunk, nskew_to_match_data, npix_sim_skew = \
        cmg8.forward_model_onespec_chunk(vel_data, norm_std, master_mask, vel_lores, flux_lores, ncovar, seed=rand, std_corr=std_corr)

    # plotting mock spectrum
    fig, ax = plt.subplots(figsize=(16, 6), sharex=sharex, sharey=sharey)
    fig.subplots_adjust(left=0.07, bottom=0.12, right=0.97, top=0.86, wspace=0, hspace=0.0)

    for i in range(ncovar):
        flux_lores_comb = np.reshape(flux_noise_ncopy[i], (nskew_to_match_data * npix_sim_skew))
        mask_nan = flux_lores_comb[:len(vel_data)] > -100
        ax.plot((vel_data/vel_factor)[mask_nan], flux_lores_comb[:len(vel_data)][mask_nan] + (i+1), 'tab:blue', alpha=fm_spec_alpha, drawstyle='steps-mid')

    ax.plot(vel_data/vel_factor, norm_flux, 'k', drawstyle='steps-mid', label=qso_namelist[iqso])

    ind_masked = np.where(master_mask == False)[0]
    for j in range(len(ind_masked)):
        if ind_masked[j] + 1 != len(vel_data):
            ax.axvspan(vel_data[ind_masked[j]]/vel_factor, vel_data[ind_masked[j] + 1]/vel_factor, facecolor='black', alpha=black_shaded_alpha)

    ax.set_ylabel(r'$F_{\mathrm{norm}}$ (+ offset)', fontsize=xylabel_fontsize)
    #ax.set_xlabel(r'Velocity $\times$ %d (km/s)' % vel_factor, fontsize=xylabel_fontsize)
    ax.set_xlabel(r'Velocity (km/s)', fontsize=xylabel_fontsize)
    ax.legend(loc=2, fontsize=legend_fontsize + 5)
    ax.tick_params(which='both', labelsize=xytick_size)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlim([vel_data.min()/vel_factor, vel_data.max()/vel_factor])
    ax.set_ylim([ymin, ymax])

    #############
    raw_data_out, masked_data_out, all_masks_out = mutils.init_onespec(iqso, redshift_bin, datapath=datapath)
    wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
    strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, telluric_gpm, master_mask = all_masks_out

    tmp_mask = pz_mask * zbin_mask

    # plot upper axis --- the CORRECT way, since vel and wave transformation is non-linear
    wave_data = wave[tmp_mask]
    def forward(x):
        return np.interp(x, vel_data/vel_factor, wave_data)
    def inverse(x):
        return np.interp(x, wave_data, vel_data/vel_factor)

    secax = ax.secondary_xaxis('top', functions=(forward, inverse))

    # if qso_namelist[iqso] == 'J0313-1806':
    #     secax.set_xticks(range(20000, 24000, 500))
    # elif qso_namelist[iqso] == 'J1342+0928':
    #     secax.set_xticks(range(20000, 23500, 500))
    # elif qso_namelist[iqso] == 'J1007+2115':
    #     secax.set_xticks(range(19500, 23500, 500))
    # elif qso_namelist[iqso] in ['J0319-1008', 'J0411-0907']:
    #     secax.set_xticks(range(19500, 21500, 500))
    # elif qso_namelist[iqso] == 'J1120+0641':
    #     secax.set_xticks(range(19500, 22500, 500))
    # elif qso_namelist[iqso] in ['J0252-0503', 'J0038-1527']:
    #     secax.set_xticks(range(20000, 22000, 500))
    if qso_namelist[iqso] == 'J0313-1806':
        secax.set_xticks(range(20000, 24000, 500))
    elif qso_namelist[iqso] == 'J1342+0928':
        secax.set_xticks(range(20000, 23500, 500))
    elif qso_namelist[iqso] == 'J1007+2115':
        secax.set_xticks(range(19500, 23500, 500))
    elif qso_namelist[iqso] in ['J0319-1008', 'J0411-0907']:
        secax.set_xticks(range(19500, 21500, 500))
    elif qso_namelist[iqso] == 'J1120+0641':
        secax.set_xticks(range(19500, 22500, 500))
    elif qso_namelist[iqso] in ['J0252-0503', 'J0038-1527', 'J0410-0139']:
        secax.set_xticks(range(20000, 22000, 500))
    elif qso_namelist[iqso] == 'newqso2':
        secax.set_xticks(range(20000, 22500, 500))

    secax.xaxis.set_minor_locator(AutoMinorLocator())
    secax.set_xlabel('obs wavelength (A)', fontsize=xylabel_fontsize, labelpad=8)
    secax.tick_params(top=True, axis="both", labelsize=xytick_size)

    if savefig:
        savefigname = 'paper_plots/10qso_revision/forward%d_%s.pdf' % (iqso, qso_namelist[iqso])
        plt.savefig(savefigname)
    else:
        plt.show()
    plt.close()