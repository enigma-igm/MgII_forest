'''
Functions here:
    - init
    - flux_pdf
    - chi_pdf
    - chi_pdf_onespec
    - plot_masked_onespec
    - do_allqso_allzbin
'''

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import sys
sys.path.append('/Users/suksientie/codes/enigma')
sys.path.append('/Users/suksientie/Research/data_redux')
sys.path.append('/Users/suksientie/Research/CIV_forest')
from enigma.reion_forest.mgii_find import MgiiFinder
from enigma.reion_forest import utils
from astropy.stats import sigma_clip, mad_std
from pypeit import utils as putils
import mutils
from matplotlib.ticker import AutoMinorLocator
from linetools.lists.linelist import LineList
from astropy import units as u
from astropy import constants as const
import pdb
from scipy import integrate

### Figure settings
font = {'family' : 'serif', 'weight' : 'normal'}#, 'size': 11}
plt.rc('font', **font)
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['xtick.minor.width'] = 1.5
mpl.rcParams['ytick.minor.width'] = 1.5
mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['ytick.minor.size'] = 4

xytick_size = 16
xylabel_fontsize = 20
legend_fontsize = 16

########## global variables ##########
seed = None

if seed != None:
    rand = np.random.RandomState(seed)
else:
    rand = np.random.RandomState()

qso_namelist = ['J0411-0907', 'J0319-1008', 'newqso1', 'newqso2', 'J0313-1806', 'J0038-1527', 'J0252-0503', \
                'J1342+0928', 'J1007+2115', 'J1120+0641']
qso_zlist = [6.826, 6.8275, 7.0, 7.1, 7.642, 7.034, 7.001, 7.541, 7.515, 7.085]
nqso_to_use = len(qso_namelist)

nires_fwhm = 111.03
mosfire_fwhm = 83.05
nires_sampling = 2.7
mosfire_sampling = 2.78
xshooter_fwhm = 150#42.8 # R=7000 in NIR arm based on telluric lines (Bosman+2017); nominal is 5300 for 0.9" slit
xshooter_sampling = 3.7 # https://www.eso.org/sci/facilities/paranal/instruments/xshooter/inst.html

qso_fwhm = [nires_fwhm, nires_fwhm, nires_fwhm, mosfire_fwhm, mosfire_fwhm, mosfire_fwhm, mosfire_fwhm, mosfire_fwhm, \
            nires_fwhm, xshooter_fwhm]
qso_sampling = [nires_sampling, nires_sampling, nires_sampling, mosfire_sampling, mosfire_sampling, mosfire_sampling, \
                mosfire_sampling, mosfire_sampling, nires_sampling, xshooter_sampling]

# chi PDF
signif_thresh = 2
signif_mask_dv = 300.0 # value used in Hennawi+2021
signif_mask_nsigma = 3

one_minF_thresh = 0.3 # flux threshold
nbins_chi = 101 #81
sig_min = 1e-3 # 1e-2
sig_max = 100.0
dsig_bin = np.ediff1d(np.linspace(sig_min, sig_max, nbins_chi))

# flux PDF
nbins_flux, oneminf_min, oneminf_max = 101, 1e-5, 1.0  # gives d(oneminf) = 0.01
color_ls = ['r', 'g', 'c', 'orange', 'm', 'gray', 'deeppink', 'lime', 'r', 'g']
ls_ls = ['-', '-', '-', '-', '-', '-', '-', '-', '--', '--']

def init(redshift_bin='all', datapath='/Users/suksientie/Research/MgII_forest/rebinned_spectra2/', do_not_apply_any_mask=False, iqso_to_use=None):
    norm_good_flux_all = []
    norm_good_std_all = []
    norm_good_ivar_all = []
    good_vel_data_all = []
    good_wave_all = []
    noise_all = []
    pz_masks_all = []
    other_masks_all = []

    if iqso_to_use is None:
        iqso_to_use = np.arange(0, nqso_to_use)

    for iqso in iqso_to_use:
        raw_data_out, _, all_masks_out = mutils.init_onespec(iqso, redshift_bin, datapath=datapath)
        wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
        strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, telluric_mask, master_mask = all_masks_out
        vel_data = mutils.obswave_to_vel_2(wave)

        if do_not_apply_any_mask: # do not apply any masks before CGM masking
            masks_for_cgm_masking = np.ones_like(wave, dtype=bool)
        else: # apply these masks before CGM masking
            masks_for_cgm_masking = mask * redshift_mask * pz_mask * zbin_mask * telluric_mask

        pz_masks_all.append(redshift_mask * pz_mask * zbin_mask)
        other_masks_all.append(mask * telluric_mask)

        # masks quantities
        norm_good_flux = (flux/fluxfit)[masks_for_cgm_masking]
        norm_good_std = (std/fluxfit)[masks_for_cgm_masking]
        norm_good_ivar = (ivar*(fluxfit**2))[masks_for_cgm_masking]
        good_wave = wave[masks_for_cgm_masking]
        good_vel_data = vel_data[masks_for_cgm_masking]

        norm_good_flux_all.append(norm_good_flux)
        norm_good_std_all.append(norm_good_std)
        norm_good_ivar_all.append(norm_good_ivar)
        good_vel_data_all.append(good_vel_data)
        good_wave_all.append(good_wave)

        if do_not_apply_any_mask:
            noise_all.append(np.ones(len(norm_good_std)) * 100)
        else: # same as mutils.plot_onespec_pdf()
            chi = (1 - norm_good_flux) / norm_good_std
            corr_factor = mad_std(chi)
            print("-----correction factor", corr_factor)
            gaussian_noise = np.random.normal(0, norm_good_std * corr_factor)
            noise_all.append(gaussian_noise)

    return good_vel_data_all, good_wave_all, norm_good_flux_all, norm_good_std_all, norm_good_ivar_all, noise_all, pz_masks_all, other_masks_all

def flux_pdf(norm_good_flux_all, noise_all, plot_ispec=None, savefig=None):
    # plot the flux PDF

    all_transmission = []
    all_noise = []
    nqso = len(norm_good_flux_all)

    #fig = plt.figure(figsize=(9, 7.5))
    fig = plt.figure(figsize=(12, 9.8))
    fig.subplots_adjust(left=0.12, bottom=0.1, right=0.97, top=0.88)

    if plot_ispec != None:
        # plot PDF for one QSO
        i = plot_ispec
        norm_good_flux, noise = norm_good_flux_all[i], noise_all[i]
        flux_bins, flux_pdf_tot = utils.pdf_calc(1.0 - norm_good_flux, oneminf_min, oneminf_max, nbins_flux)
        _, flux_pdf_noise = utils.pdf_calc(noise, oneminf_min, oneminf_max, nbins_flux)

        plt.plot(flux_bins, flux_pdf_tot, drawstyle='steps-mid', alpha=1.0, lw=2, color=color_ls[i], ls=ls_ls[i], label=qso_namelist[i])
        plt.plot(flux_bins, flux_pdf_noise, drawstyle='steps-mid', color='b', alpha=1.0, label='gaussian noise')

        plt.axvline(one_minF_thresh, color='k', ls='--', lw=2)
        plt.axvspan(one_minF_thresh, oneminf_max, facecolor='k', alpha=0.2, label='masked')

    else:
        # plot PDF for all qso
        for i in range(nqso):
            norm_good_flux, noise = norm_good_flux_all[i], noise_all[i]
            flux_bins, flux_pdf_tot = utils.pdf_calc(1.0 - norm_good_flux, oneminf_min, oneminf_max, nbins_flux)
            plt.plot(flux_bins, flux_pdf_tot, drawstyle='steps-mid', alpha=0.5, color=color_ls[i], ls=ls_ls[i])#, label=qso_namelist[i])

            all_transmission.extend(norm_good_flux)
            all_noise.extend(noise)

        # plot PDF for all qso
        all_transmission = np.array(all_transmission)
        all_noise = np.array(all_noise)

        flux_bins, flux_pdf_tot = utils.pdf_calc(1.0 - all_transmission, oneminf_min, oneminf_max, nbins_flux)
        _, flux_pdf_noise = utils.pdf_calc(all_noise, oneminf_min, oneminf_max, nbins_flux)

        plt.plot(flux_bins, flux_pdf_noise, drawstyle='steps-mid', color='b', alpha=1.0, label='all gaussian noise')
        plt.plot(flux_bins, flux_pdf_tot, drawstyle='steps-mid', color='k', alpha=1.0, lw=4, label='all spectra')
        plt.axvline(one_minF_thresh, color='k', ls='--', lw=2)
        plt.axvspan(one_minF_thresh, oneminf_max, facecolor='k', alpha=0.2, label='masked')

    xytick_size = 16 + 8
    xylabel_fontsize = 20 + 8
    legend_fontsize = 16 + 8

    plt.legend(loc=2, fontsize=legend_fontsize)
    plt.xlim([1e-4, 1.0])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('1$-$F', fontsize=xylabel_fontsize)
    plt.ylabel('PDF', fontsize=xylabel_fontsize)
    plt.gca().tick_params(axis="x", labelsize=xytick_size)
    plt.gca().tick_params(axis="y", labelsize=xytick_size)

    strong_lines = LineList('Strong', verbose=False)
    wave_blue = strong_lines['MgII 2796']['wrest']
    #fwhm_avg = np.mean(qso_fwhm)
    #sampling_avg = np.mean(qso_sampling)
    #Wfactor = ((fwhm_avg / sampling_avg) * u.km / u.s / const.c).decompose() * wave_blue.value
    #print("fwhm_avg", fwhm_avg)
    #print("sampling_avg", sampling_avg)
    #print("Wfactor", Wfactor)
    dvpix = 40
    Wfactor = (dvpix * u.km / u.s / const.c).decompose() * wave_blue.value
    print("Wfactor", Wfactor)

    Wmin_top, Wmax_top = Wfactor * oneminf_min, Wfactor * oneminf_max  # top axis
    ymin, ymax = 1e-3, 1.5

    atwin = plt.twiny()
    atwin.set_xlabel(r'$W_{{\lambda, \mathrm{{pix}}}}$ [$\mathrm{{\AA}}]$', labelpad=10, fontsize=xylabel_fontsize)
    atwin.xaxis.tick_top()
    atwin.set_xscale('log')
    atwin.axis([Wmin_top, Wmax_top, ymin, ymax])
    #atwin.tick_params(top=True, labelsize=xytick_size)
    atwin.tick_params(top=True, axis="both", labelsize=xytick_size)

    plt.tight_layout()
    if savefig != None:
        plt.savefig(savefig)
    plt.show()

def chi_pdf(vel_data_all, norm_good_flux_all, norm_good_ivar_all, noise_all, plot=False, savefig=None):
    # plot the chi PDF and returns the mgii_finder object

    all_chi = []
    all_chi_noise = []
    mgii_tot_all = []
    nqso = len(norm_good_flux_all)

    if plot:
        #fig = plt.figure(figsize=(9, 6.7))
        fig = plt.figure(figsize=(12, 9))
        fig.subplots_adjust(left=0.12, bottom=0.1, right=0.97, top=0.88)

    # PDF for each qso
    for i in range(nqso):
        vel_data = vel_data_all[i]
        norm_good_flux = norm_good_flux_all[i]
        norm_good_ivar = norm_good_ivar_all[i]
        noise = noise_all[i]

        # reshaping to be compatible with MgiiFinder
        norm_good_flux = norm_good_flux.reshape((1, len(norm_good_flux)))
        norm_good_ivar = norm_good_ivar.reshape((1, len(norm_good_ivar)))
        noise = noise.reshape((1, len(noise)))
        fwhm = qso_fwhm[i]

        mgii_tot = MgiiFinder(vel_data, norm_good_flux, norm_good_ivar, fwhm, signif_thresh,
                              signif_mask_nsigma=signif_mask_nsigma,
                              signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh)
        mgii_tot_all.append(mgii_tot)

        if plot:
            # plotting chi PDF of each QSO
            sig_bins, sig_pdf_tot = utils.pdf_calc(mgii_tot.signif, sig_min, sig_max, nbins_chi)
            plt.plot(sig_bins, sig_pdf_tot, drawstyle='steps-mid', alpha=0.5, color=color_ls[i], ls=ls_ls[i])#, label=qso_namelist[i])

            # chi PDF of pure noise
            mgii_noise = MgiiFinder(vel_data, 1.0 + noise, norm_good_ivar, fwhm, signif_thresh,
                                    signif_mask_nsigma=signif_mask_nsigma,
                                    signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh)

            all_chi.extend(mgii_tot.signif[0])
            all_chi_noise.extend(mgii_noise.signif[0])

    if plot:
        # plot PDF of all qso
        all_chi = np.array(all_chi)
        all_chi_noise = np.array(all_chi_noise)

        sig_bins, sig_pdf_tot = utils.pdf_calc(all_chi, sig_min, sig_max, nbins_chi)
        _, sig_pdf_noise = utils.pdf_calc(all_chi_noise, sig_min, sig_max, nbins_chi)

        plt.plot(sig_bins, sig_pdf_noise, drawstyle='steps-mid', color='b', alpha=1.0, label='all gaussian noise')
        plt.plot(sig_bins, sig_pdf_tot, drawstyle='steps-mid', color='k', alpha=1.0, lw=4, label='all spectra')
        plt.axvline(signif_mask_nsigma, color='k', ls='--', lw=2)
        plt.axvspan(signif_mask_nsigma, sig_max, facecolor='k', alpha=0.2, label='masked')

        xytick_size = 16 + 8
        xylabel_fontsize = 20 + 8
        legend_fontsize = 16 + 8

        plt.legend(loc=2, fontsize=legend_fontsize)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim([1e-3, 50])
        plt.ylim(top=0.8)
        plt.xlabel(r'$\chi$', fontsize=xylabel_fontsize)
        plt.ylabel('PDF', fontsize=xylabel_fontsize)
        plt.gca().tick_params(axis="both", labelsize=xytick_size)

        plt.tight_layout()
        if savefig != None:
            plt.savefig(savefig)
        plt.show()

    return mgii_tot_all

def chi_pdf2(vel_data_all, norm_good_flux_all, norm_good_ivar_all, noise_all, pz_masks_all, other_masks_all, plot=False, savefig=None):
    # same as chi_pdf() but uses gpm in MgiiFinder
    all_chi = []
    all_chi_noise = []
    mgii_tot_all = []
    nqso = len(norm_good_flux_all)

    if plot:
        #fig = plt.figure(figsize=(9, 6.7))
        fig = plt.figure(figsize=(12, 9))
        fig.subplots_adjust(left=0.12, bottom=0.1, right=0.97, top=0.88)

    # PDF for each qso
    for i in range(nqso):
        vel_data = vel_data_all[i]
        norm_good_flux = norm_good_flux_all[i]
        norm_good_ivar = norm_good_ivar_all[i]
        noise = noise_all[i]
        gpm = pz_masks_all[i] * other_masks_all[i]

        # reshaping to be compatible with MgiiFinder
        norm_good_flux = norm_good_flux.reshape((1, len(norm_good_flux)))
        norm_good_ivar = norm_good_ivar.reshape((1, len(norm_good_ivar)))
        noise = noise.reshape((1, len(noise)))
        gpm = gpm.reshape((1, len(gpm)))
        fwhm = qso_fwhm[i]

        mgii_tot = MgiiFinder(vel_data, norm_good_flux, norm_good_ivar, fwhm, signif_thresh,
                              gpm=gpm,
                              signif_mask_nsigma=signif_mask_nsigma,
                              signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh)
        mgii_tot_all.append(mgii_tot)

        if plot:
            # plotting chi PDF of each QSO
            sig_bins, sig_pdf_tot = utils.pdf_calc(mgii_tot.signif, sig_min, sig_max, nbins_chi)
            plt.plot(sig_bins, sig_pdf_tot, drawstyle='steps-mid', alpha=0.5, color=color_ls[i], ls=ls_ls[i])#, label=qso_namelist[i])

            # chi PDF of pure noise
            mgii_noise = MgiiFinder(vel_data, 1.0 + noise, norm_good_ivar, fwhm, signif_thresh,
                                    signif_mask_nsigma=signif_mask_nsigma,
                                    signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh)

            all_chi.extend(mgii_tot.signif[0])
            all_chi_noise.extend(mgii_noise.signif[0])

    if plot:
        # plot PDF of all qso
        all_chi = np.array(all_chi)
        all_chi_noise = np.array(all_chi_noise)

        sig_bins, sig_pdf_tot = utils.pdf_calc(all_chi, sig_min, sig_max, nbins_chi)
        _, sig_pdf_noise = utils.pdf_calc(all_chi_noise, sig_min, sig_max, nbins_chi)

        plt.plot(sig_bins, sig_pdf_noise, drawstyle='steps-mid', color='b', alpha=1.0, label='all gaussian noise')
        plt.plot(sig_bins, sig_pdf_tot, drawstyle='steps-mid', color='k', alpha=1.0, lw=4, label='all spectra')
        plt.axvline(signif_mask_nsigma, color='k', ls='--', lw=2)
        plt.axvspan(signif_mask_nsigma, sig_max, facecolor='k', alpha=0.2, label='masked')

        xytick_size = 16 + 8
        xylabel_fontsize = 20 + 8
        legend_fontsize = 16 + 8

        plt.legend(loc=2, fontsize=legend_fontsize)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim([1e-3, 50])
        plt.ylim(top=0.8)
        plt.xlabel(r'$\chi$', fontsize=xylabel_fontsize)
        plt.ylabel('PDF', fontsize=xylabel_fontsize)
        plt.gca().tick_params(axis="both", labelsize=xytick_size)

        plt.tight_layout()
        if savefig != None:
            plt.savefig(savefig)
        plt.show()

    return mgii_tot_all

def chi_pdf_onespec(vel_data_all, norm_good_flux_all, norm_good_ivar_all, noise_all, ispec, plot=False):
    # same as chi_pdf() but doing it for one spec

    i = ispec
    vel_data = vel_data_all[i]
    norm_good_flux = norm_good_flux_all[i]
    norm_good_ivar = norm_good_ivar_all[i]
    noise = noise_all[i]

    # reshaping to be compatible with MgiiFinder
    norm_good_flux = norm_good_flux.reshape((1, len(norm_good_flux)))
    norm_good_ivar = norm_good_ivar.reshape((1, len(norm_good_ivar)))
    noise = noise.reshape((1, len(noise)))
    fwhm = qso_fwhm[i]

    mgii_tot = MgiiFinder(vel_data, norm_good_flux, norm_good_ivar, fwhm, signif_thresh,
                          signif_mask_nsigma=signif_mask_nsigma,
                          signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh)

    # chi PDF of pure noise
    mgii_noise = MgiiFinder(vel_data, 1.0 + noise, norm_good_ivar, fwhm, signif_thresh,
                            signif_mask_nsigma=signif_mask_nsigma,
                            signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh)

    sig_bins, sig_pdf_tot = utils.pdf_calc(mgii_tot.signif, sig_min, sig_max, nbins_chi)
    _, sig_pdf_noise = utils.pdf_calc(mgii_noise.signif, sig_min, sig_max, nbins_chi)

    if plot:
        plt.plot(sig_bins, sig_pdf_tot, drawstyle='steps-mid', alpha=1.0, lw=2, label=qso_namelist[i], zorder=10)
        plt.plot(sig_bins, sig_pdf_noise, drawstyle='steps-mid', color='k', alpha=0.7, label='gaussian noise')

        plt.axvline(signif_mask_nsigma, color='k', ls='--', lw=2)
        plt.axvspan(signif_mask_nsigma, sig_max, facecolor='k', alpha=0.2, label='masked')

        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\chi$')
        plt.ylabel('PDF')
        plt.tight_layout()
        plt.show()

    return mgii_tot

def plot_masked_onespec(mgii_tot_all, wave_data_all, vel_data_all, norm_good_flux_all, norm_good_std_all, iqso, chi_max, savefig=None, saveout=None):

    mgii_tot = mgii_tot_all[iqso]
    vel_data = vel_data_all[iqso]
    wave_data = wave_data_all[iqso]
    norm_good_flux = norm_good_flux_all[iqso]
    norm_good_std = norm_good_std_all[iqso]
    qso_name = qso_namelist[iqso]
    qso_z = qso_zlist[iqso]

    f_mask = np.invert(mgii_tot.flux_gpm[0])
    s_mask = np.invert(mgii_tot.signif_gpm[0])
    fs_mask = np.invert(mgii_tot.fit_gpm[0])

    f_mask_frac = np.sum(f_mask)/len(f_mask)
    s_mask_frac = np.sum(s_mask)/len(s_mask)
    fs_mask_frac = np.sum(fs_mask)/len(fs_mask)
    print(f_mask_frac, s_mask_frac, fs_mask_frac)

    if saveout is not None:
        i_abs_found = np.argwhere(s_mask == True).squeeze()
        #np.savetxt(saveout, i_abs_found, delimiter=",")
        np.savetxt(saveout, np.vstack((i_abs_found, wave_data[i_abs_found], norm_good_flux[i_abs_found], norm_good_std[i_abs_found])).T, delimiter=",")

    fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 10), sharex=True)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.98, top=0.93, wspace=0, hspace=0.)
    flux_min, flux_max = -0.05, 1.8
    chi_min = -3.0 #, chi_max = -3.0, 8.4

    #ax1.annotate(qso_name + '\n', xy=(vel_data.min()+500, flux_max-0.3), fontsize=18)
    ax1.annotate(qso_name, xy=(vel_data.min() + 500, flux_max * 0.88), bbox=dict(boxstyle='round', ec="k", fc="white"), fontsize=18)
    ax1.plot(vel_data, norm_good_flux, drawstyle='steps-mid', color='k', linewidth=1.5, zorder=1)
    ax1.plot(vel_data, norm_good_std, drawstyle='steps-mid', color='k', linewidth=1.0, alpha=0.5)
    ax1.axhline(y = 1 - one_minF_thresh, color='green', linestyle='dashed', linewidth=2, label=r'$1 - \rm{F} = %0.1f$ (%0.2f pixels masked)' % (one_minF_thresh, f_mask_frac))
    ax1.plot(vel_data[s_mask], norm_good_flux[s_mask], color='magenta', markersize=7, markeredgewidth=2.5, linestyle='none', alpha=0.7, zorder=2, marker='|')
    ax1.plot(vel_data[f_mask], norm_good_flux[f_mask], color = 'green', markersize = 7, markeredgewidth = 2.5, linestyle = 'none', alpha = 0.7, zorder = 3, marker = '|')
    ax1.set_ylabel(r'$F_{\mathrm{norm}}$', fontsize=xylabel_fontsize)
    ax1.set_xlim([vel_data.min(), vel_data.max()])
    ax1.set_ylim([flux_min, flux_max])
    ax1.tick_params(which='both', labelsize=xytick_size)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.legend(fontsize=legend_fontsize, loc=1)

    neg = np.zeros_like(vel_data) - 100
    ax2.plot(vel_data, mgii_tot.signif[0], drawstyle='steps-mid', color='k')
    ax2.axhline(y=signif_mask_nsigma, color='magenta', linestyle='dashed', linewidth=2, label=r'$\chi$ = %d (%0.2f pixels masked)' % (signif_mask_nsigma, s_mask_frac))
    ax2.fill_between(vel_data, neg, mgii_tot.signif[0], where=s_mask, step = 'mid', facecolor = 'magenta', alpha = 0.5)
    ax2.set_xlabel('v (km/s)', fontsize=xylabel_fontsize)
    ax2.set_ylabel(r'$\chi$', fontsize=xylabel_fontsize)
    ax2.set_xlim([vel_data.min(), vel_data.max()])
    ax2.set_ylim([chi_min, chi_max])
    ax2.tick_params(which='both', top=True, labelsize=xytick_size)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.legend(fontsize=legend_fontsize, loc=1)

    # plot upper axis --- the CORRECT way, since vel and wave transformation is non-linear
    def forward(x):
        return np.interp(x, vel_data, wave_data)

    def inverse(x):
        return np.interp(x, wave_data, vel_data)

    secax = ax1.secondary_xaxis('top', functions=(forward, inverse))
    if qso_name in ['J0313-1806', 'J1342+0928']:
        secax.set_xticks(range(20000, 24000, 500))
    elif qso_name in ['J0319-1008', 'J0411-0907']:
        secax.set_xticks(range(20000, 22000, 500))
    else:
        secax.set_xticks(range(20000, 22500, 500))
    secax.xaxis.set_minor_locator(AutoMinorLocator())
    secax.set_xlabel('obs wavelength (A)', fontsize=xylabel_fontsize, labelpad=8)
    secax.tick_params(top=True, axis="both", labelsize=xytick_size)

    #plt.tight_layout()
    if savefig != None:
        plt.savefig(savefig)
    else:
        plt.show()

def plot_masked_onespec2(mgii_tot_all, wave_data_all, vel_data_all, norm_good_flux_all, norm_good_std_all, pz_masks_all, other_masks_all, iqso, chi_max, savefig=None, saveout=None):

    pz_masks = pz_masks_all[iqso]
    other_masks = other_masks_all[iqso]
    gpm = pz_masks * other_masks

    mgii_tot = mgii_tot_all[iqso]
    vel_data = vel_data_all[iqso]
    wave_data = wave_data_all[iqso]
    norm_good_flux = norm_good_flux_all[iqso]
    norm_good_std = norm_good_std_all[iqso]
    qso_name = qso_namelist[iqso]
    qso_z = qso_zlist[iqso]

    f_mask = np.invert(mgii_tot.flux_gpm[0])
    s_mask = np.invert(mgii_tot.signif_gpm[0])
    fs_mask = np.invert(mgii_tot.fit_gpm[0])

    # calculating the masked pixel fraction within unmasked regions only
    f_mask_frac = np.sum(f_mask[pz_masks * other_masks])/len(f_mask)
    s_mask_frac = np.sum(s_mask[pz_masks * other_masks])/len(s_mask)
    fs_mask_frac = np.sum(fs_mask[pz_masks * other_masks])/len(fs_mask)
    print(qso_namelist[iqso], f_mask_frac, s_mask_frac, fs_mask_frac)

    if saveout is not None:
        i_abs_found = np.argwhere(s_mask == True).squeeze()
        #np.savetxt(saveout, i_abs_found, delimiter=",")
        np.savetxt(saveout, np.vstack((i_abs_found, wave_data[i_abs_found], norm_good_flux[i_abs_found], norm_good_std[i_abs_found])).T, delimiter=",")

    #fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 10), sharex=True)
    #fig.subplots_adjust(left=0.1, bottom=0.1, right=0.98, top=0.93, wspace=0, hspace=0.)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 7), sharex=True)
    fig.subplots_adjust(left=0.085, bottom=0.11, right=0.95, top=0.89, wspace=0, hspace=0.)
    flux_min, flux_max = -0.05, 1.8
    if qso_name in ['J1007+2115', 'J0319-1008']:
        flux_max = 2.0
    chi_min = -3.0

    xytick_size = 16
    xylabel_fontsize = 20
    legend_fontsize = 16

    # xy=(vel_data.min() + 500, flux_max * 0.88)
    ax1.annotate(qso_name, xy=(vel_data.min() + 700, flux_max * 0.85), bbox=dict(boxstyle='round', ec="k", fc="white"), fontsize=legend_fontsize + 5)
    ax1.plot(vel_data, norm_good_flux, drawstyle='steps-mid', color='k', linewidth=1.5, zorder=1)
    ax1.plot(vel_data, norm_good_std, drawstyle='steps-mid', color='k', linewidth=1.0, alpha=0.5)
    ax1.axhline(y = 1 - one_minF_thresh, color='green', linestyle='dashed', linewidth=2, label=r'$1 - \rm{F} = %0.1f$ (%0.2f pixels masked)' % (one_minF_thresh, f_mask_frac))
    ax1.plot(vel_data[s_mask * gpm], norm_good_flux[s_mask * gpm], color='magenta', markersize=10, markeredgewidth=3, linestyle='none', alpha=0.7, zorder=2, marker='|')
    ax1.plot(vel_data[f_mask * gpm], norm_good_flux[f_mask * gpm], color = 'green', markersize = 15, markeredgewidth = 6, linestyle = 'none', alpha = 0.7, zorder = 3, marker = '|')
    ax1.set_ylabel(r'$F_{\mathrm{norm}}$', fontsize=xylabel_fontsize)
    #ax1.set_xlim([vel_data.min(), vel_data.max()])
    ax1.set_xlim([vel_data[pz_masks].min(), vel_data[pz_masks].max()])
    ax1.set_ylim([flux_min, flux_max])
    ax1.tick_params(which='both', labelsize=xytick_size)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.legend(fontsize=legend_fontsize, loc=1)

    neg = np.zeros_like(vel_data) - 100
    ax2.plot(vel_data, mgii_tot.signif[0], drawstyle='steps-mid', color='k')
    ax2.axhline(y=signif_mask_nsigma, color='magenta', linestyle='dashed', linewidth=2, label=r'$\chi$ = %d (%0.2f pixels masked)' % (signif_mask_nsigma, s_mask_frac))
    ax2.fill_between(vel_data, neg, mgii_tot.signif[0], where=s_mask * gpm, step = 'mid', facecolor = 'magenta', alpha = 0.5)
    ax2.set_xlabel('v (km/s)', fontsize=xylabel_fontsize)
    ax2.set_ylabel(r'$\chi$', fontsize=xylabel_fontsize)
    #ax2.set_xlim([vel_data.min(), vel_data.max()])
    ax2.set_xlim([vel_data[pz_masks].min(), vel_data[pz_masks].max()])
    ax2.set_ylim([chi_min, chi_max])
    ax2.tick_params(which='both', top=True, labelsize=xytick_size)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.legend(fontsize=legend_fontsize, loc=1)

    ind_masked = np.where(other_masks == False)[0]
    for j in range(len(ind_masked)):  # bad way to plot masked pixels
        ax1.axvline(vel_data[ind_masked[j]], c='k', alpha=0.1, lw=2)
        ax2.axvline(vel_data[ind_masked[j]], c='k', alpha=0.1, lw=2)

    if qso_namelist[iqso] == 'J1120+0641':
        """
        bosman_abs = [6.1711, 6.21845, 6.40671]
        for i_babs, babs in enumerate(bosman_abs):
            wave_blue = 2796 * (1 + babs)
            wave_red = 2804 * (1 + babs)
            iblue = np.argmin(np.abs(wave_data - wave_blue))
            ired = np.argmin(np.abs(wave_data - wave_red))
            ax1.axvline(x=vel_data[iblue], color='blue', linestyle=':', linewidth=2, zorder=1)
            ax1.axvline(x=vel_data[ired], color='red', linestyle=':', linewidth=2, zorder=1)
            ax2.axvline(x=vel_data[iblue], color='blue', linestyle=':', linewidth=2, zorder=1)
            ax2.axvline(x=vel_data[ired], color='red', linestyle=':', linewidth=2, zorder=1)
        """
        bluered_mask_gpm, abs_mask_gpm = bosman_J1120([4, 4, 3.5])
        ax1.plot(vel_data[np.invert(abs_mask_gpm)], norm_good_flux[np.invert(abs_mask_gpm)], color='blue', markersize=15,
                 markeredgewidth=8, linestyle='none', alpha=1.0, zorder=3, marker='|', label='Bosman et al. (2017)')
        ax1.legend(fontsize=legend_fontsize, loc=1)
        #ax2.fill_between(vel_data, neg, mgii_tot.signif[0], where=np.invert(abs_mask_gpm), step='mid', facecolor='blue', alpha=0.5)

    # plot upper axis --- the CORRECT way, since vel and wave transformation is non-linear
    vel_data2 = vel_data[pz_masks]
    wave_data2 = wave_data[pz_masks]
    def forward(x):
        return np.interp(x, vel_data2, wave_data2)

    def inverse(x):
        return np.interp(x, wave_data2, vel_data2)

    secax = ax1.secondary_xaxis('top', functions=(forward, inverse))
    """"
    if qso_name in ['J0313-1806', 'J1342+0928']:
        secax.set_xticks(range(20000, 24000, 500))
    elif qso_name in ['J0319-1008', 'J0411-0907']:
        secax.set_xticks(range(20000, 22000, 500))
    else:
        secax.set_xticks(range(20000, 22500, 500))
    """
    if qso_name == 'J0313-1806':
        secax.set_xticks(range(20000, 24000, 500))
    elif qso_name == 'J1342+0928':
        secax.set_xticks(range(20000, 23500, 500))
    elif qso_name == 'J1007+2115':
        secax.set_xticks(range(19500, 23500, 500))
    elif qso_name in ['J0319-1008', 'J0411-0907']:
        secax.set_xticks(range(19500, 21500, 500))
    elif qso_name == 'J1120+0641':
        secax.set_xticks(range(19500, 22500, 500))
    elif qso_name in ['J0252-0503', 'J0038-1527']:
        secax.set_xticks(range(20000, 22000, 500))

    secax.xaxis.set_minor_locator(AutoMinorLocator())
    secax.set_xlabel('obs wavelength (A)', fontsize=xylabel_fontsize, labelpad=8)
    secax.tick_params(top=True, axis="both", labelsize=xytick_size)

    #plt.tight_layout()
    if savefig != None:
        plt.savefig(savefig)
    else:
        plt.show()
        plt.close

def do_allqso_allzbin(datapath, do_not_apply_any_mask=False):
    # get mgii_tot_all for all qso and for all redshift bins

    #do_not_apply_any_mask = False

    redshift_bin = 'low'
    lowz_good_vel_data_all, lowz_good_wave_data_all, lowz_norm_good_flux_all, lowz_norm_good_std_all, lowz_good_ivar_all, lowz_noise_all, _, _ = \
        init(redshift_bin, datapath=datapath, do_not_apply_any_mask=do_not_apply_any_mask)
    lowz_mgii_tot_all = chi_pdf(lowz_good_vel_data_all, lowz_norm_good_flux_all, lowz_good_ivar_all, lowz_noise_all, plot=False)

    redshift_bin = 'high'
    highz_good_vel_data_all, highz_good_wave_data_all, highz_norm_good_flux_all, highz_norm_good_std_all, highz_good_ivar_all, highz_noise_all, _, _ = \
        init(redshift_bin, datapath=datapath, do_not_apply_any_mask=do_not_apply_any_mask)
    highz_mgii_tot_all = chi_pdf(highz_good_vel_data_all, highz_norm_good_flux_all, highz_good_ivar_all, highz_noise_all, plot=False)

    redshift_bin = 'all'
    allz_good_vel_data_all, allz_good_wave_data_all, allz_norm_good_flux_all, allz_norm_good_std_all, allz_good_ivar_all, allz_noise_all, _, _ = \
        init(redshift_bin, datapath=datapath, do_not_apply_any_mask=do_not_apply_any_mask)
    allz_mgii_tot_all = chi_pdf(allz_good_vel_data_all, allz_norm_good_flux_all, allz_good_ivar_all, allz_noise_all, plot=False)

    return lowz_mgii_tot_all, highz_mgii_tot_all, allz_mgii_tot_all

def bosman_J1120(dwave_ls, wave=None, norm_flux=None, datapath='/Users/suksientie/Research/MgII_forest/rebinned_spectra2/'):

    if wave is None and norm_flux is None:
        raw_data_out, _, all_masks_out = mutils.init_onespec(9, 'all', datapath=datapath)
        wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
        norm_flux = flux / fluxfit

    bosman_abs = [6.1711, 6.21845, 6.40671]
    bosman_ew = [0.258, 0.139, 0.094] # [4, 4, 3.5] approximately reproduces Bosman EW = [0.235, 0.13, 0.11]

    #blue_mask_all = []
    #red_mask_all = []
    bluered_mask_all = []

    for i_babs, babs in enumerate(bosman_abs):
        dwave = dwave_ls[i_babs]
        wave_blue = 2796 * (1 + babs)
        wm_blue = (wave <= wave_blue + dwave) * (wave >= wave_blue - dwave)
        flux_mask_blue = norm_flux < 1
        rest_wb = wave / (1 + babs) #wave[wm][flux_mask_blue] / (1 + babs)
        ew_blue = integrate.simps(1.0 - norm_flux[wm_blue * flux_mask_blue], rest_wb[wm_blue * flux_mask_blue]) #integrate.simps(1.0 - norm_flux[wm][flux_mask_blue], rest_wb)
        #blue_mask_all.append(wm * flux_mask_blue)

        wave_red = 2804 * (1 + babs)
        wm_red = (wave <= wave_red + dwave) * (wave >= wave_red - dwave)
        flux_mask_red = norm_flux < 1 #norm_flux[wm] < 1
        rest_wr = wave / (1 + babs) #wave[wm][flux_mask_red] / (1 + babs)
        ew_red = integrate.simps(1.0 - norm_flux[wm_red * flux_mask_red], rest_wr[wm_red * flux_mask_red])  #integrate.simps(1.0 - norm_flux[wm][flux_mask_red], rest_wr)
        #red_mask_all.append(wm * flux_mask_red)

        bluered_mask_all.append(np.ma.mask_or(wm_blue * flux_mask_blue, wm_red * flux_mask_red))
        # print(np.sum(flux_mask_blue), np.sum(flux_mask_red))
        print(ew_blue + ew_red, bosman_ew[i_babs])

    abs_mask_all = np.sum(bluered_mask_all, axis=0).astype(bool)
    abs_mask_gpm = np.invert(abs_mask_all)
    bluered_mask_gpm = np.invert(bluered_mask_all)

    return bluered_mask_gpm, abs_mask_gpm

######################## old stuffs ########################
"""
fitsfile_list = ['/Users/suksientie/Research/data_redux/wavegrid_vel/J0313-1806/vel1234_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/wavegrid_vel/J1342+0928/vel123_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/wavegrid_vel/J0252-0503/vel12_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/wavegrid_vel/J0038-1527/vel1_tellcorr_pad.fits', \
                 '/Users/suksientie/Research/data_redux/wavegrid_vel/J0038-0653/vel1_tellcorr_pad.fits']

def old_init():
    # masked
    norm_good_flux_all = []
    norm_good_std_all = []
    good_ivar_all = []
    good_wave_all = []
    vel_data_all = []
    noise_all = []

    for iqso, fitsfile in enumerate(fitsfile_list):
        #wave, flux, ivar, mask, std, fluxfit, outmask, sset, tell = mutils.extract_and_norm(fitsfile, everyn_break_list[iqso], qso_namelist[iqso])
        wave, flux, ivar, mask, std, fluxfit, outmask, sset, tell = mutils.extract_and_norm(fitsfile, everyn_break_list[iqso], '')

        # masks
        redshift_mask = wave[outmask] <= (2800 * (1 + qso_zlist[iqso]))  # removing spectral region beyond qso redshift
        obs_wave_max = (2800 - exclude_restwave) * (1 + qso_zlist[iqso])
        proximity_zone_mask = wave[outmask] < obs_wave_max

        # masked quantities
        good_wave = wave[outmask][redshift_mask * proximity_zone_mask]
        good_flux = flux[outmask][redshift_mask * proximity_zone_mask]
        good_ivar = ivar[outmask][redshift_mask * proximity_zone_mask]
        good_std = std[outmask][redshift_mask * proximity_zone_mask]
        vel_data = mutils.obswave_to_vel_2(good_wave)

        norm_good_flux = good_flux / fluxfit[redshift_mask * proximity_zone_mask]
        norm_good_std = good_std / fluxfit[redshift_mask * proximity_zone_mask]

        norm_good_flux_all.append(norm_good_flux)
        norm_good_std_all.append(norm_good_std)
        good_ivar_all.append(good_ivar)
        good_wave_all.append(good_wave)
        vel_data_all.append(vel_data)
        noise_all.append(rand.normal(0, norm_good_std))

    return norm_good_flux_all, norm_good_std_all, good_ivar_all, vel_data_all, good_wave_all, noise_all

def bosman_J1120_old(npix_ls, plot=False):
    # npix_ls = list of npix to select around absorbers; must match length of "bosman_abs", i.e. one for each abs

    raw_data_out, _, all_masks_out = mutils.init_onespec(9, 'all')
    wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
    norm_flux = flux/fluxfit

    bosman_abs = [6.1711, 6.21845, 6.40671]
    bosman_ew = [0.258, 0.139, 0.094]

    iblue_all = []
    ired_all = []
    for i_babs, babs in enumerate(bosman_abs):

        wave_blue = 2796 * (1 + babs)
        wave_red = 2804 * (1 + babs)
        
        #npix = npix_ls[i_babs]
        #wm = (wave <= wave_blue + npix) * (wave >= wave_blue - npix)
        #flux_mask_blue = norm_flux[wm] < 1
        #flux_mask_red = norm_flux[wm] < 1

        #rest_wb = wave[wm][flux_mask_blue]/(1 + babs)
        #rest_wr = wave[wm][flux_mask_red] / (1 + babs)

        #ew_blue = integrate.simps(1.0 - norm_flux[wm][flux_mask_blue], rest_wb)
        #ew_red = integrate.simps(1.0 - norm_flux[wm][flux_mask_red], rest_wr)
        
        iblue = np.argmin(np.abs(wave - wave_blue))
        ired = np.argmin(np.abs(wave - wave_red))

        npix = npix_ls[i_babs]
        flux_mask_blue = norm_flux[iblue-npix:iblue+npix] < 1.
        flux_mask_red = norm_flux[ired - npix:ired + npix] < 1.

        rest_wb = wave[iblue-npix:iblue+npix][flux_mask_blue]/(1 + babs)
        rest_wr = wave[ired - npix:ired + npix][flux_mask_red] / (1 + babs)

        ew_blue = integrate.simps(1.0 - norm_flux[iblue-npix:iblue+npix][flux_mask_blue], rest_wb)
        ew_red = integrate.simps(1.0 - norm_flux[ired - npix:ired + npix][flux_mask_red], rest_wr)

        print(ew_blue + ew_red, bosman_ew[i_babs])

        iblue_all.append(iblue)
        ired_all.append(ired)

    if plot:
        plt.plot(wave, norm_flux, drawstyle='steps-mid')
        for i in range(len(iblue_all)):
            plt.axvspan(wave[iblue_all[i]-npix], wave[iblue_all[i]+npix], facecolor='b', alpha=0.5)
            plt.axvspan(wave[ired_all[i] - npix], wave[ired_all[i] + npix], facecolor='r', alpha=0.5)

        plt.ylim([-0.5, 2.5])
        plt.show()

    return iblue_all, ired_all
"""