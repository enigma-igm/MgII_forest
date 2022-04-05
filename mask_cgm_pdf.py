import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('/Users/suksientie/codes/enigma')
sys.path.append('/Users/suksientie/Research/data_redux')
sys.path.append('/Users/suksientie/Research/CIV_forest')
from enigma.reion_forest.mgii_find import MgiiFinder
from enigma.reion_forest import utils
from pypeit import utils as putils
import misc # from CIV_forest
from scripts import rdx_utils
import mutils
from matplotlib.ticker import AutoMinorLocator

########## global variables ##########
fwhm = 90
seed = 4101877 # seed for generating random realizations of qso noise
if seed != None:
    rand = np.random.RandomState(seed)
else:
    rand = np.random.RandomState()

fitsfile_list = ['/Users/suksientie/Research/data_redux/wavegrid_vel/J0313-1806/vel1234_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/wavegrid_vel/J1342+0928/vel123_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/wavegrid_vel/J0252-0503/vel12_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/wavegrid_vel/J0038-1527/vel1_tellcorr_pad.fits']

qso_namelist = ['J0313-1806', 'J1342+0928', 'J0252-0503', 'J0038-1527']
qso_zlist = [7.642, 7.541, 7.001, 7.034] # precise redshifts from Yang+2021
everyn_break_list = [20, 20, 20, 20] # placing a breakpoint at every 20-th array element (more docs in mutils.continuum_normalize)
                                     # this results in dwave_breakpoint ~ 40 A --> 600 km/s
exclude_restwave = 1216 - 1185 # excluding proximity zones; see mutils.qso_exclude_proximity_zone

# chi PDF
fwhm = 90
signif_thresh = 4.0
signif_mask_dv = 300.0 # value used in Hennawi+2021
signif_mask_nsigma = 8 # chi threshold
one_minF_thresh = 0.3 # flux threshold
nbins_chi = 81
sig_min = 1e-2
sig_max = 100.0
dsig_bin = np.ediff1d(np.linspace(sig_min, sig_max, nbins_chi))
#print(dsig_bin)

# flux PDF
nbins_flux, oneminf_min, oneminf_max = 101, 1e-5, 1.0  # gives d(oneminf) = 0.01

def init():
    # masked
    norm_good_flux_all = []
    norm_good_std_all = []
    good_ivar_all = []
    good_wave_all = []
    vel_data_all = []
    noise_all = []

    for iqso, fitsfile in enumerate(fitsfile_list):
        wave, flux, ivar, mask, std, fluxfit, outmask, sset, tell = mutils.extract_and_norm(fitsfile, everyn_break_list[iqso], qso_namelist[iqso])

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


def flux_pdf(norm_good_flux_all, noise_all):

    all_transmission = []
    all_noise = []
    nqso = len(norm_good_flux_all)

    # plot PDF for each qso
    for i in range(nqso):

        norm_good_flux, noise = norm_good_flux_all[i], noise_all[i]
        flux_bins, flux_pdf_tot = utils.pdf_calc(1.0 - norm_good_flux, oneminf_min, oneminf_max, nbins_flux)
        plt.plot(flux_bins, flux_pdf_tot, drawstyle='steps-mid', alpha=0.5, label=qso_namelist[i])

        all_transmission.extend(norm_good_flux)
        all_noise.extend(noise)

    # plot PDF for all qso
    all_transmission = np.array(all_transmission)
    all_noise = np.array(all_noise)

    flux_bins, flux_pdf_tot = utils.pdf_calc(1.0 - all_transmission, oneminf_min, oneminf_max, nbins_flux)
    _, flux_pdf_noise = utils.pdf_calc(all_noise, oneminf_min, oneminf_max, nbins_flux)
    # _, flux_pdf_gnoise = utils.pdf_calc(gaussian_noise, oneminf_min, oneminf_max, nbins)

    plt.plot(flux_bins, flux_pdf_noise, drawstyle='steps-mid', color='b', alpha=0.7, label='all noise')
    plt.plot(flux_bins, flux_pdf_tot, drawstyle='steps-mid', color='k', alpha=1.0, lw=2, label='all spectra')
    # plt.plot(flux_bins, flux_pdf_gnoise, drawstyle='steps-mid', color='y', alpha=1.0, lw=2, label='gaussian noise')
    plt.axvline(one_minF_thresh, color='k', ls='--', lw=2)
    plt.axvspan(one_minF_thresh, oneminf_max, facecolor='k', alpha=0.2, label='masked')

    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('1$-$F')
    plt.ylabel('PDF')
    plt.tight_layout()
    plt.show()

def chi_pdf(vel_data_all, norm_good_flux_all, good_ivar_all, noise_all):

    all_chi = []
    all_chi_noise = []
    mgii_tot_all = []
    nqso = len(norm_good_flux_all)

    # plot PDF for each qso
    for i in range(nqso):
        vel_data = vel_data_all[i]
        norm_good_flux = norm_good_flux_all[i]
        good_ivar = good_ivar_all[i]
        noise = noise_all[i]

        # reshaping to be compatible with MgiiFinder
        norm_good_flux = norm_good_flux.reshape((1, len(norm_good_flux)))
        good_ivar = good_ivar.reshape((1, len(good_ivar)))
        noise = noise.reshape((1, len(noise)))

        mgii_tot = MgiiFinder(vel_data, norm_good_flux, good_ivar, fwhm, signif_thresh,
                              signif_mask_nsigma=signif_mask_nsigma,
                              signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh)
        mgii_tot_all.append(mgii_tot)

        # plotting chi PDF of each QSO
        sig_bins, sig_pdf_tot = utils.pdf_calc(mgii_tot.signif, sig_min, sig_max, nbins_chi)
        plt.plot(sig_bins, sig_pdf_tot, drawstyle='steps-mid', alpha=0.5, label=qso_namelist[i])

        # chi PDF of pure noise
        mgii_noise = MgiiFinder(vel_data, 1.0 + noise, good_ivar, fwhm, signif_thresh,
                                signif_mask_nsigma=signif_mask_nsigma,
                                signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh)

        all_chi.extend(mgii_tot.signif[0])
        all_chi_noise.extend(mgii_noise.signif[0])

    # plot PDF of all qso
    all_chi = np.array(all_chi)
    all_chi_noise = np.array(all_chi_noise)

    sig_bins, sig_pdf_tot = utils.pdf_calc(all_chi, sig_min, sig_max, nbins_chi)
    _, sig_pdf_noise = utils.pdf_calc(all_chi_noise, sig_min, sig_max, nbins_chi)

    plt.plot(sig_bins, sig_pdf_noise, drawstyle='steps-mid', color='b', alpha=0.7, label='all noise')
    plt.plot(sig_bins, sig_pdf_tot, drawstyle='steps-mid', color='k', alpha=1.0, lw=2, label='all spectra')
    plt.axvline(signif_mask_nsigma, color='k', ls='--', lw=2)
    plt.axvspan(signif_mask_nsigma, sig_max, facecolor='k', alpha=0.2, label='masked')

    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\chi$')
    plt.ylabel('PDF')
    plt.tight_layout()
    plt.show()

    return mgii_tot_all

def plot_masked_onespec(mgii_tot_all, vel_data_all, norm_good_flux_all, norm_good_std_all, iqso):
    # vel_data_all can be substituted with good_wave_all

    mgii_tot = mgii_tot_all[iqso]
    vel_data = vel_data_all[iqso]
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

    fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 6), sharex=True)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.98, top=0.93, wspace=0, hspace=0.)
    flux_min, flux_max = 0., 1.9
    chi_min, chi_max = -5.0, 23

    ax1.set_title(qso_name + ' (z = %0.2f)' % qso_z, fontsize=18)
    ax1.plot(vel_data, norm_good_flux, drawstyle='steps-mid', color='k', linewidth=1.5, zorder=1)
    ax1.plot(vel_data, norm_good_std, drawstyle='steps-mid', color='k', linewidth=1.0, alpha=0.5)
    ax1.axhline(y = 1 - one_minF_thresh, color='green', linestyle='dashed', linewidth=1.5, label=r'$1 - \rm{F} = %0.1f$ (%0.2f masked)' % (one_minF_thresh, f_mask_frac))
    ax1.plot(vel_data[s_mask], norm_good_flux[s_mask], color='magenta', markersize=7, markeredgewidth=2.5, linestyle='none', alpha=0.7, zorder=2, marker='|')
    ax1.plot(vel_data[f_mask], norm_good_flux[f_mask], color = 'green', markersize = 7, markeredgewidth = 2.5, linestyle = 'none', alpha = 0.7, zorder = 3, marker = '|')
    ax1.set_ylabel(r'$F_{\mathrm{norm}}$', fontsize=18)
    ax1.set_xlim([vel_data.min(), vel_data.max()])
    ax1.set_ylim([flux_min, flux_max])
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.legend(fontsize=13)

    neg = np.zeros_like(vel_data) - 100
    ax2.plot(vel_data, mgii_tot.signif[0], drawstyle='steps-mid', color='k')
    ax2.axhline(y=signif_mask_nsigma, color='magenta', linestyle='dashed', linewidth=1.5, label=r'$\chi$ = %d (%0.2f masked)' % (signif_mask_nsigma, s_mask_frac))
    ax2.fill_between(vel_data, neg, mgii_tot.signif[0], where=s_mask, step = 'mid', facecolor = 'magenta', alpha = 0.5)
    #ax2.set_xlabel('v (km/s)', fontsize=18)
    ax2.set_xlabel('obs wavelength (A)', fontsize=18)
    ax2.set_ylabel(r'$\chi$', fontsize=18)
    ax2.set_xlim([vel_data.min(), vel_data.max()])
    ax2.set_ylim([chi_min, chi_max])
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.legend(fontsize=13)

    plt.tight_layout()
    plt.show()


