from astropy.io import fits
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

mosfire_res = 3610 # K-band for 0.7" slit (https://www2.keck.hawaii.edu/inst/mosfire/grating.html)
fwhm = misc.convert_resolution(mosfire_res).value

"""
fitsfile_list = ['/Users/suksientie/Research/data_redux/mgii_stack_fits/J0313-1806_stacked_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/mgii_stack_fits/J1342+0928_stacked_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/mgii_stack_fits/J0252-0503_stacked_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/2010_done/Redux/J0038-1527_201024_done/J0038-1527_coadd_tellcorr.fits']
qso_zlist = [7.6, 7.54, 7.0, 7.0]
"""

fitsfile = '/Users/suksientie/Research/data_redux/2010_done/Redux/J0038-1527_201024_done/J0038-1527_coadd_tellcorr.fits'
qso_z = 7.
wave_zeropoint_value = (1 + 6) * 2800
vel_zeropoint = True
everyn_break = 20

signif_thresh = 4.0
signif_mask_dv = 300.0
signif_mask_nsigma = 8
one_minF_thresh = 0.3
nbins = 81
sig_min = 1e-2
sig_max = 100.0

def mask_cgm_old():
    #wave, flux, ivar, std, mask, cont_flux, norm_std = rdx_utils.continuum_normalize(fitsfile, qso_z)
    wave, flux, ivar, mask, std = mutils.extract_data(fitsfile)
    cont_flux = flux
    norm_std = std

    vel = mutils.obswave_to_vel(wave[mask], vel_zeropoint=vel_zeropoint, wave_zeropoint_value=wave_zeropoint_value)
    x_mask = wave[mask] <= (2800 * (1 + qso_z))

    # masking bad pixels
    vel = vel[x_mask]
    cont_flux = cont_flux[mask][x_mask]
    ivar = ivar[mask][x_mask]
    norm_std = norm_std[mask][x_mask]

    # reshaping to be compatible with MgiiFinder
    cont_flux = cont_flux.reshape((1, len(cont_flux)))
    ivar = ivar.reshape((1, len(ivar)))
    norm_std = norm_std.reshape((1, len(norm_std)))

    mgii_tot = MgiiFinder(vel, cont_flux, ivar, fwhm, signif_thresh, signif_mask_nsigma=signif_mask_nsigma, signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh)

    return mgii_tot, vel, cont_flux

def mask_cgm():

    wave, flux, ivar, mask, std, fluxfit, outmask = mutils.extract_and_norm(fitsfile, everyn_break)
    good_wave, good_flux, good_std = wave[outmask], flux[outmask], std[outmask]
    good_ivar = ivar[outmask]
    norm_good_flux = good_flux / fluxfit
    norm_good_std = good_std / fluxfit
    vel = mutils.obswave_to_vel(good_wave, vel_zeropoint=vel_zeropoint, wave_zeropoint_value=wave_zeropoint_value)

    # reshaping to be compatible with MgiiFinder
    norm_good_flux = norm_good_flux.reshape((1, len(norm_good_flux)))
    good_ivar = good_ivar.reshape((1, len(good_ivar)))
    norm_good_std = norm_good_std.reshape((1, len(norm_good_std)))

    mgii_tot = MgiiFinder(vel, norm_good_flux, good_ivar, fwhm, signif_thresh, signif_mask_nsigma=signif_mask_nsigma,
                          signif_mask_dv=signif_mask_dv, one_minF_thresh=one_minF_thresh)

    return mgii_tot, vel, norm_good_flux

def plot_chi(mgii_tot, vel, cont_flux):

    s_mask = np.invert(mgii_tot.signif_gpm[0])
    s_mask_frac = np.sum(s_mask)/len(s_mask)
    print(s_mask_frac)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 6), sharex=True)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.98, top=0.93, wspace=0, hspace=0.)
    chi_min, chi_max = -0.2, 230

    ax1.set_title(fitsfile.split('/')[-1].split('_')[0] + ' (z = %0.2f)' % qso_z, fontsize=18)
    ax1.plot(vel, cont_flux[0], drawstyle='steps-mid', color='k', linewidth=1.5, zorder=1)
    #ax1.axhline(y = 1 - one_minF_thresh, color='green', linestyle='dashed', linewidth=1.5)
    ax1.plot(vel[s_mask], cont_flux[0][s_mask], color='magenta', markersize=7, markeredgewidth=2.5, linestyle='none', alpha=0.7, zorder=2, marker='|')
    #ax1.plot(vel[f_mask], cont_flux[0][f_mask], color = 'green', markersize = 7, markeredgewidth = 2.5, linestyle = 'none', alpha = 0.7, zorder = 3, marker = '|')
    ax1.set_ylabel('Raw flux', fontsize=18)
    #ax1.set_ylim([flux_min, flux_max])
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())

    neg = np.zeros_like(vel) - 100
    ax2.plot(vel, mgii_tot.signif[0], drawstyle='steps-mid', color='k')
    ax2.axhline(y=signif_mask_nsigma, color='magenta', linestyle='dashed', linewidth=1.5)
    ax2.fill_between(vel, neg, mgii_tot.signif[0], where=s_mask, step = 'mid', facecolor = 'magenta', alpha = 0.5)
    ax2.set_xlabel('v (km/s)', fontsize=18)
    ax2.set_ylabel(r'$\chi$', fontsize=18)
    ax2.set_ylim([chi_min, chi_max])
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())

    plt.show()

def plot_both(mgii_tot, vel, cont_flux):

    f_mask = np.invert(mgii_tot.flux_gpm[0])
    s_mask = np.invert(mgii_tot.signif_gpm[0])

    f_mask_frac = np.sum(f_mask)/len(f_mask)
    s_mask_frac = np.sum(s_mask)/len(s_mask)
    print(f_mask_frac, s_mask_frac)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 6), sharex=True)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.98, top=0.93, wspace=0, hspace=0.)
    flux_min, flux_max = 0.2, 1.9
    chi_min, chi_max = -0.2, 23

    ax1.set_title(fitsfile.split('/')[-1].split('_')[0] + ' (z = %0.2f)' % qso_z, fontsize=18)
    ax1.plot(vel, cont_flux[0], drawstyle='steps-mid', color='k', linewidth=1.5, zorder=1)
    ax1.axhline(y = 1 - one_minF_thresh, color='green', linestyle='dashed', linewidth=1.5)
    ax1.plot(vel[s_mask], cont_flux[0][s_mask], color='magenta', markersize=7, markeredgewidth=2.5, linestyle='none', alpha=0.7, zorder=2, marker='|')
    ax1.plot(vel[f_mask], cont_flux[0][f_mask], color = 'green', markersize = 7, markeredgewidth = 2.5, linestyle = 'none', alpha = 0.7, zorder = 3, marker = '|')
    ax1.set_ylabel(r'$F_{\mathrm{norm}}$', fontsize=18)
    ax1.set_ylim([flux_min, flux_max])
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())

    neg = np.zeros_like(vel) - 100
    ax2.plot(vel, mgii_tot.signif[0], drawstyle='steps-mid', color='k')
    ax2.axhline(y=signif_mask_nsigma, color='magenta', linestyle='dashed', linewidth=1.5)
    ax2.fill_between(vel, neg, mgii_tot.signif[0], where=s_mask, step = 'mid', facecolor = 'magenta', alpha = 0.5)
    ax2.set_xlabel('v (km/s)', fontsize=18)
    ax2.set_ylabel(r'$\chi$', fontsize=18)
    ax2.set_ylim([chi_min, chi_max])
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())

    plt.show()