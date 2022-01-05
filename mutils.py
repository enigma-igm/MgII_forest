'''
Functions here:
    - plot_allspec
    - obswave_to_vel
    - extract_data
    - continuum_normalize
    - scipy_spline
    - custom_mask_J0313
    - custom_mask_J1342
    - custom_mask_J0038
    - extract_and_norm
'''

import sys
sys.path.append('/Users/suksientie/codes/enigma')
sys.path.append('/Users/suksientie/Research/data_redux')
sys.path.append('/Users/suksientie/Research/CIV_forest')
sys.path.append('/home/sstie/codes/PypeIt') # for running on IGM cluster
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from astropy.io import fits
from pypeit.core.fitting import iterfit, robust_fit
from pypeit import utils as putils
import scipy.interpolate as interpolate
from astropy import constants as const
from astropy.table import Table
from enigma.reion_forest import utils

def plot_allspec(wave_arr, flux_arr, qso_namelist, qso_zlist, vel_unit=False, vel_zeropoint=True, wave_zeropoint_value=None):

    wave_min, wave_max = 19500, 24000

    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(111)
    zmin = wave_min / 2800 - 1
    zmax = wave_max / 2800 - 1
    ymin = 0.5
    ymax = 3.5

    for i in range(len(wave_arr)):
        if vel_unit:
            x_arr = obswave_to_vel(wave_arr[i], vel_zeropoint=vel_zeropoint, wave_zeropoint_value=wave_zeropoint_value)
            xmin, xmax = np.min(x_arr.value), np.max(x_arr.value)
            xlabel = 'v - v_zeropoint (km/s)'
        else:
            x_arr = wave_arr[i]
            xmin, xmax = wave_min, wave_max
            xlabel = 'obs wavelength (A)'

        x_mask = wave_arr[i] <= (2800 * (1 + qso_zlist[i]))
        yoffset = i * 0.5
        ax1.plot(x_arr[x_mask], flux_arr[i][x_mask] + yoffset, label=qso_namelist[i], drawstyle='steps-mid')

    ax1.set_xlim([xmin, xmax])
    ax1.set_ylim([ymin, ymax])
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('normalized flux')
    ax1.legend()

    if not vel_unit:
        atwin = ax1.twiny()
        atwin.set_xlabel('absorber redshift')
        atwin.axis([zmin, zmax, ymin, ymax])
        atwin.tick_params(top=True, axis="x")
        atwin.xaxis.set_minor_locator(AutoMinorLocator())

    plt.tight_layout()
    plt.show()

def obswave_to_vel(wave_arr, vel_zeropoint=False, wave_zeropoint_value=None):
    # wave in Angstrom
    zabs_mean = wave_arr/2800 - 1 # between the two doublet

    cosmo = FlatLambdaCDM(H0=100.0 * 0.67, Om0=0.3192, Ob0=0.04964) # Nyx cosmology
    comov_dist = cosmo.comoving_distance(zabs_mean)
    Hz = cosmo.H(zabs_mean)
    a = 1 / (1 + zabs_mean)
    vel = comov_dist * a * Hz

    if vel_zeropoint:
        if wave_zeropoint_value != None:
            min_wave = wave_zeropoint_value
        else:
            min_wave = np.min(wave_arr)

        min_zabs = min_wave / 2800 - 1
        min_vel = cosmo.comoving_distance(min_zabs) * (1 / (1 + min_zabs)) * cosmo.H(min_zabs)
        vel = vel - min_vel

    return vel.value

def obswave_to_vel_2(wave_arr):

    # dv =  c * d(log_lambda), where log is natural log
    c_kms = const.c.to('km/s').value
    log10_wave = np.log10(wave_arr)
    diff_log10_wave = np.diff(log10_wave) # d(log10_lambda)
    diff_log10_wave = np.append(diff_log10_wave, diff_log10_wave[-1]) # appending the last value twice to make array same size as wave_arr
    dv = c_kms * np.log(10) * diff_log10_wave
    #vel = np.zeros(len(wave_arr))
    #vel[1:] = np.cumsum(dv)
    vel = np.cumsum(dv)

    return vel

def extract_data(fitsfile):
    data = fits.open(fitsfile)[1].data
    #wave_arr = data['wave'].astype('float64')
    wave_arr = data['wave_grid_mid'].astype('float64')
    flux_arr = data['flux'].astype('float64')
    ivar_arr = data['ivar'].astype('float64')
    mask_arr = data['mask'].astype('bool')
    std_arr = np.sqrt(putils.inverse(ivar_arr))

    return wave_arr, flux_arr, ivar_arr, mask_arr, std_arr

def continuum_normalize(wave_arr, flux_arr, ivar_arr, mask_arr, std_arr, nbkpt, bkpt=None, plot=False):

    #if bkpt == None:
    #    bkpt = wave_arr[::nbkpt]
    #   bkpt = bkpt.astype('float64')

    #(sset, outmask) = iterfit(wave_arr, flux_arr, invvar=ivar_arr, inmask=mask_arr, upper=3, lower=3, x2=None,
    #            maxiter=10, nord=4, bkpt=bkpt, fullbkpt=None)

    (sset, outmask) = iterfit(wave_arr, flux_arr, invvar=ivar_arr, inmask=mask_arr, upper=3, lower=3, x2=None,
                              maxiter=10, nord=4, bkpt=None, fullbkpt=None, kwargs_bspline = {'everyn': nbkpt})

    _, flux_fit = sset.fit(wave_arr[outmask], flux_arr[outmask], ivar_arr[outmask])

    if plot:
        plt.plot(wave_arr, flux_arr, c='b', alpha=0.3, drawstyle='steps-mid')
        plt.plot(wave_arr[outmask], flux_arr[outmask], c='b', drawstyle='steps-mid')
        plt.plot(wave_arr[outmask], flux_fit, c='r', drawstyle='steps-mid')
        plt.plot(wave_arr[outmask], std_arr[outmask], c='k', drawstyle='steps-mid')
        plt.ylim([0, 0.50])
        plt.show()

    return flux_fit, outmask, sset

def scipy_spline(fitsfile):
    data = fits.open(fitsfile)[1].data
    wave_arr = data['wave']
    flux_arr = data['flux']

    nknots = 6
    quartile_loc = np.linspace(0, 1, nknots + 2)[1:-1]
    knots_wave = np.quantile(wave_arr, quartile_loc)
    knots, coeff, k = interpolate.splrep(wave_arr, flux_arr, t=knots_wave, k=4)

    spline = interpolate.BSpline(knots, coeff, k, extrapolate=False)
    flux_spline = spline(wave_arr)

    plt.plot(wave_arr, flux_arr, 'k')
    plt.plot(wave_arr, flux_spline, 'r')
    plt.show()

def save_data_sigma(fitsfile_list, everyn_breakpoint, savefits):
    # 11/22/2021: maybe obsolete

    hdulist = fits.HDUList()
    all_norm_std_flat = []
    for i, fitsfile in enumerate(fitsfile_list):
        wave, flux, ivar, mask, std, fluxfit, outmask, sset = extract_and_norm(fitsfile, everyn_breakpoint)
        good_wave, good_flux, good_std = wave[outmask], flux[outmask], std[outmask]
        norm_good_std = good_std / fluxfit
        name = fitsfile.split('/')[-1].split('_')[0]
        hdulist.append(fits.ImageHDU(data=norm_good_std, name=name))
        all_norm_std_flat.extend(norm_good_std)

    hdulist.append(fits.ImageHDU(data=all_norm_std_flat, name='flattened'))
    hdulist.writeto(savefits, overwrite=True)

def temp(z, delta_z):
    vside_lores = 6577.75
    c_light = (const.c.to('km/s')).value
    z_min = z - delta_z
    z_eff = (z + z_min) / 2.0
    dv_path = (z - z_min) / (1.0 + z_eff) * c_light
    nqsos = 1
    npath_float = nqsos * dv_path / vside_lores
    npath = int(np.round(npath_float))
    print(npath)

###### by-eye strong absorbers masks ######
def custom_mask_J0313(plot=False):
    #fitsfile = '/Users/suksientie/Research/data_redux/mgii_stack_fits/J0313-1806_stacked_coadd_tellcorr.fits'
    fitsfile =  '/Users/suksientie/Research/data_redux/wavegrid_vel/J0313-1806/vel123_coadd_tellcorr.fits'
    wave, flux, ivar, mask, std = extract_data(fitsfile)

    mask_wave1 = [19815, 19825]
    mask_wave2 = [19865, 19870]
    mask_wave3 = [23303, 23325]
    mask_wave4 = [23370, 23387]

    all_mask_wave = [mask_wave1, mask_wave2, mask_wave3, mask_wave4]
    out_gpm = []
    for mask_wave_i in all_mask_wave:
        a = mask_wave_i[0] < wave
        b = wave < mask_wave_i[1]
        gpm = np.invert(a * b)
        out_gpm.append(gpm)
        mask *= gpm

    if plot:
        plt.plot(wave, flux, c='b', alpha=0.3, drawstyle='steps-mid')
        plt.plot(wave, std, c='k', alpha=0.3, drawstyle='steps-mid')
        plt.plot(wave[mask], flux[mask], c='b', drawstyle='steps-mid')
        plt.plot(wave[mask], std[mask], c='k', drawstyle='steps-mid')
        plt.show()

    return wave, flux, ivar, mask, std, out_gpm

def custom_mask_J1342(plot=False):
    #fitsfile = '/Users/suksientie/Research/data_redux/mgii_stack_fits/J1342+0928_stacked_coadd_tellcorr.fits'
    fitsfile = '/Users/suksientie/Research/data_redux/wavegrid_vel/J1342+0928/vel123_coadd_tellcorr.fits'
    wave, flux, ivar, mask, std = extract_data(fitsfile)

    # visually-identified strong absorbers
    mask_wave1 = [21920, 21940]
    mask_wave2 = [21972, 22000]
    mask_wave3 = [20320, 20335]
    mask_wave4 = [20375, 20400]

    all_mask_wave = [mask_wave1, mask_wave2, mask_wave3, mask_wave4]
    out_gpm = []
    for mask_wave_i in all_mask_wave:
        a = mask_wave_i[0] < wave
        b = wave < mask_wave_i[1]
        gpm = np.invert(a * b)
        out_gpm.append(gpm)
        mask *= gpm

    if plot:
        plt.plot(wave, flux, c='b', alpha=0.3, drawstyle='steps-mid')
        plt.plot(wave, std, c='k', alpha=0.3, drawstyle='steps-mid')
        plt.plot(wave[mask], flux[mask], c='b', drawstyle='steps-mid')
        plt.plot(wave[mask], std[mask], c='k', drawstyle='steps-mid')
        plt.show()

    return wave, flux, ivar, mask, std, out_gpm

def custom_mask_J0038(plot=False):
    #fitsfile = '/Users/suksientie/Research/data_redux/2010_done/Redux/J0038-1527_201024_done/J0038-1527_coadd_tellcorr.fits'
    fitsfile = '/Users/suksientie/Research/data_redux/wavegrid_vel/J0038-1527/vel1_tellcorr.fits'
    wave, flux, ivar, mask, std = extract_data(fitsfile)

    # visually-identified strong absorbers
    mask_wave1 = [19777, 19796]
    mask_wave2 = [19828, 19855]

    all_mask_wave = [mask_wave1, mask_wave2]
    out_gpm = []
    for mask_wave_i in all_mask_wave:
        a = mask_wave_i[0] < wave
        b = wave < mask_wave_i[1]
        gpm = np.invert(a * b)
        out_gpm.append(gpm)
        mask *= gpm

    if plot:
        plt.plot(wave, flux, c='b', alpha=0.3, drawstyle='steps-mid')
        plt.plot(wave, std, c='k', alpha=0.3, drawstyle='steps-mid')
        plt.plot(wave[mask], flux[mask], c='b', drawstyle='steps-mid')
        plt.plot(wave[mask], std[mask], c='k', drawstyle='steps-mid')
        plt.show()

    return wave, flux, ivar, mask, std, out_gpm

def extract_and_norm(fitsfile, everyn_bkpt):
    # combining extract_data() and continuum_normalize()
    wave, flux, ivar, mask, std = extract_data(fitsfile)
    qso_name = fitsfile.split('/')[-1].split('_')[0]

    if qso_name == 'J0313-1806':
        wave, flux, ivar, mask, std, out_gpm = custom_mask_J0313()

    elif qso_name == 'J1342+0928':
        wave, flux, ivar, mask, std, out_gpm = custom_mask_J1342()

    elif qso_name == 'J0252-0503':
        pass

    elif qso_name == 'J0038-1527':
        wave, flux, ivar, mask, std, out_gpm = custom_mask_J0038()

    fluxfit, outmask, sset = continuum_normalize(wave, flux, ivar, mask, std, everyn_bkpt)

    return wave, flux, ivar, mask, std, fluxfit, outmask, sset

def init_skewers_compute_model_grid():
    file = 'ran_skewers_z75_OVT_xHI_0.50_tau.fits'
    params = Table.read(file, hdu=1)
    skewers = Table.read(file, hdu=2)

    fwhm = 90 # 83
    sampling = 3
    logZ = -3.50
    vel_lores, (flux_lores, flux_lores_igm, flux_lores_cgm, _, _), vel_hires, (
    flux_hires, flux_hires_igm, flux_hires_cgm, _, _), \
    (oden, v_los, T, xHI), cgm_tuple = utils.create_mgii_forest(params, skewers, logZ, fwhm, sampling=sampling)

    vmin_corr, vmax_corr, dv_corr = 10, 2000, 60

    mean_flux_nless = np.mean(flux_lores)
    delta_f_nless = (flux_lores - mean_flux_nless) / mean_flux_nless
    (vel_mid, xi_nless, npix, xi_nless_zero_lag) = utils.compute_xi(delta_f_nless, vel_lores, vmin_corr, vmax_corr, dv_corr)
    xi_mean = np.mean(xi_nless, axis=0)

    return vel_lores, flux_lores, vel_mid, xi_mean

###### old scripts ######
def plot_allspec_old(wave_arr, flux_arr):

    fig, ax1 = plt.subplots()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    for i in range(len(wave_arr)):
        yoffset = 1 + i*0.2
        if i == 0:
            ax1.plot(wave_arr[i], flux_arr[i] + yoffset)
        else:
            plt.plot(wave_arr[i], flux_arr[i] + yoffset)

    new_tick_locations = np.arange(20000, 24000, 1000) # wave_min, wave_max = 19531.613598001197, 23957.550659619443
    zabs_mean = new_tick_locations/2800 - 1

    def tick_function(X):
        zabs_mean = X/2800 - 1
        return ["%.2f" % z for z in zabs_mean]

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels(tick_function(new_tick_locations))
    plt.show()

def continuum_normalize_old(fitsfile, nbkpt, bkpt=None):

    data = fits.open(fitsfile)[1].data
    wave_arr = data['wave'].astype('float64')
    flux_arr = data['flux'].astype('float64')
    ivar_arr = data['ivar'].astype('float64')
    mask_arr = data['mask'].astype('bool')

    if bkpt == None:
        bkpt = wave_arr[::nbkpt]
        bkpt = bkpt.astype('float64')

    (sset, outmask) = iterfit(wave_arr, flux_arr, invvar=ivar_arr, inmask=mask_arr, upper=3, lower=3, x2=None,
                maxiter=10, nord=4, bkpt=bkpt, fullbkpt=None)

    _, flux_fit = sset.fit(wave_arr, flux_arr, ivar_arr)
    plt.plot(wave_arr, flux_arr)
    plt.plot(wave_arr, flux_fit, 'r')
    plt.ylim([0, 0.50])
    plt.show()

    """
    obj_model = data['obj_model'] # continuum x mean flux ?
    telluric = data['telluric']

    std_arr = np.sqrt(putils.inverse(ivar_arr))

    transmission = flux_arr/obj_model
    norm_std = std_arr/obj_model

    if plot:
        plt.plot(wave_arr, transmission, drawstyle='steps-mid')
        plt.plot(wave_arr, std_arr, drawstyle='steps-mid')
        plt.xlabel('obs-wavelength (A)')
        plt.ylabel('continuum-normalized flux')
        plt.ylim([0, 1.8])
        plt.show()

    return wave_arr, flux_arr, ivar_arr, std_arr, mask_arr, transmission, norm_std
    """
