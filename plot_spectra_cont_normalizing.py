import numpy as np
from astropy.cosmology import FlatLambdaCDM
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from astropy.io import fits
import sys
sys.path.append('/Users/suksientie/Research/data_redux')
from scripts import rdx_utils
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
vel_unit = False # always off for now
show_breakpoints = False # always off for now

### Start plotting
#fitsfile_list = ['/Users/suksientie/Research/data_redux/mgii_stack_fits/J0313-1806_stacked_coadd_tellcorr.fits', \
#                 '/Users/suksientie/Research/data_redux/mgii_stack_fits/J1342+0928_stacked_coadd_tellcorr.fits', \
#                 '/Users/suksientie/Research/data_redux/mgii_stack_fits/J0252-0503_stacked_coadd_tellcorr.fits', \
#                 '/Users/suksientie/Research/data_redux/2010_done/Redux/J0038-1527_201024_done/J0038-1527_coadd_tellcorr.fits']

fitsfile_list = ['/Users/suksientie/Research/data_redux/wavegrid_vel/J0313-1806/vel1234_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/wavegrid_vel/J1342+0928/vel123_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/wavegrid_vel/J0252-0503/vel12_coadd_tellcorr.fits', \
                 '/Users/suksientie/Research/data_redux/wavegrid_vel/J0038-1527/vel1_tellcorr_pad.fits']

qso_namelist =['J0313-1806', 'J1342+0928', 'J0252-0503', 'J0038-1527']
qso_zlist = [7.6, 7.54, 7.0, 7.0]
exclude_restwave = 1216 - 1185 # excluding proximity zones; see mutils.qso_exclude_proximity_zone

fig, (ax1, ax2, ax3, ax4) = plt.subplots(len(fitsfile_list), figsize=(16, 10), sharex=True)
fig.subplots_adjust(left=0.1, bottom=0.07, right=0.96, top=0.93, wspace=0, hspace=0.)
color_ls = ['r', 'g', 'b', 'orange']

if plot_normalized:
    ymin, ymax = -0.05, 1.9
else:
    ymin, ymax = -0.05, 0.55

#wave_min, wave_max = 19500, 24000
wave_min, wave_max = 19300, 24300
zmin, zmax = wave_min / 2800 - 1, wave_max / 2800 - 1
everyn_break_list = [20, 20, 20, 20]

all_wave = []
all_cont_flux = []
all_norm_std = []
all_mask = []
all_xmask = []

for i, fitsfile in enumerate(fitsfile_list):

    wave, flux, ivar, mask, std, tell = mutils.extract_data(fitsfile)
    qso_name = qso_namelist[i]
    z_mask = wave[mask] <= (2800 * (1 + qso_zlist[i]))  # redshift cutoff

    if qso_name == 'J0313-1806':
        wave, flux, ivar, mask, std, out_gpm = mutils.custom_mask_J0313()

    elif qso_name == 'J1342+0928':
        wave, flux, ivar, mask, std, out_gpm = mutils.custom_mask_J1342()

    elif qso_name == 'J0252-0503':
        pass

    elif qso_name == 'J0038-1527':
        wave, flux, ivar, mask, std, out_gpm = mutils.custom_mask_J0038()

    fluxfit, outmask, sset = mutils.continuum_normalize(wave, flux, ivar, mask, std, everyn_break_list[i])
    #print(len(fluxfit), np.sum(outmask))

    good_wave, good_flux, good_std = wave[outmask], flux[outmask], std[outmask]
    norm_good_flux = good_flux/fluxfit
    norm_good_std = good_std/fluxfit
    norm_good_snr = norm_good_flux/norm_good_std
    print(np.mean(norm_good_snr), np.median(norm_good_snr))

    obs_wave_max = (2800 - exclude_restwave) * (1 + qso_zlist[i])

    if vel_unit:
        print("using velocity unit in plot")
        #good_vel = mutils.obswave_to_vel(good_wave, vel_zeropoint=False, wave_zeropoint_value=None) # good wave
        #vel = mutils.obswave_to_vel(wave, vel_zeropoint=False, wave_zeropoint_value=None) # raw wave
        #breakpoints = mutils.obswave_to_vel(sset.breakpoints, vel_zeropoint=False, wave_zeropoint_value=None)
        #good_wave, wave, breakpoints = good_vel / 1e6, vel / 1e6, breakpoints / 1e6

        good_vel = mutils.obswave_to_vel_2(good_wave)
        vel = mutils.obswave_to_vel_2(wave)  # raw wave
        breakpoints = mutils.obswave_to_vel_2(sset.breakpoints)
        good_wave, wave, breakpoints = good_vel, vel, breakpoints
        #print(good_wave[0:3], good_wave[-3:])
        #print(breakpoints[0:3], breakpoints[-3:])

    else:
        breakpoints = sset.breakpoints

    if i == 0: plot_ax = ax1
    elif i == 1: plot_ax = ax2
    elif i == 2: plot_ax = ax3
    elif i == 3: plot_ax = ax4

    if plot_normalized:
        plot_ax.plot(good_wave, norm_good_flux, c=color_ls[i], drawstyle='steps-mid', label=qso_name + ' (z=%0.2f)' % qso_zlist[i])
        plot_ax.plot(good_wave, norm_good_std, c='k', alpha=0.7, drawstyle='steps-mid')
        plot_ax.set_ylabel(r'$F_{\mathrm{norm}}$', fontsize=xylabel_fontsize)
        if show_breakpoints:
            if not vel_unit:
                plot_ax.vlines(breakpoints, ymin=ymin, ymax=ymax, alpha=0.3)

    else:
        plot_ax.plot(wave, flux, c=color_ls[i], alpha=0.3, drawstyle='steps-mid') # not masking strong absorber
        plot_ax.plot(wave, tell*(ymax-0.02), alpha=0.3, drawstyle='steps-mid') # telluric
        plot_ax.plot(good_wave, good_flux, c=color_ls[i], drawstyle='steps-mid', label=qso_name + ' (z=%0.2f)' % qso_zlist[i])
        plot_ax.plot(good_wave, fluxfit, c='k', lw=2.5, drawstyle='steps-mid') #, label='bspline with everyn=%d' % everyn_break_list[i])
        plot_ax.plot(good_wave, good_std, c='k', alpha=0.7, drawstyle='steps-mid')
        if show_breakpoints:
            if not vel_unit:
                plot_ax.vlines(breakpoints, ymin=ymin, ymax=ymax, alpha=0.3) # different zeropoint in breakpoints vs. zeropoint in wave
        plot_ax.set_ylabel('Flux', fontsize=xylabel_fontsize)

    plot_ax.xaxis.set_minor_locator(AutoMinorLocator())
    plot_ax.yaxis.set_minor_locator(AutoMinorLocator())
    plot_ax.tick_params(top=True, which='both', labelsize=xytick_size)
    if not vel_unit:
        plot_ax.axvline((qso_zlist[i] + 1) * 2800, ls='--', c=color_ls[i], lw=2)
        plot_ax.axvline(obs_wave_max, ls='--', c='k', lw=2)
        plot_ax.set_xlim([wave_min, wave_max])

    plot_ax.set_ylim([ymin, ymax])
    plot_ax.legend(fontsize=legend_fontsize, loc=1)
    if i == 3:
        if vel_unit:
            plot_ax.set_xlabel('vel (km/s)', fontsize=xylabel_fontsize)
        else:
            plot_ax.set_xlabel(r'obs wavelength ($\mathrm{{\AA}}$)', fontsize=xylabel_fontsize)

if not vel_unit:
    atwin = ax1.twiny()
    atwin.set_xlabel('absorber redshift', fontsize=xylabel_fontsize)
    atwin.axis([zmin, zmax, ymin, ymax])
    atwin.tick_params(top=True, axis="x", labelsize=xytick_size)
    atwin.xaxis.set_minor_locator(AutoMinorLocator())

plt.tight_layout()
#plt.savefig(savefig)
plt.show()
