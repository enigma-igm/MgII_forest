import numpy as np
from astropy.cosmology import FlatLambdaCDM
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def plot_allspec1(wave_arr, flux_arr):

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

def plot_allspec2(wave_arr, flux_arr, qso_namelist, qso_zlist, vel_unit=False, vel_zeropoint=True, wave_zeropoint_value=None):

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

def obswave_to_vel(wave_arr, vel_zeropoint=True, wave_zeropoint_value=None):
    # wave in Angstrom
    zabs_mean = wave_arr/2800 - 1 # between the two doublet

    cosmo = FlatLambdaCDM(H0=100.0 * 0.67, Om0=0.3192, Ob0=0.04964)
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

