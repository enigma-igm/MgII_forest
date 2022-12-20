import numpy as np
from sklearn.neighbors import KDTree
import xi_sum_cython

def compute_xi_weights(delta_f_in, vel_spec, vmin, vmax, dv, log_bin=False, given_bins=None, gpm=None, progress=False, weights_in=None):
    """

    Args:
        delta_f_in (float ndarray), shape (nskew, nspec) or (nspec,):
            Flux contrast array
        vel_spec (float ndarray): shape (nspec,)
            Velocities for flux contrast
        vmin (float):
            Minimum velocity for correlation function velocity grid. This should be a positive number that shold not
            be set to zero, since we deal with the zero lag velocity correlation function separately.
        vmax (float):
            Maximum velocity for correlation function velocity grid. Must be a positive number.
        dv (float):
            Velocity binsize for corrleation functino velocity grid
        log_bin (boolean): optional
            if set to True, then use logarithmic binning between np.log10(vmin) and np.log10(vmax),
            where now dv should be interpreted as dlogv.
        given_bins (2 1d-arrays): shape = (ncorr, ncorr), optional
            the left and right edges of the bins, where both left and right edges are 1d-arrays.
            if provided, then ignore the required vmin, vmax, and dv inputs, but instead use the given bin.
        gpm (boolean ndarray), same shape as delta_f, Optional
            Good pixel mask (True= Good) for the delta_f_in array. Bad pixels will not be used for correlation function
            computation.
        progress (bool): Optional
            If True then show a progress bar

    Returns:
        v_mid, xi, npix, xi_zero_lag

        v_mid (float ndarray): shape = (ncorr,)
             Midpoint of the bins in the velocity grid for which the correlation function is evaluated. Here
             ncorr = (int(round((vmax - vmin)/dv) + 1)
        xi (float ndarray): shape = (nskew, ncorr)
             Correlation function of each of the nskew input spectra
        npix (float ndarray): shape = (ncorr,)
             Number of spectra pixels contributing to the correlation function estimate in each of the ncorr
             correlation function velocity bins
        xi_zero_lag (float ndarray): shape = (nskew,)
             The zero lage correlation function of each input skewer.

    """

    if weights_in is None:
        weights_in = np.ones_like(delta_f_in, dtype=int)

    # This deals with the case where the input delta_f is a single spectrum
    if(len(delta_f_in.shape)==1):
        delta_f = delta_f_in.reshape(1,delta_f_in.size)
        weights = weights_in.reshape(1, weights_in.size)
        gpm_use = np.ones_like(delta_f,dtype=bool) if gpm is None else gpm.reshape(1,delta_f_in.size)
    else:
        delta_f = delta_f_in
        weights = weights_in
        gpm_use =  np.ones_like(delta_f,dtype=bool) if gpm is None else gpm
    nskew, nspec = delta_f.shape

    # Check that the velocity grid has the right size
    if vel_spec.shape[0] != nspec:
        raise ValueError('vel_spec and delta_f_in do not have matching shapes')

    # Correlation function velocity grid, using the mid point values
    if log_bin:
        log_vmin = np.log10(vmin)
        log_vmax = np.log10(vmax)
        ngrid = int(round((log_vmax - log_vmin) / dv) + 1)  # number of grid points including vmin and vmax
        log_v_corr = log_vmin + dv * np.arange(ngrid)
        log_v_lo = log_v_corr[:-1]  # excluding the last point (=vmax)
        log_v_hi = log_v_corr[1:]  # excluding the first point (=vmin)
        v_lo = np.power(10, log_v_lo)
        v_hi = np.power(10, log_v_hi)
        v_mid = np.power(10, ((log_v_hi + log_v_lo) / 2.0))
    elif given_bins is not None:
        (v_lo, v_hi) = given_bins
        v_mid = (v_hi + v_lo)/2.0
    else:
        ngrid = int(round((vmax - vmin)/dv) + 1) # number of grid points including vmin and vmax
        v_corr = vmin + dv * np.arange(ngrid)
        v_lo = v_corr[:-1]  # excluding the last point (=vmax)
        v_hi = v_corr[1:]  # excluding the first point (=vmin)
        v_mid = (v_hi + v_lo)/2.0
    ncorr = v_mid.size

    # This computes all pairs of distances
    data = np.array([vel_spec])
    data = data.transpose()
    tree = KDTree(data)
    npix_forest = len(vel_spec)

    xi = np.zeros((nskew, ncorr)) # storing the CF of each skewer, rather than the CF of all skewers
    w = np.zeros((nskew, ncorr)) #, dtype=int)

    #import time
    #start = time.process_time()
    # looping through each velocity bin and computing the 2PCF
    for iv in range(ncorr):
        # Grab the list of pixel neighbors within this separation
        ind, dist = tree.query_radius(data, v_hi[iv], return_distance=True)
        xi[:, iv], w[:, iv] = xi_sum_cython.xi_sum_weights(ind, dist, delta_f, weights, gpm_use, v_lo[iv], v_hi[iv], nskew, npix_forest)

    #end = time.process_time()
    #print("xi_sum_weights", (end-start))
    ngood = np.sum(gpm_use, axis=1)
    xi_zero_lag = (ngood > 0)*np.sum(delta_f*delta_f*gpm_use, axis=1)/(ngood + (ngood == 0.0))

    return (v_mid, xi, w, xi_zero_lag)
