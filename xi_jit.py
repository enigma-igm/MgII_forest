import numpy as np
from sklearn.neighbors import KDTree
import time
from numba import jit, njit, float64, int32, boolean

@njit('UniTuple(float64[:], 2)(float64[:], float64[:], float64[:], float64[:], boolean[:], float64, float64, int32, int32)')
def xi_sum_weights(ind, dist, delta_f, weights, gpm, v_lo, v_hi, nskew, npix_forest):

    flux_sum = np.zeros(nskew)
    w_sum = np.zeros(nskew)

    for idx in range(npix_forest):

        ibin = (dist[idx] > v_lo) & (dist[idx] <= v_hi)
        #n_neigh = np.sum(ibin)
        ind_neigh = (ind[idx])[ibin] # currently failing here
        n_neigh = len(ind_neigh)

        a = delta_f[:, idx]*gpm[:,idx]
        a_tile = np.empty((n_neigh, *a.shape), a.dtype)
        a_tile[...] = a
        a2 = delta_f[:, ind_neigh] * gpm[:, ind_neigh]
        tmp = a_tile.T * a2

        b = weights[:, idx]*gpm[:,idx]
        b_tile = np.empty((n_neigh, *b.shape), b.dtype)
        b_tile[...] = b
        b2 = weights[:, ind_neigh]*gpm[:, ind_neigh]
        tmp_w = b_tile.T * b2

        flux_sum += np.nansum(tmp * tmp_w, axis=1)
        w_sum += np.nansum(tmp_w, axis=1)

    xi = (w_sum > 0) * (flux_sum / (w_sum + (w_sum == 0)))

    return xi, w_sum

def compute_xi_weights_init(delta_f_in, vel_spec, vmin, vmax, dv, log_bin=False, given_bins=None, gpm=None, progress=False, weights_in=None):
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
    elif given_bins:
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

    all_ind = []
    all_dist = []
    for iv in range(ncorr):
        # Grab the list of pixel neighbors within this separation
        ind, dist = tree.query_radius(data, v_hi[iv], return_distance=True)
        all_ind.append(ind)
        all_dist.append(dist)

    all_ind = np.array(all_ind)
    all_dist = np.array(all_dist)

    return ncorr, v_lo, v_hi, nskew, npix_forest, all_ind, all_dist, delta_f, weights, gpm_use, v_mid

@njit('UniTuple(float64[:], 4)(int32, float64[:], float64[:], int32, int32, float64[:], float64[:], float64[:], float64[:], boolean[:], float64[:])')
def compute_xi_weights(ncorr, v_lo, v_hi, nskew, npix_forest, all_ind, all_dist, delta_f, weights, gpm_use, v_mid):

    xi = np.zeros((nskew, ncorr))  # storing the CF of each skewer, rather than the CF of all skewers
    w = np.zeros((nskew, ncorr))

    start = time.process_time()
    # looping through each velocity bin and computing the 2PCF
    for iv in range(ncorr):
        ind = all_ind[iv]
        dist = all_dist[iv]
        xi[:, iv], w[:, iv] = xi_sum_weights(ind, dist, delta_f, weights, gpm_use, v_lo[iv], v_hi[iv], nskew,
                                                 npix_forest)
        # if progress:
        #    pbar.update(1)
    end = time.process_time()
    print("xi_sum_weights", (end - start))
    ngood = np.sum(gpm_use, axis=1)
    xi_zero_lag = (ngood > 0) * np.sum(delta_f * delta_f * gpm_use, axis=1) / (ngood + (ngood == 0.0))

    return (v_mid, xi, w, xi_zero_lag)
