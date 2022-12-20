import numpy as np
cimport numpy as np

cpdef xi_sum_weights(np.ndarray ind, np.ndarray dist, np.ndarray delta_f, np.ndarray weights, np.ndarray gpm, int v_lo, int v_hi, np.int nskew, np.int npix_forest):

    cdef np.ndarray flux_sum = np.zeros(nskew, dtype=np.float)
    cdef np.ndarray w_sum = np.zeros(nskew, dtype=np.float)
    cdef int n_neigh
    cdef np.ndarray ind_neigh
    cdef np.ndarray a, a_tile, a2, tmp, b, b_tile, b2, tmp_w, xi

    #flux_sum = np.zeros(nskew)
    #w_sum = np.zeros(nskew)

    # Py_ssize_t is the proper C type for Python array indices.
    cdef Py_ssize_t idx

    for idx in range(npix_forest):
        ibin = (dist[idx] > v_lo) & (dist[idx] <= v_hi)
        n_neigh = np.sum(ibin)
        ind_neigh = (ind[idx])[ibin]

        a = delta_f[:, idx]*gpm[:,idx]
        a_tile = np.empty((n_neigh, a.shape[0]), np.float)
        a_tile[...] = a
        a2 = delta_f[:, ind_neigh] * gpm[:, ind_neigh]
        tmp = a_tile.T * a2

        b = weights[:, idx]*gpm[:,idx]
        b_tile = np.empty((n_neigh, b.shape[0]), np.float)
        b_tile[...] = b
        b2 = weights[:, ind_neigh]*gpm[:, ind_neigh]
        tmp_w = b_tile.T * b2

        #tmp = np.tile(delta_f[:, idx]*gpm[:,idx], (n_neigh, 1)).T * (delta_f[:, ind_neigh]*gpm[:, ind_neigh])
        #tmp_w = np.tile(weights[:, idx]*gpm[:,idx], (n_neigh, 1)).T * (weights[:, ind_neigh]*gpm[:, ind_neigh])
        flux_sum += np.nansum(tmp * tmp_w, axis=1)
        w_sum += np.nansum(tmp_w, axis=1)

    xi = (w_sum > 0) * (flux_sum / (w_sum + (w_sum == 0)))

    return xi, w_sum

