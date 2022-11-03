import numpy as np
from matplotlib import pyplot as plt
import compute_cf_data as ccf
import sys
sys.path.append('/Users/suksientie/codes/enigma')
from enigma.reion_forest import utils as reion_utils
import argparse
import mutils

parser = argparse.ArgumentParser()
parser.add_argument('--zbin', type=str, help="options: all, high, low")
args = parser.parse_args()
redshift_bin = args.zbin

given_bins = ccf.custom_cf_bin4(dv1=80)
vmin_corr, vmax_corr, dv_corr = 10, 3500, 40 # dummy values because we're now using custom binning
nqso = 8

lowz_cgm_fit_gpm, highz_cgm_fit_gpm, allz_cgm_fit_gpm = ccf.init_cgm_fit_gpm()
weights = mutils.reweight_factors(nqso, redshift_bin)

if redshift_bin == 'low':
    cgm_fit_gpm = lowz_cgm_fit_gpm
elif redshift_bin == 'high':
    cgm_fit_gpm = highz_cgm_fit_gpm
elif redshift_bin == 'all':
    cgm_fit_gpm = allz_cgm_fit_gpm

xi_all = []
for iqso in range(nqso):
    raw_data_out, _, all_masks_out = mutils.init_onespec(iqso, redshift_bin)
    wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
    strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, telluric_mask, master_mask = all_masks_out

    apply_mask = [mask, \
                  mask * redshift_mask, \
                  mask * redshift_mask * pz_mask, \
                  mask * redshift_mask * pz_mask * telluric_mask, \
                  mask * redshift_mask * pz_mask * telluric_mask * cgm_fit_gpm[iqso]]

    xi_apply_mask = []
    for i, mask_item in enumerate(apply_mask):
        all_masks = mask_item

        norm_good_flux = (flux / fluxfit)[all_masks]
        norm_flux = flux / fluxfit
        vel = mutils.obswave_to_vel_2(wave)
        meanflux_tot = np.mean(norm_good_flux)
        deltaf_tot = (norm_flux - meanflux_tot) / meanflux_tot

        vel_mid, xi_tot, npix_tot, _ = reion_utils.compute_xi(deltaf_tot, vel, vmin_corr, vmax_corr, dv_corr,
                                                              given_bins=given_bins, gpm=all_masks)
        xi_mean_tot = np.mean(xi_tot, axis=0)  # not really averaging here since it's one spectrum (i.e. xi_mean_tot = xi_tot)
        xi_apply_mask.append(xi_mean_tot)

    xi_all.append(xi_apply_mask)

xi_all = np.array(xi_all)
n_qso, n_mask, n_corr = xi_all.shape

label = ['data', 'data+z', 'data+z+pz', 'data+z+pz+tell', 'data+z+pz+tell+cgm']
factor = 1e5
ymin = factor*(-0.001)
ymax = factor*(0.002)

for i in range(n_mask):
    xi_mean_all = np.average(xi_all, axis=0, weights=weights)
    plt.plot(vel_mid, xi_mean_all[i] * factor, linewidth=1.5, label=label[i])

plt.legend()
plt.axhline(0, color='k', ls='--')
plt.axvline(768.469, color='r', ls='--')
plt.axvline(1000, color='k', ls=':')
plt.ylabel(r'$\xi(\Delta v)\times 10^5$')
plt.ylim([ymin, ymax])
plt.tight_layout()
plt.show()