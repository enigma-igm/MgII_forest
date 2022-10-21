import mutils
import compute_cf_data as ccf
from matplotlib import pyplot as plt

nqso = 5
wall = mutils.reweight_factors(nqso, 'all')
whigh = mutils.reweight_factors(nqso, 'high')
wlow = mutils.reweight_factors(nqso, 'low')

given_bins = ccf.custom_cf_bin4()
lowz_cgm_fit_gpm, highz_cgm_fit_gpm, allz_cgm_fit_gpm = ccf.init_cgm_fit_gpm()

# all z
vel_mid, xi_mean_unmask_no_w_allz, xi_mean_mask_no_w_allz, _, _, _, _ = \
    ccf.allspec(nqso, 'all', allz_cgm_fit_gpm, given_bins=given_bins)

vel_mid, xi_mean_unmask_w_allz, xi_mean_mask_w_allz, _, _, _, _ = \
    ccf.allspec(nqso, 'all', allz_cgm_fit_gpm, given_bins=given_bins, weights=wall)

# high z
vel_mid, xi_mean_unmask_no_w_highz, xi_mean_mask_no_w_highz, _, _, _, _ = \
    ccf.allspec(nqso, 'high', highz_cgm_fit_gpm, given_bins=given_bins)

vel_mid, xi_mean_unmask_w_highz, xi_mean_mask_w_highz, _, _, _, _ = \
    ccf.allspec(nqso, 'high', highz_cgm_fit_gpm, given_bins=given_bins, weights=whigh)

# low z
vel_mid, xi_mean_unmask_no_w_lowz, xi_mean_mask_no_w_lowz, _, _, _, _ = \
    ccf.allspec(nqso, 'low', lowz_cgm_fit_gpm, given_bins=given_bins)

vel_mid, xi_mean_unmask_w_lowz, xi_mean_mask_w_lowz, _, _, _, _ = \
    ccf.allspec(nqso, 'low', lowz_cgm_fit_gpm, given_bins=given_bins, weights=wlow)

f = 1e5
ymin = f*(-0.001)
ymax = f*(0.002)

# plots
plt.figure()
plt.suptitle("All z")
plt.subplot(121)
plt.plot(vel_mid, xi_mean_unmask_no_w_allz * f, 'ko-', label='unmasked, no weights')
plt.plot(vel_mid, xi_mean_unmask_w_allz * f, 'ro-', label='unmasked, weights')
plt.legend()
plt.xlabel(r'$\Delta v$ (km/s)')
plt.ylabel(r'$\xi(\Delta v)\times 10^5$')
plt.ylim([ymin, ymax])

plt.subplot(122)
plt.plot(vel_mid, xi_mean_mask_no_w_allz * f, 'ko-', label='masked, no weights')
plt.plot(vel_mid, xi_mean_mask_w_allz * f, 'ro-', label='masked, weights')
plt.legend()
plt.xlabel(r'$\Delta v$ (km/s)')
plt.ylim([ymin, ymax])

plt.figure()
plt.suptitle("High z")
plt.subplot(121)
plt.plot(vel_mid, xi_mean_unmask_no_w_highz * f, 'ko-', label='unmasked, no weights')
plt.plot(vel_mid, xi_mean_unmask_w_highz * f, 'ro-', label='unmasked, weights')
plt.legend()
plt.xlabel(r'$\Delta v$ (km/s)')
plt.ylabel(r'$\xi(\Delta v)\times 10^5$')
plt.ylim([ymin, ymax])

plt.subplot(122)
plt.plot(vel_mid, xi_mean_mask_no_w_highz * f, 'ko-', label='masked, no weights')
plt.plot(vel_mid, xi_mean_mask_w_highz * f, 'ro-', label='masked, weights')
plt.legend()
plt.xlabel(r'$\Delta v$ (km/s)')
plt.ylim([ymin, ymax])

plt.figure()
plt.suptitle("Low z")
plt.subplot(121)
plt.plot(vel_mid, xi_mean_unmask_no_w_lowz * f, 'ko-', label='unmasked, no weights')
plt.plot(vel_mid, xi_mean_unmask_w_lowz * f, 'ro-', label='unmasked, weights')
plt.legend()
plt.xlabel(r'$\Delta v$ (km/s)')
plt.ylabel(r'$\xi(\Delta v)\times 10^5$')
plt.ylim([ymin, ymax])

plt.subplot(122)
plt.plot(vel_mid, xi_mean_mask_no_w_lowz * f, 'ko-', label='masked, no weights')
plt.plot(vel_mid, xi_mean_mask_w_lowz * f, 'ro-', label='masked, weights')
plt.legend()
plt.xlabel(r'$\Delta v$ (km/s)')
plt.ylim([ymin, ymax])

plt.show()