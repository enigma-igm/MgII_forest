import numpy as np
import compute_cf_data as ccf
import sys
sys.path.append('/Users/suksientie/codes/enigma')
import argparse
import compute_model_grid_8qso_fast as cmg8
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--zbin', required=True, type=str, help="options: all, high, low")
parser.add_argument('--ivarweights', action='store_true', default=False)
args = parser.parse_args()
redshift_bin = args.zbin

qso_namelist = ['J0411-0907', 'J0319-1008', 'J0410-0139', 'J0038-0653', 'J0313-1806', 'J0038-1527', 'J0252-0503', 'J1342+0928']
nqso = len(qso_namelist)
median_z = 6.50
given_bins = ccf.custom_cf_bin4(dv1=80)
savefig = 'paper_plots/8qso/cf_%sz_%dqso_ivar.pdf' % (redshift_bin, nqso)
savefig = None
iqso_to_use = None #np.array([0])
ivar_weights = args.ivarweights

if iqso_to_use is None:
    iqso_to_use = np.arange(0, nqso)

#######
lowz_cgm_fit_gpm, highz_cgm_fit_gpm, allz_cgm_fit_gpm = ccf.init_cgm_fit_gpm(do_not_apply_any_mask=True)

if redshift_bin == 'low':
    cgm_fit_gpm = lowz_cgm_fit_gpm
elif redshift_bin == 'high':
    cgm_fit_gpm = highz_cgm_fit_gpm
elif redshift_bin == 'all':
    cgm_fit_gpm = allz_cgm_fit_gpm

vel_mid, xi_mean_unmask, xi_mean_mask, xi_noise_unmask, xi_noise_mask, xi_unmask, xi_mask, w_masked, w_unmasked = \
    ccf.allspec(nqso, redshift_bin, cgm_fit_gpm, given_bins=given_bins, iqso_to_use=iqso_to_use, ivar_weights=ivar_weights)

#######
output = cmg8.test_compute_model()
ihi, iZ, vel_mid, xi_mock_keep, xi_mean, covar, icovar, logdet, w_mock_ncopy, w_mock_ncopy_noiseless, w_mock_nskew_ncopy_allqso = output

#######
imock = 0
plt.figure(figsize=(14, 8))
for iqso in range(nqso):
    w_mock_qso = w_mock_nskew_ncopy_allqso[iqso][imock]
    nskew, ncorr = w_mock_qso.shape

    plt.subplot(2,4,iqso+1)
    plt.plot(vel_mid, w_masked[iqso], 'r', lw=2, label=qso_namelist[iqso])
    plt.plot(vel_mid, w_mock_ncopy[iqso][imock], 'k', lw=2, label='total nskew')
    plt.xlabel('vel')
    plt.ylabel('npixpair_corr')

    for iskew in range(nskew):
        #if np.sum(w_mock_qso[iskew]) > 0:
        plt.plot(vel_mid, w_mock_qso[iskew], alpha=0.5)#, label='iskew %d' % iskew)

    plt.tight_layout()
    plt.legend(loc=2, fontsize=8)
plt.show()

#######
imock = 0
plt.figure(figsize=(8, 5))

for iqso in range(nqso):
    w_mock_qso = w_mock_nskew_ncopy_allqso[iqso][imock]
    nskew, ncorr = w_mock_qso.shape

    ratio = w_masked[iqso]/(w_mock_ncopy[iqso][imock])
    # print(ratio)

    plt.plot(vel_mid, np.sqrt(ratio), label=qso_namelist[iqso])
    plt.xlabel('vel')
    plt.ylabel('sqrt(npair_data/npair_mock)')

    plt.tight_layout()
    plt.legend(loc=2, fontsize=8)

plt.axvline(768, ls='--')
plt.grid()
plt.show()


