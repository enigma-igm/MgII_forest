import mutils
import mutils2 as m2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from astropy import constants as const
from pypeit.core import coadd
from pypeit.core.wavecal import wvutils
from astropy.io import fits

sqldb = 'highzqso.sqlite'
con = m2.create_connection(sqldb)
df = pd.read_sql_query("select id, redshift, path, instrument from qso order by redshift", con)
arr = df.to_numpy()

rebin_spec_path = '/Users/suksientie/Research/MgII_forest/rebinned_spectra2/'
new_all = []

for i in range(len(arr)):
    qsoid, qsoz, _, instr = arr[i][0], arr[i][1], arr[i][2], arr[i][3]
    fitsfile = rebin_spec_path + qsoid +'_dv40_coadd_tellcorr.fits'

    data = fits.open(fitsfile)[1].data
    wavegrid_mid = data['wave_grid_mid'].astype('float64')
    wave_rebin = data['wave'].astype('float64')
    flux_rebin = data['flux'].astype('float64')
    ivar_rebin = data['ivar'].astype('float64')
    mask_rebin = data['mask'].astype('bool')

    redshift_mask_new, pz_mask_new, obs_wave_max_new = mutils.qso_redshift_and_pz_mask(wave_rebin, qsoz)
    telluric_gpm = mutils.telluric_mask(wave_rebin)

    final_mask_new = mask_rebin * redshift_mask_new * pz_mask_new * telluric_gpm

    snr_new = flux_rebin * np.sqrt(ivar_rebin)
    new_all.append([wave_rebin, snr_new, final_mask_new])


new_median_snr = []
goodzpix_all = []
median_snr_all = []
dx_all = []

plt.figure(figsize=(14,6))

for i in range(len(arr)):
    qsoid, qsoz, fitsfile, instr = arr[i][0], arr[i][1], arr[i][2], arr[i][3]
    wave_rebin, snr_new, final_mask_new = new_all[i][0], new_all[i][1], new_all[i][2]

    new_median_snr.append(np.median(snr_new[final_mask_new]))
    #print(qsoid, qsoz, np.median(snr_new[final_mask_new]))

    if np.median(snr_new[final_mask_new]) >= 3:
        good_zpix = wave_rebin[final_mask_new] / 2800 - 1
        goodzpix_all.extend(good_zpix)
        zlow, zhigh = good_zpix.min(), good_zpix.max()
        dx = mutils.abspath(zhigh, zlow)
        dx_all.append(dx)
        print(qsoid, qsoz, np.median(snr_new[final_mask_new]), dx, zhigh-zlow)

    plt.plot(wave_rebin[final_mask_new], snr_new[final_mask_new], drawstyle='steps-mid', \
             label=qsoid + ' z=%0.2f' % qsoz + ' (%s)' % instr)
    plt.xlabel('obs wave (dv=40 km/s)')
    plt.ylabel('snr')
    plt.legend()

print("good zpix = median:%0.3f" %  np.median(goodzpix_all))
print("good zpix = min:%0.3f" %  np.min(goodzpix_all))
print("good zpix = max:%0.3f" %  np.max(goodzpix_all))
print("dx total", np.sum(dx_all))

plt.figure(figsize=(12,8))
for i in range(len(arr)):
    qsoid, qsoz, fitsfile, instr  = arr[i][0], arr[i][1], arr[i][2], arr[i][3]
    x, y = i, new_median_snr[i]
    plt.scatter(x, y)
    plt.annotate(qsoid + '\n' + 'z=%0.2f' % qsoz + '\n' + '(%s)' % instr, (x-0.5, y+0.5), fontsize=8)
    plt.ylabel('median snr (dv=40 km/s)')

plt.grid(True)
plt.show()






