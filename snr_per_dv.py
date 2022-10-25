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

rebin_spec_path = '/Users/suksientie/Research/MgII_forest/rebinned_spectra/'
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
    final_mask_new = mask_rebin * redshift_mask_new * pz_mask_new

    snr_new = flux_rebin * np.sqrt(ivar_rebin)
    new_all.append([wave_rebin, snr_new, final_mask_new])

new_median_snr = []
plt.figure(figsize=(14,6))
for i in range(len(arr)):
    qsoid, qsoz, fitsfile, instr = arr[i][0], arr[i][1], arr[i][2], arr[i][3]
    wave_rebin, snr_new, final_mask_new = new_all[i][0], new_all[i][1], new_all[i][2]

    new_median_snr.append(np.median(snr_new[final_mask_new]))
    plt.plot(wave_rebin[final_mask_new], snr_new[final_mask_new], drawstyle='steps-mid', \
             label=qsoid + ' z=%0.2f' % qsoz + ' (%s)' % instr)
    plt.xlabel('obs wave (dv=40 km/s)')
    plt.ylabel('snr')
    plt.legend()

plt.figure(figsize=(12,8))
for i in range(len(arr)):
    qsoid, qsoz, fitsfile = arr[i][0], arr[i][1], arr[i][2]
    x, y = i, new_median_snr[i]
    plt.scatter(x, y)
    plt.annotate(qsoid, (x-0.5, y+0.1))
    plt.ylabel('median snr (dv=40 km/s)')

plt.show()






