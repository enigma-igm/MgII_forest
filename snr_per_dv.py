import mutils
import mutils2 as m2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from astropy import constants as const
from pypeit.core import coadd
from pypeit.core.wavecal import wvutils

sqldb = 'highzqso.sqlite'
con = m2.create_connection(sqldb)
df = pd.read_sql_query("select id, redshift, path, instrument from qso order by redshift", con)
arr = df.to_numpy()

#nires_fitsfiles, nires_qsoid, nires_z, nires_texp = m2.nires_qso()
#xshooter_fitsfiles, xshooter_qsoid, xshooter_z, xshooter_texp = m2.xshooter_qso()
#mosfire_fitsfiles, mosfire_qsoid, mosfire_z, mosfire_texp = m2.mosfire_qso()

c_kms = const.c.to('km/s').value
new_all = []

for i in range(len(arr)):
    qsoid, qsoz, fitsfile, instr = arr[i][0], arr[i][1], arr[i][2], arr[i][3]
    wavegrid_mid, flux, ivar, mask, std, tell = mutils.extract_data(fitsfile)

    new_wavegrid, new_wavegrid_mid, dsamp = wvutils.get_wave_grid(wavegrid_mid, masks=None, wave_method='velocity', dv=40)
    flux_new, ivar_new, gpm_new = coadd.interp_oned(new_wavegrid_mid, wavegrid_mid, flux, ivar, mask, sensfunc=False)
    redshift_mask_new, pz_mask_new, obs_wave_max_new = mutils.qso_redshift_and_pz_mask(new_wavegrid_mid, qsoz)
    final_mask_new = gpm_new * redshift_mask_new * pz_mask_new

    dwave_new, dloglam_new, resln_guess_new, pix_per_sigma_new = wvutils.get_sampling(new_wavegrid_mid)
    dv_new = c_kms * np.log(10) * dloglam_new
    snr_new = flux_new * np.sqrt(ivar_new)

    new_all.append([new_wavegrid_mid, snr_new, final_mask_new])

new_median_snr = []
plt.figure(figsize=(14,6))
for i in range(len(arr)):
    qsoid, qsoz, fitsfile, instr = arr[i][0], arr[i][1], arr[i][2], arr[i][3]
    new_wavegrid_mid, snr_new, final_mask_new = new_all[i][0], new_all[i][1], new_all[i][2]

    new_median_snr.append(np.median(snr_new[final_mask_new]))
    plt.plot(new_wavegrid_mid[final_mask_new], snr_new[final_mask_new], drawstyle='steps-mid', \
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






