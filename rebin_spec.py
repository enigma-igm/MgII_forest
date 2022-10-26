import mutils
import mutils2 as m2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from astropy import constants as const
from pypeit.core import coadd
from pypeit.core.wavecal import wvutils
from pypeit.onespec import OneSpec
from astropy.io import fits

sqldb = 'highzqso.sqlite'
con = m2.create_connection(sqldb)
df = pd.read_sql_query("select id, redshift, path, instrument from qso order by redshift", con)
arr = df.to_numpy()

wave_method = 'velocity'
dv = 40
mg2forest_wavemin = None #19500
plot = False
saveout = True
savepath = '/Users/suksientie/Research/MgII_forest/rebinned_spectra/'

for i in range(len(arr)):
    qsoid, qsoz, fitsfile, instr = arr[i][0], arr[i][1], arr[i][2], arr[i][3]

    # these fitsfiles are _coadd_tellcorr.fits
    data = fits.open(fitsfile)[1].data
    wavegrid_mid = data['wave_grid_mid'].astype('float64')
    wave_coadd = data['wave'].astype('float64')
    flux_coadd = data['flux'].astype('float64')
    ivar_coadd = data['ivar'].astype('float64')
    mask_coadd = data['mask'].astype('bool')
    telluric_coadd = data['telluric'].astype('float64')
    obj_model_coadd = data['obj_model'].astype('float64')

    # creating a new wavelength grid uniform in velocity scale having dv spacing
    new_wavegrid, new_wavegrid_mid, dsamp = wvutils.get_wave_grid(wave_coadd, masks=None, wave_method=wave_method, dv=dv, wave_grid_min=mg2forest_wavemin)

    # interpolate flux and ivar on this new wavelength grid
    flux_new, ivar_new, gpm_new = coadd.interp_oned(new_wavegrid, wave_coadd, flux_coadd, ivar_coadd, mask_coadd, sensfunc=False)
    tell_new, _, _ = coadd.interp_oned(new_wavegrid, wave_coadd, telluric_coadd, ivar_coadd, mask_coadd, sensfunc=False)
    objmodel_new, _, _ = coadd.interp_oned(new_wavegrid, wave_coadd, obj_model_coadd, ivar_coadd, mask_coadd, sensfunc=False)

    flux_new = np.nan_to_num(flux_new, nan=-100)
    ivar_new = np.nan_to_num(ivar_new, nan=1e-5)

    if plot:
        print(np.sum(mask_coadd), len(wave_coadd))
        plt.figure(figsize=(14,5))
        plt.subplot(121)
        plt.plot(wave_coadd, flux_coadd, 'k', drawstyle='steps-mid')
        plt.plot(wave_coadd, 1/np.sqrt(ivar_coadd), 'k', alpha=0.5, drawstyle='steps-mid')
        plt.plot(wave_coadd, telluric_coadd, 'b', alpha=0.5, drawstyle='steps-mid')
        plt.plot(wave_coadd, obj_model_coadd, 'r', alpha=0.5, drawstyle='steps-mid')
        plt.ylim([-0.2, 2])

        plt.subplot(122)
        plt.plot(new_wavegrid, flux_new, 'k', drawstyle='steps-mid')
        plt.plot(new_wavegrid, 1 / np.sqrt(ivar_new), 'k', alpha=0.5, drawstyle='steps-mid')
        plt.plot(new_wavegrid, tell_new, 'b', alpha=0.5, drawstyle='steps-mid')
        plt.plot(new_wavegrid, objmodel_new, 'r', alpha=0.5, drawstyle='steps-mid')
        plt.ylim([-0.2, 2])
        plt.show()

    if saveout:
        savefits = savepath + qsoid + '_dv%d_' % dv + 'coadd_tellcorr.fits'
        wave_gpm = new_wavegrid > 1.0
        mask_write = gpm_new.astype(int)
        onespec = OneSpec(wave=new_wavegrid[wave_gpm], wave_grid_mid=new_wavegrid_mid[wave_gpm[1:]], flux=flux_new[wave_gpm],
                          ivar=ivar_new[wave_gpm], mask=mask_write[wave_gpm], telluric=tell_new[wave_gpm], obj_model=objmodel_new[wave_gpm])

        # random header
        header = fits.getheader(fitsfile)
        onespec.head0 = header
        onespec.to_file(savefits, overwrite=True)
