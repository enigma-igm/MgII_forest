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
savepath = '/Users/suksientie/Research/MgII_forest/rebinned_spectra/'

for i in range(len(arr)):
    qsoid, qsoz, fitsfile, instr = arr[i][0], arr[i][1], arr[i][2], arr[i][3]

    data = fits.open(fitsfile)[1].data
    wavegrid_mid = data['wave_grid_mid'].astype('float64')
    wave_coadd = data['wave'].astype('float64')
    flux_coadd = data['flux'].astype('float64')
    ivar_coadd = data['ivar'].astype('float64')
    mask_coadd = data['mask'].astype('bool')

    #wavegrid_mid, flux, ivar, mask, std, tell = mutils.extract_data(fitsfile)

    new_wavegrid, new_wavegrid_mid, dsamp = wvutils.get_wave_grid(wave_coadd, masks=None, wave_method=wave_method,
                                                                  dv=dv)
    flux_new, ivar_new, gpm_new = coadd.interp_oned(new_wavegrid, wave_coadd, flux_coadd, ivar_coadd, mask_coadd, sensfunc=False)

    """
    new_wavegrid, new_wavegrid_mid, dsamp = wvutils.get_wave_grid(wavegrid_mid, masks=None, wave_method=wave_method, dv=dv)
    flux_new, ivar_new, gpm_new = coadd.interp_oned(new_wavegrid_mid, wavegrid_mid, flux, ivar, mask, sensfunc=False)

    # saving rebinned spectra
    savefits = savepath + qsoid + '_dv%d_' % dv + 'coadd_tellcorr.fits'
    wave_gpm = new_wavegrid_mid > 1.0
    mask_write = gpm_new.astype(int)
    onespec = OneSpec(wave=None, wave_grid_mid=new_wavegrid_mid[wave_gpm], flux=flux_new[wave_gpm],
                      ivar=ivar_new[wave_gpm], mask=mask_write[wave_gpm])
    """

    savefits = savepath + qsoid + '_dv%d_' % dv + 'coadd_tellcorr.fits'
    wave_gpm = new_wavegrid > 1.0
    mask_write = gpm_new.astype(int)
    onespec = OneSpec(wave=new_wavegrid[wave_gpm], wave_grid_mid=new_wavegrid_mid[wave_gpm[1:]], flux=flux_new[wave_gpm],
                      ivar=ivar_new[wave_gpm], mask=mask_write[wave_gpm])

    #header = fits.getheader(fitsfile)  # random header
    #onespec.head0 = header
    onespec.to_file(savefits, overwrite=True)
