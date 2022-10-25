import mutils
from astropy.io import fits
from matplotlib import pyplot as plt

#tellcorr_fits1 = '/Users/suksientie/Research/data_redux/silvia/J0020-3653_XShooter_NIR_coadd_tellcorr.fits'
tellcorr_fits1 = '/Users/suksientie/Research/data_redux/silvia/J0410-0139_NIRES_coadd_tellcorr.fits'
tellcorr_fits2 = '/Users/suksientie/Research/data_redux/wavegrid_vel/J0313-1806/vel1234_coadd_tellcorr.fits'
#tellcorr_fits2 = '/Users/suksientie/Research/data_redux/wavegrid_vel/J1342+0928/vel123_coadd_tellcorr.fits'

wave1, flux1, ivar1, mask1, std1, tell1 = mutils.extract_data(tellcorr_fits1)
wave2, flux2, ivar2, mask2, std2, tell2 = mutils.extract_data(tellcorr_fits2)

plt.plot(wave1, tell1, drawstyle='steps-mid')
plt.plot(wave2, tell2, drawstyle='steps-mid')
plt.xlabel('obs wave')
plt.ylabel('atm transmission')
plt.show()