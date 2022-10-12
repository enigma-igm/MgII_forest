import mutils
import numpy as np

datapath='/Users/suksientie/Research/data_redux/'
fitsfile_list = [datapath + 'wavegrid_vel/J0313-1806/vel1234_coadd_tellcorr.fits', \
                     datapath + 'wavegrid_vel/J1342+0928/vel123_coadd_tellcorr.fits', \
                     datapath + 'wavegrid_vel/J0252-0503/vel12_coadd_tellcorr.fits', \
                     datapath + 'wavegrid_vel/J0038-1527/vel1_tellcorr.fits', \
                     datapath + 'wavegrid_vel/J0038-0653/vel1_tellcorr.fits']

qso_namelist = ['J0313-1806', 'J1342+0928', 'J0252-0503', 'J0038-1527', 'J0038-0653']
qso_zlist = [7.642, 7.541, 7.001, 7.034, 7.1]
everyn_break_list = [20, 20, 20, 20, 20]
exclude_restwave = 1216 - 1185

goodzpix_all = []
median_snr_all = []

for iqso, fitsfile in enumerate(fitsfile_list):
    # pixel redshifts
    good_zpix = mutils.final_qso_pathlength(fitsfile, qso_namelist[iqso], qso_zlist[iqso], exclude_rest=exclude_restwave)
    goodzpix_all.extend(good_zpix)

    # snr
    wave, flux, ivar, mask, std, tell, fluxfit, strong_abs_gpm = \
        mutils.extract_and_norm(fitsfile, everyn_break_list[iqso], qso_namelist[iqso])
    redshift_mask, pz_mask, obs_wave_max = mutils.qso_redshift_and_pz_mask(wave, qso_zlist[iqso], exclude_restwave)
    master_mask = mask * redshift_mask * pz_mask
    good_snr = (flux / std)[master_mask]
    median_snr_all.append(np.median(good_snr))

print("%0.3f, %0.3f, %0.3f" % (np.median(goodzpix_all), np.min(goodzpix_all), np.max(goodzpix_all)))
print(median_snr_all)
