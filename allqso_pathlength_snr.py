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
redshift_bin = 'low'

goodzpix_all = []
median_snr_all = []
dx_all = []

for iqso, fitsfile in enumerate(fitsfile_list):
    raw_data_out, _, all_masks_out = mutils.init_onespec(iqso, redshift_bin)
    wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
    strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, master_mask = all_masks_out

    """
    # pixel redshifts
    good_zpix = mutils.final_qso_pathlength(fitsfile, qso_namelist[iqso], qso_zlist[iqso], exclude_rest=exclude_restwave)
    goodzpix_all.extend(good_zpix)
    """
    good_zpix = wave[master_mask] / 2800 - 1
    goodzpix_all.extend(good_zpix)
    zlow, zhigh = good_zpix.min(), good_zpix.max()
    dx = mutils.abspath(zhigh, zlow)
    dx_all.append(dx)

    """
    # snr
    wave, flux, ivar, mask, std, tell, fluxfit, strong_abs_gpm = \
        mutils.extract_and_norm(fitsfile, everyn_break_list[iqso], qso_namelist[iqso])
    redshift_mask, pz_mask, obs_wave_max = mutils.qso_redshift_and_pz_mask(wave, qso_zlist[iqso], exclude_restwave)
    master_mask = mask * redshift_mask * pz_mask
    """
    good_snr = (flux / std)[master_mask]
    median_snr_all.append(np.median(good_snr))

print("##############")
print("good zpix = median:%0.3f, min: %0.3f, max: %0.3f" % (np.median(goodzpix_all), np.min(goodzpix_all), np.max(goodzpix_all)))
print("median snr:", median_snr_all)
print("dx total", np.sum(dx_all))
print("dx:", dx_all)
