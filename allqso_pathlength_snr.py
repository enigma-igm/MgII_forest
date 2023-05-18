import mutils
import numpy as np
import mask_cgm_pdf

##### March 2023: No longer need this code; see plot_cf_allspec_new.py #####

"""
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
"""

datapath = '/Users/suksientie/Research/MgII_forest/rebinned_spectra/'
fitsfile_list = [datapath + 'J0411-0907_dv40_coadd_tellcorr.fits', \
                     datapath + 'J0319-1008_dv40_coadd_tellcorr.fits', \
                     datapath + 'J0410-0139_dv40_coadd_tellcorr.fits', \
                     datapath + 'J0038-0653_dv40_coadd_tellcorr.fits', \
                     datapath + 'J0313-1806_dv40_coadd_tellcorr.fits', \
                     datapath + 'J0038-1527_dv40_coadd_tellcorr.fits', \
                     datapath + 'J0252-0503_dv40_coadd_tellcorr.fits', \
                     datapath + 'J1342+0928_dv40_coadd_tellcorr.fits', \
                     datapath + 'J1007+2115_dv40_coadd_tellcorr.fits', \
                     datapath + 'J1120+0641_dv40_coadd_tellcorr.fits']

qso_namelist = ['J0411-0907', 'J0319-1008', 'J0410-0139', 'J0038-0653', 'J0313-1806', 'J0038-1527', 'J0252-0503', \
                'J1342+0928', 'J1007+2115', 'J1120+0641']
qso_zlist = [6.826, 6.8275, 7.0, 7.1, 7.642, 7.034, 7.001, 7.541, 7.515, 7.085]
everyn_break_list = (np.ones(len(qso_namelist)) * 60).astype('int')
exclude_restwave = 1216 - 1185
redshift_bin = 'all'

goodzpix_all = []
median_snr_all = []
dx_all = []

# CGM masks
good_vel_data_all, good_wave_all, norm_good_flux_all, norm_good_std_all, norm_good_ivar_all, noise_all, pz_masks_all, other_masks_all = \
    mask_cgm_pdf.init(redshift_bin=redshift_bin, do_not_apply_any_mask=True, datapath=datapath)
mgii_tot_all = mask_cgm_pdf.chi_pdf(good_vel_data_all, norm_good_flux_all, norm_good_ivar_all, noise_all, plot=False, savefig=None)


for iqso, fitsfile in enumerate(fitsfile_list):
    raw_data_out, _, all_masks_out = mutils.init_onespec(iqso, redshift_bin)
    wave, flux, ivar, mask, std, tell, fluxfit = raw_data_out
    strong_abs_gpm, redshift_mask, pz_mask, obs_wave_max, zbin_mask, telluric_gpm, master_mask = all_masks_out

    mgii_tot = mgii_tot_all[iqso]
    fs_mask = mgii_tot.fit_gpm[0]
    master_mask = master_mask * fs_mask

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

    print(qso_namelist[iqso], qso_zlist[iqso], np.median(good_snr), dx)

print("##############")
print("good zpix = median:%0.3f, min: %0.3f, max: %0.3f" % (np.median(goodzpix_all), np.min(goodzpix_all), np.max(goodzpix_all)))
print("dx total", np.sum(dx_all))
