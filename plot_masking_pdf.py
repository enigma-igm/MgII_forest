import mask_cgm_pdf

good_vel_data_all, good_wave_all, norm_good_flux_all, norm_good_std_all, norm_good_ivar_all, noise_all = mask_cgm_pdf.init(redshift_bin='all')

#savefig = 'paper_plots/flux_pdf.pdf'
#mask_cgm_pdf.flux_pdf(norm_good_flux_all, noise_all, plot_ispec=None, savefig=savefig)

#savefig = 'paper_plots/chi_pdf.pdf'
#mgii_tot_all = mask_cgm_pdf.chi_pdf(good_vel_data_all, norm_good_flux_all, norm_good_ivar_all, noise_all, plot=True, savefig=savefig)

mgii_tot_all = mask_cgm_pdf.chi_pdf(good_vel_data_all, norm_good_flux_all, norm_good_ivar_all, noise_all, plot=False, savefig=None)
chi_max = [8.4, 10.5, 4.5, 10.5] # for plotting only
for iqso in range(4):
    savefig = 'paper_plots/masked%d.pdf' % iqso
    mask_cgm_pdf.plot_masked_onespec(mgii_tot_all, good_wave_all, good_vel_data_all, norm_good_flux_all, norm_good_std_all, iqso, chi_max[iqso], savefig=savefig)