import mask_cgm_pdf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--plotpdf', action='store_true', default=False)
parser.add_argument('--plotmaskedspec', action='store_true', default=False)

args = parser.parse_args()
plotpdf = args.plotpdf
plotmaskedspec = args.plotmaskedspec

datapath = '/Users/suksientie/Research/MgII_forest/rebinned_spectra2/'
qso_namelist = ['J0411-0907', 'J0319-1008', 'newqso1', 'newqso2', 'J0313-1806', 'J0038-1527', 'J0252-0503', \
                'J1342+0928', 'J1007+2115', 'J1120+0641']

if plotpdf:
    good_vel_data_all, good_wave_all, norm_good_flux_all, norm_good_std_all, norm_good_ivar_all, noise_all, _, _ = \
        mask_cgm_pdf.init(redshift_bin='all', datapath=datapath)

    savefig = 'paper_plots/10qso_revision/flux_pdf.pdf'
    mask_cgm_pdf.flux_pdf(norm_good_flux_all, noise_all, plot_ispec=None, savefig=savefig)

    #savefig = 'paper_plots/10qso/chi_pdf.pdf'
    #mgii_tot_all = mask_cgm_pdf.chi_pdf(good_vel_data_all, norm_good_flux_all, norm_good_ivar_all, noise_all, plot=True, savefig=savefig)

if plotmaskedspec:
    good_vel_data_all, good_wave_all, norm_good_flux_all, norm_good_std_all, norm_good_ivar_all, noise_all, pz_masks_all, other_masks_all \
        = mask_cgm_pdf.init(redshift_bin='all', datapath=datapath, do_not_apply_any_mask=True)

    mgii_tot_all = mask_cgm_pdf.chi_pdf(good_vel_data_all, norm_good_flux_all, norm_good_ivar_all, noise_all, plot=False, savefig=None)
    chi_max = [4.3, 4.3, 6.3, 4.3, 6.7, 6.4, 4.3, 8.4, 4.7, 4.7] # for plotting only

    #for iqso in range(len(qso_namelist)):
    for iqso in [9]:
        savefig = 'paper_plots/10qso/masked%d_%s.pdf' % (iqso, qso_namelist[iqso])
        saveout = None
        mask_cgm_pdf.plot_masked_onespec2(mgii_tot_all, good_wave_all, good_vel_data_all, norm_good_flux_all, \
                                          norm_good_std_all, pz_masks_all, other_masks_all, iqso, chi_max[iqso], \
                                          savefig=savefig, saveout=saveout)