import compute_cf_data as ccf
from matplotlib import pyplot as plt
from astropy.io import fits
from sklearn.neighbors import KDTree
import compute_cf_data as ccf
import mutils
import numpy as np
import compute_model_grid_8qso_fast as cmg8
from astropy.table import Table
import sys
sys.path.append('/Users/suksientie/codes/enigma')
from enigma.reion_forest import utils
import mcmc_inference as mcmc
from enigma.reion_forest.utils import find_closest
import time
import scipy
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from astropy.io import fits
import matplotlib as mpl

mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['ytick.major.size'] = 5

def init_var(redshift_bin):

    (v_lo, v_hi) = ccf.custom_cf_bin4(dv1=80)
    v_mid = (v_hi + v_lo) / 2.0

    if redshift_bin == 'all':
        #xi_file = fits.open('save_cf/paper/xi_mean_mask_10qso_everyn60_corr.fits')
        xi_file = fits.open('save_cf/paper_new/xi_10qso_everyn60_corr_allz.fits')
    elif redshift_bin == 'low':
        #xi_file = fits.open('save_cf/paper/xi_mean_mask_10qso_everyn60_lowz.fits')
        xi_file = fits.open('save_cf/paper_new/xi_10qso_everyn60_corr_lowz.fits')
    elif redshift_bin == 'high':
        #xi_file = fits.open('save_cf/paper/xi_mean_mask_10qso_everyn60_highz.fits')
        xi_file = fits.open('save_cf/paper_new/xi_10qso_everyn60_corr_highz.fits')

    xi_real_data = xi_file['XI_MEAN_MASK'].data
    xi_mask_allqso = xi_file['XI_MASK'].data
    #xi_data_allmocks = np.load('xi_mock_keep_%sz_0.50_-4.50.npy' % redshift_bin) # computed using cmg8.test_compute_model()
    xi_data_allmocks = np.load('xi_mock_keep_%sz_0.50_-4.50_new_interp.npy' % redshift_bin)

    return v_mid, xi_real_data, xi_mask_allqso, xi_data_allmocks

# https://www.astrobetter.com/blog/2014/02/10/visualization-fun-with-python-2d-histogram-with-1d-histograms-on-axes/
def ellipse(ra,rb,ang,x0,y0,Nb=100):
    xpos,ypos=x0,y0
    radm,radn=ra,rb
    an=ang
    co,si=np.cos(an),np.sin(an)
    the=np.linspace(0,2*np.pi,Nb)
    X=radm*np.cos(the)*co-si*radn*np.sin(the)+xpos
    Y=radm*np.cos(the)*si+co*radn*np.sin(the)+ypos
    return X,Y

def plot_cf_corr_sigma_ellipses1(x, y):
    xcenter = np.mean(x)
    ycenter = np.mean(y)
    ra = np.std(x)
    rb = np.std(y)
    ang = 0

    X, Y = ellipse(ra, rb, ang, xcenter, ycenter)
    fig, ax_nstd = plt.subplots(figsize=(6, 6))
    ax_nstd.scatter(x, y, s=3)
    ax_nstd.plot(X, Y, "k:", ms=1, linewidth=2.0)

    X, Y = ellipse(2*ra, 2*rb, ang, xcenter, ycenter)
    ax_nstd.plot(X, Y, "r:", ms=1, linewidth=2.0)

    X, Y = ellipse(3* ra, 3* rb, ang, xcenter, ycenter)
    ax_nstd.plot(X, Y, "g:", ms=1, linewidth=2.0)
    plt.show()

# https://stackoverflow.com/questions/37031356/check-if-points-are-inside-ellipse-faster-than-contains-point-method
def dist_to_ellcenter(xdata, ydata, xcenter, ycenter, ell_width, ell_height, ell_ang):
    # normalised distance of the point from the cell centre,
    # where a distance of 1 would be on the ellipse, less than 1 is inside, and more than 1 is outside.
    cos_angle = np.cos(np.radians(180. - ell_ang))
    sin_angle = np.sin(np.radians(180. - ell_ang))

    dx_data = xdata - xcenter
    dy_data = ydata - ycenter

    xct_data = dx_data * cos_angle - dy_data * sin_angle
    yct_data = dx_data * sin_angle + dy_data * cos_angle

    #ell_width, ell_height = 2 * (3 * ra), 2 * (3 * rb)  # times 2 because we want the total
    rad_cc = (xct_data ** 2 / (ell_width / 2.) ** 2) + (yct_data ** 2 / (ell_height / 2.) ** 2)
    return rad_cc

def cfbin_corr_one(xi_real_data, xi_data_allmocks, ibin, v_mid, xi_mask_allqso=None, plot=False, saveplot=False):

    #plot_w, plot_h = 13, 8 # 6x5
    #nrow, ncol = 5, 6

    scale_fac = 1e3
    plot_w, plot_h = 9, 13
    nrow, ncol = 7, 4

    nbin_outside_ell = 0
    if plot:
        #plt.figure(figsize=(plot_w, plot_h), sharex=True, sharey=True)
        fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(plot_w, plot_h), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.13, bottom=0.15, right=0.98, top=0.93, hspace=0.1, wspace=0.25)
        ax_plot = axes.flatten()
        isubplot = 0

    nsigma = 3
    for j in range(len(v_mid)):
        if j != ibin:
            x, y = xi_data_allmocks[:, ibin] * scale_fac, xi_data_allmocks[:, j] * scale_fac

            xcenter = np.mean(x)
            ycenter = np.mean(y)
            ra = np.std(x)
            rb = np.std(y)
            ang = 0
            rad_cc = dist_to_ellcenter(xi_real_data[ibin] * scale_fac, xi_real_data[j] * scale_fac, xcenter, ycenter, 2*(nsigma * ra), 2*(nsigma * rb), ang)

            if rad_cc > 1:
                nbin_outside_ell += 1
                data_marker = 'r+'
            else:
                data_marker = 'g+'

            if plot:
                #isubplot += 1
                #ax = plt.subplot(nrow, ncol, isubplot)
                ax = ax_plot[isubplot]
                ax.plot(x, y, 'kx', alpha=0.5)

                X, Y = ellipse(ra, rb, ang, xcenter, ycenter) # 1-sigma
                ax.plot(X, Y, "b-", ms=1, linewidth=2)
                X, Y = ellipse(2 * ra, 2 * rb, ang, xcenter, ycenter) # 2-sigma
                ax.plot(X, Y, "b-", ms=1, linewidth=2)
                X, Y = ellipse(3 * ra, 3 * rb, ang, xcenter, ycenter) # 3-sigma
                ax.plot(X, Y, "b-", ms=1, linewidth=1.5)

                ax.plot(xi_real_data[ibin] * scale_fac, xi_real_data[j] * scale_fac, data_marker, ms=12, markeredgewidth=3)

                if xi_mask_allqso is not None:
                    for iqso in range(len(xi_mask_allqso)):
                        ax.scatter(xi_mask_allqso[iqso][ibin], xi_mask_allqso[iqso][j], label=iqso, s=10, zorder=10)

                if isubplot in (24, 25, 26, 27):
                    ax.set_xlabel(r'$\xi (dv=%d) \times 10^3$' % v_mid[ibin], fontsize=10)
                    #ax.set_xlabel(r'$\xi \times 10^3$' + '\n' + r'$(dv=%d)$' % v_mid[ibin])

                #if isubplot in (0, 4, 8, 13, 17, 21, 25):
                if isubplot % 4 == 0:
                    ax.set_ylabel(r'$\xi (dv=%d) \times 10^3$' % v_mid[j], fontsize=10)
                else:
                    plt.setp(ax.get_yticklabels(), visible=False)
                    ax.set_ylabel(r'$\xi (dv=%d) \times 10^3$' % v_mid[j] % v_mid[j], labelpad=6, fontsize=10)


                isubplot += 1

    #plt.tight_layout()
    if saveplot:
        #plt.savefig('paper_plots/10qso/debug/allz2/cf_corr_dv%d.png' % v_mid[ibin])
        plt.savefig('paper_plots/10qso/cfbin_masking/allz/cf_corr_dv%d.png' % v_mid[ibin])
        plt.close()
    else:
        plt.show()

    print(v_mid[ibin], nbin_outside_ell)
    return v_mid[ibin], nbin_outside_ell

def cfbin_corr_all(xi_real_data, xi_data_allmocks, v_mid, xi_mask_allqso=None, plot=False, saveplot=False):

    mask_bin = []
    mask_ibin = []
    mask_bin_noutside = []

    for ibin in range(len(v_mid)):
        v_mid_ibin, nbin_outside_ell= cfbin_corr_one(xi_real_data, xi_data_allmocks, ibin, v_mid, \
                                                     xi_mask_allqso=xi_mask_allqso, plot=plot, saveplot=saveplot)

        if nbin_outside_ell >= 20: # definition adopted for paper
            mask_bin.append(v_mid_ibin)
            mask_ibin.append(ibin)
            mask_bin_noutside.append(nbin_outside_ell)

    return mask_bin, mask_bin_noutside, mask_ibin

def old_plot_cf_corr_new(xi_real_data, xi_data_allmocks, ibin, v_mid, saveplot=False, xi_mask_allqso=None):

    nbin_outside_ell = 0
    plt.figure(figsize=(13, 8)) # 6x5

    for j in range(len(v_mid)):
        ax = plt.subplot(5, 6, j + 1)

        if j != ibin:
            x, y = xi_data_allmocks[:, ibin], xi_data_allmocks[:, j]
            ax.plot(x, y, 'kx')

            xcenter = np.mean(x)
            ycenter = np.mean(y)
            ra = np.std(x)
            rb = np.std(y)
            ang = 0

            X, Y = ellipse(ra, rb, ang, xcenter, ycenter) # 1-sigma
            ax.plot(X, Y, "b:", ms=1, linewidth=1.5)
            X, Y = ellipse(2 * ra, 2 * rb, ang, xcenter, ycenter) # 2-sigma
            ax.plot(X, Y, "b--", ms=1, linewidth=1.5)
            X, Y = ellipse(3 * ra, 3 * rb, ang, xcenter, ycenter) # 3-sigma
            ax.plot(X, Y, "b-", ms=1, linewidth=1.5)

            rad_cc = dist_to_ellcenter(xi_real_data[ibin], xi_real_data[j], xcenter, ycenter, 2*(3 * ra), 2*(3 * rb), ang)

            """
            cos_angle = np.cos(np.radians(180. - ang))
            sin_angle = np.sin(np.radians(180. - ang))

            dx_data = xi_real_data[ibin] - xcenter
            dy_data = xi_real_data[j] - ycenter

            xct_data = dx_data * cos_angle - dy_data * sin_angle
            yct_data = dx_data * sin_angle + dy_data * cos_angle

            ell_width, ell_height = 2*(3 * ra), 2*(3 * rb) # times 2 because we want the total
            rad_cc = (xct_data ** 2 / (ell_width / 2.) ** 2) + (yct_data ** 2 / (ell_height / 2.) ** 2)
            """

            if rad_cc > 1:
                ax.plot(xi_real_data[ibin], xi_real_data[j], 'r+', ms=10, markeredgewidth=2)
                nbin_outside_ell += 1
            else:
                ax.plot(xi_real_data[ibin], xi_real_data[j], 'g+', ms=10, markeredgewidth=2)

            if xi_mask_allqso is not None:
                for iqso in range(len(xi_mask_allqso)):
                    ax.scatter(xi_mask_allqso[iqso][ibin], xi_mask_allqso[iqso][j], label=iqso, s=10, zorder=10)

            ax.set_xlabel(r'$\xi$(dv=%d)' % v_mid[ibin])
            ax.set_ylabel(r'$\xi$(dv=%d)' % v_mid[j])
            #ax.axes.xaxis.set_ticklabels([])
            #ax.axes.yaxis.set_ticklabels([])

    print(v_mid[ibin], nbin_outside_ell)
    #print()
    #print(v_mid[nbin_outside_ell > round(len(v_mid)*0.9)])

    plt.tight_layout()
    if saveplot:
        plt.savefig('paper_plots/10qso/debug/highz/cf_corr_dv%d.png' % v_mid[ibin])
        plt.close()

##########################
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    # https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html

    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)

    return ellipse, ax.add_patch(ellipse)

def plot_cf_corr_sigma_ellipses2(x, y):
    fig, ax_nstd = plt.subplots(figsize=(6, 6))
    ax_nstd.scatter(x, y, s=3)
    confidence_ellipse(x, y, ax_nstd, n_std=1,
                       label=r'$1\sigma$', edgecolor='firebrick')
    confidence_ellipse(x, y, ax_nstd, n_std=2,
                       label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
    confidence_ellipse(x, y, ax_nstd, n_std=3,
                       label=r'$3\sigma$', edgecolor='blue', linestyle=':')
    plt.show()

def plot_cf_corr_new2(xi_real_data, xi_data_allmocks, ibin, v_mid, saveplot=False, xi_mask_allqso=None):

    nbin_outside_ell = 0
    plt.figure(figsize=(13, 8)) # 6x5

    for j in range(len(v_mid)):
        ax = plt.subplot(5, 6, j + 1)

        if j != ibin:
            x, y = xi_data_allmocks[:, ibin], xi_data_allmocks[:, j]
            ax.plot(x, y, 'kx')

            kwargs = dict(zorder=10)
            c = 'blue'
            ell1, _ = confidence_ellipse(x, y, ax, n_std=1, edgecolor=c, linestyle=':', **kwargs)
            ell2, _ = confidence_ellipse(x, y, ax, n_std=2, edgecolor=c, linestyle='--', **kwargs)
            ell3, _ = confidence_ellipse(x, y, ax, n_std=3, edgecolor=c, linestyle='-', **kwargs)

            xcenter = np.mean(x)
            ycenter = np.mean(y)
            rad_cc = dist_to_ellcenter(xi_real_data[ibin], xi_real_data[j], xcenter, ycenter, ell3.width, ell3.height, ell3.angle)

            if rad_cc > 1:
                ax.plot(xi_real_data[ibin], xi_real_data[j], 'r+', ms=10, markeredgewidth=2)
                nbin_outside_ell += 1
            else:
                ax.plot(xi_real_data[ibin], xi_real_data[j], 'g+', ms=10, markeredgewidth=2)

            if xi_mask_allqso is not None:
                for iqso in range(len(xi_mask_allqso)):
                    ax.scatter(xi_mask_allqso[iqso][ibin], xi_mask_allqso[iqso][j], label=iqso, s=10, zorder=10)

            ax.set_xlabel(r'$\xi$(dv=%d)' % v_mid[ibin])
            ax.set_ylabel(r'$\xi$(dv=%d)' % v_mid[j])
            #ax.axes.xaxis.set_ticklabels([])
            #ax.axes.yaxis.set_ticklabels([])

    print(v_mid[ibin], nbin_outside_ell)

    plt.tight_layout()
    #plt.legend()
    if saveplot:
        plt.savefig('paper_plots/10qso/debug/allz2/cf_corr_dv%d.png' % v_mid[ibin])
    #else:
    #    plt.show()