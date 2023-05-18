import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import csv
import numpy as np

### Figure settings
font = {'family' : 'serif', 'weight' : 'normal'}#, 'size': 11}
plt.rc('font', **font)
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['xtick.minor.width'] = 1.5
mpl.rcParams['ytick.minor.width'] = 1.5
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.minor.size'] = 4

xytick_size = 16
xylabel_fontsize = 20
legend_fontsize = 18
savefig = 'paper_plots/10qso/evol.pdf'

#this_work_zmedian = 6.57
#this_work_upperlim = -3.38
this_work_z = [6.52, 6.79, 6.26]
this_work_upplim = [-3.6376, -3.4869, -3.7132]

# trying to recreate the black regions in Fig 13 (right plot) of Schaye et al. (2003)
def schaye_black_regions():

    z = np.linspace(1.5, 4.5, 6)
    median = -3.47 + 0.08 * (z - 3)

    # fits from the abstract or Table 2
    amin, amax = -3.47 - 0.06, -3.47 + 0.07
    bmin, bmax = 0.08 - 0.10, 0.08 + 0.08

    lbound = []
    ubound = []
    for iz in z:
        pos_values = [amin + bmin * (iz - 3), amin + bmax * (iz - 3), amax + bmin * (iz - 3), amax + bmax * (iz - 3)]
        lbound.append(np.min(pos_values))
        ubound.append(np.max(pos_values))
    lbound = np.array(lbound)
    ubound = np.array(ubound)

    return z, median, lbound, ubound

schaye_z, median, lb, ub = schaye_black_regions()

# red points in Fig 16 of Simcoe (2011)
f = open('Simcoe2011-red-data.csv')
csvr = csv.reader(f)
next(csvr)
z = []
logZ = []
for row in csvr:
    z.append(np.float(row[0]))
    logZ.append(np.float(row[1]))
f.close()

# select black squares in that figure
f = open('square-data.csv')
csvr = csv.reader(f)
next(csvr)
zsq = []
logZ_sq = []
for row in csvr:
    zsq.append(np.float(row[0]))
    logZ_sq.append(np.float(row[1]))
f.close()

# putting everything together
plt.figure(figsize=(10, 5.5))
plt.plot(z, logZ, 'ro', ms=10, label='[C/H]: Simcoe (2011)')
plt.plot(zsq, logZ_sq, 'ks', ms=6, mfc='none')

plt.fill_between(schaye_z, lb, ub, color='k', alpha=0.2, ec=None, label='[C/H]: Schaye+(2003)')
for i, elem in enumerate(this_work_z):
    if i == 0:
        plt.errorbar(this_work_z[i], this_work_upplim[i], xerr=0.1, yerr=0.5, uplims = this_work_upplim[i], color='k', lw=3.0, label='[Mg/H]: this work')
    else:
        plt.errorbar(this_work_z[i], this_work_upplim[i], xerr=0.1, yerr=0.5, uplims=this_work_upplim[i], color='k', lw=3.0)
plt.legend(fontsize=legend_fontsize)
plt.xlabel('Redshift', fontsize=xylabel_fontsize)
plt.ylabel('[X/H]', fontsize=xylabel_fontsize)
plt.xticks(np.arange(2.0, 7.5, 0.5))
plt.yticks(np.arange(-5, 1, 1))
plt.xlim([1.8, 7])
plt.gca().tick_params(which='both', labelsize=xytick_size)
plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
plt.gca().set_aspect(aspect=0.5)

if savefig is not None:
    plt.savefig(savefig)
plt.show()
plt.close()

