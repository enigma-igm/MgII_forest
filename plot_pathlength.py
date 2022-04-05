import sys
sys.path.append('/Users/suksientie/codes/enigma')
#sys.path.append('/Users/suksientie/Research/data_redux')
#sys.path.append('/Users/suksientie/Research/CIV_forest')
sys.path.append('/home/sstie/codes/PypeIt') # for running on IGM cluster
import numpy as np
from matplotlib import pyplot as plt
import mutils

## in progress -- very crude (3/31/22)
datapath = '/Users/suksientie/Research/data_redux/'
#datapath = '/mnt/quasar/sstie/MgII_forest/z75/' # on IGM

fitsfile_list = [datapath + 'wavegrid_vel/J0313-1806/vel1234_coadd_tellcorr.fits', \
                 datapath + 'wavegrid_vel/J1342+0928/vel123_coadd_tellcorr.fits', \
                 datapath + 'wavegrid_vel/J0252-0503/vel12_coadd_tellcorr.fits', \
                 datapath + 'wavegrid_vel/J0038-1527/vel1_tellcorr_pad.fits']

qso_namelist = ['J0313-1806', 'J1342+0928', 'J0252-0503', 'J0038-1527']
qso_zlist = [7.642, 7.541, 7.001, 7.034] # precise redshifts from Yang+2021
exclude_restwave = 1216 - 1185 # excluding proximity zones; see mutils.qso_exclude_proximity_zone
mgii_line = 2800 # taking the midpoint between 2796 and 2804

good_z_all = []
good_z_all2 = []

for iqso, fitsfile in enumerate(fitsfile_list):
    gz, outmask, z_mask, pz_mask = mutils.qso_pathlength(fitsfile, '', qso_zlist[iqso])
    good_z_all.append(gz)
    good_z_all2.extend(gz)

    plt.plot(gz, (iqso+1)*np.ones(len(gz)), 'k.')
    plt.plot(qso_zlist[iqso], iqso + 1, 'ko')

median_z = np.median(good_z_all2)
plt.axvline(median_z)
plt.xlabel('z')
plt.ylabel('QSO sightline')

plt.show()
