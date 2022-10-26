import sys
sys.path.append('/Users/suksientie/codes/enigma')
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from astropy.io import fits
from pypeit.core.fitting import iterfit, robust_fit
from pypeit import utils as putils
from pypeit import bspline
import scipy.interpolate as interpolate
from astropy import constants as const
from astropy.table import Table
from astropy.stats import sigma_clip, mad_std
from astropy.modeling.models import Gaussian1D
from enigma.reion_forest import utils
import compute_cf_data as ccf
import pdb
from astropy.cosmology import FlatLambdaCDM
import mutils
import glob
import sqlite3
from sqlite3 import Error
import pandas as pd

def nires_qso():
    fitsfiles = glob.glob('/Users/suksientie/Research/data_redux/silvia/*NIRES*fits')
    qsoid = []
    for file in fitsfiles:
        qsoid.append(file.split('/')[-1].split('_')[0])
    qso_z = [6.8260, 6.8275, 7.0, 6.8]
    texp = [360, 360, 360, 300 * 4 + 360 * 32]

    return fitsfiles, qsoid, qso_z, texp

def xshooter_qso():
    fitsfiles = glob.glob('/Users/suksientie/Research/data_redux/silvia/*XShooter*fits')
    qsoid = []
    for file in fitsfiles:
        qsoid.append(file.split('/')[-1].split('_')[0])
    qso_z = [6.8340, 6.9020, 6.8876, 6.8230, 6.8449]
    texp = [600 * 4 + 1200 * 2, 1800 * 2 + 1400 * 4, 1200, 1200, 600 * 4 + 1200 * 28]

    return fitsfiles, qsoid, qso_z, texp

def mosfire_qso():
    fitsfiles = glob.glob('/Users/suksientie/Research/data_redux/wavegrid_vel/*/vel*_tellcorr.fits')
    qsoid = []
    for file in fitsfiles:
        qsoid.append(file.split('/wavegrid_vel/')[-1].split('/')[0])
    qso_z = [7.1, 7.642, 7.034, 7.001, 7.541]
    texp = [180 * 32, 180 * 22 * 2 + 180 * 60 + 180 * 4 + 180 * 36, 180 * 40, 180 * 40 + 180 * 36, 180 * 8 + 180 * 12 + 180 * 52]

    return fitsfiles, qsoid, qso_z, texp

###############
def create_connection(path):
    try:
        connection = sqlite3.connect(path)
        print("connection successful")
    except Error:
        connection = None
        print("error:", Error)

    return connection

def execute_query(connection, query):
    cursor = connection.cursor()
    cursor.execute(query)
    connection.commit()
    print("query executed successfully")

def create_qso_table():
    query = """
    CREATE TABLE IF NOT EXISTS qso (\n
    id TEXT, \n
    redshift FLOAT, \n
    instrument TEXT, \n
    texp INTEGER, \n
    path TEXT);"""

    return query

def insert_into_qso(qsoid, redshift, instrument, texp, path):
    query = """
    INSERT INTO
      qso (id, redshift, instrument, texp, path)
    VALUES
      ("%s", %f, "%s", %d, "%s"); """ % (qsoid, redshift, instrument, texp, path)

    return query

def run_sql_all():
    nires_fitsfiles, nires_qsoid, nires_z, nires_texp = nires_qso()
    xshooter_fitsfiles, xshooter_qsoid, xshooter_z, xshooter_texp = xshooter_qso()
    mosfire_fitsfiles, mosfire_qsoid, mosfire_z, mosfire_texp = mosfire_qso()

    sqlpath = 'highzqso.sqlite'
    con = create_connection(sqlpath)
    create_table = create_qso_table()
    execute_query(con, create_table)

    for i in range(len(nires_fitsfiles)):
        q = insert_into_qso(nires_qsoid[i], nires_z[i], 'nires', nires_texp[i], nires_fitsfiles[i])
        execute_query(con, q)

    for i in range(len(xshooter_fitsfiles)):
        q = insert_into_qso(xshooter_qsoid[i], xshooter_z[i], 'xshooter', xshooter_texp[i], xshooter_fitsfiles[i])
        execute_query(con, q)

    for i in range(len(mosfire_fitsfiles)):
        q = insert_into_qso(mosfire_qsoid[i], mosfire_z[i], 'mosfire', mosfire_texp[i], mosfire_fitsfiles[i])
        execute_query(con, q)

    con.close()

def add_col_qso(new_col, new_col_type):
    query = """
    ALTER TABLE qso
    ADD %s %s; 
    """ % (new_col, new_col_type)

    return query

###############
def final_qso_list(sqldb='highzqso.sqlite'):

    final_qso_id = ['J0411-0907', 'J0319-1008', 'J0410-0139', 'J0252-0503', 'J0038-1527', 'J0038-0653', 'J1342+0928', 'J0313-1806']
    con = create_connection(sqldb)
    df = pd.read_sql_query("select id, redshift, instrument from qso where id in " + str(tuple(final_qso_id)), con)
    arr = df.to_numpy()
    con.close()

    final_qso_id = arr[:,0]
    final_qso_z = arr[:,1]
    final_qso_instr = arr[:,2]
    final_qso_fitsfile_rebin = []

    rebin_path = '/Users/suksientie/Research/MgII_forest/rebinned_spectra/'

    for qso in final_qso_id:
        f = glob.glob(rebin_path + qso + '*')[0]
        final_qso_fitsfile_rebin.append(f)

    return final_qso_id, final_qso_z, final_qso_instr, final_qso_fitsfile_rebin

def average_telluric(final_qso_id):
    rebin_spec = glob.glob('/Users/suksientie/Research/MgII_forest/rebinned_spectra/*fits')

    wave_all = []
    tell_all = []
    for spec in rebin_spec:
        if spec.split('/')[-1].split('_')[0] in final_qso_id:
            data = fits.open(spec)[1].data
            wave = data['wave'].astype('float64')
            telluric = data['telluric'].astype('float64')

            mask = wave >= 19500
            wave_all.append(wave[mask])
            tell_all.append(telluric[mask])
            plt.plot(wave.squeeze(), telluric.squeeze(), drawstyle='steps-mid')

    return wave_all, tell_all

###############
def continuum_normalize_all():

    final_qso_id, final_qso_z, final_qso_instr, final_qso_fitsfile_rebin = final_qso_list()

    everyn_break_list = 20
    exclude_restwave = 1216 - 1185

    for i in range(len(final_qso_id)):
        fitsfile = final_qso_fitsfile_rebin[i]
        qsoid = final_qso_id[i]
        qsoz = final_qso_z[i]

        wave, flux, ivar, mask, std, tell, fluxfit, strong_abs_gpm = mutils.extract_and_norm(fitsfile, everyn_break_list, qsoid)
        redshift_mask, pz_mask, obs_wave_max = mutils.qso_redshift_and_pz_mask(wave, qsoz, exclude_restwave)

        plt.figure(figsize=(14, 8))
        plt.suptitle("%s (z=%0.2f)" % (qsoid, qsoz))
        plt.subplot(211)
        plt.plot(wave, flux, c='k', drawstyle='steps-mid')
        plt.plot(wave, fluxfit, c='r', drawstyle='steps-mid', label='continuum fit')
        plt.plot(wave, std, c='b', alpha=0.5, drawstyle='steps-mid', label='sigma')
        plt.legend()
        plt.xlim([19500, 24100])
        plt.ylim([-0.05, 0.7])
        plt.xlabel('Obs wave')
        plt.ylabel('Flux')

        plt.subplot(212)
        plt.plot(wave, flux / fluxfit, c='k', drawstyle='steps-mid')
        plt.plot(wave, std / fluxfit, c='b', alpha=0.5, drawstyle='steps-mid')
        plt.xlim([19500, 24100])
        plt.ylim([-0.05, 2.3])
        plt.xlabel('Obs wave')
        plt.ylabel('Normalized flux')

    plt.show()


def continuum_normalize_nires(sqldb='highzqso.sqlite'):
    con = create_connection(sqldb)
    df = pd.read_sql_query("select id, redshift, path from qso where instrument in ('nires') order by redshift", con)
    arr = df.to_numpy()

    everyn_break_list = 20
    exclude_restwave = 1216 - 1185

    instr = 'nires'
    for i in range(len(arr)):
        qsoid, qsoz, fitsfile = arr[i][0], arr[i][1], arr[i][2]
        wave, flux, ivar, mask, std, tell, fluxfit, strong_abs_gpm = mutils.extract_and_norm(fitsfile, everyn_break_list,qsoid)
        redshift_mask, pz_mask, obs_wave_max = mutils.qso_redshift_and_pz_mask(wave, qsoz, exclude_restwave)

        plt.figure(figsize=(14,8))
        plt.suptitle("%s (z=%0.2f), %s" % (qsoid, qsoz, instr))
        plt.subplot(211)
        plt.plot(wave, flux, c='k', drawstyle='steps-mid')
        plt.plot(wave, fluxfit, c='r', drawstyle='steps-mid', label='continuum fit')
        plt.plot(wave, std, c='b', alpha=0.5, drawstyle='steps-mid', label='sigma')
        plt.legend()
        plt.xlim([19500, 24100])
        plt.ylim([-0.05, 0.7])
        plt.xlabel('Obs wave')
        plt.ylabel('Flux')

        plt.subplot(212)
        plt.plot(wave, flux/fluxfit, c='k', drawstyle='steps-mid')
        plt.plot(wave, std/fluxfit, c='b', alpha=0.5, drawstyle='steps-mid')
        plt.xlim([19500, 24100])
        plt.ylim([-0.05, 2.3])
        plt.xlabel('Obs wave')
        plt.ylabel('Normalized flux')

    plt.show()

def continuum_normalize_xshooter(sqldb='highzqso.sqlite'):
    con = create_connection(sqldb)
    df = pd.read_sql_query("select id, redshift, path from qso where instrument in ('xshooter') order by redshift", con)
    arr = df.to_numpy()

    everyn_break_list = 20
    exclude_restwave = 1216 - 1185

    instr = 'xshooter'
    for i in range(len(arr)):
        qsoid, qsoz, fitsfile = arr[i][0], arr[i][1], arr[i][2]
        wave, flux, ivar, mask, std, tell, fluxfit, strong_abs_gpm = mutils.extract_and_norm(fitsfile, everyn_break_list, qsoid)
        redshift_mask, pz_mask, obs_wave_max = mutils.qso_redshift_and_pz_mask(wave, qsoz, exclude_restwave)

        plt.figure(figsize=(14,8))
        plt.suptitle("%s (z=%0.2f), %s" % (qsoid, qsoz, instr))
        plt.subplot(211)
        plt.plot(wave, flux, c='k', drawstyle='steps-mid')
        plt.plot(wave, fluxfit, c='r', drawstyle='steps-mid', label='continuum fit')
        plt.plot(wave, std, c='b', alpha=0.5, drawstyle='steps-mid', label='sigma')
        plt.legend()
        plt.xlim([19500, 24100])
        plt.ylim([-0.2, 0.9])
        plt.xlabel('Obs wave')
        plt.ylabel('Flux')

        plt.subplot(212)
        plt.plot(wave, flux/fluxfit, c='k', drawstyle='steps-mid')
        plt.plot(wave, std/fluxfit, c='b', alpha=0.5, drawstyle='steps-mid')
        plt.xlim([19500, 24100])
        plt.ylim([-1, 3.0])
        plt.xlabel('Obs wave')
        plt.ylabel('Normalized flux')
    plt.show()
