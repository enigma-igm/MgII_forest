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
    qso_z = [7.642, 7.541, 7.001, 7.034, 7.1]
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

###############
def run_all():
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

# df = pd.read_sql_query("select * from qso order by redshift", con)