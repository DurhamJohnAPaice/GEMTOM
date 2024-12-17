#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:45:23 2024

@author: hugo

Usage: BG_quick_selection.py [ndays=1] [snrlim=6.5] [fwhmlim=6] [run_sql=1] [nuclear|offnuclear|star|hostless]

List of useful files (see https://drive.google.com/drive/u/3/folders/13vas0SyV6zR2D0e2-K6QjTqXjkfiNzxp):
- DELVE catalog: delvedr2_mag21_withAGN.fits
- MOC of DESI and DECAPS sky coverage: MOC_DESI-Legacy-Surveys_DR10.fits, MOC_DECAPS_DR2.fits
"""

# print("Usage: BG_quick_selection.py [ndays=1] [snrlim=6.5] [fwhmlim=6] [run_sql=1] [nuclear|offnuclear|star|hostless]")

# Basic imports

from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time, TimeDelta
# import sys
import os
from pathlib import Path
# import getpass
import time
# import json
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from matplotlib.patches import Circle
# from astropy.io import fits
# from astropy.wcs import WCS
# from tqdm import tqdm
# from mocpy import MOC
import warnings
# import requests
# from PIL import Image
# from io import BytesIO
# from urllib.request import urlretrieve
warnings.filterwarnings('ignore')

ST = time.time()
querycat = 1

# BG-related imports

from blackpy import BlackGEM
from blackpy.catalogs.blackgem import TransientsCatalog
from blackpy.catalogs.gaia import GaiaCatalog

user_home = str(Path.home())
creds_user_file = user_home + "/.bg_follow_user_john_creds"
creds_db_file = user_home + "/.bg_follow_transientsdb_creds"
bg = BlackGEM(creds_user_file=creds_user_file, creds_db_file=creds_db_file)


tc = TransientsCatalog(bg)



####################
### Requirements ###
# Minimum RB score
min_rb = 0.8
# Minimum number of detections
min_ndet = 1
# Minimum SNR_zogy
min_snr = 6.5
# Maximum FWHM of the gaussian fit to source image
max_fwhm = 6
querycat = 1
pipeline = 'hostless'

####################

print("Running: BG_hostless_selection.py %s %s %d %s"%(min_snr,max_fwhm,querycat,pipeline))

ndays = 1
days_ago = 0

time0 = (date.today()-timedelta(days=days_ago+ndays)).isoformat()+" 12:00:00"
time0min = (date.today()-timedelta(days=days_ago+ndays+30)).isoformat()+" 12:00:00"
time1 = (date.today()-timedelta(days=days_ago)).isoformat()+" 12:00:00"

# folder to save files
# bgfiles = user_home + "/make-tom/GEMTOM_alt2/GEMTOM/data/history_transients/"
bgfiles = user_home + "/GEMTOM/GEMTOM/data/history_transients/"

## === Make sure we're running... ===
with open(bgfiles+'run_check.txt', 'w') as f:
    f.write(str(time.time()))
    f.write("\nRunning!")
f.close()

if not(os.path.isdir(bgfiles)):
    os.mkdir(bgfiles)

print("\nOutput directory:")
print(bgfiles)

### Save selection to the following filename:
fileout = bgfiles+"_BlackGEM_transients.csv"

min_det_fraction = 0.5
min_runcat = 0

if querycat:
    # print("\nQuerying period",time0, time1)
    print("\n")
    print("--- Initial Query: ---")


    st = time.time()
    # params = {'min_snr': min_snr,
    #           'min_ndet': min_ndet,
    #           'min_rb': min_rb,
    #           'max_fwhm': max_fwhm}
    params = {'min_snr': 6.5,
              'max_xyerr': 0.3,
              'min_ndet': min_ndet,
              'min_runcat': min_runcat,
              'max_fwhm': max_fwhm,
              'min_rb': 0.7,
              'time0': time0,
              'time0min': time0min,
              'time1': time1,
              'max_radec_std': 0.3/3600, # arcsec
              'min_det_fraction': min_det_fraction}

    print(params)

    # Query below explained:
    # Select source id and associated (first) detection id
    # from sources list, associations list, detections list, images list
    # compliant with number of detections requirement
    # and being associated to a detection compliant with both RB score and S/N requirements
    # which was taken in the required time interval

    qu = """\
    SELECT  r.id
           ,r.ra_deg
           ,r.dec_deg
           ,AVG(x.ra_psf_d)
           ,AVG(x.dec_psf_d)
           ,r.xtrsrc
           ,r.datapoints
           ,  MIN(CASE WHEN x.filter = 'q' THEN mag_zogy END) AS   min_mag_zogy_q
           ,  MAX(CASE WHEN x.filter = 'q' THEN mag_zogy END) AS   max_mag_zogy_q
           ,  AVG(CASE WHEN x.filter = 'q' THEN mag_zogy END) AS   avg_mag_zogy_q
           ,  AVG(CASE WHEN x.filter = 'q' THEN class_real END) AS avg_rb_q
           ,COUNT(CASE WHEN x.filter = 'q' THEN mag_zogy END) AS num_datapoints_q
           ,  MIN(CASE WHEN x.filter = 'u' THEN mag_zogy END) AS   min_mag_zogy_u
           ,  MAX(CASE WHEN x.filter = 'u' THEN mag_zogy END) AS   max_mag_zogy_u
           ,  AVG(CASE WHEN x.filter = 'u' THEN mag_zogy END) AS   avg_mag_zogy_u
           ,  AVG(CASE WHEN x.filter = 'u' THEN class_real END) AS avg_rb_u
           ,COUNT(CASE WHEN x.filter = 'u' THEN mag_zogy END) AS num_datapoints_u
           ,  MIN(CASE WHEN x.filter = 'i' THEN mag_zogy END) AS   min_mag_zogy_i
           ,  MAX(CASE WHEN x.filter = 'i' THEN mag_zogy END) AS   max_mag_zogy_i
           ,  AVG(CASE WHEN x.filter = 'i' THEN mag_zogy END) AS   avg_mag_zogy_i
           ,  AVG(CASE WHEN x.filter = 'i' THEN class_real END) AS avg_rb_i
           ,COUNT(CASE WHEN x.filter = 'i' THEN mag_zogy END) AS num_datapoints_i
           ,COUNT(mag_zogy) AS num_datapoints_all
         FROM runcat r
              ,assoc a
              ,extractedsource x
              ,image i
         WHERE r.datapoints >= %(min_ndet)s
           AND a.runcat > %(min_runcat)s
           AND r.id = a.runcat
           AND a.xtrsrc = x.id
           AND x.class_real > %(min_rb)s
           AND x.snr_zogy > %(min_snr)s
           AND x.xerr_psf_d < %(max_xyerr)s
           AND x.yerr_psf_d < %(max_xyerr)s
           AND x.chi2_psf_d BETWEEN 0 AND 2
           AND x.image = i.id
           AND x.mag_zogy < 99
           AND i."date-obs" BETWEEN '%(time0)s'
                                AND '%(time1)s'
        GROUP BY r.id, r.xtrsrc, r.ra_deg, r.dec_deg, datapoints
        HAVING STDDEV_POP(x.ra_psf_d)*COS(RADIANS(AVG(x.dec_psf_d))) < %(max_radec_std)s
           AND STDDEV_POP(x.dec_psf_d) < %(max_radec_std)s
        ORDER BY r.id
    """
    query = qu % (params)

    # Call the run_query function that can execute any sql query
    l_results = bg.run_query(query)

    et = time.time()
    print("Query complete. Elapsed time:",et-st,"seconds")


    # For display purposes, cast the list to a pandas dataframe
    df_new_runcats = pd.DataFrame(l_results, columns=[
        'runcat_id',
        'ra',
        'dec',
        'ra_psf',
        'dec_psf',
        'xtrsrc',
        'n_datapoints',
        'q_min',
        'q_max',
        'q_avg',
        'q_rb_avg',
        'q_num',
        'u_min',
        'u_max',
        'u_avg',
        'u_rb_avg',
        'u_num',
        'i_min',
        'i_max',
        'i_avg',
        'i_rb_avg',
        'i_num',
        'all_num_datapoints',
        ])
    n_new = len(np.unique(df_new_runcats['runcat_id']))
    print("Found %d transients"%n_new)

    # Exclude asteroids
    print("\n")
    print("--- Gate 1: Number of Detections ---")
    print(" Current: %d"%n_new)
    st = time.time()

    df_new_runcats = df_new_runcats[df_new_runcats.all_num_datapoints > min_ndet]
    n_new = len(np.unique(df_new_runcats['runcat_id']))

    if min_ndet == 1:   ndet_plural = "s"
    else:               ndet_plural = ""
    print("%d transients have >%d valid datapoint%s"%(n_new, min_ndet,ndet_plural))

    if len(df_new_runcats)==0:
        raise SystemExit("No transient matching requirements. Stopping code here.")

    et = time.time()
    print("Elapsed time:",et-st,"seconds")

    print(" Remaining: %d"%n_new)


    # Exclude asteroids
    print("\n")
    print("--- Gate 2: Asteroids ---")
    print(" Current: %d"%n_new)
    st = time.time()
    xtrsrc_ids = tuple(df_new_runcats.xtrsrc)
    runcat_ids = tuple(df_new_runcats.runcat_id)
    #l_columns, l_results = tc.transients_have_asteroid_detection(runcat_ids)
    l_columns, l_results = tc.xtrsrcs_are_asteroid(xtrsrc_ids)
    df_asteroids = pd.DataFrame(l_results, columns=l_columns)
    asteroidmask = df_new_runcats.xtrsrc.isin(tuple(df_asteroids['tx.id']))
    #print(list(df_asteroids['ta.runcat']))
    #asteroidmask = df_new_runcats.runcat_id.isin(tuple(df_asteroids['ta.runcat']))
    df_new_runcats = df_new_runcats[~asteroidmask]

    num_asteroids = len(asteroidmask[asteroidmask==True])
    if num_asteroids == 1: print("Found %d asteroid"%num_asteroids)
    else: print("Found %d asteroids"%num_asteroids)
    et = time.time()
    print("Elapsed time:",et-st,"seconds")
    if len(df_new_runcats)==0:
        raise SystemExit("All detections are of asteroids. Stopping code here.")

    iu = np.unique(df_new_runcats['runcat_id'],return_index=True)[1] # unique runcat_id
    df_new_runcats = df_new_runcats.iloc[iu]
    # print("remaining_transients:",len(df_new_runcats))
    print(" Remaining: %d"%len(df_new_runcats))


    # EXCLUDE OLD GOOD RUNCATS, MATCHING ALL REQUIREMENTS EXCEPT DATE
    print("\n")
    print("--- Gate 3: Old, Good Runcats ---")
    print(" Current: %d"%len(df_new_runcats))
    qu = """\
    SELECT r.id
         FROM runcat r
              ,extractedsource x
              ,image i
         WHERE r.datapoints >= %(min_ndet)s
           AND r.xtrsrc = x.id
           AND x.class_real > %(min_rb)s
           AND x.snr_zogy > %(min_snr)s
           AND x.fwhm_gauss_d < %(max_fwhm)s
           AND x.image = i.id
           AND i."date-obs"< '%(time0)s'
        GROUP BY r.id
    """
    query = qu % (params)


    # Call the run_query function that can execute any sql query
    l_results = bg.run_query(query)

    # For display purposes, cast the list to a pandas dataframe
    df_old_good_runcats = pd.DataFrame(l_results, columns=['runcat_id'])
    et = time.time()
    print("Found %d transients with older good detections"%len(df_old_good_runcats))
    df_new_runcats = df_new_runcats.iloc[~np.isin(df_new_runcats['runcat_id'],df_old_good_runcats['runcat_id'])]

    print("Elapsed time:",et-st,"seconds")
    print(" Remaining: %d"%len(df_new_runcats))


    # EXCLUDE OLD BAD RUNCATS, DETECTED A LONG TIME AGO (>30 days before query) WITH POOR RB OR SNR
    print("\n")
    print("--- Gate 4: Old, Bad Runcats ---")
    print(" Current: %d"%len(df_new_runcats))
    qu = """\
    SELECT r.id
         FROM runcat r
              ,extractedsource x
              ,image i
         WHERE r.datapoints >= 5
           AND r.xtrsrc = x.id
           AND x.image = i.id
           AND i."date-obs"< '%(time0min)s'
        GROUP BY r.id
    """
    query = qu % (params)


    # Call the run_query function that can execute any sql query
    l_results = bg.run_query(query)

    # For display purposes, cast the list to a pandas dataframe
    df_old_bad_runcats = pd.DataFrame(l_results, columns=['runcat_id',])
    et = time.time()
    print("Found %d transients with old (>30d before query), bad detections"%len(df_old_bad_runcats))
    df_new_runcats = df_new_runcats.iloc[~np.isin(df_new_runcats['runcat_id'],df_old_bad_runcats['runcat_id'])]

    print("Elapsed time:",et-st,"seconds")
    print(" Remaining: %d"%len(df_new_runcats))

    # Save file
    df_new_runcats.to_csv(fileout)

# MATCH WITH GAIA AND GLADE
print("\n")
print("--- Gate 5: Gaia Crossmatch ---")

st = time.time()
res = df_new_runcats
print(" Current: %d"%len(res))
# print("Loaded file of %d entries"%len(res))

filegaia = fileout.replace(".csv","_gaia.csv")

rad = 2 # arcsec

cmd = 'stilts cdsskymatch in=%s ra="ra" dec="dec" cdstable="Gaia DR3 (Epoch 2016)" find=all radius=%f  out="%s"'%(fileout,rad,filegaia)
print(cmd)
# if not(os.path.isfile(filegaia)):
#     print("Gaia file does not exist. Crossmatching...")
#     os.system(cmd)
#     df_gaia = pd.read_csv(filegaia)
# else:
#     print("Gaia file already exists; using that one.")
#     df_gaia = pd.read_csv(filegaia)
os.system(cmd)
df_gaia = pd.read_csv(filegaia)


star = df_gaia.loc[(df_gaia['PSS']+df_gaia['PQSO'])>0.5]
star.reset_index(drop=True, inplace=True)

mask = np.isin(res['runcat_id'],star['runcat_id'])

if pipeline=="star":
    res = res.loc[mask]
    res.reset_index(drop=True, inplace=True)
elif pipeline=="hostless":
    print("Hostless pipeline active; removing all Gaia sources")
    mask = np.isin(res['runcat_id'],df_gaia['runcat_id'])
    res = res.loc[~mask]
    res.reset_index(drop=True, inplace=True)
else:
    res = res.loc[~mask]
    res.reset_index(drop=True, inplace=True)
    print("%d entries not found in Gaia."%len(res))
et = time.time()
print("Elapsed time:",et-st,"seconds")
print(" Remaining: %d"%len(res))



### CROSSMATCH WITH DELVE
print("\n")
print("--- Gate 6: Delve Crossmatch ---")
print(" Current: %d"%len(res))
st = time.time()

if pipeline!='star':


    max_sep = 8 # arcsec.
    nuc_sep = 2 # arcsec.

    print("Looking for %d entries in Delve..."%len(res))
    # g = Table.read('/Users/JohnAPaice/make-tom/GEMTOM_alt2/GEMTOM/data/blackgem_crossmatch/delvedr2_mag21_withAGN.fits')
    g = Table.read(user_home + '/GEMTOM/GEMTOM/data/blackgem_crossmatch/delvedr2_mag21_withAGN.fits')
    print("delve catalog is read")

    ra1 = np.asarray(res['ra'])
    dec1 = np.asarray(res['dec'])
    ramin, ramax = ra1.min(), ra1.max()
    decmin, decmax = dec1.min(), dec1.max()
    g = g[g['ra']>ramin-1e-2]
    g = g[g['ra']<ramax+1e-2]
    g = g[g['dec']>decmin-1e-2]
    g = g[g['dec']<decmax+1e-2]
    ra2 = g['ra']
    dec2 = g['dec']

    c1 = SkyCoord(ra=ra1,dec=dec1,frame="icrs",unit="deg") # coordinates of BG objects
    c2 = SkyCoord(ra=ra2,dec=dec2,frame="icrs",unit="deg") # coordinates of galaxies
    idx, d2d, d3d = c1.match_to_catalog_sky(c2)

    # compare each separation to the matching radius
    if pipeline=='nuclear':
        posmatch = (d2d.arcsec < nuc_sep)
    elif pipeline=='offnuclear':
        posmatch = np.logical_and(d2d.arcsec > nuc_sep, d2d.arcsec < max_sep)
    else:
        posmatch = d2d.arcsec > max_sep

    # absmag = res['min']-5*np.log10(g[idx]['D']*1e6)+5
    absmag = res['q_min']-5*np.log10(g[idx]['D'])+5

    # select SNe, ILOTs and bright novae
    if pipeline in ['nuclear','offnuclear']:
        magselect = abs(absmag+14)<=11
        selection = np.logical_and(list(posmatch), list(magselect))
        print("%d entries found in Delve"%len(selection))
    else:
        selection = posmatch
        print("%d entries not found in Delve"%len(selection))

else:
    selection = range(len(res))
    print("Star pipeline active; not looking in Delve.")


et = time.time()
print("Elapsed time:",et-st,"seconds")
print(" Remaining: %d"%len(selection))

res_select = res.loc[selection]


df_new_runcats = res_select.copy()


## REMOVE ANY SOURCES WITH THE BULK OF DETECTIONS >1.5 ARCSEC AWAY
print("\n")
print("--- Gate 7: Removing transients with too-far bulk detections...... ---")
print(" Current: %d"%len(df_new_runcats))
st = time.time()

df_new_runcats["det_sep"] = list( \
    SkyCoord(np.array(df_new_runcats["ra_psf"])*u.deg, np.array(df_new_runcats["dec_psf"])*u.deg).separation( \
    SkyCoord(np.array(df_new_runcats["ra"])*u.deg,     np.array(df_new_runcats["dec"])*u.deg) \
    ).arcsecond)
df_new_runcats = df_new_runcats[df_new_runcats.det_sep < 1.5]
print("%d entries with bulk detections >1.5 arcsec"%len(df_new_runcats[df_new_runcats.det_sep >= 1.5]))

et = time.time()
print("Elapsed time:",et-st,"seconds")
print(" Remaining: %d"%len(df_new_runcats))



do_new_fiddling = False
if do_new_fiddling:

    print("\n")
    print("--- Gate 8: Hugo's new fiddling... ---")
    print(" Current: %d"%len(res))
    st = time.time()

    if len(res_select)>0:
        # Check upper limits
        # list of detections for selected runcats
        params['runcatids'] = tuple(df_new_runcats['runcat_id']) + (df_new_runcats['runcat_id'].iloc[0],)
        qu = """ SELECT a.runcat
                      , i.id
                      , i.object
                      , i.filter
                      , i."date-obs"
                   FROM assoc a
                      , extractedsource x
                      , image i
                  WHERE a.runcat IN %(runcatids)s
                    AND a.xtrsrc = x.id
                    AND x.image = i.id
         """
        query = qu % (params)
        l_results = bg.run_query(query)
        df_detections = pd.DataFrame(l_results, columns=['runcat_id','image','fieldid','filter','date'])

        params['fieldids'] = tuple(np.unique(df_detections.fieldid)) + (np.unique(df_detections.fieldid)[0],)

        # list of observations for selected runcats
        qu = """ SELECT i.id
                      , i.object
                      , i.filter
                      , i."date-obs"
                   FROM image i
                  WHERE i.object IN %(fieldids)s
        """
        query = qu % (params)
        l_results = bg.run_query(query)
        df_observations = pd.DataFrame(l_results, columns=['image','fieldid','filter','date'])

        mask = []

        ## For each ID,
        for runcat in df_new_runcats['runcat_id']:
            keep = False

            ## Get all its detections, and all the filters.
            runcat_det = df_detections.loc[df_detections['runcat_id']==runcat]
            flters = np.unique(runcat_det['filter'])

            ## For each filter,
            for flter in flters:

                ## Only get detections in that filter.
                runcat_det_filt = runcat_det.loc[runcat_det['filter']==flter]

                ## Only get observations in that filter, with that runcat,
                runcat_obs_filt = df_observations.loc[np.logical_and(np.isin(df_observations['fieldid'],runcat_det['fieldid']),
                                                                     df_observations['filter']==flter)]
                runcat_obs_filt_postdet = runcat_obs_filt.loc[runcat_obs_filt['date']>=runcat_det_filt['date'].min()]
                if len(runcat_det_filt)/len(runcat_obs_filt_postdet)>min_det_fraction:
                    keep = True

    et = time.time()
    print("Elapsed time:",et-st,"seconds")
    print(" Remaining: %d"%len(df_new_runcats))

res_select = df_new_runcats

print("\n")
print("--- Result ---")

print("%d sources found!"%len(res_select))

# res_select = res_select.iloc[np.argsort(res_select['D_gal'])]
#
# res_select = res_select.sort_values(by=['runcat_id'])
# print(res_select)
print(res_select)
#
# ## Get the magnitudes in each filter
# def get_filter_mags(df):
#     for filt in ['q','u','i']:
#         print("Finding "+filt+"-band datapoints...")
#         params = {  'runcatids' : tuple(list(df.runcat_id)),
#                     'filter'    : filt,
#                     'min_snr'   : min_snr,
#                     'min_ndet'  : min_ndet,
#                     'time0'     : time0,
#                     'time1'     : time1,
#                     'time0min'  : time0min,
#                     'min_rb'    : min_rb,
#                     'max_fwhm'  : max_fwhm
#         }
#
#         # print(params)
#
#         # Query below explained:
#         # Select source id and associated (first) detection id
#         # from sources list, associations list, detections list, images list
#         # compliant with number of detections requirement
#         # and being associated to a detection compliant with both RB score and S/N requirements
#         # which was taken in the required time interval
#
#
#         num_datapoints = np.empty(len(df))
#         magmin = np.empty(len(df))
#         magmax = np.empty(len(df))
#         magavg = np.empty(len(df))
#         num_datapoints[:] = np.nan
#         magmin[:] = np.nan
#         magmax[:] = np.nan
#         magavg[:] = np.nan
#
#         qu = """\
#         SELECT a.runcat runcat_id
#                    ,MIN(x.mag_zogy) AS magmin
#                    ,MAX(x.mag_zogy) AS magmax
#                    ,AVG(x.mag_zogy) AS magavg
#                    ,COUNT(x.mag_zogy) as num_datapoints
#              FROM runcat r
#                   ,assoc a
#                   ,extractedsource x
#                   ,image i
#              WHERE a.runcat IN %(runcatids)s
#                AND x.class_real > %(min_rb)s
#                AND x.snr_zogy > %(min_snr)s
#                AND x.image = i.id
#                AND i."date-obs" BETWEEN '%(time0)s'
#                                     AND '%(time1)s'
#                AND a.xtrsrc = x.id
#                AND x.filter = '%(filter)s'
#                AND x.mag_zogy < 99
#              GROUP BY a.runcat
#            ORDER BY runcat
#         """
#
#         """
#                AND x.fwhm_gauss_d BETWEEN 3 AND %(max_fwhm)s
#         """
#
#         query = qu % (params)
#
#         # Call the run_query function that can execute any sql query
#         l_results = bg.run_query(query)
#
#         df_filt = pd.DataFrame(l_results, columns=['runcat_id',
#                                                    'magmin',
#                                                    'magmax',
#                                                    'magavg',
#                                                    'num_datapoints'
#         ])
#
#         i,i1,i2 = np.intersect1d(df['runcat_id'],
#                                  df_filt['runcat_id'],
#                                  return_indices = True)
#
#         magmin[i1] = df_filt.loc[i2]['magmin']
#         magmax[i1] = df_filt.loc[i2]['magmax']
#         magavg[i1] = df_filt.loc[i2]['magavg']
#         num_datapoints[i1] = df_filt.loc[i2]['num_datapoints']
#
#         df['%s_min'%filt] = magmin
#         df['%s_max'%filt] = magmax
#         df['%s_avg'%filt] = magavg
#         df['%s_dif'%filt] = magmax-magmin
#         df['%s_num_datapoints'%filt] = num_datapoints
#
#     return df
#
# res_select = get_filter_mags(res_select)
# res_unselect = get_filter_mags(res_unselect)

# res_select['q_abs'] = res_select['q_avg']-5*np.log10(res_select['D_gal'])+5
# res_select['u_abs'] = res_select['u_avg']-5*np.log10(res_select['D_gal'])+5
# res_select['i_abs'] = res_select['i_avg']-5*np.log10(res_select['D_gal'])+5

# filebgselection = fileout.replace(".csv","_selected.csv")
# filebgunselection = fileout.replace(".csv","_selected_"+str(ndays)+"_"+str(time1)[:10]+".csv")
# res_select.to_csv(filebgselection)



# ##  --- TNS functions ---
# ## NOT USED. TNS doesn't like you searching for more than about 10 sources at a time, so let's leave this for now.
#
# from collections import OrderedDict
#
# ## Get TNS token
# print("Loading dotenv...")
# from dotenv import load_dotenv, dotenv_values
# load_dotenv()
# print(dotenv_values())
# print("Dotenv loaded.")
#
# TNS                 = "www.wis-tns.org"
# url_tns_api         = "https://" + TNS + "/api/get"
#
# TNS_BOT_ID          = "187806"
# TNS_BOT_NAME        = "BotGEM"
# TNS_API_KEY         = os.getenv('TNS_API_TOKEN', 'TNS_API_TOKEN not set')
#
# def set_bot_tns_marker():
#     tns_marker = 'tns_marker{"tns_id": "' + str(TNS_BOT_ID) + '", "type": "bot", "name": "' + TNS_BOT_NAME + '"}'
#     return tns_marker
#
# def search(search_obj):
#     search_url = url_tns_api + "/search"
#     tns_marker = set_bot_tns_marker()
#     headers = {'User-Agent': tns_marker}
#     json_file = OrderedDict(search_obj)
#     search_data = {'api_key': TNS_API_KEY, 'data': json.dumps(json_file)}
#     response = requests.post(search_url, headers = headers, data = search_data)
#     print("TNS API Key:", TNS_API_KEY)
#
#     return response
#
#
# # def get(get_obj):
# #     get_url = url_tns_api + "/object"
# #     tns_marker = set_bot_tns_marker()
# #     headers = {'User-Agent': tns_marker}
# #     json_file = OrderedDict(get_obj)
# #     get_data = {'api_key': TNS_API_KEY, 'data': json.dumps(json_file)}
# #     response = requests.post(get_url, headers = headers, data = get_data)
# #     return response
#
# def format_to_json(source):
#     parsed = json.loads(source, object_pairs_hook = OrderedDict)
#     result = json.dumps(parsed, indent = 4)
#     return result
#
# def get_tns_from_ra_dec(ra, dec, radius):
#
#     search_obj          = [("ra", str(ra)), ("dec", str(dec)), ("radius", str(radius)), ("units", "arcsec"), ("objname", ""),
#                        ("objname_exact_match", 0), ("internal_name", ""),
#                        ("internal_name_exact_match", 0), ("objid", ""), ("public_timestamp", "")]
#
#     response = search(search_obj)
#     json_data = format_to_json(response.text)
#     # print(json_data)
#     json_data = json.loads(json_data)
#     print("ID Code:", json_data["id_code"])
#     if json_data["id_code"] == 429:
#         return "Too many requests!"
#     elif json_data["id_code"] == 401:
#         return "Unauthorised!"
#     else:
#         print("ID Code:", json_data["id_code"])
#         print("ID Code:", json_data["id_code"])
#         print(json_data.keys())
#         print(json_data["data"]["reply"])
#         print(len(json_data["data"]["reply"]))
#         return json_data

# search_radius = 10
# tns_assoc = []
# for ra, dec in zip(res_select['ra'], res_select['dec']):
#     print("Querying RA %.3f"%ra, "Dec %.3f"%dec, "...")
#     tns_data = get_tns_from_ra_dec(ra, dec, search_radius)
#     if tns_data == "Too many requests!":
#         tns_text = "Too many TNS requests. Please check later."
#         tns_list = []
#     elif tns_data == "Unauthorised!":
#         tns_text = "Note: TNS Unauthorised. Please check."
#         tns_list = []
#     else:
#         tns_reply = tns_data["data"]["reply"]
#         tns_reply_length = len(tns_data["data"]["reply"])
#         if tns_reply_length == 0:
#             tns_text = "No TNS object found within " + str(search_radius) + " arcseconds."
#             tns_assoc.append("")
#         else:
#             tns_text = "TNS results within " + str(search_radius) + " arcseconds"
#             tns_list = tns_reply
#             tns_assoc.append(tns_data["data"]["reply"][0]['objname'])
#
# res_select["tns"] = tns_assoc






obs_date = str(time0)[:4]+str(time0)[5:7]+str(time0)[8:10]
# fileout = user_home + "/make-tom/GEMTOM_alt2/GEMTOM/data/history_transients/" + obs_date + "_" + pipeline + ".csv"
fileout = user_home + "/GEMTOM/GEMTOM/data/history_transients/" + obs_date + "_" + pipeline + ".csv"

# res_select = res_select.sort_values(by=['n_datapoints'], ascending=False)
res_select = res_select.sort_values(by=['i_rb_avg'], ascending=False)
res_select = res_select.sort_values(by=['u_rb_avg'], ascending=False)
res_select = res_select.sort_values(by=['q_rb_avg'], ascending=False)

res_select.to_csv(fileout)
print("File outputted")

# plt.figure()
# plt.hist(res_select['q_min'],bins=30)
# plt.xlabel('peak mag')
# plt.savefig(bgfiles+'hist_mag.png')
# plt.figure()
# plt.hist(res_select['n_datapoints'],bins=np.geomspace(1,1000,30))
# plt.gca().set_xscale('log')
# plt.xlabel('N datapoints')
# plt.savefig(bgfiles+'hist_datapoints.png')

ET = time.time()
print("total elapsed time:",ET-ST,"seconds")


# 38635490.0, 78.49326426407464, -69.70990428477359, 78.49352575950519, -69.70983874006888, 176720624.0, 349.0, 16.59682273864746, 16.59682273864746, 16.59682273864746, 1.0, 16.008560180664062, 16.398263931274414, 16.21450585653616, 215.0, 16.174360275268555, 17.79743766784668, 16.574434280395508, 15.0, 231.0, -0.35992842385934765
