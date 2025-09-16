#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Basic imports
import numpy as np
import pandas as pd
import warnings
from astropy.time import Time
from pathlib import Path

warnings.filterwarnings('ignore')

# BG-related imports
from blackpy import BlackGEM
from blackpy.catalogs.blackgem import TransientsCatalog
from blackpy.catalogs.gaia import GaiaCatalog

user_home = str(Path.home())
creds_user_file = user_home + "/.bg_follow_user_john_creds"
creds_db_file = user_home + "/.bg_follow_transientsdb_creds"
bg = BlackGEM(creds_user_file=creds_user_file, creds_db_file=creds_db_file)

tc = TransientsCatalog(bg)



#####

## Read in all the targets
# df_targets = pd.read_csv("./GEMTOM/data/target_watchlist_ids.csv")
df_targets = pd.read_csv("./data/target_watchlist_ids.csv")

## Query for the latest magnitude
qu = """\
    SELECT runcat, mag_zogy, mjd, filter
    FROM (
        SELECT a.runcat,
               x.mag_zogy,
               i."mjd-obs" AS mjd,
               i.filter AS filter,
               ROW_NUMBER() OVER (PARTITION BY a.runcat ORDER BY i."mjd-obs" DESC) AS rn
        FROM assoc a
        JOIN extractedsource x ON a.xtrsrc = x.id
        JOIN image i ON x.image = i.id
       WHERE a.runcat IN %(id_list)s
         AND x.mag_zogy < 99
    ) sub
    WHERE rn = 1;
"""

clean_ID_list = [int(x) for x in df_targets["BlackGEM ID"] if str(x) != 'nan']
clean_ID_list = [int(x) for x in clean_ID_list if len(str(x)) != 0]

params = {'id_list' : tuple(clean_ID_list)}
query = qu % (params)
print("Running Query...")
l_results = bg.run_query(query)

df_targets_2 = pd.DataFrame(l_results, columns=['BlackGEM ID','latest_mag','last_obs','filter'])


## Combine the two dataframes together
df_targets['BlackGEM ID'] = df_targets['BlackGEM ID'].fillna(0)
df_targets["BlackGEM ID"] = df_targets["BlackGEM ID"].astype(int)

df_targets_all = pd.merge(df_targets, df_targets_2, on="BlackGEM ID")#, how="left")

## Get the dataframes ready for the table
df_targets_all = df_targets_all.sort_values(by=['last_obs'], ascending=False).reset_index(drop=True)

df_targets_all['BlackGEM ID'] = df_targets_all['BlackGEM ID'].fillna(0)

df_targets_all['GEMTOM_Link'] = df_targets_all['id'].apply(lambda x: f'[Target Page](https://gemtom.blackgem.org/targets/{str(x)}/)')
df_targets_all['BGEM_ID_Link'] = df_targets_all['BlackGEM ID'].apply(
    lambda x: f'[{str(x)}](https://gemtom.blackgem.org/transients/{str(x)}/)' if x != 0 else x
)
df_targets_all['last_obs'] = Time(df_targets_all['last_obs'], format='mjd').iso
df_targets_all['last_obs'] = [x.split(" ")[0] for x in df_targets_all['last_obs']]

## Export
# df_targets_all.to_csv("./GEMTOM/data/target_watchlist_latestmags.csv", index=False)
df_targets_all.to_csv("./data/target_watchlist_latestmags.csv", index=False)
