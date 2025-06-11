#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNSS Calibration Tool
Created on Mon Dec 30 15:34:07 2024

@author: diego
"""
import os
import time
import datetime
import numpy as np
from conf import (config, file_a, file_b, file_nav, pos_a, pos_b, delays_a, 
                  delays_b
)

from GNSS_cal_tools_subs import (
    OExyz, dfSTAgen, dfNAVgen, C1P1, outputs,
    ElevationReject, DIFgen, figures, loader, calibration, DIFgen1
)

# Limitations:
# Only one RINEX file per station
# No LZ files (the case when the two receivers don't have the same reference)
# Tested only for GPS

# Start time
start_time = time.time()

# Version
VERSION = '2/1/25'

# See if files and output folders are there
for f in [file_a, file_b, file_nav]:
    if not os.path.exists(f):
        raise FileNotFoundError(f'File not found: {f}')

if not os.path.exists('outputs'):
    os.makedirs('outputs')

# Date
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

# Loading data from files
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
      ': Loading NAV and OBS files')
nav = loader(file_nav, config)
sta_a = loader(file_a, config)
sta_b = loader(file_b, config)
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ': DONE\n')

# Generation of dataframes
df_sta_a = dfSTAgen(sta_a)
df_sta_b = dfSTAgen(sta_b)
dfnav = dfNAVgen(nav)

# Positions, distance and interval
x = pos_b - pos_a
dist = np.linalg.norm(x)

# Create a reduced dataframe of ephemeris, with only one entry per sat, per day
# Keep first non-NAN entrance of each sv:

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
      ': Creation of dataframe for epehemeris')    
first_occurrence_idx = dfnav.groupby('sv').apply(lambda x: x.index[0])
dfnav_first = dfnav.loc[first_occurrence_idx]
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ': DONE')


# Adding of EARTH FIXED COORDINATES (subroutine OExyz of dclrinex)
df_sta_a = OExyz(dfnav_first, df_sta_a)
df_sta_b = OExyz(dfnav_first, df_sta_b)

# Rejection at low elevation (line 1554 of dclrinex)
df_sta_a = ElevationReject(df_sta_a, pos_a, config, sta_a.filename)
df_sta_b = ElevationReject(df_sta_b, pos_b, config, sta_b.filename)

# Add C1P1 bias
sta_a = C1P1(sta_a,df_sta_a)
sta_b = C1P1(sta_b,df_sta_b)


# Genero diferencias
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
      ': Creation of dataframe of time differences')  
dif = DIFgen1(df_sta_a, df_sta_b, config, pos_a, pos_b)
#dif = DIFgen(df_sta_a, df_sta_b, config, pos_a, pos_b)
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ': DONE\n')

# Text Outputs and rawdif calculation. rawdiff = a - b
rawdiff = outputs(VERSION, st, nav, sta_a, sta_b, file_nav, dist, config, dif)

# Results of calibration (optional)
if config['calculate_delays']:
    delays_b = calibration(rawdiff, delays_a, delays_b, sta_a, sta_b)

# Figure Outputs
figures(dif, config, ts)

# Stop time
stop_time = time.time()
print(f"Execution time: {stop_time - start_time:.4f} seconds")
