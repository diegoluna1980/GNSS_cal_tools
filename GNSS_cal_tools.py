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
    OExyz, dfSTAgen, dfNAVgen, C1P1, outputsG, outputsE,
    ElevationReject, figuresG, figuresE, loader, calibration, DIFgenG, DIFgenE,
    APOcorrection, multipath
)

# Limitations:
# Only one RINEX file per station
# No LZ files (the case when the two receivers don't have the same reference)

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
nav, nav_hdr = loader(file_nav, config)
sta_a, sta_a_hdr = loader(file_a, config)
sta_b, sta_b_hdr = loader(file_b, config)
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ': DONE\n')

# Generation of dataframes
df_sta_a = dfSTAgen(sta_a)
df_sta_b = dfSTAgen(sta_b)
dfnav = dfNAVgen(nav)

# Apply Antenna Phase offsets correction of RECEIVER
pos_a = APOcorrection(pos_a,sta_a_hdr)
pos_b = APOcorrection(pos_b,sta_b_hdr)

# Positions, distance and interval
x = pos_b - pos_a
dist = np.linalg.norm(x)

# Create a reduced dataframe of ephemeris, with only one entry per sat, per day
# Keep first non-NAN entrance of each sv:
first_occurrence_idx = dfnav.groupby('sv').apply(lambda x: x.index[0])
dfnav_first = dfnav.loc[first_occurrence_idx]


# Adding of EARTH FIXED COORDINATES (subroutine OExyz of dclrinex)
# and removing unhealthy satellites
df_sta_a = OExyz(dfnav_first, df_sta_a, sta_a.filename)
df_sta_b = OExyz(dfnav_first, df_sta_b, sta_b.filename)


# Rejection at low elevation (line 1554 of dclrinex)
df_sta_a = ElevationReject(df_sta_a, pos_a, config, sta_a.filename, st)
df_sta_b = ElevationReject(df_sta_b, pos_b, config, sta_b.filename, st)

#Add Multipath Error estimation https://ieeexplore.ieee.org/document/8316317
# https://www.nature.com/articles/s44172-025-00355-z
df_sta_a, sta_a = multipath(df_sta_a, config, sta_a, st)
df_sta_b, sta_b = multipath(df_sta_b, config, sta_b, st)


# Add C1P1 bias
if config['SYS'] == 'G':
    sta_a = C1P1(sta_a,df_sta_a)
    sta_b = C1P1(sta_b,df_sta_b)

# Generation of differences, text Outputs and rawdif calculation.
# rawdiff = a - b

if config['SYS'] == 'G':
    dif = DIFgenG(df_sta_a, df_sta_b, config, pos_a, pos_b)
    rawdiff = outputsG(VERSION, st, nav, sta_a, sta_b, file_nav, dist, config, dif)
 
if config['SYS'] == 'E':
    dif = DIFgenE(df_sta_a, df_sta_b, config, pos_a, pos_b)
    rawdiff = outputsE(VERSION, st, nav, sta_a, sta_b, file_nav, dist, config, dif)


# Results of calibration (optional)
if config['calculate_delays'] and config['SYS'] == 'G':
    delays_b = calibration(rawdiff, delays_a, delays_b, sta_a, sta_b)

# Figure Outputs
if config['SYS'] == 'G':
    figuresG(dif, config, ts, sta_a, sta_b)
if config['SYS'] == 'E':
    figuresE(dif, config, ts, sta_a, sta_b)


# Stop time
stop_time = time.time()
print(f"Execution time: {stop_time - start_time:.4f} seconds")


