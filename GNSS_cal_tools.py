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

from GNSS_cal_tools_subs import (
    OExyz, dfSTAgen, dfNAVgen, C1P1, outputs,
    ElevationReject, DIFgen, figures, loader, calibration
)

# Limitations:
# Only one RINEX file per station
# No LZ files (the case when the two receivers don't have the same reference)
# Tested only for GPS

# =============================================================================
# Start of inputs
# =============================================================================

config = {
    'elmin': 5,    # Elevation minimum, in degrees
    'intcod': 300, # Default: one code average every 300 s
    'ithr': 20,    # Default code threshold = 20 ns
    'thres': 0.05, # Option to read residual threshold (L336)
    'SYS': 'G',    # System to calibrate (GPS:G, Galileo:R, Glonass:R, Beidu:C)
    # PLOT AND CALCULATION OPTIONS
    'plotelevations': True,        # Plot histograms of elevations
    'timeplots': True,             # Plot time differneces and allan deviations
    'calculate_delays': True,      # Calculation of delays in DUT receiver 
}

# RINEX OBS files
file_a = 'AGGO2350.24O'
file_b = 'SIMr2350.24O'  # The station that will be calibrated

# RINEX navigation file
file_nav = 'BRDC00IGS_R_20242350000_01D_MN.rnx'

# Positions extracted from NRCan PPP solutions
pos_a = np.array([2765121.467, -4449250.973, -3626403.769])
pos_b = np.array([2765129.907, -4449245.382, -3626402.075])

# Delays in receivers (optional)
# delays_a are the values in the calibrated receiver
delays_a = {
    'INTdlyC1': 31.9,
    'INTdlyP1': 30.1,
    'INTdlyP2': 028.3,    
    'CABdly': 207.9,
    'REFdly': 12.3,
    }

# delays_b are the values in the DUT receiver
delays_b = {
    'INTdlyC1': np.nan,  # Leave NaN. Will be calculated
    'INTdlyP1': np.nan,  # Leave NaN. Will be calculated
    'INTdlyP2': np.nan,  # Leave NaN. Will be calculated
    'CABdly': 328.3,
    'REFdly': 13.7,
    }

# =============================================================================
# End of inputs
# =============================================================================

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
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ': DONE')

# Generation of dataframes
df_sta_a = dfSTAgen(sta_a)
df_sta_b = dfSTAgen(sta_b)
dfnav = dfNAVgen(nav)

# Positions, distance and interval
x = pos_b - pos_a
dist = np.linalg.norm(x)

# Create a reduced dataframe of ephemeris, with only one entry per sat, per day
# Keep first non-NAN entrance of each sv:
first_occurrence_idx = dfnav.groupby('sv').apply(lambda x: x.index[0])
dfnav_first = dfnav.loc[first_occurrence_idx]

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
dif = DIFgen(df_sta_a, df_sta_b, config, pos_a, pos_b)

# Text Outputs and rawdif calculation. rawdiff = a - b
rawdiff = outputs(VERSION, st, nav, sta_a, sta_b, file_nav, dist, config, dif)

# Results of calibration (optional)
if config['calculate_delays']:
    delays_b = calibration(rawdiff, delays_a, delays_b)

# Figure Outputs
figures(dif, config, ts)

# Stop time
stop_time = time.time()
print(f"Tiempo de ejecución: {stop_time - start_time:.4f} segundos")
