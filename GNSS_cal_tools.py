#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 15:34:07 2024

@author: diego
"""
import os
import time
import datetime
import numpy as np
from GNSS_cal_tools_subs import OExyz, dfSTAgen, dfNAVgen, C1P1, outputs
from GNSS_cal_tools_subs import ElevationReject, DIFgen, figures, loader

inicio = time.time()

# Limitations:
# Only one RINEX file per station
# No LZ files (the case when the two receivers don't have the same reference)



# Version
VERSION = '2/1/25'

# =============================================================================
# Start of inputs
# =============================================================================

config = {
    'elmin': 5,  # Elevation minimum, in degrees
    'intcod': 300,  # Default: one code average every 300 s
    'ithr': 20,  # Default code threshold = 20 ns
    'thres': 0.05,  # Option to read residual threshold (L336)
    'SYS': 'G',  # System to calibrate (GPS:G, Galileo:R, Glonass:R, Beidu:C)
    # PLOT OPTIONS
    'plotelevations': True,
    'timeplots': True,
}

# RINEX OBS and NAV files
file_a = 'SIMr2350.24O'  # Usually the travelling station
file_b = 'AGGO2350.24O'
file_nav = 'BRDC00IGS_R_20242350000_01D_MN.rnx'

# Positions extracted from NRCan PPP solutions
pos1 = np.array([2765129.907, -4449245.382, -3626402.075])
pos2 = np.array([2765121.467, -4449250.973, -3626403.769])

# =============================================================================
# End of inputs
# =============================================================================


# See if files are there
for f in [file_a, file_b, file_nav]:
    if not os.path.exists(f):
        raise FileNotFoundError(f'File not found: {f}')

# Date
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

# Loading data from files
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
      ': Loading NAV and OBS files')

nav = loader(file_nav, config)
sta1 = loader(file_a, config)
sta2 = loader(file_b, config)
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ': DONE')

# Generation of dataframes
df_sta1 = dfSTAgen(sta1)
df_sta2 = dfSTAgen(sta2)
dfnav = dfNAVgen(nav)

# Positions, distance and interval
x = pos2-pos1
dist = np.linalg.norm(x)

# Create a reduced dataframe of ephemeris, with only one entry per sat, per day
# Keep first non-NAN entrance of each sv:
first_occurrence_idx = dfnav.groupby('sv').apply(lambda x: x.index[0])
dfnav_first = dfnav.loc[first_occurrence_idx]

# Adding of EARTH FIXED COORDINATES (subroutine OExyz of dclrinex)
df_sta1 = OExyz(dfnav_first, df_sta1)
df_sta2 = OExyz(dfnav_first, df_sta2)

# Rejection at low elevation (line 1554 of dclrinex)
df_sta1 = ElevationReject(df_sta1, pos1, config, sta1.filename)
df_sta2 = ElevationReject(df_sta2, pos2, config, sta2.filename)

# Add C1P1 bias
sta1 = C1P1(sta1,df_sta1)
sta2 = C1P1(sta2,df_sta2)

# Genero diferencias
dif = DIFgen(df_sta1, df_sta2, config, pos1, pos2)

# Text Outputs
outputs(VERSION, st, nav, sta1, sta2, file_nav, dist, config, dif)

# Figure Outputs
figures(dif, config, ts)


fin = time.time()
print(f"Tiempo de ejecución: {fin - inicio:.4f} segundos")
