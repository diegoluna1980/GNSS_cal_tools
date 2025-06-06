#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 14:58:22 2025

@author: diego
"""
import numpy as np

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