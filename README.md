# GNSS_cal_tools

GNSS_cal_tools is a set of tools to transfer the calibration from one GNSS station to another.

## Description

This Python script transfers the calibration from a reference GNSS station to a Device Under Test (DUT) station by comparing observations between the two receivers. The tool calculates internal receiver delays for the DUT station based on a known calibrated reference station.

## Dependencies

```python
pandas numpy matplotlib georinex allantools
```

## Features

- Processes RINEX observation files from two GNSS stations

- Uses ephemeris (RINEX NAV) for satellite positions

- Calculates code biases and internal delays

- Generates time difference plots and Allan deviation analysis

- Supports elevation-based data filtering

- Currently tested only with GPS data



## Usage

Copy in one folder:

-  GNSS_cal_tools.py *and* GNSS_cal_tools_subs.py.

- The two RINEX observation files of the stations.

- The RINEX navigation file.   

Edit the configuration section at the top of GNSS_cal_tools.py to set your parameters.

```python
config = {
 'elmin': 5, # Minimum elevation angle (degrees)
 'intcod': 300, # Interval for code averaging (seconds)
 'ithr': 20, # Code threshold (ns)
 'thres': 0.05, # Residual threshold
 'SYS': 'G', # GNSS system (G=GPS)
 'plotelevations': True, # Enable elevation histograms
 'timeplots': True, # Enable time difference plots
 'calculate_delays': True, # Enable delay calculation
}
```

Edit the names of your RINEX files in GNSS_cal_tools.py and enter the Cartesian coordinates of the stations. This positions can be calculated, for example, using the  Precise Point Positioning service at []([Precise Point Positioning](https://webapp.csrs-scrs.nrcan-rncan.gc.ca/geod/tools-outils/ppp.php))



```python

# RINEX OBS files
file_a = 'AGGO2350.24O'
file_b = 'SIMr2350.24O'  # The station that will be calibrated

# RINEX navigation file
file_nav = 'BRDC00IGS_R_20242350000_01D_MN.rnx'

# Positions extracted from NRCan PPP solutions
pos_a = np.array([2765121.467, -4449250.973, -3626403.769])
pos_b = np.array([2765129.907, -4449245.382, -3626402.075])

```

## Authors

Diego Luna
