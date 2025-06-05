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

### What you need

Copy in one folder:

- GNSS_cal_tools.py *and* GNSS_cal_tools_subs.py.

- The two RINEX observation files of the stations.

- The RINEX navigation file. 
  
  

### Configuration

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



1. **RINEX Observation Files**:
   
   - `file_a`: Reference station (calibrated)
   
   - `file_b`: DUT station (to be calibrated)

2. **RINEX Navigation File**:
   
   - `file_nav`: Broadcast ephemeris file

3. **Position Inputs:**
   
   - `pos_a`: Cartesian coordinates of station a
   - `pos_b`: Cartesian coordinates of station b



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

(Optional) Enter the delay values for both receivers. Leave np.nan for internal delays in station b. They will be calculated at the end.

```python
delays_a = {  # Known delays for reference station
    'INTdlyC1': 31.9,  # Internal C1 delay (ns)
    'INTdlyP1': 30.1,  # Internal P1 delay (ns)
    'INTdlyP2': 28.3,  # Internal P2 delay (ns)    
    'CABdly': 207.9,   # Cable delay (ns)
    'REFdly': 12.3,    # Reference delay (ns)
}

delays_b = {  # DUT station (unknown values should be NaN)
    'INTdlyC1': np.nan,  # Will be calculated
    'INTdlyP1': np.nan,  # Will be calculated
    'INTdlyP2': np.nan,  # Will be calculated
    'CABdly': 328.3,    # Known cable delay
    'REFdly': 13.7,     # Known reference delay
}
```

## Execution

Run the script directly from your IDE or from the console: 

```bash
python GNSS_calibration_transfer.py
```



## Authors

Diego Luna
