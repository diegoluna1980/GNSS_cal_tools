#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 13:57:59 2025

@author: diego
"""
import numpy as np
from scipy.optimize import fsolve
import pandas as pd
from astropy.time import Time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import allantools
import georinex as gr
from pyproj import CRS, Transformer

# =============================================================================
#  Define constants
# =============================================================================

# Standard Gravitational Parameter (μ) for Earth in m³/s² 
# for GPS users https://www.unoosa.org/pdf/icg/2012/template/WGS_84.pdf
MU = 3.9860050e14 

# OMEGAE - WGS84 value of the Earth's rotation rate in rad/sec
OMEGAE = 7.292115e-5

def multipathG(dfSTA,config,sta,st):
    
    """
    Calculate multipath error for GPS L1 observations.
    
    Uses linear combination of code and phase observables to isolate multipath effects
    by eliminating geometric, ionospheric, and clock terms.
    
    https://ieeexplore.ieee.org/document/8316317
    https://www.nature.com/articles/s44172-025-00355-z
    
    Parameters:
    -----------
    dfSTA : DataFrame
        DataFrame containing RINEX observations for a station.
        Must include columns: 'C1', 'P1', 'P2' (pseudoranges on L1, P1 and P2)
    config : dict
        Configuration dictionary with processing parameters.
        Must contain: 'plot_mp_errors' (bool) and 'SYS' (GNSS system)
    
    Returns:
    --------
    DataFrame
        Original DataFrame with additional 'MP1' column containing
        multipath error in nanoseconds
    """
   
    # Constantes GPS            
    f1, f2 = 1575.42e6, 1227.60e6  # Hz
    c = 299792458 # m/s
    l1 = c / f1 # wavelength L1 in m
    l2 = c / f2 # wavelength L2 in m
    
    alpha = (f1/f2)**2
    K1 = 1 + 2/(alpha - 1)  # 4.091
    K2 = 2/(alpha - 1)      # 3.091
    
    # Corrects phase slips
    dfSTA['GF'] = dfSTA['L1C']*l1 - dfSTA['L2W']*l2
    dfSTA['GF_diff'] = dfSTA.groupby('sv')['GF'].diff().abs()
    
    # slip  
    slip_threshold = l1 / 2
    dfSTA['slip'] = dfSTA['GF_diff'] > slip_threshold
    
    # Create Segment
    dfSTA['segment'] = dfSTA.groupby('sv')['slip'].cumsum()
    
    # Calculate MP1
    dfSTA['MP1'] = dfSTA['C1'] - K1*dfSTA['L1C']*l1 + K2*dfSTA['L2W']*l2

    # Corregir ambiguity by segment
    dfSTA['MP1_corr'] = dfSTA.groupby(['sv','segment'])['MP1'].transform(
                        lambda x: x - x.mean()  )
                
    # Convert from meters to ns
    dfSTA['MP1_corr'] = dfSTA['MP1_corr'] * 3.33564095 
    
    # Crear figura y ejes
    fig, axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, 5),
        dpi=200,
        facecolor='white'
    )
    
    # --- Subplot 1: MP1_corr vs MJD ---
    axs[0].plot(dfSTA['MJD'], dfSTA['MP1_corr'], 'k.')
    axs[0].set_xlabel('MJD', fontsize=14)
    axs[0].set_ylabel('L1 Multipath Error / ns', fontsize=14)
    axs[0].set_ylim([-5, 5])
    axs[0].grid(True)
    
    axs[1].plot(dfSTA['elevation'], dfSTA['MP1_corr'], 'k.')
    axs[1].set_xlabel('Elevation / degrees', fontsize=14)
    axs[1].set_ylabel('L1 Multipath Error / ns', fontsize=14)
    axs[1].set_ylim([-5, 5])
    axs[1].grid(True)
    
    axs[2].hist(
        dfSTA.loc[dfSTA['MP1_corr'].abs() < 5, 'MP1_corr'],
        bins=30,
        alpha=0.7
    )
    axs[2].set_xlabel('L1 Multipath Error / ns', fontsize=14)
    axs[2].set_ylabel('Occurrence', fontsize=14)
    axs[2].grid(True)
    
    fig.text(
        0.98, 0.5,
        f'GNSS_cal_tools \n Computed at: {st} UTC-3',
        rotation=90,
        va='center',
        ha='right',
        fontweight="bold"
    )
    
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    
    fig.savefig(
        f'./outputs/MultipathError_{sta.filename}_G.jpg',
        dpi=300,
        bbox_inches='tight',
        facecolor='0.9'
    )
    
    plt.close(fig)

    
    MP1_mean = dfSTA.loc[dfSTA['MP1_corr'].abs() < 5, 'MP1_corr'].mean()
    MP1_std = dfSTA.loc[dfSTA['MP1_corr'].abs() < 5, 'MP1_corr'].std()

    sta['MP1_mean'] = float(MP1_mean)
    sta['MP1_std'] = float(MP1_std)


    return (dfSTA, sta)

def multipathE(dfSTA,config,sta,st):
    
    """
    Calculate multipath error for Galileo E1 observations.
    
    Uses linear combination of code and phase observables to isolate multipath effects
    by eliminating geometric, ionospheric, and clock terms.
    
    https://ieeexplore.ieee.org/document/8316317
    https://www.nature.com/articles/s44172-025-00355-z
    
    Parameters:
    -----------
    dfSTA : DataFrame
        DataFrame containing RINEX observations for a station.
        Must include columns: 'C1', 'P1', 'P2' (pseudoranges on L1, P1 and P2)
    config : dict
        Configuration dictionary with processing parameters.
        Must contain: 'plot_mp_errors' (bool) and 'SYS' (GNSS system)
    
    Returns:
    --------
    DataFrame
        Original DataFrame with additional 'MP1' column containing
        multipath error in nanoseconds
    """
   
    # Constants Galileo            
    f1, f2 = 1575.42e6, 1176.45e6  # Hz
    c = 299792458 # m/s
    l1 = c / f1 # wavelength E1 in m
    l2 = c / f2 # wavelength E5 in m

    alpha = (f1/f2)**2
    K1 = 1 + 2/(alpha - 1)  #
    K2 = 2/(alpha - 1)      #
    

    # # Corrects phase slips
    dfSTA['GF'] = dfSTA['L1C']*l1 - dfSTA['L5Q']*l2
    dfSTA['GF_diff'] = dfSTA.groupby('sv')['GF'].diff().abs()
    
    # slip if GF over 0.05 m 
    slip_threshold = l1 / 4
    dfSTA['slip'] = dfSTA['GF_diff'] > slip_threshold
    
    # Create Segment
    dfSTA['segment'] = dfSTA.groupby('sv')['slip'].cumsum()
    
    # Calculate MP1
    dfSTA['MP1'] = dfSTA['E1'] - K1*dfSTA['L1C']*l1 + K2*dfSTA['L5Q']*l2

    # Corregir ambiguity by segment
    dfSTA['MP1_corr'] = dfSTA.groupby(['sv','segment'])['MP1'].transform(
                        lambda x: x - x.mean()  )
                
    # Convert from meters to ns
    dfSTA['MP1_corr'] = dfSTA['MP1_corr'] * 3.33564095 
    
    # Crear figura y ejes
    fig, axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, 5),
        dpi=200,
        facecolor='white'
    )
    
    # --- Subplot 1: MP1_corr vs MJD ---
    axs[0].plot(dfSTA['MJD'], dfSTA['MP1_corr'], 'k.')
    axs[0].set_xlabel('MJD', fontsize=14)
    axs[0].set_ylabel('E1 Multipath Error / ns', fontsize=14)
    axs[0].set_ylim([-5, 5])
    axs[0].grid(True)
    
    axs[1].plot(dfSTA['elevation'], dfSTA['MP1_corr'], 'k.')
    axs[1].set_xlabel('Elevation / degrees', fontsize=14)
    axs[1].set_ylabel('E1 Multipath Error / ns', fontsize=14)
    axs[1].set_ylim([-5, 5])
    axs[1].grid(True)
    
    axs[2].hist(
        dfSTA.loc[dfSTA['MP1_corr'].abs() < 5, 'MP1_corr'],
        bins=30,
        alpha=0.7
    )
    axs[2].set_xlabel('E1 Multipath Error / ns', fontsize=14)
    axs[2].set_ylabel('Occurrence', fontsize=14)
    axs[2].grid(True)
    
    fig.text(
        0.98, 0.5,
        f'GNSS_cal_tools \n Computed at: {st} UTC-3',
        rotation=90,
        va='center',
        ha='right',
        fontweight="bold"
    )
    
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    
    fig.savefig(
        f'./outputs/MultipathError_{sta.filename}_E.jpg',
        dpi=300,
        bbox_inches='tight',
        facecolor='0.9'
    )
    
    plt.close(fig)

    
    MP1_mean = dfSTA.loc[dfSTA['MP1_corr'].abs() < 5, 'MP1_corr'].mean()
    MP1_std = dfSTA.loc[dfSTA['MP1_corr'].abs() < 5, 'MP1_corr'].std()

    sta['MP1_mean'] = float(MP1_mean)
    sta['MP1_std'] = float(MP1_std)

    return (dfSTA, sta)


def multipathC(dfSTA, config, sta, st):

    """
    Calculate multipath error for BeiDou B1I observations.

    Uses B1I (1561.098 MHz) and B2I (1207.14 MHz).

    Required columns:
        'C2I', 'L2I', 'L7I'
    """

    # Frequencies (BDS-2 / compatible signals)
    f1 = 1561.098e6   # B1I
    f2 = 1207.14e6    # B2I

    c = 299792458
    l1 = c / f1
    l2 = c / f2

    alpha = (f1 / f2) ** 2
    K1 = 1 + 2 / (alpha - 1)
    K2 = 2 / (alpha - 1)

    # Geometry-free combination (cycle slip detection)
    dfSTA['GF'] = dfSTA['L2I'] * l1 - dfSTA['L7I'] * l2
    dfSTA['GF_diff'] = dfSTA.groupby('sv')['GF'].diff().abs()

    slip_threshold = l1 / 4
    dfSTA['slip'] = dfSTA['GF_diff'] > slip_threshold

    # Segment arcs
    dfSTA['segment'] = dfSTA.groupby('sv')['slip'].cumsum()

    # Multipath combination (MP1 for B1I)
    dfSTA['MP1'] = dfSTA['C2I'] - K1 * dfSTA['L2I'] * l1 + K2 * dfSTA['L7I'] * l2

    # Remove ambiguity per arc
    dfSTA['MP1_corr'] = dfSTA.groupby(['sv', 'segment'])['MP1'].transform(
        lambda x: x - x.mean()
    )

    # Convert meters → ns
    dfSTA['MP1_corr'] = dfSTA['MP1_corr'] * 3.33564095

    # --- Plotting ---
    fig, axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, 5),
        dpi=200,
        facecolor='white'
    )

    axs[0].plot(dfSTA['MJD'], dfSTA['MP1_corr'], 'k.')
    axs[0].set_xlabel('MJD', fontsize=14)
    axs[0].set_ylabel('B1I Multipath Error / ns', fontsize=14)
    axs[0].set_ylim([-5, 5])
    axs[0].grid(True)

    axs[1].plot(dfSTA['elevation'], dfSTA['MP1_corr'], 'k.')
    axs[1].set_xlabel('Elevation / degrees', fontsize=14)
    axs[1].set_ylabel('B1I Multipath Error / ns', fontsize=14)
    axs[1].set_ylim([-5, 5])
    axs[1].grid(True)

    axs[2].hist(
        dfSTA.loc[dfSTA['MP1_corr'].abs() < 5, 'MP1_corr'],
        bins=30,
        alpha=0.7
    )
    axs[2].set_xlabel('B1I Multipath Error / ns', fontsize=14)
    axs[2].set_ylabel('Occurrence', fontsize=14)
    axs[2].grid(True)

    fig.text(
        0.98, 0.5,
        f'GNSS_cal_tools \n Computed at: {st} UTC-3',
        rotation=90,
        va='center',
        ha='right',
        fontweight="bold"
    )

    fig.tight_layout(rect=[0, 0, 0.95, 1])

    fig.savefig(
        f'./outputs/MultipathError_{sta.filename}_C.jpg',
        dpi=300,
        bbox_inches='tight',
        facecolor='0.9'
    )

    plt.close(fig)

    # Statistics
    MP1_mean = dfSTA.loc[dfSTA['MP1_corr'].abs() < 5, 'MP1_corr'].mean()
    MP1_std = dfSTA.loc[dfSTA['MP1_corr'].abs() < 5, 'MP1_corr'].std()

    sta['MP1_mean'] = float(MP1_mean)
    sta['MP1_std'] = float(MP1_std)

    return (dfSTA, sta)


def APOcorrection(pos,sta_hdr):
    
    # Define transformations
    crs_ecef = CRS.from_epsg(4978)   # ECEF (WGS84)
    crs_geo  = CRS.from_epsg(4979)   # Geodetic lat/lon/ellipsoidal height (WGS84)
    transformer = Transformer.from_crs(crs_ecef, crs_geo, always_xy=True)
    lon, lat, h = transformer.transform(pos[0], pos[1],pos[2])

    APO_str = sta_hdr['ANTENNA: DELTA H/E/N']
    
    # Phase center offset (ENU frame)
    u_offset, e_offset, n_offset = [float(x) for x in APO_str.split() if x]

    # Convert to radians
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)

    # ENU to ECEF rotation matrix
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)

    # Convert offset to ECEF
    dx = -sin_lon * e_offset - sin_lat * cos_lon * n_offset + cos_lat * cos_lon * u_offset
    dy = cos_lon * e_offset - sin_lat * sin_lon * n_offset + cos_lat * sin_lon * u_offset
    dz = cos_lat * n_offset + sin_lat * u_offset
    
    pos = pos + [dx,dy,dz]
    return(pos)

def calibration(rawdiff, delays_a, delays_b, sta_a, sta_b):
    
    """
    Adjusts the internal delays (INTdly) of device B based on the difference 
    between device A and B's cable delays (CABdly) and reference delays (REFdly),
    as well as raw timing differences (rawdiff).
    
    Args:
        rawdiff (dict): Contains median timing differences between devices A and B
                        for channels C1, P1, and P2 (keys: 'medianC1', 'medianP1', 'medianP2').
        delays_a (dict): Delay values for device A, including:
                        - CABdly: Cable delay
                        - REFdly: Reference delay
                        - INTdlyC1/INTdlyP1/INTdlyP2: Internal delays for different channels
        delays_b (dict): Delay values for device B (same structure as delays_a) that will be updated
    
    Returns:
        dict: The updated delays_b dictionary with adjusted INTdly values for channels C1, P1, and P2.
    """
    
    # Calculate the difference in cable delays (CABdly) between device A and B  
    deltaCABdly = delays_a['CABdly'] - delays_b['CABdly']

    # Calculate the difference in reference delays (REFdly) between device A and B    
    deltaREFdly = delays_a['REFdly'] - delays_b['REFdly']

    # Calculate the internal delay differences for each channel by adjusting the raw differences
    # with the cable and reference delay differences    
    deltaINTdlyC1 = rawdiff['medianC1'] - deltaCABdly + deltaREFdly 
    deltaINTdlyP1 = rawdiff['medianP1'] - deltaCABdly + deltaREFdly 
    deltaINTdlyP2 = rawdiff['medianP2'] - deltaCABdly + deltaREFdly 
    
    # Update device B's internal delays by subtracting the calculated differences
    # from device A's internal delays    
    delays_b['INTdlyC1'] = round(delays_a['INTdlyC1'] - deltaINTdlyC1,1) 
    delays_b['INTdlyP1'] = round(delays_a['INTdlyP1'] - deltaINTdlyP1,1)
    delays_b['INTdlyP2'] = round(delays_a['INTdlyP2'] - deltaINTdlyP2,1)
    
    # Calibration outputs
    
    filename = sta_a.filename.partition(".")[0] + sta_b.filename.partition(".")[0]
    #file_sum = open('./outputs/' + filename + '_results.txt', 'w')
    
    with open('./outputs/' + filename + '_results.txt', "a") as file:
        file.write('\nCalculated delays in station ' + sta_b.filename + '(DUT station):\n')
        for key, value in delays_b.items():
            file.write(f"{key}: {value}\n")
        file.write('\nDelays in station ' + sta_a.filename + '(Reference station):\n')
        for key, value in delays_a.items():
            file.write(f"{key}: {value}\n")
        file.write('\n')

    
    return(delays_b)



    
    
def loader(file,config):
    """
    Reads a RINEX file (version 2 or 3) and returns the dataset.
    If the file is an observation file, it only loads C1, P1, and P2 observables.
    
    Parameters:
    -----------
    file : str
        filename of the RINEX file (e.g., "example.21o" or "example.obs").
    config : dict
        configuration dictionary
        
    Returns:
    --------
    dataset
        Dataset containing the RINEX data.
    """

    file_hdr = gr.rinexheader(file)
    if (file_hdr['filetype'] == 'N'):
        dataset = gr.load(file,use=config['SYS'],
                          meas=['sqrtA', 'DeltaN', 'M0', 'DeltaN'])
    
    if (file_hdr['filetype'] == 'O'):
        if (file_hdr['version'] > 3):
            if config['SYS'] == 'G':
                #dataset = gr.load(file, use=config['SYS'],
                #                   meas=['C1C', 'C1W', 'C2W'])
                dataset = gr.load(file, use=config['SYS'],
                                   meas=['C1C', 'C1W', 'C2W','L2W','L1C'])
                dataset = dataset.rename({"C1C": "C1"})
                dataset = dataset.rename({"C1W": "P1"})
                dataset = dataset.rename({"C2W": "P2"})
            if config['SYS'] == 'E':
                dataset = gr.load(file, use=config['SYS'],
                                  meas=['C1C', 'C5Q', 'L5Q','L1C'])
                dataset = dataset.rename({"C1C": "E1"})
                dataset = dataset.rename({"C5Q": "E5"})                
            if config['SYS'] == 'C':
                dataset = gr.load(file, use=config['SYS'],
                                  meas=['C2I', 'L2I', 'L7I','C7I'])
        else:
            dataset = gr.load(file, use=config['SYS'], meas=['C1', 'P1', 'P2'])


    return(dataset, file_hdr)

def figuresC(dif, config, ts, sta_a, sta_b):
    
    """
    Generates plots of time series and TDEV for BeiDou signals:
    B1I (C2I) and B2I (L7I).
    """
    
    if config['timeplots']:

        # meters → ns
        k = 0.299792458
        
        MJD = dif.MJD.unique()
        pop1 = dif.groupby(['MJD']).median()
        
        # Convert to ns
        B1 = pop1['C2I_corr'].to_numpy() / k
        B2 = pop1['C7I_corr'].to_numpy() / k
        #B2 = pop1['L7I_corr'].to_numpy() / k
        
        # --- TDEV ---
        (B1_tau_tdev, B1_tdev, B1_tdeverr, n_tdev) = allantools.tdev(
            B1, rate=1/config['intcod'], data_type="phase", taus='octave'
        )
        
        (B2_tau_tdev, B2_tdev, B2_tdeverr, n_tdev) = allantools.tdev(
            B2, rate=1/config['intcod'], data_type="phase", taus='octave'
        )
        
        # --- Figure ---
        fig1 = plt.figure(1, figsize=(12, 8))
        plt.subplots_adjust(hspace=.3)
        
        # Timestamp
        plt.figtext(
            0.95, 0.5,
            'Computed at: ' + datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') + ' UTC-3\n',
            rotation=90
        )
        
        plt.figtext(
            0.05, 0.01,
            'Raw differences of ' + sta_a.filename + ' - ' + sta_b.filename
        )
        
        # --- B1 time series ---
        plt.subplot(221)
        plt.plot(MJD, B1, 'b.', markeredgewidth=0.0, zorder=4, label='B1')
        plt.title('Median: (' + str(round(np.median(B1), 2)) +
                  '+/-' + str(round(B1.std(), 2)) + ') ns')
        plt.ylabel('Time / ns', size=14)
        plt.xlabel('MJD', size=14)
        plt.legend(loc=0, prop={'size': 12}, framealpha=1)
        plt.grid(linestyle='dashed')
        plt.xticks(rotation=30, size=12)
        plt.yticks(size=12)
        xx, _ = plt.xticks()
        plt.xticks(xx, ['%.0f' % a for a in xx])
        plt.tick_params(direction="in")

        # --- B2 time series ---
        plt.subplot(222)
        plt.plot(MJD, B2, 'b.', markeredgewidth=0.0, zorder=4, label='B2')
        plt.title('Median: (' + str(round(np.median(B2), 2)) +
                  '+/-' + str(round(B2.std(), 2)) + ') ns')
        plt.xlabel('MJD', size=14)
        plt.legend(loc=0, prop={'size': 12}, framealpha=1)
        plt.grid(linestyle='dashed')
        plt.xticks(rotation=30, size=12)
        plt.yticks(size=12)
        xx, _ = plt.xticks()
        plt.xticks(xx, ['%.0f' % a for a in xx])
        plt.tick_params(direction="in")

        # --- TDEV B1 ---
        plt.subplot(223)
        plt.loglog(B1_tau_tdev, B1_tdev, '-ko', markeredgewidth=0.0, zorder=4)
        plt.axhline(y=0.1, color='r', linestyle='--')
        plt.ylabel('Time deviation / ns', size=14)
        plt.xlabel('Time / s', size=14)
        plt.grid(linestyle='dashed')
        plt.tick_params(direction="in")
        plt.xticks(size=12)
        plt.yticks(size=12)

        # --- TDEV B2 ---
        plt.subplot(224)
        plt.loglog(B2_tau_tdev, B2_tdev, '-ko', markeredgewidth=0.0, zorder=4)
        plt.axhline(y=0.1, color='r', linestyle='--')
        plt.xlabel('Time / s', size=14)
        plt.grid(linestyle='dashed')
        plt.tick_params(direction="in")
        plt.xticks(size=12)
        plt.yticks(size=12)

        # --- Save ---
        plt.suptitle('B1 and B2 plots - GNSS_cal_tools.py',
                     fontsize=16, fontweight='bold')

        destino = './outputs/B1B2plots_' + sta_a.filename + '_' + sta_b.filename + '.pdf'

        fig1.savefig(destino, facecolor='0.9', dpi=200)
        plt.close()


def figuresE(dif, config, ts, sta_a, sta_b):
    
    """
    Generates plots of time series and time deviations (TDEV) 
    for E1 and E5 Galileo code corrections, and saves them to a PDF.

    Parameters:
    - dif: DataFrame containing corrected GNSS code data with MJD index.
    - config: dict with configuration options 
    - ts: Unix timestamp indicating when the computation was performed.
    - filename_a
    - filename_b
    """
    
    if config['timeplots']:

        # Conversion factor from kilometers to nanoseconds
        k = 0.299792458
        
        # List of unique Modified Julian Dates (MJD)
        MJD = dif.MJD.unique()
        
        # Group data by MJD and compute median values
        pop1 = dif.groupby(['MJD']).median()
        
        # Convert median corrected values from km to ns
        E1 = pop1['E1_corr'].to_numpy() / k
        E5 = pop1['E5_corr'].to_numpy() / k
               
        # Compute Time Deviation (TDEV) using allantools for each observable
        (E1_tau_tdev, E1_tdev, E1_tdeverr, n_tdev) = allantools.tdev(E1, rate= 1/config['intcod'], data_type="phase", taus='octave')
        (E5_tau_tdev, E5_tdev, E5_tdeverr, n_tdev) = allantools.tdev(E5, rate= 1/config['intcod'], data_type="phase", taus='octave')
        
        # Create figure and layout        
        fig1 = plt.figure(1,figsize=(12,8))
        plt.subplots_adjust(hspace = .3)
        
        # Add timestamp to right margin
        plt.figtext(0.95, 0.5,  'Computed at: ' + datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') + ' UTC-3\n', rotation=90)
        plt.figtext(0.05,0.01,'Rawdifferences of ' + sta_a.filename + ' - ' + sta_b.filename)
        # Plot time series for E1        
        plt.subplot(221)
        plt.plot(MJD, E1, 'b.',markeredgewidth=0.0,zorder=4,label='E1')
        plt.title('Median: (' + str(round(np.median(E1),2)) + '+/-' + str(round(E1.std(),2)) + ') ns')
        plt.legend(loc=0, prop={'size': 12}, framealpha=1)
        plt.ylabel('Time / ns', size = 14)
        plt.xlabel('MJD', size = 14)
        plt.grid(linestyle='dashed')
        locs,labels = plt.xticks()
        plt.xticks( rotation=30,size=12)
        plt.yticks(size=12)
        xx, locs = plt.xticks()
        ll = ['%.0f' % a for a in xx]
        plt.xticks(xx, ll)
        plt.tick_params(direction="in")

        # Plot time series for E5
        plt.subplot(222)
        plt.plot(MJD, E5, 'b.',markeredgewidth=0.0,zorder=4,label='E5')
        plt.title('Median: (' + str(round(np.median(E5),2)) + '+/-' + str(round(E5.std(),2)) + ') ns')
        plt.xlabel('MJD', size = 14)
        plt.legend(loc=0, prop={'size': 12}, framealpha=1)
        plt.grid(linestyle='dashed')
        locs,labels = plt.xticks()
        plt.xticks( rotation=30,size=12)
        plt.yticks(size=12)
        xx, locs = plt.xticks()
        ll = ['%.0f' % a for a in xx]
        plt.xticks(xx, ll)
        plt.tick_params(direction="in")
        
        # Plot time series for P2
        # plt.subplot(233)
        # #plt.plot(MJD, P2, 'b.',markeredgewidth=0.0,zorder=4,label='P2')
        # #plt.title('Median: (' + str(round(np.median(P2),1)) + '+/-' + str(round(P2.std(),1)) + ') ns')
        # plt.xlabel('MJD', size = 14)
        # plt.legend(loc=0, prop={'size': 12}, framealpha=1)
        # plt.grid(linestyle='dashed')
        # locs,labels = plt.xticks()
        # plt.xticks( rotation=30,size=12)
        # plt.yticks(size=12)
        # xx, locs = plt.xticks()
        # ll = ['%.0f' % a for a in xx]
        # plt.xticks(xx, ll)
        # plt.tick_params(direction="in")
        
        # Plot TDEV for E1
        plt.subplot(223)
        plt.loglog(E1_tau_tdev, E1_tdev, '-ko',markeredgewidth=0.0,zorder=4)
        plt.axhline(y=0.1, color='r', linestyle='--')  # Red dashed line at y=5
        plt.ylabel('Time deviation / ns', size = 14)
        plt.xlabel('Time / s', size = 14)
        plt.yticks(size=12)
        plt.xticks(size=12)
        plt.grid(linestyle='dashed')
        plt.tick_params(direction="in")
        
        # Plot TDEV for E5
        plt.subplot(224)
        plt.loglog(E5_tau_tdev, E5_tdev, '-ko',markeredgewidth=0.0,zorder=4)
        plt.axhline(y=0.1, color='r', linestyle='--')  # Red dashed line at y=5
        plt.xlabel('Time / s', size = 14)
        #plt.title('P1_alllan')
        plt.yticks(size=12)
        plt.xticks(size=12)
        plt.grid(linestyle='dashed')
        plt.tick_params(direction="in")
        
        # # Plot TDEV for P2
        # plt.subplot(236)
        # #plt.loglog(P2_tau_tdev, P2_tdev, '-ko',markeredgewidth=0.0,zorder=4)
        # plt.axhline(y=0.1, color='r', linestyle='--')  # Red dashed line at y=5
        # plt.xlabel('Time / s', size = 14)
        # #plt.title('P2_alllan')
        # plt.yticks(size=12)
        # plt.xticks(size=12)
        # plt.grid(linestyle='dashed')
        # plt.tick_params(direction="in")
        
        # Global title and save
        plt.suptitle('E1 and E5 plots - GNSS_cal_tools.py', fontsize=16,  fontweight='bold')
        destino = './outputs/E1E5plots_' + sta_a.filename + '_' + sta_b.filename + '.pdf'       

        fig1.savefig(destino,facecolor='0.9', dpi = 200)
        plt.close()


def figuresG(dif,config,ts, sta_a, sta_b):
    
    """
    Generates plots of time series and time deviations (TDEV) 
    for C1, P1, and P2 GNSS code corrections, and saves them to a PDF.

    Parameters:
    - dif: DataFrame containing corrected GNSS code data with MJD index.
    - config: dict with configuration options 
    - ts: Unix timestamp indicating when the computation was performed.
    """
    
    
    if config['timeplots']:

        # Conversion factor from meters to nanoseconds
        k = 0.299792458
        
        # List of unique Modified Julian Dates (MJD)
        MJD = dif.MJD.unique()
        
#########################################################        
        # Group data by MJD and compute median values
        pop1 = dif.groupby(['MJD']).median()
        
        
        # Convert median corrected values from m to ns
        C1 = pop1['C1_corr'].to_numpy() / k
        P1 = pop1['P1_corr'].to_numpy() / k
        P2 = pop1['P2_corr'].to_numpy() / k
##########################################################               
        
        # C1 = dif.groupby('MJD').apply(lambda g: np.average(g['C1_corr'],
        #     weights=np.sin(np.deg2rad(g['elevation']))**2        ))

        # P1 = dif.groupby('MJD').apply(lambda g: np.average(g['P1_corr'],
        #     weights=np.sin(np.deg2rad(g['elevation']))**2        ))

        # P2 = dif.groupby('MJD').apply(lambda g: np.average(g['P2_corr'],
        #     weights=np.sin(np.deg2rad(g['elevation']))**2       ))

        # C1 = C1.to_numpy() / k
        # P1 = P1.to_numpy() / k
        # P2 = P2.to_numpy() / k

##########################################################               

        # Compute Time Deviation (TDEV) using allantools for each observable
        (C1_tau_tdev, C1_tdev, C1_tdeverr, n_tdev) = allantools.tdev(C1, rate= 1/config['intcod'], data_type="phase", taus='octave')
        (P1_tau_tdev, P1_tdev, P1_tdeverr, n_tdev) = allantools.tdev(P1, rate= 1/config['intcod'], data_type="phase", taus='octave')
        (P2_tau_tdev, P2_tdev, P2_tdeverr, n_tdev) = allantools.tdev(P2, rate= 1/config['intcod'], data_type="phase", taus='octave')
        
        # Create figure and layout        
        fig1 = plt.figure(1,figsize=(12,8))
        plt.subplots_adjust(hspace = .3)
        
        # Add timestamp to right margin
        plt.figtext(0.95, 0.5,  'Computed at: ' + datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') + ' UTC-3\n', rotation=90)
        plt.figtext(0.05,0.01,'Rawdifferences of ' + sta_a.filename + ' - ' + sta_b.filename)

        # Plot time series for C1        
        plt.subplot(231)
        plt.plot(MJD, C1, 'b.',markeredgewidth=0.0,zorder=4,label='C1')
        plt.title('Median: (' + str(round(np.median(C1),2)) + '+/-' + str(round(C1.std(),2)) + ') ns')
        plt.legend(loc=0, prop={'size': 12}, framealpha=1)
        plt.ylabel('Time / ns', size = 14)
        plt.xlabel('MJD', size = 14)
        plt.grid(linestyle='dashed')
        locs,labels = plt.xticks()
        plt.xticks( rotation=30,size=12)
        plt.yticks(size=12)
        xx, locs = plt.xticks()
        ll = ['%.0f' % a for a in xx]
        plt.xticks(xx, ll)
        plt.tick_params(direction="in")

        # Plot time series for P1
        plt.subplot(232)
        plt.plot(MJD, P1, 'b.',markeredgewidth=0.0,zorder=4,label='P1')
        plt.title('Median: (' + str(round(np.median(P1),2)) + '+/-' + str(round(P1.std(),2)) + ') ns')
        plt.xlabel('MJD', size = 14)
        plt.legend(loc=0, prop={'size': 12}, framealpha=1)
        plt.grid(linestyle='dashed')
        locs,labels = plt.xticks()
        plt.xticks( rotation=30,size=12)
        plt.yticks(size=12)
        xx, locs = plt.xticks()
        ll = ['%.0f' % a for a in xx]
        plt.xticks(xx, ll)
        plt.tick_params(direction="in")
        
        # Plot time series for P2
        plt.subplot(233)
        plt.plot(MJD, P2, 'b.',markeredgewidth=0.0,zorder=4,label='P2')
        plt.title('Median: (' + str(round(np.median(P2),2)) + '+/-' + str(round(P2.std(),2)) + ') ns')
        plt.xlabel('MJD', size = 14)
        plt.legend(loc=0, prop={'size': 12}, framealpha=1)
        plt.grid(linestyle='dashed')
        locs,labels = plt.xticks()
        plt.xticks( rotation=30,size=12)
        plt.yticks(size=12)
        xx, locs = plt.xticks()
        ll = ['%.0f' % a for a in xx]
        plt.xticks(xx, ll)
        plt.tick_params(direction="in")
        
        # Plot TDEV for C1
        plt.subplot(234)
        plt.loglog(C1_tau_tdev, C1_tdev, '-ko',markeredgewidth=0.0,zorder=4)
        plt.axhline(y=0.1, color='r', linestyle='--')  # Red dashed line at y=5
        plt.ylabel('Time deviation / ns', size = 14)
        plt.xlabel('Time / s', size = 14)
        plt.yticks(size=12)
        plt.xticks(size=12)
        plt.grid(linestyle='dashed')
        plt.tick_params(direction="in")
        
        # Plot TDEV for P1
        plt.subplot(235)
        plt.loglog(P1_tau_tdev, P1_tdev, '-ko',markeredgewidth=0.0,zorder=4)
        plt.axhline(y=0.1, color='r', linestyle='--')  # Red dashed line at y=5
        plt.xlabel('Time / s', size = 14)
        #plt.title('P1_alllan')
        plt.yticks(size=12)
        plt.xticks(size=12)
        plt.grid(linestyle='dashed')
        plt.tick_params(direction="in")
        
        # Plot TDEV for P2
        plt.subplot(236)
        plt.loglog(P2_tau_tdev, P2_tdev, '-ko',markeredgewidth=0.0,zorder=4)
        plt.axhline(y=0.1, color='r', linestyle='--')  # Red dashed line at y=5
        plt.xlabel('Time / s', size = 14)
        #plt.title('P2_alllan')
        plt.yticks(size=12)
        plt.xticks(size=12)
        plt.grid(linestyle='dashed')
        plt.tick_params(direction="in")
        
        # Global title and save
        plt.suptitle('C1, P1, and P2 plots - GNSS_cal_tools.py', fontsize=16,  fontweight='bold')
        destino = './outputs/C1P1P2plots_' + sta_a.filename + '_' + sta_b.filename + '.pdf'       
        fig1.savefig(destino,facecolor='0.9', dpi = 200)
        plt.close()
        
        
def DIFgenE(dfSTA1, dfSTA2, config, pos1, pos2):
    """
    Computes observation differences between two Galileo stations after temporal alignment
    and satellite-based geometric corrections.

    Parameters
    ----------
    dfSTA1 : pd.DataFrame
        Observation data from station 1.
    dfSTA2 : pd.DataFrame
        Observation data from station 2.
        config : dict
        Configuration dictionary, must include 'intcod' (integration time in seconds).
    pos1 : np.ndarray
        ECEF coordinates of station 1 (3-element array).
    pos2 : np.ndarray
        ECEF coordinates of station 2 (3-element array).

    Returns
    -------
    pd.DataFrame
        DataFrame containing aligned satellite observations, differences and geometry-corrected values.
    """

    codint = config['intcod'] / 86400  # Convert integration interval to days

    # Round MJD to the nearest integration time
    dfSTA1['MJD_bin'] = (dfSTA1['MJD'] / codint).round() * codint
    dfSTA2['MJD_bin'] = (dfSTA2['MJD'] / codint).round() * codint

    # Median values per (MJD_bin, sv) for both stations
    grp_cols = ['MJD_bin', 'sv']
    agg_cols1 = ['E1', 'E5', 'X', 'Y', 'Z', 'elevation']
    agg_cols2 = ['E1', 'E5']

    dat1 = dfSTA1.groupby(grp_cols)[agg_cols1].median().reset_index()
    dat2 = dfSTA2.groupby(grp_cols)[agg_cols2].median().reset_index()

    # Merge aligned records
    dif = pd.merge(dat1, dat2, on=grp_cols, suffixes=('_1', '_2'))

    # Calculate observation differences
    dif['E1'] = dif['E1_1'] - dif['E1_2']
    dif['E5'] = dif['E5_1'] - dif['E5_2']

    # Remove gross outliers
    dif = dif[(dif[['E1', 'E5']].abs() <= 300).all(axis=1)]

    # Median Absolute Deviation (MAD) filtering
    def mad_filter(col, u=3):
        med = col.median()
        mad = 1.4826 * np.median(np.abs(col - med))
        return (col - med).abs() <= u * mad

    for col in ['E1', 'E5']:
        dif = dif[mad_filter(dif[col])]

    # Geometry correction
    x = pos2 - pos1
    xsat = dif['X'] - pos1[0]
    ysat = dif['Y'] - pos1[1]
    zsat = dif['Z'] - pos1[2]
    r = np.sqrt(xsat**2 + ysat**2 + zsat**2)
    corg = (x[0]*xsat + x[1]*ysat + x[2]*zsat) / r

    dif['E1_corr'] = dif['E1'] - corg
    dif['E5_corr'] = dif['E5'] - corg

    dif = dif.rename(columns={'MJD_bin': 'MJD'})
    
    # Keep only relevant columns
    
    return dif[['MJD', 'sv', 'X', 'Y', 'Z', 'elevation', 'E1', 'E5', 'E1_corr', 'E5_corr']]


def DIFgenC(dfSTA1, dfSTA2, config, pos1, pos2):
    """
    Computes observation differences between two stations for BeiDou signals
    using C2I, L2I, L7I observables.

    Parameters
    ----------
    dfSTA1 : pd.DataFrame
        Must include 'MJD', 'sv', 'C2I', 'L2I', 'L7I', 'X', 'Y', 'Z', 'elevation'
    dfSTA2 : pd.DataFrame
        Must include 'MJD', 'sv', 'C2I', 'L2I', 'L7I'
    config : dict
        Must include 'intcod'
    pos1, pos2 : np.ndarray
        Station coordinates (ECEF)

    Returns
    -------
    pd.DataFrame
    """

    codint = config['intcod'] / 86400  # days

    # Time binning
    dfSTA1['MJD_bin'] = (dfSTA1['MJD'] / codint).round() * codint
    dfSTA2['MJD_bin'] = (dfSTA2['MJD'] / codint).round() * codint

    grp_cols = ['MJD_bin', 'sv']

    agg_cols1 = ['C7I', 'C2I', 'L2I', 'L7I', 'X', 'Y', 'Z', 'elevation']
    agg_cols2 = ['C7I', 'C2I', 'L2I', 'L7I']

    dat1 = dfSTA1.groupby(grp_cols)[agg_cols1].median().reset_index()
    dat2 = dfSTA2.groupby(grp_cols)[agg_cols2].median().reset_index()

    # Merge
    dif = pd.merge(dat1, dat2, on=grp_cols, suffixes=('_1', '_2'))

    # Differences
    dif['C7I'] = dif['C7I_1'] - dif['C7I_2']
    dif['C2I'] = dif['C2I_1'] - dif['C2I_2']
    dif['L2I'] = dif['L2I_1'] - dif['L2I_2']
    dif['L7I'] = dif['L7I_1'] - dif['L7I_2']

    # Dual-frequency combination (phase difference)
    dif['L2I-L7I'] = dif['L2I'] - dif['L7I']

    # --- Outlier rejection ---
    dif = dif[(dif[['C2I', 'L2I', 'L7I']].abs() <= 300).all(axis=1)]
    dif = dif[dif['L2I-L7I'].abs() <= 30]

    # MAD filter
    def mad_filter(col, u=3):
        med = col.median()
        mad = 1.4826 * np.median(np.abs(col - med))
        return (col - med).abs() <= u * mad

    for col in ['C7I', 'C2I', 'L2I', 'L7I']:
        dif = dif[mad_filter(dif[col])]

    # --- Geometry correction ---
    x = pos2 - pos1

    xsat = dif['X'] - pos1[0]
    ysat = dif['Y'] - pos1[1]
    zsat = dif['Z'] - pos1[2]

    r = np.sqrt(xsat**2 + ysat**2 + zsat**2)

    corg = (x[0]*xsat + x[1]*ysat + x[2]*zsat) / r

    dif['C7I_corr'] = dif['C7I'] - corg
    dif['C2I_corr'] = dif['C2I'] - corg
    dif['L2I_corr'] = dif['L2I'] - corg
    dif['L7I_corr'] = dif['L7I'] - corg

    dif = dif.rename(columns={'MJD_bin': 'MJD'})

    return dif[['MJD', 'sv', 'X', 'Y', 'Z', 'elevation',
                'C2I', 'L2I', 'L7I', 'L2I-L7I', 'C7I',
                'C2I_corr', 'L2I_corr', 'L7I_corr','C7I_corr']]

def DIFgenG(dfSTA1, dfSTA2, config, pos1, pos2):
    """
    Computes observation differences between two GNSS stations after temporal alignment
    and satellite-based geometric corrections.

    Parameters
    ----------
    dfSTA1 : pd.DataFrame
        Observation data from station 1. Must include 'MJD', 'sv', 'C1', 'P1', 'P2', 'X', 'Y', 'Z', 'elevation'.
    dfSTA2 : pd.DataFrame
        Observation data from station 2. Must include 'MJD', 'sv', 'C1', 'P1', 'P2'.
    config : dict
        Configuration dictionary, must include 'intcod' (integration time in seconds).
    pos1 : np.ndarray
        ECEF coordinates of station 1 (3-element array).
    pos2 : np.ndarray
        ECEF coordinates of station 2 (3-element array).

    Returns
    -------
    pd.DataFrame
        DataFrame containing aligned satellite observations, differences and geometry-corrected values.
    """

    codint = config['intcod'] / 86400  # Convert integration interval to days

    # Round MJD to the nearest integration time
    dfSTA1['MJD_bin'] = (dfSTA1['MJD'] / codint).round() * codint
    dfSTA2['MJD_bin'] = (dfSTA2['MJD'] / codint).round() * codint

    # Median values per (MJD_bin, sv) for both stations
    grp_cols = ['MJD_bin', 'sv']
    agg_cols1 = ['C1', 'P1', 'P2', 'X', 'Y', 'Z', 'elevation']
    agg_cols2 = ['C1', 'P1', 'P2']

    dat1 = dfSTA1.groupby(grp_cols)[agg_cols1].median().reset_index()
    dat2 = dfSTA2.groupby(grp_cols)[agg_cols2].median().reset_index()

    # Merge aligned records
    dif = pd.merge(dat1, dat2, on=grp_cols, suffixes=('_1', '_2'))

    # Calculate observation differences
    dif['C1'] = dif['C1_1'] - dif['C1_2']
    dif['P1'] = dif['P1_1'] - dif['P1_2']
    dif['P2'] = dif['P2_1'] - dif['P2_2']
    dif['P1-P2'] = dif['P1'] - dif['P2']

    # Remove gross outliers
    dif = dif[(dif[['C1', 'P1', 'P2']].abs() <= 300).all(axis=1)]
    dif = dif[dif['P1-P2'].abs() <= 30]

    # Median Absolute Deviation (MAD) filtering
    def mad_filter(col, u=3):
        med = col.median()
        mad = 1.4826 * np.median(np.abs(col - med))
        return (col - med).abs() <= u * mad

    for col in ['C1', 'P1', 'P2']:
        dif = dif[mad_filter(dif[col])]

    # Geometry correction
    x = pos2 - pos1
    xsat = dif['X'] - pos1[0]
    ysat = dif['Y'] - pos1[1]
    zsat = dif['Z'] - pos1[2]
    r = np.sqrt(xsat**2 + ysat**2 + zsat**2)
    corg = (x[0]*xsat + x[1]*ysat + x[2]*zsat) / r

    dif['C1_corr'] = dif['C1'] - corg
    dif['P1_corr'] = dif['P1'] - corg
    dif['P2_corr'] = dif['P2'] - corg

    dif = dif.rename(columns={'MJD_bin': 'MJD'})
    
    # Keep only relevant columns
    
    return dif[['MJD', 'sv', 'X', 'Y', 'Z', 'elevation', 'C1', 'P1', 'P2', 'P1-P2', 'C1_corr', 'P1_corr', 'P2_corr']]


def outputsE(VERSION, st, nav, sta1, sta2, file_nav, dist, config, dif):

    # Open file
    filename = sta1.filename.partition(".")[0] + sta2.filename.partition(".")[0]
    file_sum = open('./outputs/' + filename + '_results_E.txt', 'w')
    
    file_sum.write(
    f" GNSS_cal_tools Version: {VERSION}\n"
    f"Processing date and time: {st} UTC-3\n"
    f"Output interval (s) = {config['intcod']}\n"
    f"Code threshold (ns) = {config['ithr']}\n"
    f"Residual threshold (m) = {config['thres']}\n"
    f"Processed system: {config['SYS']}. (GPS:G, Galileo:R, Glonass:R, Beidu:C)\n"
    f"Min Elevation (deg): {config['elmin']}\n\n"
    f"INPUT FILES\n"
    f" {file_nav}\tRINEX version: {nav.version}\n"
    f" {sta1.filename}\tRINEX version: {sta1.version}\n"
    f" {sta2.filename}\tRINEX version: {sta2.version}\n\n"
    )


    file_sum.write(
    f"Distance from headers is {dist:.2f} m\n"
    f"Interval of {sta1.filename} is {sta1.interval} s\n"
    f"Interval of {sta2.filename} is {sta2.interval} s\n\n"
    )

    print(
        f"Distance read from headers is: {dist:.2f} m\n"
        f"Interval of file1 is {sta1.interval} s\n"
        f"Interval of file2 is {sta2.interval} s"
    )


    if dist > 1000:
        file_sum.write('WARNING: Distance read from headers is ' + dist + ' m!\n')
        print('WARNING: Distance read from headers is ' + dist + ' m!')
    
    if sta1.interval != sta2.interval:
        file_sum.write('Not the same data interval\n')
        print('Not the same data interval')
        

    if config['plot_mp_errors']:
        print(
            f"Mean and stdev of E1 Multipath Error in {sta1.filename}: "
            f"({round(float(sta1.MP1_mean), 2)} +/- "
            f"{round(float(sta1.MP1_std), 2)}) ns\n"
            f"Mean and stdev of E1 Multipath Error in {sta2.filename}: "
            f"({round(float(sta2.MP1_mean), 2)} +/- "
            f"{round(float(sta2.MP1_std), 2)}) ns\n"
                )
        file_sum.write(
        f"Mean and stdev of E1 Multipath Error in {sta1.filename}: "
        f"({round(float(sta1.MP1_mean), 2)} +/- "
        f"{round(float(sta1.MP1_std), 2)}) ns\n"
        f"Mean and stdev of E1 Multipath Error in {sta2.filename}: "
        f"({round(float(sta2.MP1_mean), 2)} +/- "
        f"{round(float(sta2.MP1_std), 2)}) ns\n\n"
        )




    pop1 = dif.groupby(['MJD']).median()

    rawdiff = {
        'medianE1' : round(pop1['E1_corr'].median()/0.299792458, 2),
        'stdE1' : round(pop1['E1_corr'].std()/0.299792458, 2),
        'medianE5' : round(pop1['E5_corr'].median()/0.299792458, 2),
        'stdE5' : round(pop1['E5_corr'].std()/0.299792458, 2),
        }


    file_sum.write(
    f"Median and stdev of E1 difference: ({rawdiff['medianE1']} +/- {rawdiff['stdE1']}) ns\n"
    f"Median and stdev of E5 difference: ({rawdiff['medianE5']} +/- {rawdiff['stdE5']}) ns\n"
    )

    print(
        f"Median and stdev of E1 difference: ({rawdiff['medianE1']} +/- {rawdiff['stdE1']}) ns\n"
        f"Median and stdev of E5 difference: ({rawdiff['medianE5']} +/- {rawdiff['stdE5']}) ns\n"
    )

    
    file_sum.close()


    cols_a_exportar = dif[['MJD', 'sv', 'E1_corr', 'E5_corr', 'elevation']].copy()
    cols_a_exportar.columns = ['MJD', 'sv', 'E1', 'E5', 'elevation']
    cols_a_exportar['E1'] = cols_a_exportar['E1']/0.299792458
    cols_a_exportar['E5'] = cols_a_exportar['E5']/0.299792458
    cols_a_exportar['MJD'] = cols_a_exportar['MJD'].map(lambda x: f"{x:.5f}")
    cols_a_exportar['E1']  = cols_a_exportar['E1'].map(lambda x: f"{x:.2f}")
    cols_a_exportar['E5']  = cols_a_exportar['E5'].map(lambda x: f"{x:.2f}")
    cols_a_exportar['elevation']  = cols_a_exportar['elevation'].map(lambda x: f"{x:.1f}")

    
    cols_a_exportar.to_csv( './outputs/' + filename + '_measurements_E.txt', sep='\t', index=False)
    

    return(rawdiff)

    
def outputsG(VERSION, st, nav, sta1, sta2, file_nav, dist, config, dif):

    # Open file
    filename = sta1.filename.partition(".")[0] + sta2.filename.partition(".")[0]
    file_sum = open('./outputs/' + filename + '_results_GPS.txt', 'w')
    
    file_sum.write(
    f" GNSS_cal_tools Version: {VERSION}\n"
    f"Processing date and time: {st} UTC-3\n"
    f"Output interval (s) = {config['intcod']}\n"
    f"Code threshold (ns) = {config['ithr']}\n"
    f"Residual threshold (m) = {config['thres']}\n"
    f"Processed system: {config['SYS']}. (GPS:G, Galileo:R, Glonass:R, Beidu:C)\n"
    f"Min Elevation (deg): {config['elmin']}\n\n"
    f"INPUT FILES\n"
    f" {file_nav}\tRINEX version: {nav.version}\n"
    f" {sta1.filename}\tRINEX version: {sta1.version}\n"
    f" {sta2.filename}\tRINEX version: {sta2.version}\n\n"
    )


    file_sum.write(
    f"Distance from headers is {dist:.2f} m\n"
    f"Interval of {sta1.filename} is {sta1.interval} s\n"
    f"Interval of {sta2.filename} is {sta2.interval} s\n\n"
    )

    print(
        f"Distance read from headers is: {dist:.2f} m\n"
        f"Interval of file1 is {sta1.interval} s\n"
        f"Interval of file2 is {sta2.interval} s"
    )


    if dist > 1000:
        file_sum.write('WARNING: Distance read from headers is ' + dist + ' m!\n')
        print('WARNING: Distance read from headers is ' + dist + ' m!')
    
    if sta1.interval != sta2.interval:
        file_sum.write('Not the same data interval\n')
        print('Not the same data interval')
    
    
    file_sum.write(
        f"Median and stdev of C1P1 bias in {sta1.filename}: "
        f"({round(sta1['c1p1_bias_median'].values / 0.299792458, 2)} +/- "
        f"{round(sta1['c1p1_bias_std'].values / 0.299792458, 2)}) ns\n"
    
        f"Median and stdev of C1P1 bias in {sta2.filename}: "
        f"({round(sta2['c1p1_bias_median'].values / 0.299792458, 2)} +/- "
        f"{round(sta2['c1p1_bias_std'].values / 0.299792458, 2)}) ns\n\n"
        
        
    )


    print(
        f"Median and stdev of C1P1 bias in {sta1.filename}: "
        f"({round(sta1['c1p1_bias_median'].values / 0.299792458, 2)} +/- "
        f"{round(sta1['c1p1_bias_std'].values / 0.299792458, 2)}) ns\n"
        
        f"Median and stdev of C1P1 bias in {sta2.filename}: "
        f"({round(sta2['c1p1_bias_median'].values / 0.299792458, 2)} +/- "
        f"{round(sta2['c1p1_bias_std'].values / 0.299792458, 2)}) ns\n"
    )

    if config['plot_mp_errors']:
        print(
            f"Mean and stdev of L1 Multipath Error in {sta1.filename}: "
            f"({round(float(sta1.MP1_mean), 2)} +/- "
            f"{round(float(sta1.MP1_std), 2)}) ns\n"
            f"Mean and stdev of L1 Multipath Error in {sta2.filename}: "
            f"({round(float(sta2.MP1_mean), 2)} +/- "
            f"{round(float(sta2.MP1_std), 2)}) ns\n"
                )
        file_sum.write(
        f"Mean and stdev of L1 Multipath Error in {sta1.filename}: "
        f"({round(float(sta1.MP1_mean), 2)} +/- "
        f"{round(float(sta1.MP1_std), 2)}) ns\n"
        f"Mean and stdev of L1 Multipath Error in {sta2.filename}: "
        f"({round(float(sta2.MP1_mean), 2)} +/- "
        f"{round(float(sta2.MP1_std), 2)}) ns\n\n"
        )

    pop1 = dif.groupby(['MJD']).median()


    rawdiff = {
        'medianC1' : round(pop1['C1_corr'].median()/0.299792458, 2),
        'stdC1' : round(pop1['C1_corr'].std()/0.299792458, 2),
        'medianP1' : round(pop1['P1_corr'].median()/0.299792458, 2),
        'stdP1' : round(pop1['P1_corr'].std()/0.299792458, 2),
        'medianP2' : round(pop1['P2_corr'].median()/0.299792458, 2),
        'stdP2' : round(pop1['P2_corr'].std()/0.299792458, 2)
        }


    file_sum.write(
    f"Median and stdev of C1 difference: ({rawdiff['medianC1']} +/- {rawdiff['stdC1']}) ns\n"
    f"Median and stdev of P1 difference: ({rawdiff['medianP1']} +/- {rawdiff['stdP1']}) ns\n"
    f"Median and stdev of P2 difference: ({rawdiff['medianP2']} +/- {rawdiff['stdP2']}) ns\n"
    )

    print(
        f"Median and stdev of C1 difference: ({rawdiff['medianC1']} +/- {rawdiff['stdC1']}) ns\n"
        f"Median and stdev of P1 difference: ({rawdiff['medianP1']} +/- {rawdiff['stdP1']}) ns\n"
        f"Mean and stdev of P2 difference:   ({rawdiff['medianP2']} +/- {rawdiff['stdP2']}) ns\n"
    )

    
    file_sum.close()


    cols_a_exportar = dif[['MJD','sv', 'C1_corr', 'P1_corr', 'P2_corr', 'elevation']].copy()
    cols_a_exportar.columns = ['MJD', 'sv', 'C1', 'P1', 'P2','elevation']
    cols_a_exportar['C1'] = cols_a_exportar['C1']/0.299792458
    cols_a_exportar['P1'] = cols_a_exportar['P1']/0.299792458
    cols_a_exportar['P2'] = cols_a_exportar['P2']/0.299792458
    cols_a_exportar['MJD'] = cols_a_exportar['MJD'].map(lambda x: f"{x:.5f}")
    cols_a_exportar['C1']  = cols_a_exportar['C1'].map(lambda x: f"{x:.2f}")
    cols_a_exportar['P1']  = cols_a_exportar['P1'].map(lambda x: f"{x:.2f}")
    cols_a_exportar['P2']  = cols_a_exportar['P2'].map(lambda x: f"{x:.2f}")
    cols_a_exportar['elevation']  = cols_a_exportar['elevation'].map(lambda x: f"{x:.1f}")

    cols_a_exportar.to_csv( './outputs/' + filename + '_measurements_GPS.txt', sep='\t', index=False)
    

    return(rawdiff)

def outputsC(VERSION, st, nav, sta1, sta2, file_nav, dist, config, dif):

    # Open file
    filename = sta1.filename.partition(".")[0] + sta2.filename.partition(".")[0]
    file_sum = open('./outputs/' + filename + '_results_C.txt', 'w')
    
    file_sum.write(
    f" GNSS_cal_tools Version: {VERSION}\n"
    f"Processing date and time: {st} UTC-3\n"
    f"Output interval (s) = {config['intcod']}\n"
    f"Code threshold (ns) = {config['ithr']}\n"
    f"Residual threshold (m) = {config['thres']}\n"
    f"Processed system: {config['SYS']}. (GPS:G, Galileo:E, Glonass:R, Beidou:C)\n"
    f"Min Elevation (deg): {config['elmin']}\n\n"
    f"INPUT FILES\n"
    f" {file_nav}\tRINEX version: {nav.version}\n"
    f" {sta1.filename}\tRINEX version: {sta1.version}\n"
    f" {sta2.filename}\tRINEX version: {sta2.version}\n\n"
    )

    file_sum.write(
    f"Distance from headers is {dist:.2f} m\n"
    f"Interval of {sta1.filename} is {sta1.interval} s\n"
    f"Interval of {sta2.filename} is {sta2.interval} s\n\n"
    )

    print(
        f"Distance read from headers is: {dist:.2f} m\n"
        f"Interval of file1 is {sta1.interval} s\n"
        f"Interval of file2 is {sta2.interval} s"
    )

    if dist > 1000:
        file_sum.write(f'WARNING: Distance read from headers is {dist} m!\n')
        print(f'WARNING: Distance read from headers is {dist} m!')
    
    if sta1.interval != sta2.interval:
        file_sum.write('Not the same data interval\n')
        print('Not the same data interval')

    # --- Multipath stats ---
    if config['plot_mp_errors']:
        print(
            f"Mean and stdev of B1I Multipath Error in {sta1.filename}: "
            f"({round(float(sta1.MP1_mean), 2)} +/- "
            f"{round(float(sta1.MP1_std), 2)}) ns\n"
            f"Mean and stdev of B1I Multipath Error in {sta2.filename}: "
            f"({round(float(sta2.MP1_mean), 2)} +/- "
            f"{round(float(sta2.MP1_std), 2)}) ns\n"
        )
        file_sum.write(
        f"Mean and stdev of B1I Multipath Error in {sta1.filename}: "
        f"({round(float(sta1.MP1_mean), 2)} +/- "
        f"{round(float(sta1.MP1_std), 2)}) ns\n"
        f"Mean and stdev of B1I Multipath Error in {sta2.filename}: "
        f"({round(float(sta2.MP1_mean), 2)} +/- "
        f"{round(float(sta2.MP1_std), 2)}) ns\n\n"
        )

    # Convert to meters
    l7 = 299792458 / 1207.14e6
    dif['L7I_corr'] = dif['L7I_corr'] * l7
    # --- Statistics ---
    pop1 = dif.groupby(['MJD']).median()

    rawdiff = {
        'medianB1': round(pop1['C2I_corr'].median() / 0.299792458, 2),
        'stdB1': round(pop1['C2I_corr'].std() / 0.299792458, 2),
        'medianB2': round(pop1['C7I_corr'].median() / 0.299792458, 2),
        'stdB2': round(pop1['C7I_corr'].std() / 0.299792458, 2),
    }

    file_sum.write(
    f"Median and stdev of B1I difference: ({rawdiff['medianB1']} +/- {rawdiff['stdB1']}) ns\n"
    f"Median and stdev of B2I difference: ({rawdiff['medianB2']} +/- {rawdiff['stdB2']}) ns\n"
    )

    print(
        f"Median and stdev of B1I difference: ({rawdiff['medianB1']} +/- {rawdiff['stdB1']}) ns\n"
        f"Median and stdev of B2I difference: ({rawdiff['medianB2']} +/- {rawdiff['stdB2']}) ns\n"
    )

    file_sum.close()

    # --- Export measurements ---
    cols_a_exportar = dif[['MJD', 'sv', 'C2I_corr', 'C7I_corr', 'elevation']].copy()
    cols_a_exportar.columns = ['MJD', 'sv', 'B1', 'B2', 'elevation']

    cols_a_exportar['B1'] = cols_a_exportar['B1'] / 0.299792458
    cols_a_exportar['B2'] = cols_a_exportar['B2']# / 0.299792458

    cols_a_exportar['MJD'] = cols_a_exportar['MJD'].map(lambda x: f"{x:.5f}")
    cols_a_exportar['B1']  = cols_a_exportar['B1'].map(lambda x: f"{x:.2f}")
    cols_a_exportar['B2']  = cols_a_exportar['B2'].map(lambda x: f"{x:.2f}")
    cols_a_exportar['elevation'] = cols_a_exportar['elevation'].map(lambda x: f"{x:.1f}")

    cols_a_exportar.to_csv(
        './outputs/' + filename + '_measurements_C.txt',
        sep='\t',
        index=False
    )

    return rawdiff



def ElevationReject(dfSTA,pos,config,name,st):
    """
    Filters satellite data based on elevation angle and optionally plots an elevation histogram.
    
    Args:
        dfSTA (pd.DataFrame): DataFrame containing satellite coordinates (X, Y, Z).
        pos (tuple): Observer's position (x, y, z) in the same coordinate system.
        config (dict): Configuration dictionary with keys:
            - 'elmin' (float): Minimum elevation threshold (degrees).
            - 'plotelevations' (bool): If True, generates an elevation histogram.
        name (str): Name of the station/satellite (used for labeling).
    
    Returns:
        pd.DataFrame: Filtered DataFrame with satellites above the elevation threshold.
    """
    
    # Extract minimum elevation threshold from config
    ielmin = config['elmin']
    xsta, ysta, zsta = pos  # Observer's position

    # Compute satellite positions relative to observer
    xsat =  dfSTA['X'].to_numpy() - xsta
    ysat =  dfSTA['Y'].to_numpy() - ysta 
    zsat =  dfSTA['Z'].to_numpy() - zsta
    
    # sinelv = (
             # (xsta*xsat + ysta*ysat + zsta*zsat)/np.linalg.norm(pos)/np.sqrt(xsat**2 + ysat**2 + zsat**2)
             #   )

    # Calculate sine of elevation angle using dot product and norms
    dot_product = xsta*xsat + ysta*ysat + zsta*zsat
    r_norm = np.sqrt(xsta**2 + ysta**2 + zsta**2)
    s_norm = np.sqrt(xsat**2 + ysat**2 + zsat**2)
    sinelv = dot_product / (r_norm * s_norm) # De las dos formas da lo mismo
    
    # Convert to degrees and store in DataFrame    
    dfSTA['elevation'] = np.arcsin(sinelv)*180/np.pi
    

    # Compute azimuth angle (degrees)
    # azimuth = arctan2(East, North) in local ENU coordinates
    # Convert (X,Y,Z) -> local (E,N,U)
    # Assuming observer position defines Up = pos / |pos|
    lat = np.arctan2(zsta, np.sqrt(xsta**2 + ysta**2))
    lon = np.arctan2(ysta, xsta)

    # Rotation matrix ECEF -> ENU
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    R = np.array([[-sin_lon,             cos_lon,              0],
                  [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat],
                  [ cos_lat*cos_lon,  cos_lat*sin_lon, sin_lat]])

    rel_xyz = np.vstack((xsat, ysat, zsat))
    enu = R @ rel_xyz
    E, N, U = enu[0, :], enu[1, :], enu[2, :]

    dfSTA['azimuth'] = (np.degrees(np.arctan2(E, N)) + 360) % 360


    # Count and filter out low-elevation satellites    
    low_elevations = (dfSTA['elevation'] < ielmin).sum()
    dfSTA = dfSTA[dfSTA['elevation'] >= ielmin]
    
    # Print rejection statistics
    print('Number of measurements below elevation thereshold (' + str(ielmin) + ' degrees):')
    print(name + ' --> REJECTED: ' + str(low_elevations) + '. ACCEPTED: ' + str(dfSTA.count().MJD) + '\n')
    

    # Generate elevation histogram if enabled in config
    if config['plotelevations']:
    
        sns.set_style("white")
        sns.set_context("paper", font_scale=1.4)
    
        # Crear figura con 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
        
        # -------------------------
        # (0,0) Elevation histogram
        # -------------------------
        ax1 = axs[0, 0]
    
        bins = np.arange(0, 100, 5)
    
        sns.histplot(
            data=dfSTA['elevation'],
            bins=bins,
            color='steelblue',
            edgecolor='white',
            linewidth=1.2,
            alpha=0.85,
            ax=ax1
        )
    
        ax1.set_title(name, fontweight='bold')
        ax1.set_xlabel('Elevation / degrees')
        ax1.set_ylabel('Number of satellites')
        ax1.set_xticks(bins)
        ax1.set_xticklabels(bins, rotation=45)
        ax1.set_xlim(0, 95)
        ax1.grid(axis='y', alpha=0.3)
        ax1.grid(axis='x', alpha=0.1)
    
        # -------------------------
        # (0,1) SKYPLOT (polar)
        # -------------------------

        fig.delaxes(axs[0,1])
        ax2 = fig.add_subplot(2, 2, 2, projection='polar')
    
        ax2.set_theta_zero_location('N')
        ax2.set_theta_direction(-1)
    
        theta = np.radians(dfSTA['azimuth'])
        r = dfSTA['elevation']
    
        ax2.scatter(theta, r, s=1)
    
        ax2.set_rlim(90, 0)
        ax2.set_rlabel_position(180)
        ax2.set_title(f'{name} – Skyplot', fontweight='bold')
    
    
        # -------------------------
        # (1,0) OBS vs MJD
        # -------------------------
        ax3 = axs[1, 0]
        
        # Conteo de observaciones por MJD
        obs_per_mjd = dfSTA.groupby('MJD').size()
        
        ax3.plot(obs_per_mjd.index, obs_per_mjd.values, linewidth=1)
        
        ax3.set_title('Observations per epoch', fontweight='bold')
        ax3.set_xlabel('MJD')
        ax3.set_ylabel('Number of observations')
        ax3.grid(alpha=0.3)
            
        # -------------------------
        # Eliminar subplots vacíos
        # -------------------------
        #fig.delaxes(axs[1, 1])
    
            # -------------------------
        # (1,1) Elevation vs MJD
        # -------------------------
        ax4 = axs[1, 1]
        
        sc = ax4.scatter(
            dfSTA['MJD'],
            dfSTA['elevation'],
            s=1,
            alpha=0.5
        )
        
        ax4.set_title('Elevation vs MJD', fontweight='bold')
        ax4.set_xlabel('MJD')
        ax4.set_ylabel('Elevation (deg)')
        ax4.set_ylim(0, 90)
        ax4.grid(alpha=0.3)
    
    
    
        fig.text(
            0.98, 0.5,
            'GNSS_cal_tools\nComputed at: ' + st + ' UTC-3',
            rotation=90,
            fontweight="bold",
            ha='right',
            va='center'
        )
    
        # Ajuste layout
        plt.tight_layout()
    
        plt.savefig(
            f'./outputs/Elevation_Skyplot_{name}_{config["SYS"]}.png',
            dpi=100,
            bbox_inches='tight',
            facecolor='0.9'
        )
    
        plt.close()

    return(dfSTA)

def C1P1(sta,df_sta):
    c1p1_diff = df_sta['C1'] - df_sta['P1']
    sta['c1p1_bias_median'] = c1p1_diff.median()
    sta['c1p1_bias_std'] = c1p1_diff.std()  
    return(sta)

def OExyz(dfnav_first, dfSTA, stafilename,config):
    
    """
     Computes satellite positions in Earth-Centered, Earth-Fixed (ECEF) coordinates 
     using broadcast ephemeris data and merges them with observation data.
    
     Args:
         dfnav_first (pd.DataFrame): DataFrame containing broadcast ephemeris parameters 
                                    (e.g., semi-major axis, eccentricity, mean anomaly).
         dfSTA (pd.DataFrame): Observation DataFrame with satellite IDs ('sv') and 
                               observation times ('MJD').
    
     Returns:
         pd.DataFrame: Augmented observation DataFrame with satellite ECEF coordinates (X, Y, Z).
     """
    
    # Merge Ephemeris and Observation Data ---
    # Compute mean motion (rad/s) and corrected mean motion

    # Add MJD_O, Corrected mean motion, Mean Anomaly an Eccentricity to 
    # observation dataframe

    # N0 - Computed mean motion in rad/s
    dfnav_first['N0'] = np.sqrt(MU)/(dfnav_first['sqrtA']**(3)) # Checked

    # N - Corrected mean motion
    dfnav_first['N'] = dfnav_first['N0'] + dfnav_first['DeltaN'] #Checked
    
    dfSTA = dfSTA.merge(dfnav_first, on="sv", how="left")
    
    # Compute Orbital Parameters
    # A - Semi-major axis a in meters
    A = dfSTA['sqrtA'].to_numpy()**2 #Checked
    #print(str(np.min(A)) + ' < A < ' + str(np.max(A)))        

    # Time since ephemeris reference epoch (seconds)
    # Calculate TK - Time from ephemeris reference epoch in sec
    dfSTA['TK'] = (dfSTA['MJD'] - dfSTA['MJD_N'])*86400 #Checked
    
    # Mean anomaly (radians): M = M0 + N * TK
    # Calculate Mean Anomaly
    dfSTA['MK'] = dfSTA['M0'] + dfSTA['N']*dfSTA['TK']
    # se muestra en 
   # https://www.gsc-europa.eu/gsc-products/galileo-rinex-navigation-parameters

    def kepler_function(E, M, e):
        """Kepler's equation: f(E) = E - e*sin(E) - M"""
        return E - e * np.sin(E) - M

    def kepler_jacobian(E, M, e):
        """Jacobian (derivative) of Kepler's equation: f'(E) = 1 - e*cos(E)"""
        return 1 - e * np.cos(E)
    
    # Numerically solve Kepler's equation for each satellite

    result = np.zeros(dfSTA.shape[0])
    for i in range(0,dfSTA.shape[0]):
        M = dfSTA['MK'][i]              # Mean anomaly in radians
        e = dfSTA['Eccentricity'][i]    # Eccentricity
        E_solution = fsolve(kepler_function, M, args=(M, e), fprime=kepler_jacobian, maxfev=25, full_output=True)
        # "Because of the small eccentricity of GPS orbits (e less 0.001), 
        # two steps are usually sufficient " Applied GPS for Engineers and Project Managers

        result[i] = E_solution[0][0]
#        salida, info, ier, mesg = E_solution
#        print(f"Iteraciones: {info['nfev']}")
#        print(f"Error residual: {info['fvec'][0]:.2e}")
#        print(f"Código de salida: {ier} - {mesg}")

    dfSTA['EK'] = result
    
    # Calculo de la true anomaly, vk. 
    # La 'EK' es eccentricity anomaly que sale de resolver numericamente 
    # la ecuación de Kepler.

    ec = dfSTA['Eccentricity'].to_numpy()
    ek = dfSTA['EK'].to_numpy()    
    cosek = np.cos(ek)
    denom = 1-ec*cosek
    VC = (cosek-ec)/denom
    VS = np.sin(ek)*np.sqrt(1-ec**2)/denom
    vk = np.arctan(VS/VC)
    # Add pi to negative elements
    vk[VC < 0] += np.pi
    
    # PHI -Argument of latitude
    phi = vk + dfSTA['omega'].to_numpy()
    
    # Harmonic corrections for orbit perturbations
    # DUK -Argument of Latitue Correction
    DUK = (
        dfSTA['Cus'].to_numpy()*np.sin(2*phi) +
        dfSTA['Cuc'].to_numpy()*np.cos(2*phi)
        )
    
    # DRK -Radius Correction
    DRK = (
        dfSTA['Crc'].to_numpy()*np.cos(2*phi) +
        dfSTA['Crs'].to_numpy()*np.sin(2*phi)
        )
    
    # DIK -Correction to Inclination
    DIK = (dfSTA['Cic'].to_numpy()*np.cos(2*phi) +
           dfSTA['Cis'].to_numpy()*np.sin(2*phi)
           )
    
    # UK - Corrected Argument of Latitude
    UK = phi + DUK
    
    # RK - Corrected Radius
    RK = A*denom + DRK

    # IK - Corrected Inclination
    IK = dfSTA['Io'].to_numpy() + DIK + dfSTA['IDOT'].to_numpy()*dfSTA['TK'].to_numpy()
    
    # POSITION IN ORBITAL PLANE
    XK = RK*np.cos(UK)
    YK = RK*np.sin(UK)
    
    # OMEGAK - Corrected Longitude of Ascending Node
    #OMEGAK=dfSTA['Omega0'].to_numpy() + (dfSTA['OmegaDot'].to_numpy()-OMEGAE)*dfSTA['TK'].to_numpy()-OMEGAE*dfSTA['Toe'].to_numpy()
    OMEGAK = (
    dfSTA['Omega0'].to_numpy()
    + (dfSTA['OmegaDot'].to_numpy() - OMEGAE) * dfSTA['TK'].to_numpy()
    - OMEGAE * dfSTA['Toe'].to_numpy()
    )
    
    # - EARTH FIXED COORDINATES -
    X = XK*np.cos(OMEGAK) - YK*np.cos(IK)*np.sin(OMEGAK)
    Y = XK*np.sin(OMEGAK) + YK*np.cos(IK)*np.cos(OMEGAK)
    Z = YK*np.sin(IK)
    
    dfSTA['X'] = X
    dfSTA['Y'] = Y
    dfSTA['Z'] = Z
    
    
    # Remove unhealthy satellites
    if config['SYS'] == 'C':
        unhealthy_count = sum(dfSTA['SatH1'] != 0)
        healthy_count = sum(dfSTA['SatH1'] == 0)
        print(f'Unhealthy/healthy sats in {stafilename}: {unhealthy_count}/{healthy_count}')
        dfSTA = dfSTA[dfSTA['SatH1'] == 0]

    else:
        unhealthy_count = sum(dfSTA['health'] != 0)
        healthy_count = sum(dfSTA['health'] == 0)
        print(f'Unhealthy/healthy sats in {stafilename}: {unhealthy_count}/{healthy_count}')
        dfSTA = dfSTA[dfSTA['health'] == 0]

    return(dfSTA)


def dfSTAgen(STA):
    # Generation of dataframes
    dfSTA = STA.to_dataframe()
    
    # Removing rows with only NANs from dataframes
    dfSTA = dfSTA.dropna(how='all')
    
    # Resetting indexes
    dfSTA = dfSTA.reset_index()

    #Adding MJD columns
    dfSTA['time'] = pd.to_datetime(dfSTA['time'])
    pop = dfSTA['time'].dt.strftime('%Y-%m-%d %H:%M:%S').to_list()
    dfSTA['MJD'] = Time(pop).mjd
        
    return(dfSTA)

def dfNAVgen(nav,config):
    # Generation of dataframes
    dfnav = nav.to_dataframe()

    # Removing rows with only NANs from dataframes
    dfnav = dfnav.dropna(how='all')
    
    # Resetting indexes
    dfnav = dfnav.reset_index()
    
    # Remove unhealthy 
    if config['SYS'] == 'C':
        unhealthy_count = sum(dfnav['SatH1'] != 0)
        healthy_count = sum(dfnav['SatH1'] == 0)
        dfnav = dfnav[dfnav['SatH1'] == 0]
        print(f'Unhealthy/healthy sats in {nav.filename}: {unhealthy_count}/{healthy_count}')
    else:    
        unhealthy_count = sum(dfnav['health'] != 0)
        healthy_count = sum(dfnav['health'] == 0)
        dfnav = dfnav[dfnav['health'] == 0]
        print(f'Unhealthy/healthy sats in {nav.filename}: {unhealthy_count}/{healthy_count}')

    #Remove satellites with clock drift bigger than 1e-11
    #dfnav = dfnav[dfnav['SVclockDrift'].abs() < 1e-11]

    
    #Adding MJD columns
    dfnav['time'] = pd.to_datetime(dfnav['time'])
    pop = dfnav['time'].dt.strftime('%Y-%m-%d %H:%M:%S').to_list()
    dfnav['MJD_N'] = Time(pop).mjd
    
    return(dfnav)
