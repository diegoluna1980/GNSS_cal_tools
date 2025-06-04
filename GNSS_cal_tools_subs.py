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


# =============================================================================
#  Define constants
# =============================================================================

# Standard Gravitational Parameter (μ) for Earth in m³/s² 
# for GPS users https://www.unoosa.org/pdf/icg/2012/template/WGS_84.pdf
MU = 3.9860050e14 

# OMEGAE - WGS84 value of the Earth's rotation rate in rad/sec
OMEGAE = 7.292115e-5

def calibration(rawdiff, delays_a, delays_b):
    
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
    delays_b['INTdlyC1'] = delays_a['INTdlyC1'] - deltaINTdlyC1 
    delays_b['INTdlyP1'] = delays_a['INTdlyP1'] - deltaINTdlyP1
    delays_b['INTdlyP2'] = delays_a['INTdlyP2'] - deltaINTdlyP2
    
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
            dataset = gr.load(file, use=config['SYS'],
                              meas=['C1C', 'C1W', 'C2W'])
            dataset = dataset.rename({"C1C": "C1"})
            dataset = dataset.rename({"C1W": "P1"})
            dataset = dataset.rename({"C2W": "P2"})
        else:
            dataset = gr.load(file, use=config['SYS'], meas=['C1', 'P1', 'P2'])
    return(dataset)

def figures(dif,config,ts):
    
    """
    Generates plots of time series and time deviations (TDEV) 
    for C1, P1, and P2 GNSS code corrections, and saves them to a PDF.

    Parameters:
    - dif: DataFrame containing corrected GNSS code data with MJD index.
    - config: dict with configuration options 
    - ts: Unix timestamp indicating when the computation was performed.
    """
    
    
    if config['timeplots']:

        # Conversion factor from kilometers to nanoseconds
        k = 0.299792458
        
        # List of unique Modified Julian Dates (MJD)
        MJD = dif.MJD.unique()
        
        # Group data by MJD and compute median values
        pop1 = dif.groupby(['MJD']).median()
        
        # Convert median corrected values from km to ns
        C1 = pop1['C1_corr'].to_numpy() / k
        P1 = pop1['P1_corr'].to_numpy() / k
        P2 = pop1['P2_corr'].to_numpy() / k
               
        # Compute Time Deviation (TDEV) using allantools for each observable
        (C1_tau_tdev, C1_tdev, C1_tdeverr, n_tdev) = allantools.tdev(C1, rate= 1/config['intcod'], data_type="phase", taus='octave')
        (P1_tau_tdev, P1_tdev, P1_tdeverr, n_tdev) = allantools.tdev(P1, rate= 1/config['intcod'], data_type="phase", taus='octave')
        (P2_tau_tdev, P2_tdev, P2_tdeverr, n_tdev) = allantools.tdev(P2, rate= 1/config['intcod'], data_type="phase", taus='octave')
        
        # Create figure and layout        
        fig1 = plt.figure(1,figsize=(12,8))
        plt.subplots_adjust(hspace = .3)
        
        # Add timestamp to right margin
        plt.figtext(0.95, 0.5,  'Computed at: ' + datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') + ' UTC-3\n', rotation=90)
        
        # Plot time series for C1        
        plt.subplot(231)
        plt.plot(MJD, C1, 'b.',markeredgewidth=0.0,zorder=4,label='C1')
        plt.title('Median: (' + str(round(np.median(C1),1)) + '+/-' + str(round(C1.std(),1)) + ') ns')
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
        plt.title('Median: (' + str(round(np.median(P1),1)) + '+/-' + str(round(P1.std(),1)) + ') ns')
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
        plt.title('Median: (' + str(round(np.median(P2),1)) + '+/-' + str(round(P2.std(),1)) + ') ns')
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
        destino = 'C1P1P2plotsGNSS_cal_tools.pdf'
        fig1.savefig(destino,facecolor='0.9', dpi = 200)
        plt.close()


def DIFgen(dfSTA1, dfSTA2, config, pos1, pos2):

    """
    Generate differential GNSS observations between two stations.

    This function computes epoch-wise differences between pseudorange observations 
    (C1, P1, P2) from two stations, aligned by time and satellite. It applies filters 
    to remove outliers and corrects the differences based on the geometric projection 
    of the baseline between the stations onto the satellite directions.

    Parameters:
    -----------
    dfSTA1 : pd.DataFrame
        DataFrame for station 1. Must contain columns:
        ['MJD', 'sv', 'C1', 'P1', 'P2', 'X', 'Y', 'Z', 'elevation']

    dfSTA2 : pd.DataFrame
        DataFrame for station 2. Must contain columns:
        ['MJD', 'sv', 'C1', 'P1', 'P2']

    config : dict
        Configuration dictionary containing:
        - 'intcod': integration time in seconds (typically 300)

    pos1 : array-like of float
        coordinates [X, Y, Z] of station 1

    pos2 : array-like of float
        coordinates [X, Y, Z] of station 2

    Returns:
    --------
    dif : pd.DataFrame
        DataFrame of differential measurements and metadata with columns:
        ['MJD', 'sv', 'X', 'Y', 'Z', 'elevation', 
         'C1', 'P1', 'P2', 'P1-P2', 
         'C1_corr', 'P1_corr', 'P2_corr']
    """

    # Convert integration time from seconds to days
    codint = config['intcod']/86400  
    dat1 = pd.DataFrame()
    dat2 = pd.DataFrame()
    
    # Generate time vector from min to max MJD, spaced by integration time
    # (usually 300 s, the value of intcod)
    dat1['MJD'] = np.arange(np.floor(dfSTA1['MJD'].min()), np.ceil(dfSTA1['MJD'].max()) + codint, codint)
    
    # Get all unique satellites
    # A las columnas con las fechas, le agrego una entrada por cada satelite
    arr = dfSTA1['sv'].unique()
    df_arr = pd.DataFrame({'sv': arr})
    
    # Cross join: each time instant x each satellite
    dat1 = dat1.merge(df_arr,how='cross')
    
    # Copio al dat2
    dat2['MJD'] = dat1['MJD'].copy()
    dat2['sv'] = dat1['sv'].copy()
    
    # Precompilar valores para asignar
    dat1[['C1', 'P1', 'P2']] = np.nan
    C1_vals = []
    P1_vals = []
    P2_vals = []
    X_vals = []
    Y_vals = []
    Z_vals = []
    ele_vals = []
    
    # Fill dat1 with median values within each time window
    for row in dat1.itertuples(index=False):
         sat = row.sv
         mjd = row.MJD
         mask = (dfSTA1['sv'] == sat) & (np.abs(dfSTA1['MJD'] - mjd) < codint/2)
         subset = dfSTA1.loc[mask]
         C1_vals.append(subset['C1'].median())
         P1_vals.append(subset['P1'].median())
         P2_vals.append(subset['P2'].median())
         X_vals.append(subset['X'].median())
         Y_vals.append(subset['Y'].median())
         Z_vals.append(subset['Z'].median())
         ele_vals.append(subset['elevation'].median())
    
    # Assign aggregated values to dat1
    dat1['C1'] = C1_vals
    dat1['P1'] = P1_vals
    dat1['P2'] = P2_vals
    dat1['X'] = X_vals
    dat1['Y'] = Y_vals
    dat1['Z'] = Z_vals
    dat1['elevation'] = ele_vals
    
    # Repeat for second station (no satellite positions needed)
    dat2[['C1', 'P1', 'P2']] = np.nan
    C1_vals = []
    P1_vals = []
    P2_vals = []
    
    
    for row in dat2.itertuples(index=False):
        sat = row.sv
        mjd = row.MJD
        mask = (dfSTA2['sv'] == sat) & (np.abs(dfSTA2['MJD'] - mjd) < codint/2)
        subset = dfSTA2.loc[mask]
        C1_vals.append(subset['C1'].median())
        P1_vals.append(subset['P1'].median())
        P2_vals.append(subset['P2'].median())

    # Asignar las columnas
    dat2['C1'] = C1_vals
    dat2['P1'] = P1_vals
    dat2['P2'] = P2_vals
    
    # Compute differences between stations
    dif = pd.DataFrame()    
    #dif = dat1[['MJD', 'sv']].copy()
    dif = dat1[['MJD', 'sv', 'X', 'Y','Z','elevation']].copy()

    # Calculo diferencias    
    dif['C1'] = dat1['C1'] - dat2['C1']
    dif['P1'] = dat1['P1'] - dat2['P1']
    dif['P2'] = dat1['P2'] - dat2['P2']
    dif['P1-P2'] = dif['P1'] - dif['P2']
    
    # Filter out large outliers (absolute value <= 300 ns)(Line 1588 dclrinex)
    dif = dif[(dif[['C1', 'P1', 'P2']].abs() <= 300).all(axis=1)]
    
    # Additional filter on ionospheric observable P1-P2 (<= 30 ns)
    # (Line 1592 dclrinex) (not really necessary in this case)
    dif = dif[(dif[['P1-P2']].abs() <= 30).all(axis=1)]

    #------------------------------------------------------------------------------
    # Apply Median Absolute Deviation (MAD) filter for C1, P1, and P2
    #------------------------------------------------------------------------------

    u = 3 # Threshold for MAD filtering (u * sigma)
    Xc = dif['C1'].median()
    S=1.4826*np.median(np.abs(dif['C1']-Xc))
    dif['C1-Xc'] = dif['C1']-Xc
    dif = dif[(dif[['C1-Xc']].abs() <= u*S).all(axis=1)]

    Xc = dif['P1'].median()
    S=1.4826*np.median(np.abs(dif['P1']-Xc))
    dif['P1-Xc'] = dif['P1']-Xc
    dif = dif[(dif[['P1-Xc']].abs() <= u*S).all(axis=1)]

    Xc = dif['P2'].median()
    S=1.4826*np.median(np.abs(dif['P2']-Xc))
    dif['P2-Xc'] = dif['P2']-Xc
    dif = dif[(dif[['P2-Xc']].abs() <= u*S).all(axis=1)]
    
    dif.drop(columns='C1-Xc', inplace=True)
    dif.drop(columns='P1-Xc', inplace=True)
    dif.drop(columns='P2-Xc', inplace=True)

    #------------------------------------------------------------------------------
    # Geometry-based correction (project baseline onto satellite direction)
    #------------------------------------------------------------------------------

    x = pos2-pos1

    xsta, ysta, zsta = pos1
    xsat =  dif['X'].to_numpy() - xsta
    ysat =  dif['Y'].to_numpy() - ysta
    zsat =  dif['Z'].to_numpy() - zsta
    r = np.sqrt(xsat**2+ysat**2+zsat**2) # 1664 in dclrinex
    
    # Projection of baseline onto satellite line-of-sight    
    corg1 = (x[0]*xsat + x[1]*ysat + x[2]*zsat)/r
    
    # Apply correction to measurements
    dif['C1_corr'] = (dif['C1'].to_numpy() - corg1)
    dif['P1_corr'] = (dif['P1'].to_numpy() - corg1)
    dif['P2_corr'] = (dif['P2'].to_numpy() - corg1)
    

    return(dif)
    
def outputs(VERSION, st, nav, sta1, sta2, file_nav, dist, config, dif):

    # Open file
    filename = sta1.filename.partition(".")[0] + sta2.filename.partition(".")[0]
    file_sum = open(filename + '_results.txt', 'w')
    
    file_sum.write(
    f" SIM_GNSS_cal Version: {VERSION}\n"
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

    pop1 = dif.groupby(['MJD']).median()

    rawdiff = {
        'medianC1' : round(pop1['C1_corr'].median()/0.299792458, 1),
        'stdC1' : round(pop1['C1_corr'].std()/0.299792458, 1),
        'medianP1' : round(pop1['P1_corr'].median()/0.299792458, 1),
        'stdP1' : round(pop1['P1_corr'].std()/0.299792458, 1),
        'medianP2' : round(pop1['P2_corr'].median()/0.299792458, 1),
        'stdP2' : round(pop1['P2_corr'].std()/0.299792458, 1)
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


    cols_a_exportar = dif[['MJD','sv', 'C1_corr', 'P1_corr', 'P2_corr']].copy()
    cols_a_exportar.columns = ['MJD', 'sv', 'C1', 'P1', 'P2']
    cols_a_exportar['MJD'] = cols_a_exportar['MJD'].map(lambda x: f"{x:.5f}")
    cols_a_exportar['C1']  = cols_a_exportar['C1'].map(lambda x: f"{x:.2f}")
    cols_a_exportar['P1']  = cols_a_exportar['P1'].map(lambda x: f"{x:.2f}")
    cols_a_exportar['P2']  = cols_a_exportar['P2'].map(lambda x: f"{x:.2f}")
    cols_a_exportar.to_csv( filename + '_measurements.txt', sep='\t', index=False)
    

    return(rawdiff)

def ElevationReject(dfSTA,pos,config,name):
    
    ielmin = config['elmin']
    xsta, ysta, zsta = pos

    xsat =  dfSTA['X'].to_numpy() - xsta
    ysat =  dfSTA['Y'].to_numpy() - ysta 
    zsat =  dfSTA['Z'].to_numpy() - zsta
    
    # sinelv = (
             # (xsta*xsat + ysta*ysat + zsta*zsat)/np.linalg.norm(pos)/np.sqrt(xsat**2 + ysat**2 + zsat**2)
             #   )
    
    dot_product = xsta*xsat + ysta*ysat + zsta*zsat
    r_norm = np.sqrt(xsta**2 + ysta**2 + zsta**2)
    s_norm = np.sqrt(xsat**2 + ysat**2 + zsat**2)
    sinelv = dot_product / (r_norm * s_norm) # De las dos formas da lo mismo
    
    dfSTA['elevation'] = np.arcsin(sinelv)*180/np.pi
    
    low_elevations = (dfSTA['elevation'] < ielmin).sum()
    dfSTA = dfSTA[dfSTA['elevation'] >= ielmin]
    
    print('Number of measurements below elevation thereshold (' + str(ielmin) + ' degrees):')
    print(name + ' --> REJECTED: ' + str(low_elevations) + '. ACCEPTED: ' + str(dfSTA.count().MJD) + '\n')
    
    # =============================================================================
    # Generation of elevation histogram
    # =============================================================================
    if config['plotelevations']:
        
        data = dfSTA['elevation']

        # Set up the figure 
        plt.figure(figsize=(10, 6), dpi=300)  # High resolution
        sns.set_style("white")  # Clean background
        sns.set_context("paper", font_scale=1.4)

        # Create histogram 
        bins = np.arange(0, 100, 5)  # 0-95 in steps of 5
        sns.histplot(data=data, bins=bins, color='steelblue', 
                            edgecolor='white', linewidth=1.2, alpha=0.85)

        # Customize the plot
        plt.title(name, pad=20, fontweight='bold')
        plt.xlabel('Elevation / degrees', labelpad=10)
        plt.ylabel('Number of satellites', labelpad=10)
        plt.xticks(bins, rotation=45)
        plt.xlim(0, 95)  
        
        # Add grid 
        plt.grid(axis='y', alpha=0.3)
        plt.grid(axis='x', alpha=0.1)
        plt.tight_layout()

        # Save image 
        plt.savefig('Elevation_' + name + 'histogram.pdf', dpi=300,
                    bbox_inches='tight')
        plt.show()    
    return(dfSTA)

def C1P1(sta,df_sta):
    sta['c1p1_bias_median'] = (df_sta['C1'] - df_sta['P1']).median()
    sta['c1p1_bias_std'] = (df_sta['C1'] - df_sta['P1']).std()    
    return(sta)

def OExyz(dfnav_first,dfSTA):
    
    # Add MJD_O, Corrected mean motion, Mean Anomaly an Eccentricity to 
    # observation dataframe

    # N0 - Computed mean motion in rad/s
    dfnav_first['N0'] = np.sqrt(MU)/(dfnav_first['sqrtA']**(3)) # Checked

    # N - Corrected mean motion
    dfnav_first['N'] = dfnav_first['N0'] + dfnav_first['DeltaN'] #Checked
    
    dfSTA = dfSTA.merge(dfnav_first, on="sv", how="left")
    
    # A - Semi-major axis a in meters
    A = dfSTA['sqrtA'].to_numpy()**2 #Checked
    #print(str(np.min(A)) + ' < A < ' + str(np.max(A)))        

    # Calculate TK - Time from ephemeris reference epoch in sec
    dfSTA['TK'] = (dfSTA['MJD'] - dfSTA['MJD_N'])*86400 #Checked
    
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
    
    result = np.zeros(dfSTA.shape[0])
    for i in range(0,dfSTA.shape[0]):
        M = dfSTA['MK'][i]              # Mean anomaly in radians
        e = dfSTA['Eccentricity'][i]    # Eccentricity
        E_solution = fsolve(kepler_function, M, args=(M, e), fprime=kepler_jacobian, maxfev=25, full_output=True)
        # "Because of the small eccentricity of GPS orbits (e less 0.001), 
        # two steps are usually sufficient " Applied GPS for Engineers and Project Managers

        result[i] = E_solution[0][0]
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
    
    # DUK -Argument of Latitue Correction
    DUK = dfSTA['Cus'].to_numpy()*np.sin(2*phi) + dfSTA['Cuc'].to_numpy()*np.cos(2*phi) 
    
    # DRK -Radius Correction
    DRK = dfSTA['Crc'].to_numpy()*np.cos(2*phi) + dfSTA['Crs'].to_numpy()*np.sin(2*phi)
    
    # DIK -Correction to Inclination
    DIK = dfSTA['Cic'].to_numpy()*np.cos(2*phi) + dfSTA['Cis'].to_numpy()*np.sin(2*phi)
    
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

def dfNAVgen(nav):
    # Generation of dataframes
    dfnav = nav.to_dataframe()
    
    # Removing rows with only NANs from dataframes
    dfnav = dfnav.dropna(how='all')
    
    # Resetting indexes
    dfnav = dfnav.reset_index()
    
    #Adding MJD columns
    dfnav['time'] = pd.to_datetime(dfnav['time'])
    pop = dfnav['time'].dt.strftime('%Y-%m-%d %H:%M:%S').to_list()
    dfnav['MJD_N'] = Time(pop).mjd
    
    return(dfnav)
