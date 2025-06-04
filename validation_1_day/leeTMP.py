#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:26:00 2024

@author: diego
"""

#def LeeCGGTTS(date, header):
    
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import allantools
import time, datetime

# Date
ts = time.time()    


df = pd.read_csv('./C1.tmp', delim_whitespace=True)
df.columns = ['FMJD', 'delay', 'sat']
# Extraigo cada una de las fechas distintas que aparecen en el FMJD
MJD = df.FMJD.unique()
# Tomo la media de cada una de las FMJD únicas
pop1 = df.groupby(['FMJD']).median()
C1 = pop1['delay'].to_numpy()
(C1_tau_tdev, C1_tdev, P1_tdeverr, n_tdev) = allantools.tdev(C1, data_type="phase", rate=0.0033333, taus='octave')


df = pd.read_csv('./P1.tmp', delim_whitespace=True)
df.columns = ['FMJD', 'delay', 'sat']
# Extraigo cada una de las fechas distintas que aparecen en el FMJD
pop = df.FMJD.unique()
# Tomo la media de cada una de las FMJD únicas
pop1 = df.groupby(['FMJD']).median()
P1 = pop1['delay'].to_numpy()
(P1_tau_tdev, P1_tdev, P1_tdeverr, n_tdev) = allantools.tdev(P1, data_type="phase", rate=0.0033333, taus='octave')


# Levanto datos como strings
df = pd.read_csv('./P2.tmp', delim_whitespace=True)
df.columns = ['FMJD', 'delay', 'sat']
# Extraigo cada una de las fechas distintas que aparecen en el FMJD
pop = df.FMJD.unique()
# Tomo la media de cada una de las FMJD únicas
pop1 = df.groupby(['FMJD']).median()
P2 = pop1['delay'].to_numpy()
(P2_tau_tdev, P2_tdev, P2_tdeverr, n_tdev) = allantools.tdev(P2, data_type="phase", rate=0.0033333, taus='octave')


#==============================================================================
# Figura 1    
#==============================================================================

fig1 = plt.figure(1,figsize=(12,8))
plt.subplots_adjust(hspace = .3)
plt.figtext(0.95, 0.5,  'Computed at: '+ datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') + ' UTC-3\n', rotation=90)

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

plt.subplot(234)
plt.loglog(C1_tau_tdev, C1_tdev, '-ko',markeredgewidth=0.0,zorder=4)
#plt.title('C1_alllan')
#plt.legend()
plt.ylabel('Time deviation / ns', size = 14)
plt.xlabel('Time / s', size = 14)
plt.yticks(size=12)
plt.xticks(size=12)
plt.grid(linestyle='dashed')
plt.tick_params(direction="in")

plt.subplot(235)
plt.loglog(P1_tau_tdev, P1_tdev, '-ko',markeredgewidth=0.0,zorder=4)
plt.xlabel('Time / s', size = 14)
#plt.title('P1_alllan')
plt.yticks(size=12)
plt.xticks(size=12)
plt.grid(linestyle='dashed')
plt.tick_params(direction="in")

plt.subplot(236)
plt.loglog(P2_tau_tdev, P2_tdev, '-ko',markeredgewidth=0.0,zorder=4)
plt.xlabel('Time / s', size = 14)
#plt.title('P2_alllan')
plt.yticks(size=12)
plt.xticks(size=12)
plt.grid(linestyle='dashed')
plt.tick_params(direction="in")

plt.suptitle('C1, P1, and P2 plots. DCLRINEX.f', fontsize=16,  fontweight='bold')
destino = 'C1P1P2plotsDCLRINEX.pdf'
fig1.savefig(destino,facecolor='0.9', dpi = 200)
plt.close()



