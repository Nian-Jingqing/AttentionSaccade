# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:52:28 2017

@author: sammirc
"""


import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import pygazeanalyser
import matplotlib
from matplotlib import pyplot as plt
from pygazeanalyser.edfreader import read_edf
from pygazeanalyser import gazeplotter
from scipy import stats, signal, ndimage, interpolate
from scipy.signal import resample
import re
import os
import mne
import copy
import cPickle
workingfolder = '/home/sammirc/Experiments/Nick/AttentionSaccade' #workstation directory
os.chdir(workingfolder)
import myfunctions

#from mpl_toolkits.mplot3d import Axes3D
#%matplotlib inline
%matplotlib
np.set_printoptions(suppress = True)

import multiprocessing
n_cores = multiprocessing.cpu_count() #8 cores on workstation


#%% load in all data from pickle (this is combined behavioural and eyetracking data, with x and y interpolated for blink reduction)

pickname = os.path.join(workingfolder, 'AttSacc_9subs_cleanedxy_BehavEyeDat.pickle')
print 'loading combined data from pickle'
with open(os.path.join(workingfolder, pickname), 'rb') as handle:
    edat = cPickle.load(handle)
print 'done!'

sublist = [1, 2, 3, 4, 5, 6, 7, 8, 9]

#%%

# get saccade time for saccade task trials
# finds the closest saccade of a tral to the array trigger (when target appears)
# takes time at which it starts, and subtracts from array onset to get saccade time

# also stores 'csac' which is the saccade closest to the target onset.
# If this is the first saccade after target (i.e appropriate response) then ['csac']==['fsac'] would be true,

# if using np.argmin then negative values mean saccades happened before target appeared!
#saccade response: start, end, duration, startx, starty, endx, endy

for sub in range(len(edat)):
    sacctask = [x for x in range(len(edat[sub])) if edat[sub][x]['behaviour']['task'] == 2.0]
    for i in sacctask:
        targettrig  = edat[sub][i]['triggers']['array'] #0 is the trackertime, 1 is the within-trial time
        fstime      = edat[sub][i]['trackertime'][0]
        trlsaccades = edat[sub][i]['events']['Esac']
        if len(trlsaccades) == 0:
            s_starts = np.array([]) #initialise empty array such that s_starts.size == 0 if no saccades in the trial
            pp_sac   = np.array([])
            c_sac    = np.array([])
        elif len(trlsaccades) == 1: #one saccade in the trial only
            s_starts    = np.array(edat[sub][i]['events']['Esac'])[0][0]   # start point of all saccades on this trial
            pp_sac      = np.squeeze(np.where(s_starts >= targettrig[0])) # all post prob saccades for this trial
            c_sac       = np.argmin(np.abs(s_starts - targettrig[0]))     # closest saccade to the target onset
        else:
            s_starts    = np.array(edat[sub][i]['events']['Esac'])[:,0]   # start point of all saccades on this trial
            pp_sac      = np.squeeze(np.where(s_starts >= targettrig[0])) # all post prob saccades for this trial
            c_sac       = np.argmin(np.abs(s_starts - targettrig[0]))     # closest saccade to the target onset
        if s_starts.size == 0: # no saccades at all in this trial
            edat[sub][i]['behaviour']['fstime']   = fstime
            edat[sub][i]['behaviour']['fsac']     = np.NaN # no first saccade as no saccades on trial
            edat[sub][i]['behaviour']['sacctime'] = np.NaN # ST is missing as no saccade on the trial
            edat[sub][i]['behaviour']['csac']     = np.NaN # no closest saccade as none on the trial
        elif s_starts.size > 0 and pp_sac.size == 0: #there were saccades on this trial, but none after the probe
            edat[sub][i]['behaviour']['fstime']   = fstime
            edat[sub][i]['behaviour']['fsac']     = np.NaN
            edat[sub][i]['behaviour']['sacctime'] = np.NaN
            edat[sub][i]['behaviour']['csac']     = trlsaccades[c_sac]
        elif s_starts.size > 0 and pp_sac.size == 1: #saccades on this trial, and only one happened after the target appeared
            edat[sub][i]['behaviour']['fstime']   = fstime
            edat[sub][i]['behaviour']['fsac']     = trlsaccades[pp_sac] # first saccade after target onset
            edat[sub][i]['behaviour']['sacctime'] = trlsaccades[pp_sac][0] - targettrig[0] #diff in tracker time between saccade onset and target onset
            edat[sub][i]['behaviour']['csac']     = trlsaccades[c_sac]     # saccade closest to the target onset
        elif s_starts.size > 0 and pp_sac.size > 1:
            edat[sub][i]['behaviour']['fstime']   = fstime
            edat[sub][i]['behaviour']['fsac']     = trlsaccades[pp_sac[0]] # first saccade after target onset
            edat[sub][i]['behaviour']['sacctime'] = trlsaccades[pp_sac[0]][0] - targettrig[0] #diff in tracker time between saccade onset and target onset
            edat[sub][i]['behaviour']['csac']     = trlsaccades[c_sac]     # saccade closest to the target onset
#%%
#pipeline:
#   1. interpolate effect of blinks on the x and y traces (this is done in the preprocessing)
#   2. baseline x and y to fixation x (960) and y (540) prior to cue period (i.e. take average of 100ms before, remove it, then add that value to recentre traces)
#   3. euclidian distance measurements
          
#%% step 2 - baselining x and y 
          
fixpos = [960, 540] #centre of the screen

for sub in range(len(edat)):
    print('baselining x and y data for subject %s' %sublist[sub])
    for trl in range(len(edat[sub])):
        cueind = edat[sub][trl]['triggers']['cue'][1] #0 is in trackertime, 1 is in trial time 
        xbline = np.nanmean(edat[sub][trl]['x'][cueind-100:cueind])
        ybline = np.nanmean(edat[sub][trl]['y'][cueind-100:cueind])
        edat[sub][trl]['x'] = np.subtract(edat[sub][trl]['x'], (xbline-fixpos[0]))
        edat[sub][trl]['y'] = np.subtract(edat[sub][trl]['y'], (ybline-fixpos[1]))

#%% step 3 - distance measurements from fixation and target

# for each trial, calculate time series for distance from fixation (dist2fix) and distance from target (dist2fix)
# this collapses across x and y into one series. - will make baselining trials easy for both dist2targ and dist2fix as can just reduce one instead of 2
# but this may be inaccurate because may need to do this before doing this distance measurement, so it fixes both rather than just pixel distance. unsure?


#this is a bit slow, and could ultimately be multi-cored at a later point.
fixpos = [960,540] #centre of the screen
for sub in range(len(edat)):
    print('working on subject %s'%sublist[sub])
    for trl in range(len(edat[sub])):
        targpos   = np.squeeze(edat[sub][trl]['behaviour']['targlocpix']).tolist()
        eyepos    = np.vstack([edat[sub][trl]['x'], edat[sub][trl]['y']]) #stack x and y into array of shape (2, length of trial in samples)
        dist2fix  = np.zeros(eyepos.shape[1])
        dist2targ = np.zeros(eyepos.shape[1])
        for i in range(eyepos.shape[1]):
            idist = myfunctions.Eucdist( fixpos[0],  fixpos[1], eyepos[0,i], eyepos[1,i])
            itarg = myfunctions.Eucdist(targpos[0], targpos[1], eyepos[0,i], eyepos[1,i])
            #dist2fix.append(idist)
            #dist2targ.append(itarg)
            dist2fix[i] = idist
            dist2targ[i] = itarg
        edat[sub][trl]['dist2fix']  = dist2fix
        edat[sub][trl]['dist2targ'] = dist2targ
        
        
#plt.figure();
#plt.plot(dist2fix)
#plt.axvline(dummy[0]['triggers']['array'][1], ls = 'dashed', color = '#636363');
#plt.axvline(dummy[0]['triggers']['cue'][1], ls = 'dashed', color = '#fc9272');
#plt.axvline(dummy[0]['behaviour']['fsac'][0]-dummy[0]['behaviour']['fstime'], ls = 'dashed', color = '#3182bd')
#
#plt.figure();
#plt.plot(dist2targ);
#plt.axvline(dummy[0]['triggers']['array'][1], ls = 'dashed', color = '#636363');
#plt.axvline(dummy[0]['triggers']['cue'][1], ls = 'dashed', color = '#fc9272');
#plt.axvline(dummy[0]['behaviour']['fsac'][0]-dummy[0]['behaviour']['fstime'], ls = 'dashed', color = '#3182bd')
        
#%%
#plot of example trial to send to Nick for help

dummy     = copy.deepcopy(edat[1])
sacctask  = [x for x in range(len(dummy)) if dummy[x]['behaviour']['task'] == 2.0]
dist2fix  = dummy[sacctask[5]]['dist2fix']
dist2targ = dummy[sacctask[5]]['dist2targ']
targpos = dummy[sacctask[5]]['behaviour']['targlocpix'].tolist()

targdistfromfix = myfunctions.Eucdist(targpos[0], targpos[1], fixpos[0], fixpos[1])


arrind    = dummy[sacctask[5]]['triggers']['array'][1] 
fsacind   = dummy[sacctask[5]]['behaviour']['fsac'][0]- dummy[sacctask[5]]['behaviour']['fstime']



plt.figure();
plt.plot(dist2fix,   color = '#41ab5d', label = 'distance from fixation point');
plt.axvline(arrind,  color = '#bdbdbd', ls = 'dashed', label = 'onset of array')
plt.axvline(fsacind, color = '#4292c6', ls = 'dashed', label = 'onset of saccade')
plt.axhline(targdistfromfix, color = '#ef3b2c', ls = 'dashed', label = 'dist of target from fixation')
plt.legend(loc = 4)
plt.xlim([1800,2360])
plt.title('euclidian distance of gaze from fixation (in pixels)')
plt.savefig(os.path.join(workingfolder, 'exampletrial_distancefromfixation.png') )


plt.figure();
plt.plot(dist2targ,   color = '#41ab5d', label = 'distance from target');
plt.axvline(arrind,  color = '#bdbdbd', ls = 'dashed', label = 'onset of array')
plt.axvline(fsacind, color = '#4292c6', ls = 'dashed', label = 'onset of saccade')
plt.legend(loc = 4)
plt.xlim([1800,2360])
plt.title('euclidian distance of gaze from target (in pixels)')
plt.savefig(os.path.join(workingfolder, 'exampletrial_distancefromtarget.png') )















#%%


# need to go through trials and check the saccades that have been made

dummy = copy.deepcopy(edat[2])

sacctask = [x for x in range(len(dummy)) if dummy[x]['behaviour']['task'] == 2.0]
tmptrl = copy.deepcopy(dummy[sacctask[0]])
    
plt.figure();
plt.plot(tmptrl['x'])
plt.axvline(tmptrl['triggers']['cue'][1], ls = 'dashed', color = '#fc9272')
plt.axvline(tmptrl['triggers']['array'][1], ls = 'dashed', color = '#636363')
plt.axvline(tmptrl['behaviour']['fsac'][0]-tmptrl['behaviour']['fstime'], ls = 'dashed', color = '#3182bd')





plt.figure();
plt.plot(edat[1][sacctask[5]]['dist2targ'], label = 'eucdist raw')
#plt.plot(np.diff(edat[1][sacctask[5]]['dist2targ']), color = 'red', label = 'diff')
plt.axvline(x = 1831, ls = 'dashed', color = '#636363')
plt.axvline(x = 2066, ls = 'dashed', color = '#636363')
plt.legend(loc = 4)







