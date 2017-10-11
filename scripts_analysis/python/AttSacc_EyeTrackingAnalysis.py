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


#%% load in all data from pickle (this is combined behavioural and eyetracking data)

pickname = os.path.join(workingfolder, 'preprocessed_eyes_9subs.pickle')
print 'loading preprocessed data from pickle'
with open(os.path.join(workingfolder, pickname), 'rb') as handle:
    edat = cPickle.load(handle)
print 'done!'
#%%

# get saccade time for saccade task trials
# finds the closest saccade of a tral to the array trigger (when target appears)
# takes time at which it starts, and subtracts from array onset to get saccade time

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
