# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:22:19 2017

@author: sammirc
"""
import numpy as np
import pandas as pd
import scipy as sp
from scipy.signal import resample
import pygazeanalyser
from pygazeanalyser.edfreader import read_edf
from pygazeanalyser import gazeplotter
import re
import os
import mne
import copy
import cPickle
np.set_printoptions(suppress = True)

workingfolder = '/home/sammirc/Experiments/Nick/AttentionSaccade' #workstation directory
os.chdir(workingfolder)
import myfunctions #this is a script containing bespoke functions for this analysis pipeline
#%%
# set relevant directories and paths
eyedat        = os.path.join(workingfolder, 'eyes')
behdat        = os.path.join(workingfolder, 'behaviour/csv')
eegdat        = os.path.join(workingfolder, 'EEG') #only on workstation though, no eeg data on laptop
eyelist       = os.listdir(eyedat)
eyelist       = np.sort(eyelist)
behlist       = np.sort(os.listdir(behdat))
saccadedat    = os.path.join(workingfolder, 'saccades')
if not os.path.exists(saccadedat):
    os.mkdir(saccadedat)
sublist = [1, 2, 3, 4, 5, 6, 7, 8, 9]

#sublist = [1, 2,    4, 5, 6, 7, 8, 9] #subject 3 has 961 trials in eyetracker data, and 960 in behavioural - check with nick. for now, remove from analysis
#re subject 3: seems to say length of it is 961, but actually edatasub3[961] is out of bounds, so doesn't exist. edatasub3[0:960] will rectify

parts   = ['a','b']

epsubs       = [3, 4, 5, 6,9] #subjects who did the task in EP
ohbasubs     = [1, 2, 7 ,8]   #subjects who did the task in OHBA
hrefsubs     = [7, 8]
lowsamp_subs = [4]

#1KHz  sample: 1 5 6 7 8 9
#250Hz sample: 4

# specs for the task
resxy = (1920,1080)
scrsize = (60,34)
scrdist = 100 # cm
#pixpcm = np.mean([resxy[0]/scrsize[0],resxy[1]/scrsize[1]])
#samplerate = 1000.0 # Hz , sample every 1ms. subject 4 has 250Hz SR, all other 1kHz
samplerate = 250.0 #for subject 4

# Data files
sep = '\t' #value separator
EDFSTART = "_BEG"
EDFSTOP  = "END"
TRIALORDER = ["B(.*)_T(.*)_BEG","B(.*)_T(.*)_CUE","B(.*)_T(.*)_ARR","B(.*)_T(.*)_RESP","B(.*)_T(.*)_END"]
INVALCODE = np.NaN
ppd = 72 #pixels per degree in current study

#%%
edat = []
for sub in range(len(sublist)):
    print 'working on S%02d'%(sublist[sub])
    if sub in range(0,2): #subjects 1 and 2 only have one file per subject
        fname    = os.path.join(eyedat, 'AttSacc_S%02d.asc'%(sublist[sub]));
        edata    = read_edf(fname, EDFSTART, EDFSTOP, missing = np.NaN, debug = False)
    elif sub in range(2,9): #all other subjects have 2 files per sub
        fname    = os.path.join(eyedat, 'AttSacc_S%02d%s.asc'%(sublist[sub], parts[0]))
        edata    = read_edf(fname, EDFSTART, EDFSTOP, missing = np.NaN, debug = False)
        if sublist[sub]==3: edata = edata[0:960] #length of the data is 961, but edata[961] doesnt exist. there's an extra trial where it started to record block 13,but we dont have block 13.
        fname2   = os.path.join(eyedat, 'AttSacc_S%02d%s.asc'%(sublist[sub], parts[1]))
        edata2   = read_edf(fname2, EDFSTART, EDFSTOP, missing = np.NaN, debug = False)
    if sub == 4:#this is subject 5. in trial 568, the time array is shifted by 1 sample, and skips sample 2762. need to realign ['time'] on this trial
            #length of trial 568 is 2936
        edata[568]['time'] = np.arange(0, len(edata[568]['x'])) #set the time to align to length of the trial
        
    #subject 4 (edat[3]) was acquired at 250hz - upsample the x, y, pupil, time and trackertime for each trial    
    if sublist[sub] == 4:
        for i in range(len(edata)): #loop through all trials
            #original length of the vector, and the length that it would be if collected at `1khz (based on trackertime samples)
            o_length = len(edata[i]['trackertime'])
            n_length = edata[i]['trackertime'][-1] - edata[i]['trackertime'][0]            
            edata[i]['x']    = resample(edata[i]['x'], n_length, window = 'boxcar') #resample to actual length of the trial, which equates to 1Khz
            edata[i]['y']    = resample(edata[i]['y'], n_length, window = 'boxcar') #resample to actual length of the trial, which equates to 1Khz
            edata[i]['size'] = resample(edata[i]['size'], n_length, window = 'boxcar') #resample to actual length of the trial, which equates to 1Khz
            edata[i]['time'] = np.arange(1,n_length + 1) #account for zero indexing in this time vector
            edata[i]['trackertime'] = np.arange(edata[i]['trackertime'][0], edata[i]['trackertime'][-1]+1) #compensate for zero indexing
    
    
    #epoch the data around the cue appearance, take 200ms before and 800ms after (1s epoch for convenience of resampling later on
    print 'epoching the data relative to the cue'    
    for trial in range(len(edata)):
        #extract trigger times
        if len(edata[trial]['events']['msg']) == 3:  # saccade trial, lacking response trigger
            cuetrig  = edata[trial]['events']['msg'][1][0]
            arrtrig  = edata[trial]['events']['msg'][2][0]
        elif len(edata[trial]['events']['msg']) == 4: # attention trial, has response trigger
            cuetrig  = edata[trial]['events']['msg'][1][0]
            arrtrig  = edata[trial]['events']['msg'][2][0]
            resptrig = edata[trial]['events']['msg'][3][0]
            respind  = np.argmin(np.abs(arrtrig - edata[trial]['trackertime'])) #find sample nearest to trigger time
        
        #find sample closest to these trigger times
        #begind  = np.argmin(np.abs(begtrig  - edata[trial]['trackertime']))
        cueind  = np.argmin(np.abs(cuetrig  - edata[trial]['trackertime']))
        arrind  = np.argmin(np.abs(arrtrig  - edata[trial]['trackertime'])) 

        #take 200 samples before, and 800 samples after, if subject is sampled at 1khz
        #if sublist[sub] != 4:
        #    epoch                = np.arange(cueind-200, cueind+800) #extra 1 for the 0 indexing
        #    edata[trial]['x']    = edata[trial]['x'][epoch]   # epoch the x coord data
        #    edata[trial]['y']    = edata[trial]['y'][epoch]   # epoch the y coord data
        #    edata[trial]['size'] = edata[trial]['size'][epoch]# epoch the pupil data
        #    edata[trial]['time'] = edata[trial]['time'][epoch] - cueind #epoch the timescale too, and rescale relative to the trigger
    edat.append(edata)
print 'saving processed data in pickle format'

pickname = 'AttentionSaccade_9subs_epochcue.pickle'
with open(os.path.join(workingfolder, pickname), 'w') as handle:
    cPickle.dump(edat, handle)
print 'done!'

#%%  epoch around the array (i.e target presentation) -- most of this is a copy of code from above cell but it runs fast so who cares

edat = []
for sub in range(len(sublist)):
    print 'working on S%02d'%(sublist[sub])
    if sub in range(0,2): #subjects 1 and 2 only have one file per subject
        fname    = os.path.join(eyedat, 'AttSacc_S%02d.asc'%(sublist[sub]));
        edata    = read_edf(fname, EDFSTART, EDFSTOP, missing = np.NaN, debug = False)
    elif sub in range(2,9): #all other subjects have 2 files per sub
        fname    = os.path.join(eyedat, 'AttSacc_S%02d%s.asc'%(sublist[sub], parts[0]))
        edata    = read_edf(fname, EDFSTART, EDFSTOP, missing = np.NaN, debug = False)
        if sublist[sub]==3: edata = edata[0:960] #length of the data is 961, but edata[961] doesnt exist. there's an extra trial where it started to record block 13,but we dont have block 13.
        fname2   = os.path.join(eyedat, 'AttSacc_S%02d%s.asc'%(sublist[sub], parts[1]))
        edata2   = read_edf(fname2, EDFSTART, EDFSTOP, missing = np.NaN, debug = False)
    if sub == 4:#this is subject 5. in trial 568, the time array is shifted by 1 sample, and skips sample 2762. need to realign ['time'] on this trial
            #length of trial 568 is 2936
        edata[568]['time'] = np.arange(0, len(edata[568]['x'])) #set the time to align to length of the trial
        
    #subject 4 (edat[3]) was acquired at 250hz - upsample the x, y, pupil, time and trackertime for each trial    
    if sublist[sub] == 4:
        for i in range(len(edata)): #loop through all trials
            #original length of the vector, and the length that it would be if collected at `1khz (based on trackertime samples)
            o_length = len(edata[i]['trackertime'])
            n_length = edata[i]['trackertime'][-1] - edata[i]['trackertime'][0]            
            edata[i]['x']    = resample(edata[i]['x'], n_length, window = 'boxcar') #resample to actual length of the trial, which equates to 1Khz
            edata[i]['y']    = resample(edata[i]['y'], n_length, window = 'boxcar') #resample to actual length of the trial, which equates to 1Khz
            edata[i]['size'] = resample(edata[i]['size'], n_length, window = 'boxcar') #resample to actual length of the trial, which equates to 1Khz
            edata[i]['time'] = np.arange(1,n_length + 1) #account for zero indexing in this time vector
            edata[i]['trackertime'] = np.arange(edata[i]['trackertime'][0], edata[i]['trackertime'][-1]+1) #compensate for zero indexing
    
    
    #epoch the data around the cue appearance, take 200ms before and 800ms after (1s epoch for convenience of resampling later on
    print 'epoching the data relative to the cue'    
    for trial in range(len(edata)):
        #extract trigger times
        if len(edata[trial]['events']['msg']) == 3:  # saccade trial, lacking response trigger
            cuetrig  = edata[trial]['events']['msg'][1][0]
            arrtrig  = edata[trial]['events']['msg'][2][0]
        elif len(edata[trial]['events']['msg']) == 4: # attention trial, has response trigger
            cuetrig  = edata[trial]['events']['msg'][1][0]
            arrtrig  = edata[trial]['events']['msg'][2][0]
            resptrig = edata[trial]['events']['msg'][3][0]
            respind  = np.argmin(np.abs(arrtrig - edata[trial]['trackertime'])) #find sample nearest to trigger time
        
        #find sample closest to these trigger times
        #begind  = np.argmin(np.abs(begtrig  - edata[trial]['trackertime']))
        cueind  = np.argmin(np.abs(cuetrig  - edata[trial]['trackertime']))
        arrind  = np.argmin(np.abs(arrtrig  - edata[trial]['trackertime'])) 

        #take 200 samples before, and 800 samples after, if subject is sampled at 1khz
        if sublist[sub] != 4:
            epoch                = np.arange(cueind-200, cueind+800) #extra 1 for the 0 indexing
            edata[trial]['x']    = edata[trial]['x'][epoch]   # epoch the x coord data
            edata[trial]['y']    = edata[trial]['y'][epoch]   # epoch the y coord data
            edata[trial]['size'] = edata[trial]['size'][epoch]# epoch the pupil data
            edata[trial]['time'] = edata[trial]['time'][epoch] - cueind #epoch the timescale too, and rescale relative to the trigger
    edat.append(edata)
print 'saving processed data in pickle format'

pickname = 'AttentionSaccade_9subs_epochcue.pickle'
with open(os.path.join(workingfolder, pickname), 'w') as handle:
    cPickle.dump(edat, handle)
print 'done!'


