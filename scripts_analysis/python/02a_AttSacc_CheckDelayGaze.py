#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 13:10:18 2018

@author: sammirc
"""

import numpy as np
import pandas as pd
import os.path as op
import matplotlib as mpl
from matplotlib import pyplot as plt
import os
import copy
import cPickle
import BCEyes
import glob
np.set_printoptions(suppress = True)
%matplotlib


workingfolder = '/home/sammirc/Experiments/Nick/AttentionSaccade' #workstation directory
os.chdir(workingfolder)


behaviour     = op.join(workingfolder, 'behaviour', 'csv')
eyedir        = op.join(workingfolder, 'eyes')
cleaned_dir   = op.join(eyedir, 'gaze_cleaned')


os.chdir(behaviour)
list_fnames = sorted(glob.glob('*.csv')) #list only .csv files for behavioural data
os.chdir(workingfolder)
#%%

#task in EP:   3, 4, 5, 6, 9, 
#task at OHBA: 1, 2, 7, 8, 10, 11, 12, 13, 14

EPsubs   = ['AttSacc_S03a', 'AttSacc_S03b',
            'AttSacc_S04a', 'AttSacc_S04b',
            'AttSacc_S05a', 'AttSacc_S05b',
            'AttSacc_S06a', 'AttSacc_S06b',
            'AttSacc_S09a', 'AttSacc_S09b']

OHBAsubs = ['AttSacc_S01' , 'AttSacc_S02' , 
            'AttSacc_S07a', 'AttSacc_S07b',
            'AttSacc_S08a', 'AttSacc_S08b',
            'AttSacc_S10a', 'AttSacc_S10b',
            'AttSacc_S11a', 'AttSacc_S11b',
            'AttSacc_S12a', 'AttSacc_S12b',
            'AttSacc_S13a', 'AttSacc_S13b',
            'AttSacc_S14a', 'AttSacc_S14b']

fid       = list_fnames[6]
fid_sub   = fid.split('.')[0]

beh_fname = op.join(behaviour, fid)
eye_fname = op.join(cleaned_dir, fid_sub+'_parsed_cleanedGaze.pickle')


print '\nloading parsed data from pickle'
with open(eye_fname, 'rb') as handle:
    ds = cPickle.load(handle)
print 'finished loading data'

#read behavioural data to identify saccade blocks and trials
df = pd.read_csv(beh_fname, sep = ',', header = 0)

if fid_sub in ['AttSacc_S01', 'AttSacc_S02']:
    nblocks, ntrials, nmeas = 24, 80, 2
else:
    nblocks, ntrials, nmeas = 12, 80, 2 #nmeas = number of things to extract from eye data, here just x and y

blockid       = np.arange(1,nblocks+1)
blockid       = np.repeat(blockid, ntrials)
df['block']   = blockid

sacctrl_df    = df.query('task == 2')
saccblocks    = pd.unique(sacctrl_df.block)
saccblockinds = np.subtract(saccblocks, 1)

#pull out only the saccade blocks of the task

sacc_ds       = copy.deepcopy(ds[saccblockinds])

#%% loop over saccade blocks and check that gaze is within 2 degrees of fixation during delay
ppd             = 72 #pixels per degree
gazethresh      = ppd * 2
baseline        = np.divide([1920,1080], 2) # coords for the centre of the screen
baselinerange   = 100 #100ms prior to trigger of interest

for i in range(len(sacc_ds)):
    blockind     = i
    blocknum     = saccblocks[i]
    block        = sacc_ds[i]
    
    events       = copy.deepcopy(block['Msg'])
    events       = np.array([x for event in events for x in event])
    cuetrig      = '_CUE'
    targtrig     = '_ARR'
    
    cuetriginds  = []
    targtriginds = []
    
    for x,y in np.ndenumerate(events):
        if cuetrig in y[2]:
            cuetriginds.append(x[0])
        elif targtrig in y[2]:
            targtriginds.append(x[0])
    
    #get just relevant triggers
    cuetrigs  = events[cuetriginds]
    targtrigs = events[targtriginds]
    
    #find start and end points of the interval for each trial (trackertime)
    starts_tt = []
    ends_tt   = []
    for x,y in np.ndenumerate(cuetrigs):
        starts_tt.append(int(y[1]))
    for x,y in np.ndenumerate(targtrigs):
        ends_tt.append(int(y[1]))
        
    #get starts in sample time not trackertime
    starts, ends = [], []
    for x,y in np.ndenumerate(starts_tt):
        ind = int(np.where(block['trackertime']==y)[0])
        starts.append(ind)
    for x,y in np.ndenumerate(ends_tt):
        ind = int(np.where(block['trackertime']==y)[0])
        ends.append(ind)
    
    usable = np.zeros([ntrials], dtype = int) #premake list of usable trials indices for block
    for x in range(len(starts)):
        start = starts[x] - 200
        end   = ends[x]

        trlepoch      = np.empty([ 4, len(np.arange(start,end)) ]) #lx, ly, rx, ry for epoch in the same array
        trlepochtimes = np.subtract(range(len(np.arange(start,end))),200) #timepoints within epoch
        
        #get the data wanted
        trlepoch[0,:] = block['lx'][start:end]
        trlepoch[1,:] = block['ly'][start:end]
        trlepoch[2,:] = block['rx'][start:end]
        trlepoch[3,:] = block['ry'][start:end]
        
        #baseline prior to averaging across the eyes (should make the average eye more reliable)
        baselinewindow = [ int(np.where(trlepochtimes==0)[0]), int(np.where(trlepochtimes==0)[0]) - baselinerange]
        lx_bl = np.nanmedian(trlepoch[0,:][baselinewindow[1]:baselinewindow[0]])
        ly_bl = np.nanmedian(trlepoch[1,:][baselinewindow[1]:baselinewindow[0]])
        rx_bl = np.nanmedian(trlepoch[2,:][baselinewindow[1]:baselinewindow[0]])
        ry_bl = np.nanmedian(trlepoch[3,:][baselinewindow[1]:baselinewindow[0]])
        
        trlepoch[0,:] = np.add(np.subtract(trlepoch[0,:], lx_bl),baseline[0])
        trlepoch[1,:] = np.add(np.subtract(trlepoch[1,:], ly_bl),baseline[1])
        trlepoch[2,:] = np.add(np.subtract(trlepoch[2,:], rx_bl),baseline[0])
        trlepoch[3,:] = np.add(np.subtract(trlepoch[3,:], ry_bl),baseline[1])
        
        trlavg = np.empty([2, len(np.arange(start,end))])
        trlavg[0,:] = np.nanmean(trlepoch[[0,2], :], axis = 0)
        trlavg[1,:] = np.nanmean(trlepoch[[1,3], :], axis = 0)
        
        #calculate distance of gaze from fixation
        trldistfromfix = np.empty([len(np.arange(start,end))])
        for y in range(trlavg.shape[1]):
            trldistfromfix[y] = BCEyes.Eucdist(baseline[0], baseline[1], trlavg[0,y], trlavg[1,y])
        over_thresh = np.where(trldistfromfix > ppd)[0]
        


#%%

# median filters can be found:
# sp.ndimage.filters.median_filter(array, size = [insert size of the window you want to use for filter])













