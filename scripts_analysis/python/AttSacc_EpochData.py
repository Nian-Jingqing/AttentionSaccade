#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:59:40 2018

@author: sammirc
"""



import numpy as np
import os
import copy
import cPickle
workingfolder = '/home/sammirc/Experiments/Nick/AttentionSaccade' #workstation directory
workingfolder = '/Users/user/Desktop/Experiments/Nick/AttentionSaccade' #laptop directory

os.chdir(workingfolder)
import BCEyes

np.set_printoptions(suppress = True)

eyedir        = os.path.join(workingfolder, 'eyes')
#cleaned_dir   = os.path.join(eyedir, 'gaze_cleaned')

#list_fnames= sorted(os.listdir(cleaned_dir))

#%%

# read in a data file to work on and test epoching with

fname = os.path.join(eyedir, 'blocked_data/AttSacc_S03a_blocked.pickle')

print '\nloading parsed data from pickle'
with open(os.path.join(fname), 'rb') as handle:
    ds = cPickle.load(handle)
print 'finished loading data'

ds = BCEyes.find_missing_periods(ds, len(ds))

for block in ds:
    for trace in ['lx','ly','rx','ry']:
        block = BCEyes.interpolateBlinks_Blocked(block, trace)




block= copy.deepcopy(ds[0])

events = copy.deepcopy(block['Msg'])

trl_events = copy.deepcopy(events[0])


trig_inds = []

for trl in events:
    for x,y in np.ndenumerate(trl):
        for a,b in np.ndenumerate(y):
            if '_CUE' in b:
                trig_inds.append(int(trl[x[0]][1]))

            
# get the time-series indices of when the cue appears (for indexing the traces)
triginds = []
for x,y in np.ndenumerate(trig_inds):
    triginds.append(int(np.squeeze(np.where(y == block['trackertime'])))) 
            
            
            
            
            
            
            