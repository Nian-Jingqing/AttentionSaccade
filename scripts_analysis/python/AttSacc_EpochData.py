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
os.chdir(workingfolder)
import BCEyes

np.set_printoptions(suppress = True)

eyedir        = os.path.join(workingfolder, 'eyes')
cleaned_dir   = os.path.join(eyedir, 'gaze_cleaned')

list_fnames= sorted(os.listdir(cleaned_dir))

#%%

# read in a data file to work on and test epoching with

fname = os.path.join(cleaned_dir, list_fnames[3])

print '\nloading parsed data from pickle'
with open(os.path.join(fname), 'rb') as handle:
    ds = cPickle.load(handle)
print 'finished loading data'

with open('')


block= copy.deepcopy(ds[0])

events = copy.deepcopy(block['Msg'])

trl_events = copy.deepcopy(events[0])


cue_inds = []
for trl in events:
    for x,y in np.ndenumerate(trl):
        for a,b in np.ndenumerate(y):
            if "_CUE" in b:
                cue_inds.append(int(y[1]))
            