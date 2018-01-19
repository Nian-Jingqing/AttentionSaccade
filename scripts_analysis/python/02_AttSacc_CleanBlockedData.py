#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:03:17 2018

@author: Sammi Chekroud
"""

import numpy as np
import os
import copy
import cPickle
workingfolder = '/home/sammirc/Experiments/Nick/AttentionSaccade' #workstation directory
os.chdir(workingfolder)
import myfunctions

np.set_printoptions(suppress = True)


#%%

eyedir        = os.path.join(workingfolder, 'eyes')
raw_dir       = os.path.join(eyedir, 'raw_data')
parsed_dir    = os.path.join(eyedir, 'blocked_data')
cleaned_dir   = os.path.join(eyedir, 'gaze_cleaned')

sublist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

sublist = [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13] #3 has a problem with data, needs fixing manually in conversion. subject 4 is downsampled. need to add a check for upsampling data and to what frequency

#sublist = [1, 2,    4, 5, 6, 7, 8, 9] #subject 3 has 961 trials in eyetracker data, and 960 in behavioural - check with nick. for now, remove from analysis
#re subject 3: seems to say length of it is 961, but actually edatasub3[961] is out of bounds, so doesn't exist. edatasub3[0:960] will rectify


parts   = ['a','b']

#re subject 3: seems to say length of it is 961, but actually edatasub3[961] is out of bounds, so doesn't exist. edatasub3[0:960] will rectify
#
#epsubs       = [3, 4, 5, 6,9] #subjects who did the task in EP
#ohbasubs     = [1, 2, 7 ,8]   #subjects who did the task in OHBA
#hrefsubs     = [7, 8]
#lowsamp_subs = [4]
#%%

list_fnames = sorted(os.listdir(parsed_dir))


for i in range(len(list_fnames)):
    fileid = list_fnames[i]
    print '\nloading parsed data from pickle'
    with open(os.path.join(parsed_dir, fileid), 'rb') as handle:
        ds = cPickle.load(handle)
    print 'finished loading data'
    
    nblocks = len(ds)
    ds = myfunctions.find_missing_periods(ds, nblocks) #find all missing data chunks in the signal
    
    count = 1
    for block in ds:
        print 'cleaning gaze data for block %02d/%02d'%(count, nblocks)
        for trace in ['lx', 'ly', 'rx', 'ry']: #iterate over the eyes and their x & y traces
            block = myfunctions.interpolateBlinks_Blocked(block, trace)
        count+=1
    subjfname = fileid.split('.')[0]
    pickname = subjfname + '_cleanedGaze.pickle'
    print 'saving cleaned data to pickle'
    with open(os.path.join(cleaned_dir, pickname), 'w') as handle:
        cPickle.dump(ds,handle)
        print 'done!'

