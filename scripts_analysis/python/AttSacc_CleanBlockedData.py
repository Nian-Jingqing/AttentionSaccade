# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 19:28:32 2018

@author: Sammi Chekroud
"""

import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import re
import os
import copy
import cPickle

workingfolder = '/home/sammirc/Experiments/Nick/AttentionSaccade' #workstation directory
os.chdir(workingfolder)

import myfunctions

%matplotlib
np.set_printoptions(suppress = True)
#%%
eyedat        = os.path.join(workingfolder, 'eyes')
sublist = [1, 2, 3, 4, 5, 6, 7, 8, 9]
parts   = ['a','b']

epsubs       = [3, 4, 5, 6,9] #subjects who did the task in EP
ohbasubs     = [1, 2, 7 ,8]   #subjects who did the task in OHBA
hrefsubs     = [7, 8]
lowsamp_subs = [4]

list_fnames = sorted(os.listdir(os.path.join(eyedat, 'blocked_data')))
#%%
for fid in range(len(list_fnames)):
    if fid not in [4,5]: # don't look at subject 4 as the data is a lower sample rate and hasn't been fixed yet
        print 'working on file %02d/%02d' %(fid+1, len(list_fnames))
    
        #load in the blocked data    
        fname    = list_fnames[fid]
        pickname = fname.split('.')[0] + '_gazeclean.pickle' #remove the previous file ending (.asc), and add new one
        if not os.path.exists(os.path.join(workingfolder, eyedat, 'gaze_cleaned', pickname)):
            print '\nloading blocked data from pickle'
            with open(os.path.join(eyedat, 'blocked_data', fname), 'rb') as handle:
                ds = cPickle.load(handle)
            print 'finished loading data'
            
            # ds contains the data for this file. it is an array of dictionaries len(ds) = nblocks for the file
            # keys for each block dictionary are:
            # Msg           - contains all the messages sent in this block (triggers for each trial, etc)
            # Sfix          - start sample points for fixations (in tracker time, using SR research methods)
            # Efix          - event coded version of fixations
            # Sblk          - start sample points for blinks (in tracker time, using SR research methods)
            # Eblk          - event coded version of blinks
            # Ssac          - start sample points of saccades (in tracker time, using SR research methods)
            # Esac          - event coded version of saccades (SR research methods)
            # trackertime   - timepoints in tracker time
            # lx            - time-series for the left x value
            # ly            - time-series for the left y value
            # lp            - time-series for the left pupil value
            # rx            - time-series for the right x value
            # ry            - time-series for the right y value
            # rp            - time-series for the right pupil value
            # av_x          - time-series averaged (pre-cleaning) x value across eyes     (this is probably noisier than necessary as it's from raw data)
            # av_y          - time-series averaged (pre-cleaning) y value across eyes     (this is probably noisier than necessary as it's from raw data)
            # av_p          - time-series averaged (pre-cleaning) pupil value across eyes (this is probably noisier than necessary as it's from raw data)
            if fid in range(2):
                nblocks = 24
            else:
                nblocks = 12
            ds = myfunctions.find_missing_periods(ds,nblocks) #identify missing periods of data for subsequent cleaning
            count = 1
            for block in ds: #iterate over all blocks in the data
                print 'cleaning gaze data for block %02d/%02d'%(count, nblocks)
                for trace in ['lx', 'ly', 'rx', 'ry']: #iterate over the eyes and their x & y traces
                    block = myfunctions.interpolateBlinks_Blocked(block, trace)
                count+=1
            #save the cleaned gaze data to pickle
            print 'saving cleaned blocked data to pickle'    
            with open(os.path.join(workingfolder, eyedat, 'gaze_cleaned', pickname), 'w') as handle:
                cPickle.dump(ds, handle)
            print 'done'

