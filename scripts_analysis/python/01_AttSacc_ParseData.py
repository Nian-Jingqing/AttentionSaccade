# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 11:51:17 2018

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
blocked_dir   = os.path.join(eyedir, 'blocked_data')

sublist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

sublist = [1, 2, 5, 6, 7, 8, 9] #3 has a problem with data, needs fixing manually in conversion. subject 4 is downsampled. need to add a check for upsampling data and to what frequency

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

list_fnames = sorted(os.listdir(raw_dir))


for i in range(len(list_fnames)):
    fileid = list_fnames[i]
    if fileid not in ['AttSacc_S04a.asc', 'AttSacc_S04b.asc', 'AttSacc_S03a.asc']: #ignore these files for now as need manually fixing at the moment
        print('\nparsing file %02d/%02d'%(i, len(list_fnames)))
        eye_fname = os.path.join(raw_dir, fileid)
        if fileid in ['AttSacc_S01.asc','AttSacc_S02.asc']:
            data = myfunctions.parse_eye_data(eye_fname, block_rec = False, trial_rec = True, nblocks = 24, ntrials = 1920)
        else:
            data = myfunctions.parse_eye_data(eye_fname, block_rec = False, trial_rec = True, nblocks = 12, ntrials = 960)
        subjname = fileid.split('.')[0]
        pickname = subjname + '_parsed.pickle'
    
        
        print ' -- saving parsed data to pickle'    
        with open(os.path.join(blocked_dir, pickname), 'w') as handle:
            cPickle.dump(data, handle)
        print '\tdone!'
        