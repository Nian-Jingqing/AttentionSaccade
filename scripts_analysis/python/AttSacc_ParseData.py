# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:52:54 2017

@author: sammirc
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

#from mpl_toolkits.mplot3d import Axes3D
#%matplotlib inline
%matplotlib
np.set_printoptions(suppress = True)

import multiprocessing
n_cores = multiprocessing.cpu_count() #8 cores on workstation

#%%

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

#re subject 3: seems to say length of it is 961, but actually edatasub3[961] is out of bounds, so doesn't exist. edatasub3[0:960] will rectify

epsubs       = [3, 4, 5, 6,9] #subjects who did the task in EP
ohbasubs     = [1, 2, 7 ,8]   #subjects who did the task in OHBA
hrefsubs     = [7, 8]
lowsamp_subs = [4]
#%%

list_fnames = sorted(os.listdir(eyedat))

for i in list_fnames:


fname = list_fnames[0]

d = open(os.path.join(eyedat, fname), 'r') #io open the file
raw_d = d.readlines() #read the data from file 
d.close() #close file

split_d = []  # split each line into separate parts instead of one long string
for i in range(len(raw_d)):
    tmp = raw_d[i].split()
    split_d.append(tmp)

#get all the inds for lines where 'START  TSTAMP ....' is seen (this is the start of a recording)
#if somebody has stopped/started the recording at every trial, then len(start_inds) should be the number of trials
# e.g. here, len(start_inds == 1920)
start_inds = [x for x in range(len(split_d)) if len(split_d[x]) == 6 and split_d[x][0] == 'START'] 
len(start_inds)
fstart_ind = start_inds[0] #first time that the start recording message is seen

end_inds   = [x for x in range(len(split_d)) if len(split_d[x]) == 7 and split_d[x][0] == 'END']
len(end_inds) # again, if stopped/started recording for each trial, then len(end_inds) == n(trials)
trig_ends = np.add(end_inds, 1) # add 1 to get the line where the trigger for end of a trial is sent
fend_ind      = end_inds[-1] 
fend_trig_ind = fend_ind + 1 

ntrials, trialsperblock = 1920, 80
nblocks = ntrials/trialsperblock
blocked_data = []

count = 1
i = 0
for i in range(len(start_inds)):
    start_line = start_inds[i]
    end_line   = end_inds[i]
    
    temp = split_d[start_line:end_line]
    #len 3 as triggers follow format of ['MSG', timestamp, trigger] :
    temp_events = [x for x in range(len(temp)) if temp[x][0] == 'MSG' and len(temp[x]) == 3]




fsamp_rec = split_d[fstart_ind][1] #this is the sample value for the first record start of the file

