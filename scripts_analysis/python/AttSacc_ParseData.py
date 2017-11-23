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
    fname = list_fnames[i]
    
#%%

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
trackertime = []
lx   = []; rx   = []
ly   = []; ry   = []
lp   = []; rp   = []
av_x = []; av_y = []
av_p = []

Efix = []; Sfix = []
Esac = []; Ssac = []
Eblk = []; Sblk = []
Msg  = []

count = 1
#i = 0
for i in range(len(start_inds)): #get trialwise data
    start_line = start_inds[i]
    end_line   = end_inds[i]
    
    itrl = split_d[start_line:end_line]
    #len 3 as triggers follow format of ['MSG', timestamp, trigger] :
    itrl_event_inds = [x for x in range(len(itrl)) if itrl[x][0] == 'MSG']        # get the line indices where trigs sent
    itrl_events     = [itrl[x] for x in itrl_event_inds]                              # get the actual triggers
    itrl_fix_inds   = [x for x in range(len(itrl)) if itrl[x][0] == 'EFIX' or itrl[x][0] == 'SFIX']
    itrl_fix        = [itrl[x] for x in itrl_fix_inds]
    itrl_sac_inds   = [x for x in range(len(itrl)) if itrl[x][0] == 'ESACC' or itrl[x][0] == 'SSACC']
    itrl_sac        = [itrl[x] for x in itrl_sac_inds]
    itrl_blink_inds = [x for x in range(len(itrl)) if itrl[x][0] == 'EBLINK' or itrl[x][0] == 'SBLINK']
    itrl_blink      = [itrl[x] for x in itrl_blink_inds]  
    
    itrl_data       = [itrl[x] for x in range(len(itrl)) if
                       x not in itrl_event_inds and
                       x not in itrl_fix_inds   and
                       x not in itrl_sac_inds   and
                       x not in itrl_blink_inds    ] # get all non-trigger lines
    
    itrl_data = itrl_data[6:] #remove first five lines which are filler from the eyetracker. temp_data now only contains the raw signal

    itrl_data = np.vstack(itrl_data) # shape of this should be the number of columns of data in the file!
    
    #before you can convert to float, need to replace missing data where its '.' as nans (in pupil this is '0.0')
    eyecols = [1,2,4,5] #leftx, lefty, rightx, right y col indices
    for col in eyecols:
        missing_inds = np.where(itrl_data[:,col] == '.') #find where data is missing in the gaze position, as probably a blink (or its missing as lost the eye)
        for i in missing_inds:
            itrl_data[i,col] = np.NaN #replace missing data ('.') with NaN
            itrl_data[i,3]   = np.NaN #replace left pupil as NaN (as in a blink)
            itrl_data[i,6]   = np.NaN #replace right pupil as NaN (as in a blink)

    
    
    itrl_data = itrl_data.astype(np.float) #convert data from string to floats for computations
     
    #for binocular data, the shape is:
    # columns: time stamp, left x, left y, left pupil, right x, right y, right pupil
    
    itrl_trackertime = itrl_data[:,0]
    itrl_lx, itrl_ly, itrl_lp = itrl_data[:,1], itrl_data[:,2], itrl_data[:,3]
    itrl_rx, itrl_ry, itrl_rp = itrl_data[:,4], itrl_data[:,5], itrl_data[:,6]
    
    #average data across the eyes --- take the nanmean though in case of missing data. we'll still save the independent eyes though as a sanity check
    itrl_x = np.vstack([itrl_lx, itrl_rx])
    itrl_x = np.nanmean(itrl_x, axis = 0)
    itrl_y = np.vstack([itrl_ly, itrl_ry])
    itrl_y = np.nanmean(itrl_y, axis = 0)    
    itrl_p = np.vstack([itrl_lp, itrl_rp])
    itrl_p = np.nanmean(itrl_p, axis = 0)


    # split Efix/Sfix and Esacc/Ssacc into separate lists
    itrl_efix = [itrl_fix[x] for x in range(len(itrl_fix)) if
                 itrl_fix[x][0] == 'EFIX']
    itrl_sfix = [itrl_fix[x] for x in range(len(itrl_fix)) if
                 itrl_fix[x][0] == 'SFIX']
                
    itrl_ssac = [itrl_sac[x] for x in range(len(itrl_sac)) if
                 itrl_sac[x][0] == 'SSACC']
    itrl_esac = [itrl_sac[x] for x in range(len(itrl_sac)) if
                 itrl_sac[x][0] == 'ESACC']
                 
    itrl_sblk = [itrl_blink[x] for x in range(len(itrl_blink)) if
                 itrl_blink[x][0] == 'SBLINK']
    itrl_eblk = [itrl_blink[x] for x in range(len(itrl_blink)) if
                 itrl_blink[x][0] == 'EBLINK']

    #append to the collection of all data now
    trackertime.append(itrl_trackertime)
    lx.append(itrl_lx)
    ly.append(itrl_ly)
    rx.append(itrl_rx)
    ry.append(itrl_ry)
    lp.append(itrl_lp)
    rp.append(itrl_rp)
    av_x.append(itrl_x)
    av_y.append(itrl_y)
    av_p.append(itrl_p)
    Efix.append(itrl_efix)
    Sfix.append(itrl_sfix)
    Ssac.append(itrl_ssac)
    Esac.append(itrl_esac)
    Sblk.append(itrl_sblk)
    Eblk.append(itrl_eblk)
    Msg.append(itrl_events)

    











#fsamp_rec = split_d[fstart_ind][1] #this is the sample value for the first record start of the file

