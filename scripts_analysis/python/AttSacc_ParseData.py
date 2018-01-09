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

list_fnames = sorted(os.listdir(os.path.join(eyedat, 'raw_data')))
    
#%%
for fileid in range(len(list_fnames)): # iterate over all files in the raw data folder
    print 'working on file %02d/%02d' %(fileid+1, len(list_fnames))
    fname = list_fnames[fileid] #get file name
    
    d = open(os.path.join(eyedat, 'raw_data', fname), 'r') #io open the file
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
    if fname == 'AttSacc_S03.asc': start_inds = start_inds[0:960]
    len(start_inds)
    fstart_ind = start_inds[0] #first time that the start recording message is seen
    
    end_inds   = [x for x in range(len(split_d)) if len(split_d[x]) == 7 and split_d[x][0] == 'END']
    if fname == 'AttSacc_S03.asc': end_inds = end_inds[0:960]
    len(end_inds) # again, if stopped/started recording for each trial, then len(end_inds) == n(trials)
    trig_ends = np.add(end_inds, 1) # add 1 to get the line where the trigger for end of a trial is sent
    fend_ind      = end_inds[-1] 
    fend_trig_ind = fend_ind + 1 
    
    if fileid in [0,1]: ntrials, trialsperblock = 1920, 80
    else: ntrials, trialsperblock = 960, 80
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
    
    rep_inds  = np.repeat(np.arange(nblocks),trialsperblock)
    #plt.hist(rep_inds, bins = nblocks) # all bars should be the same height now if its work? @ trials per block (80)
       
    
    iblock = {
    'trackertime': [],
    'av_x': [], 'av_y': [], 'av_p': [],
    'lx'  : [], 'ly'  : [], 'lp'  : [],
    'rx'  : [], 'ry'  : [], 'rp'  : [],
    'Efix': [], 'Sfix': [],
    'Esac': [], 'Ssac': [],
    'Eblk': [], 'Sblk': [],
    'Msg' : []
    }
    dummy = copy.deepcopy(iblock)
    blocked_data = np.repeat(dummy, nblocks)
    
    for i in np.arange(nblocks):
        inds = np.squeeze(np.where(rep_inds == i))
        iblock_data = copy.deepcopy(iblock)
        for ind in inds: #add trialwise info into the blocked structure, continuous data rather than sectioned
            iblock_data['trackertime'].append(trackertime[ind])
            iblock_data['av_x'].append(av_x[ind])
            iblock_data['av_y'].append(av_y[ind])
            iblock_data['av_p'].append(av_p[ind])
            iblock_data['lx'].append(lx[ind])
            iblock_data['ly'].append(ly[ind])
            iblock_data['lp'].append(lp[ind])
            iblock_data['rx'].append(rx[ind])
            iblock_data['ry'].append(ry[ind])
            iblock_data['rp'].append(rp[ind])
            iblock_data['Efix'].append(Efix[ind])
            iblock_data['Sfix'].append(Sfix[ind])
            iblock_data['Esac'].append(Esac[ind])
            iblock_data['Ssac'].append(Ssac[ind])
            iblock_data['Eblk'].append(Eblk[ind])
            iblock_data['Sblk'].append(Sblk[ind])
            iblock_data['Msg'].append(Msg[ind])
            blocked_data[i] = iblock_data
            
    
    # this is the structure of blocked_data:
    # len(blocked_data) = nblocks
    # len(blocked_data[block])                    = 17 (17 variables within it, e.g. trackertime)
    # len(blocked_data[block]['variable'])        = ntrials
    # len(blocked_data[block]['variable'][trial]) = trial length
    
    for block in range(len(blocked_data)): #concatenate trialwise signals into whole block traces to make artefact removal easier
        blocked_data[block]['trackertime'] = np.hstack(blocked_data[block]['trackertime'])
        blocked_data[block]['av_x']        = np.hstack(blocked_data[block]['av_x']       )
        blocked_data[block]['av_y']        = np.hstack(blocked_data[block]['av_y']       )
        blocked_data[block]['av_p']        = np.hstack(blocked_data[block]['av_p']       )
        blocked_data[block]['lx']          = np.hstack(blocked_data[block]['lx']         )
        blocked_data[block]['ly']          = np.hstack(blocked_data[block]['ly']         )
        blocked_data[block]['lp']          = np.hstack(blocked_data[block]['lp']         )
        blocked_data[block]['rx']          = np.hstack(blocked_data[block]['rx']         )
        blocked_data[block]['ry']          = np.hstack(blocked_data[block]['ry']         )
        blocked_data[block]['rp']          = np.hstack(blocked_data[block]['rp']         )
        
    # # # # save subject data (trials concatenated into blocks) to pickle file to read in to next script
    
    print 'saving blocked data to pickle'    
    pickname = fname.split('.')[0] + '_blocked.pickle' #remove the previous file ending (.asc), and add new one
    with open(os.path.join(workingfolder, eyedat, 'blocked_data', pickname), 'w') as handle:
        cPickle.dump(blocked_data, handle)
    print 'done'
        
        
#%%


temp = copy.deepcopy(blocked_data)
for block in range(len(temp)):
    temp_ttime = temp[block]['trackertime']
    lx = temp[block]['lx']
    ly = temp[block]['ly']
    lp = temp[block]['lp']
    rx = temp[block]['rx']
    ry = temp[block]['ry']
    rp = temp[block]['rp']


    
plt.figure()
plt.plot(blocked_data[0]['trackertime'], blocked_data[0]['lx'], color = 'green')
plt.plot(blocked_data[0]['trackertime'], blocked_data[0]['rx'], color = 'blue')
plt.title('trial eye-x data across whole of block 1, triggers are blinks')
for i in range(len(blocked_data[0]['Sblk'])):
    if len(blocked_data[0]['Sblk'][i]) > 0:
        for blink in range(len(blocked_data[0]['Sblk'][i])):
            plt.axvline(int(blocked_data[0]['Sblk'][i][blink][2]), ls = 'dashed', color = 'red')
    

